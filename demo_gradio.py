import os

#os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import numpy as np
import argparse

from PIL import Image
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from utils import load_model, get_num_frames, section_title_update, get_image_np_pt

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer = load_model()
high_vram = get_cuda_free_memory_gb(gpu) > 60

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, sampling, seed, total_latent_sections, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, section_keyframes, paddings):
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape if input_image is not None else end_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np, input_image_pt = get_image_np_pt(input_image, width, height)
        
        #Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        if end_image is not None:
            end_image_np, end_image_pt = get_image_np_pt(end_image, width, height)

        key_frames_pt, key_frames_np = [], []
        for i in range(total_latent_sections):
            section_keyframe_np, section_keyframe_pt = get_image_np_pt(section_keyframes[i], width, height)
            key_frames_pt.append(section_keyframe_pt)
            key_frames_np.append(section_keyframe_np)

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        if input_image_pt is None:
            end_latent = vae_encode(end_image_pt, vae)
            start_latent = torch.zeros_like(end_latent)
        else:
            start_latent = vae_encode(input_image_pt, vae)
            if end_image_pt is not None:
                end_latent = vae_encode(end_image_pt, vae)
            else:
                end_latent = torch.zeros_like(start_latent)

        key_frame_latents = []
        for key_frame_pt in key_frames_pt:
            if key_frame_pt is not None:
                key_frame_latent = vae_encode(key_frame_pt, vae)
                key_frame_latents.append(key_frame_latent)
            else:
                if sampling == 'inverse':
                    key_frame_latents.append(start_latent)
                elif sampling == 'vanilla':
                    key_frame_latents.append(end_latent)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np if input_image_np is not None else end_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        key_frame_image_encoder_last_hidden_states = []
        for key_frame_np in key_frames_np:
            if key_frame_np is not None:
                key_frame_image_encoder_output = hf_clip_vision_encode(key_frame_np, feature_extractor, image_encoder)
                key_frame_image_encoder_last_hidden_states.append(key_frame_image_encoder_output.last_hidden_state.to(transformer.dtype))
            else:
                key_frame_image_encoder_last_hidden_states.append(image_encoder_last_hidden_state.to(transformer.dtype))

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = get_num_frames(latent_window_size)

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        if end_image is not None and sampling == 'inverse':
            history_latents[:, :, :1] = end_latent.to(history_latents)
        if sampling == 'vanilla':
            history_latents[:, :, -1:] = start_latent.to(history_latents)
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            if sampling == 'inverse':
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            elif sampling == "vanilla":
                latent_paddings = [0, 1] + [2] * (total_latent_sections - 3) + [3]

        for i in range(total_latent_sections):
            if paddings[i] >= 0:
                latent_paddings[i] = paddings[i]

        for i, latent_padding in enumerate(latent_paddings):
            is_first_section = i == 0
            is_last_section = i == (total_latent_sections - 1)
            latent_padding_size = int(latent_padding * latent_window_size)

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            if sampling == 'inverse':
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = key_frame_latents[i].to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            elif sampling == 'vanilla':
                clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_indices_post, latent_indices, blank_indices, clean_latent_indices_pre = indices.split([16, 2, 1, latent_window_size, latent_padding_size, 1], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = key_frame_latents[i].to(history_latents)
                clean_latents_4x, clean_latents_2x, clean_latents_post = history_latents[:, :, -(1 + 2 + 16):, :, :].split([16, 2, 1], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Section {i+1}/{total_latent_sections}, Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=key_frame_image_encoder_last_hidden_states[i],
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            if sampling == 'inverse':
                if is_last_section and paddings[i] == 0:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                total_generated_latent_frames += int(generated_latents.shape[2])
                
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = num_frames

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            elif sampling == 'vanilla':
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = num_frames

                    current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, end_image, prompt, n_prompt, sampling, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, *args):
    global stream
    #assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    section_keyframes = args[0:32]
    paddings = args[32:64]

    async_run(worker, input_image, end_image, prompt, n_prompt, sampling, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, section_keyframes, paddings)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                end_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            sampling = gr.Radio(label="Sampling Method", choices=["inverse", "vanilla"], value="inverse")

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_latent_sections = gr.Slider(label="Total Video Length (Sections)", minimum=1, maximum=120, value=5, step=1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
            
            with gr.Group():
                first_total_sections = 5
                section_title = gr.Markdown(f'## 総セクション数は{first_total_sections}だよーん。')
                section_keyframes = []
                paddings = []
                for i in range(32):
                    with gr.Row():
                        section_keyframes.append(gr.Image(sources='upload', type="numpy", label="Image", height=320, visible= i < first_total_sections))
                        paddings.append(gr.Number(label=f"Section {i} Padding", minimum=-0.1, maximum=20, value=-0.1, step=0.1, visible= i < first_total_sections))

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, end_image, prompt, n_prompt, sampling, seed, total_latent_sections, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf] + section_keyframes + paddings
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

    total_latent_sections.change(
        fn=section_title_update,
        inputs=[total_latent_sections],
        outputs=[section_title] + section_keyframes + paddings,
    )


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
