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
from diffusers_helper.utils import save_bcthw_as_png, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
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
parser.add_argument("--f1", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer = load_model(args.f1)
high_vram = get_cuda_free_memory_gb(gpu) > 60

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(target_indice, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, keyframes, indices, strengths):
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

        H, W, _ = keyframes[0]["background"].shape
        height, width = find_nearest_bucket(H, W, resolution=640)

        key_frames_pt, key_frames_np = [], []
        key_frame_masks = []
        for keyframe in keyframes:
            section_keyframe_np, section_keyframe_pt = get_image_np_pt(keyframe["background"][:,:,:3], width, height)
            key_frames_pt.append(section_keyframe_pt)
            key_frames_np.append(section_keyframe_np)

            key_frame_mask = keyframe["layers"][0][:,:,3]
            key_frame_mask = np.repeat(key_frame_mask[..., None], 3, axis=2)
            _, key_frame_mask_pt = get_image_np_pt(key_frame_mask, width // 8, height // 8)
            key_frame_mask_pt = 1 - ((key_frame_mask_pt + 1) / 2)
            key_frame_masks.append(key_frame_mask_pt[:, :1, :, :])

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        key_frame_latents = []
        input_indices = []
        for key_frame_pt, key_frame_mask, strangth, indice in zip(key_frames_pt, key_frame_masks, strengths, indices):
            if key_frame_pt is not None:
                key_frame_latent = vae_encode(key_frame_pt, vae)
                key_frame_latent = key_frame_latent * key_frame_mask.to(key_frame_latent)
                key_frame_latents.append(key_frame_latent * strangth)
                input_indices.append(indice)
            else:
                key_frame_latents.append(torch.zeros((1, 16, 1, height // 8, width // 8), dtype=torch.float32).cpu())
                input_indices.append(indice)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(key_frames_np[0], feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            return
        
        clean_latent_indices = torch.tensor(input_indices).unsqueeze(0)
        clean_latents = torch.cat(key_frame_latents, dim=2)
        latent_indices = torch.tensor([target_indice]).unsqueeze(0)

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
            hint = f'Sampling {current_step}/{steps}'
            stream.output_queue.push(('progress', (preview, None, make_progress_bar_html(percentage, hint))))
            return

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=1,
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
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            callback=callback,
        )

        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)

        history_pixels = vae_decode(generated_latents, vae).cpu()


        if not high_vram:
            unload_complete_models()

        output_filename = os.path.join(outputs_folder, f'{job_id}.png')

        save_bcthw_as_png(history_pixels, output_filename)
        stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(target_indice, num_input_frames, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, *args):
    global stream
    #assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    keyframes = args[0:num_input_frames]
    indices = args[4:4+num_input_frames]
    strangths = args[8:8+num_input_frames]

    async_run(worker, target_indice, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, keyframes, indices, strangths)

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

def input_frames_update(num_input_frames):
    visibles = [gr.update(visible=True) for _ in range(num_input_frames)] + [gr.update(visible=False) for _ in range(4 - num_input_frames)]
    return visibles * 3

with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            with gr.Group():
                first_total_frames = 1
                target_indice = gr.Number(label=f"target indice", minimum=0, maximum=256, value=0, step=1)
                num_input_frames = gr.Slider(label="Number of input Frames", minimum=1, maximum=4, value=first_total_frames, step=1)
                keyframes = []
                indices = []
                strengths = []
                for i in range(4):
                    with gr.Row():
                        keyframes.append(gr.ImageMask(sources='upload', type="numpy", label="Image", height=640, visible= i < first_total_frames))
                        with gr.Column():
                            indices.append(gr.Number(label=f"Section {i} Padding", minimum=0, maximum=256, value=0, step=1, visible= i < first_total_frames))
                            strengths.append(gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01, info='The strength of the image. 0 means no effect, 1 means full effect.'))
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Image(label="Finished Frames")
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [target_indice, num_input_frames, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache] + keyframes + indices + strengths
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

    num_input_frames.change(
        fn=input_frames_update,
        inputs=[num_input_frames],
        outputs=keyframes + indices + strengths,
    )


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
