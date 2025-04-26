import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from diffusers_helper.utils import resize_and_center_crop

import gradio as gr

def load_model():
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f'Free VRAM {free_mem_gb} GB')
    print(f'High-VRAM Mode: {high_vram}')

    text_encoder = LlamaModel.from_pretrained("furusu/hv_llama_nf4", torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('furusu/framepack_transformer_nf4', torch_dtype=torch.bfloat16).cpu()

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    #transformer.to(dtype=torch.bfloat16) # comment out for bnb 4bit
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    #text_encoder.to(dtype=torch.float16) # comment out for bnb 4bit
    text_encoder_2.to(dtype=torch.float16)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if not high_vram:
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

    return (
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        feature_extractor,
        image_encoder,
        transformer
    )

def get_num_frames(latent_window_size):
    return latent_window_size * 4 - 3

def section_title_update(total_latent_sections):
    visibles = [gr.update(visible=True) for _ in range(total_latent_sections)] + [gr.update(visible=False) for _ in range(32 - total_latent_sections)]
    output = [f'## 総セクション数は{total_latent_sections}だよーん。'] + visibles * 2
    return output

def get_image_np_pt(image, width, height):
    if image is None:
        return None, None
    else:
        image_np = resize_and_center_crop(image, target_width=width, target_height=height)
        image_pt = torch.from_numpy(image_np).float() / 127.5 - 1
        image_pt = image_pt.permute(2, 0, 1)[None, :, None]
        return image_np, image_pt
    