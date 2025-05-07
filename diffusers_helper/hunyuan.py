import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask


@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images

def hook_forward_conv3d(self):
    # replace HunyuanVideoCausalConv3d.forward
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        if self.time_causal_padding[4] > 0:
            if hasattr(self, "cache") and self.cache is not None:
                hidden_states[:, :, :self.time_causal_padding[4]] = self.cache.clone() # copy cache to the top frames
            self.cache = hidden_states[:, :, -self.time_causal_padding[4]:].clone() # cache the last frames
        return self.conv(hidden_states)
    return forward

def hook_forward_upsample(self):
    # replace HunyuanVideoUpsampleCausal3D.forward
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.conv, "cache") and self.conv.cache is not None:
            # upsample all frames if cache is used
            hidden_states = torch.nn.functional.interpolate(hidden_states.contiguous(), scale_factor=self.upsample_factor, mode="nearest")
        else:
            num_frames = hidden_states.size(2)

            first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
            first_frame = torch.nn.functional.interpolate(
                first_frame.squeeze(2), scale_factor=self.upsample_factor[1:], mode="nearest"
            ).unsqueeze(2)

            if num_frames > 1:
                # See: https://github.com/pytorch/pytorch/issues/81665
                # Unless you have a version of pytorch where non-contiguous implementation of F.interpolate
                # is fixed, this will raise either a runtime error, or fail silently with bad outputs.
                # If you are encountering an error here, make sure to try running encoding/decoding with
                # `vae.enable_tiling()` first. If that doesn't work, open an issue at:
                # https://github.com/huggingface/diffusers/issues
                other_frames = other_frames.contiguous()
                other_frames = torch.nn.functional.interpolate(other_frames, scale_factor=self.upsample_factor, mode="nearest")
                hidden_states = torch.cat((first_frame, other_frames), dim=2)
            else:
                hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states
    return forward

def hook_vae(vae):
    vae._original_use_framewise_decoding = vae.use_framewise_decoding
    vae._original_use_slicing = vae.use_slicing
    vae._original_use_tiling = vae.use_tiling
    vae.use_framewise_decoding = False
    vae.use_slicing = False
    vae.use_tiling = False
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module._orginal_forward = module.forward
            module.forward = hook_forward_conv3d(module)
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module._orginal_forward = module.forward
            module.forward = hook_forward_upsample(module)

def restore_vae(vae):
    vae.use_framewise_decoding = vae._original_use_framewise_decoding
    vae.use_slicing = vae._original_use_slicing
    vae.use_tiling = vae._original_use_tiling

    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module.forward = module._orginal_forward
            module.cache = None
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module.forward = module._orginal_forward

@torch.no_grad()
def vae_decode(latents, vae):
    latents = latents / vae.config.scaling_factor
    frames = latents.shape[2]
    hook_vae(vae)
    
    for i in range(frames):
        latents_slice = latents[:, :, i:i+1, :, :]
        image_slice = vae.decode(latents_slice.to(device=vae.device, dtype=vae.dtype)).sample
        
        if i == 0:
            image = image_slice
        else:
            image = torch.cat((image, image_slice), dim=2)
    restore_vae(vae)

    return image


@torch.no_grad()
def vae_encode(image, vae):
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents
