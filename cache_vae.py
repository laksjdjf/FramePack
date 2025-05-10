import torch
import torch.nn.functional as F
from typing import Optional

def hook_forward_conv3d(self):
    # replace HunyuanVideoCausalConv3d.forward
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if self.time_causal_padding[4] > 0:
            t = self.time_causal_padding[4]
            padding = (self.time_causal_padding[0], self.time_causal_padding[1], self.time_causal_padding[2], self.time_causal_padding[3], 0, 0)
            
            if hasattr(self, "cache") and self.cache is not None:
                cache = self.cache.to(hidden_states)
                hidden_states = torch.cat([cache, hidden_states], dim=2)
            else:
                hidden_states = F.pad(hidden_states, (0, 0, 0, 0, t, 0), mode=self.pad_mode)
            self.cache = hidden_states[:, :, -t:].clone()
            hidden_states = F.pad(hidden_states, padding, mode=self.pad_mode)
        else:
            hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)
    return forward

def hook_forward_upsample(self):
    # replace HunyuanVideoUpsampleCausal3D.forward
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.conv, "cache") and self.conv.cache is not None:
            # upsample all frames if cache is used
            hidden_states = F.interpolate(hidden_states.contiguous(), scale_factor=self.upsample_factor, mode="nearest")
        else:
            num_frames = hidden_states.size(2)

            first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
            first_frame = F.interpolate(
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
                other_frames = F.interpolate(other_frames, scale_factor=self.upsample_factor, mode="nearest")
                hidden_states = torch.cat((first_frame, other_frames), dim=2)
            else:
                hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states
    return forward

def hook_forward_groupnorm(org_forward, h, w):
    # replace HunyuanVideoGroupNorm.forward
    def forward(x: torch.Tensor) -> torch.Tensor:
        nonlocal h, w
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = x.transpose(1, 2).reshape(b * t, c, h, w)
            x = org_forward(x)
            x = x.reshape(b, t, c, h, w).transpose(1, 2)
        else:
            b, c, n = x.shape
            x = x.reshape(b, c, -1, h * w).transpose(1, 2).reshape(-1, c, h * w)
            x = org_forward(x)
            x = x.reshape(b, -1, c, h * w).transpose(1, 2).reshape(b, c, n)
        return x
    return forward

# AttnProcessor2_0 with KVCache
class AttnProcessor2_0_KVCache:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        self.k_cache = None
        self.v_cache = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.k_cache is not None:
            key = torch.cat([self.k_cache, key], dim=2)
            value = torch.cat([self.v_cache, value], dim=2)
            attention_mask = torch.cat(
                [torch.zeros(attention_mask.shape[0], attention_mask.shape[1],  attention_mask.shape[2], self.k_cache.shape[2]).to(attention_mask), attention_mask], dim=3
            )
        
        self.k_cache = key
        self.v_cache = value

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
def hook_vae(vae):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module._orginal_forward = module.forward
            module.forward = hook_forward_conv3d(module)
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module._orginal_forward = module.forward
            module.forward = hook_forward_upsample(module)
        if module.__class__.__name__ == "Attention":
            module._orginal_processor = module.processor
            module.processor = AttnProcessor2_0_KVCache()

def restore_vae(vae):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module.forward = module._orginal_forward
            module.cache = None
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module.forward = module._orginal_forward
        if module.__class__.__name__ == "Attention":
            module.processor.k_cache = None
            module.processor.v_cache = None
            module.processor = module._orginal_processor

def fix_groupnorm(vae, h, w):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "GroupNorm":
            module._orginal_forward = module.forward
            module.forward = hook_forward_groupnorm(module._orginal_forward, h, w)

def restore_groupnorm(vae):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "GroupNorm":
            module.forward = module._orginal_forward
            module._orginal_forward = None

@torch.no_grad()
def vae_decode_cache(latents, vae):
    latents = latents / vae.config.scaling_factor
    frames = latents.shape[2]
    hook_vae(vae)
    
    tile_rate = 1
    latents = latents.to(device=vae.device, dtype=vae.dtype)
    images = []
    for i in range(0, frames, tile_rate):
        z = vae.post_quant_conv(latents[:, :, i:i + tile_rate, :, :])
        dec = vae.decoder(z)
        
        images.append(dec)
    image = torch.cat(images, dim=2)
    restore_vae(vae)

    return image

@torch.no_grad()
def vae_decode_tiling(latents, vae):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    return image

@torch.no_grad()
def vae_decode(latents, vae):
    latents = latents / vae.config.scaling_factor
    z = vae.post_quant_conv(latents.to(device=vae.device, dtype=vae.dtype))
    image = vae.decoder(z)
    return image
    
if __name__ == "__main__":
    from diffusers import AutoencoderKLHunyuanVideo
    import torch
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float32).cuda().eval()

    x = torch.randn(1, 16, 9, 32, 32).cuda()
    fix_groupnorm(vae, 32, 32)

    with torch.no_grad():
        decoded_o = vae_decode(x, vae)
        decoded_c = vae_decode_cache(x, vae)

    print((decoded_o - decoded_c).abs().mean(dim=(0,1,3,4)).cpu().numpy())