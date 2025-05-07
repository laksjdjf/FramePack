import torch
import torch.nn.functional as F
from typing import Optional

'''
class HunyuanVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)
'''

def hook_forward_conv3d(self):
    # replace HunyuanVideoCausalConv3d.forward
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
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
        
        
        self.k_cache = key.clone()
        self.v_cache = value.clone()

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
        if module.__class__.__name__ == "Attention":
            module._orginal_processor = module.processor
            module.processor = AttnProcessor2_0_KVCache()

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
            module.conv.cache = None
        if module.__class__.__name__ == "Attention":
            module.processor.k_cache = None
            module.processor.v_cache = None
            module.processor = module._orginal_processor
    
@torch.no_grad()
def vae_decode_cache(latents, vae):
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

    
if __name__ == "__main__":
    from diffusers import AutoencoderKLHunyuanVideo
    from diffusers_helper.bucket_tools import find_nearest_bucket
    from diffusers_helper.utils import resize_and_center_crop, save_bcthw_as_mp4
    from diffusers_helper.hunyuan import vae_decode, vae_encode
    import torch
    import cv2
    from PIL import Image
    import time
    import numpy as np

    def mp4_to_pil_images(video_path):
        cap = cv2.VideoCapture(video_path)
        images = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # BGR（OpenCV）→ RGB（PIL）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            images.append(pil_image)

        cap.release()
        return images
    
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cuda()
    vae.eval().requires_grad_(False)
    vae.enable_slicing()
    vae.enable_tiling()

    video_file = 'outputs/250428_092032_789_7426_19.mp4'
    image_list = mp4_to_pil_images(video_file)

    movie_pt = []
    height, width = find_nearest_bucket(image_list[0].size[1], image_list[0].size[0], resolution=640)
    for image in image_list:
        image_np = resize_and_center_crop(np.array(image), target_width=width, target_height=height)
        image_pt = torch.from_numpy(image_np).float() / 127.5 - 1
        image_pt = image_pt.permute(2, 0, 1)[None, :, None]
        movie_pt.append(image_pt)
    movie_pt = torch.cat(movie_pt, dim=2).cuda().half()

    with torch.no_grad():
        latents = vae_encode(movie_pt, vae)

    print("encoded latents shape:", latents.shape)

    # original vae_decode
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.no_grad():
        start = time.time()
        images_o = vae_decode(latents, vae)
        torch.cuda.synchronize()  # GPU処理の同期
        end = time.time()
        
        mem_o = torch.cuda.max_memory_allocated()
        print(f"vae_decode() 使用メモリ: {mem_o / (1024**2):.2f} MB 実行時間{end - start:.4f} 秒")

    # vae_decode_cache
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.no_grad():
        start = time.time()
        images_c = vae_decode_cache(latents, vae)
        torch.cuda.synchronize()
        end = time.time()
        
        mem_c = torch.cuda.max_memory_allocated()
        print(f"vae_decode_cache() 使用メモリ: {mem_c / (1024**2):.2f} MB 実行時間{end - start:.4f} 秒")

    print((images_o-images_c).abs().mean())
    x = save_bcthw_as_mp4(torch.cat([movie_pt, images_o, images_c]), "test.mp4", 30, 16)
