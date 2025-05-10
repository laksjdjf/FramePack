import torch
import torch.nn.functional as F

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
            self.cache = hidden_states[:, :, -t:].to(torch.float8_e4m3fn)
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

    
def hook_vae(vae):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module._orginal_forward = module.forward
            module.forward = hook_forward_conv3d(module)
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module._orginal_forward = module.forward
            module.forward = hook_forward_upsample(module)

def restore_vae(vae):
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module.forward = module._orginal_forward
            module.cache = None
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module.forward = module._orginal_forward

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
    from diffusers_helper.bucket_tools import find_nearest_bucket
    from diffusers_helper.utils import resize_and_center_crop, save_bcthw_as_mp4
    from diffusers_helper.hunyuan import vae_encode
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

    video_file = 'outputs/250510_093026_750_7653_19.mp4'
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
        images_o = vae_decode_tiling(latents, vae)
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