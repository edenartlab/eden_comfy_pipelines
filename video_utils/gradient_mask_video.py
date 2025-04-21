import torch
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

class KeyframeBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"image_frames": ("IMAGE",),
                     "keyframe_ip_adapter_features": ("EMBEDS",),
                     "n_frames": ("INT", {"default": 50}),
                     "denoise_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                     "ip_adapter_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "EMBEDS")
    RETURN_NAMES = ("keyframe_blend", "denoising_masks", "ip_adapter_attention_masks", "denoising_mask_curve", "ip_adapter_trajectory")
    FUNCTION = "blend_keyframes"
    CATEGORY = "Video Effects"

    def plot_denoising_values(self, denoising_values):
        fig, ax = plt.subplots()
        ax.plot(denoising_values)
        ax.set(xlabel='Frame Number', ylabel='Denoising Value', title='Denoising Mask Curve')
        ax.grid()
        ax.set_ylim(0, 1)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        tensor_image = torch.from_numpy(image_from_plot).float().unsqueeze(0) / 255.0
        plt.close(fig)
        return tensor_image

    def blend_keyframes(self, image_frames, keyframe_ip_adapter_features, n_frames, denoise_gamma, ip_adapter_gamma):
        num_keyframes, height, width, channels = image_frames.shape
        _, n_features, feature_dim = keyframe_ip_adapter_features.shape
        device = image_frames.device

        transition_frames = [n_frames // (num_keyframes - 1)] * (num_keyframes - 1)
        remainder = n_frames % (num_keyframes - 1)
        for i in range(remainder):
            transition_frames[i] += 1

        blended_video = torch.zeros(n_frames, height, width, 3, device=device)
        denoising_masks = torch.zeros(n_frames, height, width, device=device)
        ip_adapter_trajectory = torch.zeros(n_frames, n_features, feature_dim, device=device)

        start_frame = 0
        denoising_values = []
        for i in range(num_keyframes - 1):
            end_frame = start_frame + transition_frames[i]
            midpoint_frame = start_frame + (end_frame - start_frame) // 2

            for j in range(start_frame, end_frame):
                alpha = (j - start_frame) / (end_frame - start_frame)
                blended_video[j] = image_frames[i] * (1 - alpha) + image_frames[i + 1] * alpha
                ip_adapter_trajectory[j] = keyframe_ip_adapter_features[i] * (1 - alpha) + keyframe_ip_adapter_features[i + 1] * alpha

                if j < midpoint_frame:
                    denoising_value = (j - start_frame) / (midpoint_frame - start_frame)
                else:
                    denoising_value = (end_frame - j) / (end_frame - midpoint_frame)

                denoising_values.append(denoising_value)
                denoising_masks[j] = torch.tensor(denoising_value, device=device).float()

            start_frame = end_frame

        denoising_values = np.array(denoising_values)**denoise_gamma
        curve_image = self.plot_denoising_values(denoising_values)

        # apply gamma corrections:
        ip_adapter_attention_masks = denoising_masks.clone()
        denoising_masks = denoising_masks ** denoise_gamma
        ip_adapter_attention_masks = ip_adapter_attention_masks ** ip_adapter_gamma

        return blended_video, denoising_masks, ip_adapter_attention_masks, curve_image, ip_adapter_trajectory
    

import os, re
import subprocess
import torch
import numpy as np
from folder_paths import get_output_directory

class MaskedRegionVideoExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "filename_prefix": ("STRING", {"default": "masked_video"}),
                "flip_mask": ("BOOLEAN", {"default": False}),
                "format": (["webm", "prores_mov"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    CATEGORY = "Video"
    FUNCTION = "export"

    def export(self, images, masks, fps, filename_prefix, flip_mask, format):
        if images.shape[0] != masks.shape[0]:
            raise ValueError("Number of images and masks must match!")

        print(f"Masking images of shape: {images.shape} with masks of shape: {masks.shape}")
        print(f"Mask max value: {masks.max()}, min value: {masks.min()}")

        output_dir = get_output_directory()
        ext = "webm" if format == "webm" else "mov"
        base_name = f"{filename_prefix}"
        existing_files = os.listdir(output_dir)
        matcher = re.compile(re.escape(base_name) + r"_(\d+)\." + ext + r"$", re.IGNORECASE)
        max_index = -1
        for f in existing_files:
            match = matcher.fullmatch(f)
            if match:
                max_index = max(max_index, int(match.group(1)))
        new_index = max_index + 1
        video_filename = f"{base_name}_{new_index:03d}.{ext}"
        video_path = os.path.join(output_dir, video_filename)

        height, width = images.shape[1:3]

        if format == "webm":
            codec = "libvpx-vp9"
            pix_fmt = "yuva420p"
            ffmpeg_args = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-c:v", codec,
                "-crf", "19", "-b:v", "0",
                "-pix_fmt", pix_fmt,
                "-auto-alt-ref", "0",
                video_path
            ]
        else:  # prores_mov
            codec = "prores_ks"
            pix_fmt = "yuva444p10le"
            ffmpeg_args = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-c:v", codec,
                "-profile:v", "4",  # ProRes 4444
                "-pix_fmt", pix_fmt,
                video_path
            ]

        frames = []
        for img, mask in zip(images, masks):
            img = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask = mask.cpu().numpy()

            if flip_mask:
                mask = 1.0 - mask

            mask = np.clip(mask, 0, 1)
            alpha = (mask * 255).astype(np.uint8)
            img[alpha == 0] = 0
            rgba = np.dstack([img, alpha])
            frames.append(rgba)

        video_data = b''.join([frame.tobytes() for frame in frames])

        try:
            subprocess.run(ffmpeg_args, input=video_data, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")

        preview = {
            "filename": video_filename,
            "subfolder": "",
            "type": "output",
            "format": f"video/{ext}",
            "frame_rate": fps,
            "workflow": "",
            "fullpath": video_path,
        }

        return {"ui": {"gifs": [preview]}, "result": (video_path,)}