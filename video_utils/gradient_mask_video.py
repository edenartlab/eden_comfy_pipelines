import os
import re
import subprocess

import numpy as np
import torch
from folder_paths import get_output_directory


class KeyframeBlender:
    DESCRIPTION = "Crossfades between keyframe images (and their IP-Adapter embeds) over n_frames, and builds per-frame denoising / attention masks that peak halfway between keyframes."

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_frames": ("IMAGE", {"tooltip": "Keyframes to blend between (at least 2)."}),
                     "keyframe_ip_adapter_features": ("EMBEDS", {"tooltip": "One IP-Adapter embed per keyframe, interpolated like the images."}),
                     "n_frames": ("INT", {"default": 50, "tooltip": "Total number of output frames."}),
                     "denoise_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "tooltip": "Gamma applied to the denoising masks; >1 keeps frames near the keyframes cleaner."}),
                     "ip_adapter_gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "tooltip": "Gamma applied to the IP-Adapter attention masks."}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "EMBEDS")
    RETURN_NAMES = ("keyframe_blend", "denoising_masks", "ip_adapter_attention_masks", "denoising_mask_curve", "ip_adapter_trajectory")
    OUTPUT_TOOLTIPS = ("Crossfaded frames.", "Per-frame denoising strength masks (0 at keyframes, 1 halfway).", "Per-frame IP-Adapter attention masks.", "Plot of the denoising curve.", "Interpolated IP-Adapter embeds, one per frame.")
    FUNCTION = "blend_keyframes"
    CATEGORY = "Eden 🌱/Video"

    def plot_denoising_values(self, denoising_values):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.plot(denoising_values)
        ax.set(xlabel='Frame Number', ylabel='Denoising Value', title='Denoising Mask Curve')
        ax.grid()
        ax.set_ylim(0, 1)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3]
        return torch.from_numpy(rgb.copy()).float().unsqueeze(0) / 255.0

    def blend_keyframes(self, image_frames, keyframe_ip_adapter_features, n_frames, denoise_gamma, ip_adapter_gamma):
        num_keyframes, height, width, _ = image_frames.shape
        device = image_frames.device

        transition_frames = [n_frames // (num_keyframes - 1)] * (num_keyframes - 1)
        for i in range(n_frames % (num_keyframes - 1)):
            transition_frames[i] += 1

        # Per output frame: which keyframe pair, the blend weight, and the denoising value
        segment, alphas, denoising_values = [], [], []
        start_frame = 0
        for i, length in enumerate(transition_frames):
            end_frame = start_frame + length
            midpoint_frame = start_frame + length // 2
            for j in range(start_frame, end_frame):
                segment.append(i)
                alphas.append((j - start_frame) / length)
                if j < midpoint_frame:
                    denoising_values.append((j - start_frame) / (midpoint_frame - start_frame))
                else:
                    denoising_values.append((end_frame - j) / (end_frame - midpoint_frame))
            start_frame = end_frame

        def interpolate(x):
            idx = torch.tensor(segment, device=x.device, dtype=torch.long)
            shape = (-1,) + (1,) * (x.dim() - 1)
            a = torch.tensor(alphas, device=x.device, dtype=x.dtype).view(shape)
            one_minus_a = torch.tensor([1 - v for v in alphas], device=x.device, dtype=x.dtype).view(shape)
            return x[idx] * one_minus_a + x[idx + 1] * a

        blended_video = interpolate(image_frames)[..., :3].float()
        ip_adapter_trajectory = interpolate(keyframe_ip_adapter_features).float().to(device)

        values = torch.tensor(denoising_values, device=device, dtype=torch.float32).view(-1, 1, 1)
        denoising_masks = (values ** denoise_gamma).expand(-1, height, width).contiguous()
        ip_adapter_attention_masks = (values ** ip_adapter_gamma).expand(-1, height, width).contiguous()

        curve_image = self.plot_denoising_values(np.array(denoising_values) ** denoise_gamma)

        return blended_video, denoising_masks, ip_adapter_attention_masks, curve_image, ip_adapter_trajectory


class MaskedRegionVideoExport:
    DESCRIPTION = "Encodes images with their masks as the alpha channel into a transparent video (VP9 webm or ProRes 4444 mov) in the output folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames."}),
                "masks": ("MASK", {"tooltip": "One mask per frame; becomes the alpha channel (0 = transparent)."}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120, "tooltip": "Frame rate of the exported video."}),
                "filename_prefix": ("STRING", {"default": "masked_video", "tooltip": "Output file name prefix; a running number is appended."}),
                "flip_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert the masks before using them as alpha."}),
                "format": (["webm", "prores_mov"], {"tooltip": "webm = VP9 with alpha, prores_mov = ProRes 4444 with alpha."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_TOOLTIPS = ("Full path of the exported video.",)
    OUTPUT_NODE = True
    CATEGORY = "Eden 🌱/Loaders"
    FUNCTION = "export"

    def export(self, images, masks, fps, filename_prefix, flip_mask, format):
        if images.shape[0] != masks.shape[0]:
            raise ValueError("Number of images and masks must match!")

        output_dir = get_output_directory()
        ext = "webm" if format == "webm" else "mov"
        matcher = re.compile(re.escape(filename_prefix) + r"_(\d+)\." + ext, re.IGNORECASE)
        indices = [int(m.group(1)) for m in map(matcher.fullmatch, os.listdir(output_dir)) if m]
        video_filename = f"{filename_prefix}_{max(indices, default=-1) + 1:03d}.{ext}"
        video_path = os.path.join(output_dir, video_filename)

        height, width = images.shape[1:3]
        input_args = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        ]
        if format == "webm":
            codec_args = ["-c:v", "libvpx-vp9", "-crf", "19", "-b:v", "0", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0"]
        else:
            codec_args = ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]

        proc = subprocess.Popen(input_args + codec_args + [video_path], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for img, mask in zip(images, masks):
                rgb = (img * 255).clamp(0, 255).to(torch.uint8)
                if flip_mask:
                    mask = 1.0 - mask
                alpha = (mask.clamp(0, 1) * 255).to(torch.uint8)
                rgb[alpha == 0] = 0
                proc.stdin.write(torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1).cpu().numpy().tobytes())
        except BrokenPipeError:
            pass
        proc.stdin.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')}")

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
