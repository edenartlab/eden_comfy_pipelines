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