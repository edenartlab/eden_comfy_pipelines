import sys

import numpy as np
import torch


def smart_depth_slicing(rgb_img, depth_img, n_slices, rgb_weight, standardize_features):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    depth_img = depth_img.cpu().numpy()
    rgb_img = rgb_img.cpu().numpy()

    if rgb_weight != 0.0:
        features = np.hstack((depth_img.reshape(-1, 1), rgb_img.reshape(-1, 3)))
        weights = np.array([1, rgb_weight, rgb_weight, rgb_weight])
    else:
        features = depth_img.reshape(-1, 1)
        weights = np.array([1])

    if standardize_features:
        features = StandardScaler().fit_transform(features)

    # Weighted euclidean k-means == plain k-means on weight-scaled features
    kmeans = KMeans(n_clusters=n_slices, random_state=42)
    kmeans.fit(features * weights)

    # Relabel clusters so slice 0 is the one with the highest depth value (closest)
    depth_centers = kmeans.cluster_centers_[:, 0]
    rank = np.empty(n_slices, dtype=np.int64)
    rank[np.argsort(depth_centers)[::-1]] = np.arange(n_slices)
    sorted_cluster_indices = rank[kmeans.labels_].reshape(depth_img.shape)

    masks = torch.from_numpy(sorted_cluster_indices)[None] == torch.arange(n_slices)[:, None, None]
    return masks.float().unsqueeze(-1).repeat(1, 1, 1, 3)


class Eden_DepthSlice_MaskVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"depth_map": ("IMAGE", {"tooltip": "Depth map (white = near). Only the first channel is used."}),
                     "slice_width": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99, "step": 0.01, "tooltip": "Thickness of the depth band that is visible in each frame (in normalized depth)."}),
                     "min_depth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Depth where the sweep starts."}),
                     "max_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Depth where the sweep ends."}),
                     "gamma_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Gamma applied to the normalized depth map before slicing."}),
                     "n_frames": ("INT", {"default": 100, "min": 1, "max": sys.maxsize, "tooltip": "Number of output mask frames."}),
                     "reverse": ("BOOLEAN", {"default": False, "tooltip": "Sweep from far to near instead of near to far."}),
                     "bounce": ("BOOLEAN", {"default": False, "tooltip": "Sweep forward then back within the same number of frames."}),
                     }
                }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("depth_slice_masks", "gamma_corrected_depth_map")
    FUNCTION = "generate_mask_video"
    CATEGORY = "Eden 🌱/Depth"
    DESCRIPTION = "Sweeps a thin depth band through the scene and outputs one mask per frame, for depth-based reveal animations."
    OUTPUT_TOOLTIPS = ("One mask per frame covering the current depth band.", "The normalized, gamma-corrected depth map.")

    def generate_mask_video(self, depth_map, slice_width, min_depth, max_depth, gamma_correction, n_frames, reverse, bounce):
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / depth_map.max()
        depth_map = depth_map ** gamma_correction

        if depth_map.shape[-1] == 3:
            depth_map = depth_map[..., 0]

        if bounce:
            n_frames = n_frames // 2

        depth = depth_map.squeeze()
        # Bounds in float64, compared in the depth dtype (same as comparing against python floats)
        steps = torch.arange(n_frames, device=depth.device, dtype=torch.float64)
        lower = min_depth + (max_depth - slice_width - min_depth) * steps / n_frames
        upper = (lower + slice_width).to(depth.dtype).view(-1, 1, 1)
        lower = lower.to(depth.dtype).view(-1, 1, 1)
        video_frames = ((depth >= lower) & (depth < upper)).float()

        if reverse:
            video_frames = video_frames.flip(0)

        if bounce:
            video_frames = torch.cat([video_frames, video_frames.flip(0)])
            if video_frames.shape[0] < n_frames:
                video_frames = torch.cat([video_frames, video_frames[-1:]])

        return (video_frames, depth_map)


class DepthSlicer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE", {"tooltip": "RGB image matching the depth map (used when rgb_weight > 0)."}),
                     "depth_map":  ("IMAGE", {"tooltip": "Depth map; only the first frame and first channel are used."}),
                     "n_slices":  ("INT", {"default": 2, "min": 1, "max": sys.maxsize, "tooltip": "Number of depth layers (k-means clusters)."}),
                     "rgb_weight":  ("FLOAT", {"default": 0.0, "step": 0.01, "tooltip": "How much color similarity counts next to depth when clustering. 0 = depth only."}),
                     "standardize_features": ("BOOLEAN", {"default": False, "tooltip": "Standardize depth and color features before clustering."}),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpainting_masks",)
    FUNCTION = "slice"
    CATEGORY = "Eden 🌱/Depth"
    DESCRIPTION = "Splits an image into depth layers with k-means clustering and returns one mask per layer, nearest first."
    OUTPUT_TOOLTIPS = ("One mask image per depth layer, ordered near to far.",)

    def slice(self, image, depth_map, n_slices, rgb_weight, standardize_features):
        return (smart_depth_slicing(image, depth_map[0, :, :, 0], n_slices, rgb_weight, standardize_features),)


class ParallaxZoom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"masks": ("IMAGE", {"tooltip": "Layer masks (e.g. from DepthSlicer); the first mask selects the foreground."}),
                 "image_slices": ("IMAGE", {"tooltip": "Two images: foreground layer first, (inpainted) background layer second."}),
                 "foreground_zoom_factor": ("FLOAT", {"default": 1.1, "step": 0.001, "tooltip": "Total zoom applied to the foreground over the animation."}),
                 "background_zoom_factor": ("FLOAT", {"default": 1.05, "step": 0.001, "tooltip": "Total zoom applied to the background over the animation (< 1 zooms out)."}),
                 "pan_left": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Total horizontal foreground shift as a fraction of the width."}),
                 "n_frames": ("INT", {"default": 25, "min": 1, "max": sys.maxsize, "tooltip": "Number of output frames."}),
                 "loop": ("BOOLEAN", {"default": False, "tooltip": "Move forward and back along a sine curve so the clip loops seamlessly."}),
                }
               }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("frames", "masks")
    FUNCTION = "zoom"
    CATEGORY = "Eden 🌱/Depth"
    DESCRIPTION = """
Apply 3D depth parallax to the input image to create a 3D video effect.
Foreground and Background zoom factors control the amount of zoom applied to the respective layers.
Pan Left controls the amount of horizontal shift applied to the image.
All these values are the total fraction (relative to resolution) applied to the image over the full animation.
"""
    OUTPUT_TOOLTIPS = ("The composited parallax frames.", "The warped foreground mask per frame.")

    @staticmethod
    def warp_affine(image, zoom_factor=1.0, shift_factor=0.0):
        import cv2
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom_factor)
        M[0, 2] += shift_factor * w
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def zoom(self, masks, image_slices, foreground_zoom_factor, background_zoom_factor, pan_left, n_frames, loop):
        masks = masks.cpu().numpy()
        image_slices = image_slices.cpu().numpy()

        foreground_image = image_slices[0]
        background_image = image_slices[1]
        foreground_mask = np.ascontiguousarray(masks[0, :, :, 0])

        frames, foreground_masks = [], []
        for i in range(n_frames):
            progress = i / (n_frames - 1) if n_frames > 1 else 0.0
            factor = (np.sin(progress * np.pi * 2) + 1) / 2 if loop else progress

            fg_zoom = 1 + (foreground_zoom_factor - 1) * factor
            if background_zoom_factor >= 1:
                bg_zoom = 1 + (background_zoom_factor - 1) * factor
            else:
                bg_zoom = 1 / background_zoom_factor - (1 / background_zoom_factor - 1) * factor
            fg_shift = pan_left/2 - pan_left * factor

            warped_foreground = self.warp_affine(foreground_image, fg_zoom, fg_shift)
            warped_mask = self.warp_affine(foreground_mask, fg_zoom, fg_shift)
            warped_background = self.warp_affine(background_image, bg_zoom)

            warped_mask = np.stack([warped_mask] * 3, axis=-1)
            foreground_masks.append(warped_mask)
            final_image = warped_foreground * warped_mask + warped_background * (1 - warped_mask)
            frames.append(final_image[:, :, :3])

        return (torch.from_numpy(np.stack(frames)), torch.from_numpy(np.stack(foreground_masks)))
