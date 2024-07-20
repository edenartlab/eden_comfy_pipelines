import torch
from PIL import Image
import sys, os, time
import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class WeightedKMeans(KMeans):
    def __init__(self, n_clusters=8, weights=None, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.weights = weights

    def fit(self, X):
        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        return super().fit(X)

    def _weighted_euclidean_distance(self, X, Y):
        return cdist(X, Y, metric='wminkowski', w=self.weights, p=2)

    def fit_predict(self, X, y=None, sample_weight=None):
        return super().fit_predict(X, sample_weight=sample_weight)

    def fit_transform(self, X, y=None, sample_weight=None):
        return super().fit_transform(X, sample_weight=sample_weight)

    def transform(self, X):
        return self._weighted_euclidean_distance(X, self.cluster_centers_)


def smart_depth_slicing(rgb_img, depth_img, n_slices, rgb_weight, standardize_features):
    depth_img = depth_img.numpy() if hasattr(depth_img, 'numpy') else depth_img
    rgb_img = rgb_img.numpy() if hasattr(rgb_img, 'numpy') else rgb_img

    # Reshape images
    depth_flat = depth_img.reshape(-1, 1)
    rgb_flat = rgb_img.reshape(-1, 3)

    if rgb_weight != 0.0:
        combined_features = np.hstack((depth_flat, rgb_flat))
        weights = np.array([1, rgb_weight, rgb_weight, rgb_weight])
    else:
        combined_features = np.hstack((depth_flat))[:,np.newaxis]
        weights = np.array([1])

    if standardize_features:
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)

    # Apply weighted k-means clustering
    kmeans = WeightedKMeans(n_clusters=n_slices, weights=weights, random_state=42)
    kmeans.fit(combined_features)

    cluster_indices = kmeans.labels_

    # Extract depth values from cluster centers
    depth_centers = kmeans.cluster_centers_[:, 0]

    # Sort the cluster centers based on depth
    sorted_indices = np.argsort(depth_centers)
    sorted_centers = depth_centers[sorted_indices]

    # Reshape cluster_indices back to original image shape
    cluster_indices = cluster_indices.reshape(depth_img.shape)

    # Create a mapping from old cluster indices to new sorted indices
    index_map = {old: new for new, old in enumerate(sorted_indices[::-1])}

    # Apply the mapping to get sorted cluster indices
    sorted_cluster_indices = np.vectorize(index_map.get)(cluster_indices)

    # Create mask_images tensor
    h, w = depth_img.shape
    mask_images = torch.zeros((n_slices, h, w, 3), dtype=torch.float32)

    # Fill mask_images with binary masks
    for i in range(n_slices):
        binary_mask = (sorted_cluster_indices == i)
        mask_images[i] = torch.from_numpy(np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2))
    
    return mask_images



class DepthSlicer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"image": ("IMAGE",),
                     "depth_map":  ("IMAGE",),
                     "n_slices":  ("INT", {"default": 2}),
                     "rgb_weight":  ("FLOAT", {"default": 0.0, "step": 0.01}),
                     "standardize_features": ("BOOLEAN", {"default": False}),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpainting_masks",)
    FUNCTION = "slice"
    CATEGORY = "Eden 🌱/Depth"

    def slice(self, image, depth_map, n_slices, rgb_weight, standardize_features):
        # Use only one channel (they're identical)
        depth = depth_map[0, :, :, 0]

        # Calculate thresholds
        masks = smart_depth_slicing(image, depth, n_slices, rgb_weight, standardize_features)

        return (masks,)

class ParallaxZoom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"masks": ("IMAGE",),
                 "image_slices": ("IMAGE",),
                 "foreground_zoom_factor": ("FLOAT", {"default": 1.1, "step": 0.001}),
                 "background_zoom_factor": ("FLOAT", {"default": 1.05, "step": 0.001}),
                 "pan_left": ("FLOAT", {"default": 0.1, "step": 0.001}),
                 "n_frames": ("INT", {"default": 25}),
                 "loop": ("BOOLEAN", {"default": False}),
                }
               }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("frames","masks")
    FUNCTION = "zoom"
    CATEGORY = "Eden 🌱/Depth"
    DESCRIPTION = """
Apply 3D depth parallax to the input image to create a 3D video effect.
Foreground and Background zoom factors control the amount of zoom applied to the respective layers.
Pan Left controls the amount of horizontal shift applied to the image.
All these values are the total fraction (relative to resolution) applied to the image over the full animation.
"""

    @staticmethod
    def warp_affine(image, mask=None, zoom_factor=1.0, shift_factor=0.0):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Create the affine transformation matrix
        M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
        M[0, 2] += shift_factor * w  # Add horizontal shift

        # Apply the affine transformation
        warped_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if mask is not None:
            warped_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return warped_image, warped_mask

        return warped_image

    def zoom(self, masks, image_slices, foreground_zoom_factor, background_zoom_factor, pan_left, n_frames, loop):
        masks = masks.numpy()
        image_slices = image_slices.numpy()

        # Extract the images and masks
        foreground_image = image_slices[0].copy()
        background_image = image_slices[1].copy()
        foreground_mask = masks[0,:,:,0].copy()

        frames, foreground_masks = [], []

        for i in range(n_frames):
            print(f"Processing frame {i+1}/{n_frames}")

            # Compute progress as a value between 0 and 1
            progress = i / (n_frames - 1)

            if loop:
                # Full sine wave cycle for looping
                angle = progress * np.pi * 2
                factor = (np.sin(angle) + 1) / 2
            else:
                # Linear progression for non-looping
                factor = progress

            # Compute zoom and shift factors
            fg_zoom = 1 + (foreground_zoom_factor - 1) * factor
            
            # Adjust background zoom behavior
            if background_zoom_factor >= 1:
                bg_zoom = 1 + (background_zoom_factor - 1) * factor
            else:
                bg_zoom = 1 / background_zoom_factor - (1 / background_zoom_factor - 1) * factor
            
            fg_shift = -pan_left * factor

            # Apply transformations
            warped_foreground, warped_mask = self.warp_affine(foreground_image, foreground_mask, fg_zoom, fg_shift)
            warped_background = self.warp_affine(background_image, zoom_factor=bg_zoom)

            # Ensure the mask has 3 channels to match the image
            warped_mask = np.stack([warped_mask] * 3, axis=-1)
            foreground_masks.append(warped_mask)

            # Combine foreground and background
            final_image = warped_foreground * warped_mask + warped_background * (1 - warped_mask)
            final_image = final_image[:, :, :3]

            frames.append(final_image)

        frames = torch.tensor(np.array(frames))
        foreground_masks = torch.tensor(np.array(foreground_masks))

        return (frames, foreground_masks)


if __name__ == "__main__":

    mask0, mask1 = cv2.imread("mask0.png"), cv2.imread("mask1.png")
    img0, img1 = cv2.imread("img0.png"), cv2.imread("img1.png")
    # Convert images from BGR to RGB
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    masks = np.stack([mask0, mask1], axis=0) / 255.
    image_slices = np.stack([img0, img1], axis=0) / 255.
    masks = torch.tensor(masks)
    image_slices = torch.tensor(image_slices)

    parallax = ParallaxZoom()
    frames, = parallax.zoom(masks, image_slices, 1.12, 1.05, 0.05, 40)
    frames = frames.numpy()

    # Save frames as a video:
    print("Writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
    out = cv2.VideoWriter('output.mp4', fourcc, 30, (1024,1024))  # Change fps to 30 and file extension to .mp4
    for i, frame in enumerate(frames):
        # Convert frame to uint8 and BGR color space
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        # Save frames as images
        if len(frames) < 10:
            cv2.imwrite(f"frame_{i}.jpg", frame_bgr)

    out.release()