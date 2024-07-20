import torch
from PIL import Image
import sys, os, time
import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans

def smart_depth_slicing(img, n_slices):
    # convert w,h torch tensor to numpy array:
    img = img.numpy()

    # Flatten the image and reshape for k-means
    flat_img = img.flatten().reshape(-1, 1)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_slices, random_state=42)
    kmeans.fit(flat_img)

    # Sort the cluster centers
    sorted_centers = np.sort(kmeans.cluster_centers_.flatten())

    print(f"Sorted cluster centers: {sorted_centers}")

    # Create slices based on the midpoints between cluster centers
    slices = []
    thresholds = [0] + list((sorted_centers[:-1] + sorted_centers[1:]) / 2) + [1.0]

    return thresholds


class DepthSlicer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"depth_map":  ("IMAGE",),
                     "n_slices":  ("INT", {"default": 3}),
                     "slope":  ("FLOAT", {"default": 1.0}),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpainting_masks",)
    FUNCTION = "slice"
    CATEGORY = "Eden 🌱/Depth"

    def slice(self, depth_map, n_slices, slope):
        print(depth_map.shape)
        # torch.Size([1, 2048, 2048, 3])

        # Use only one channel (they're identical)
        depth = depth_map[0, :, :, 0]

        # Calculate min and max depth values
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)

        # Calculate the range and adjust it with the slope
        depth_range = depth_max - depth_min
        depth_range *= slope

        # Calculate thresholds
        thresholds = torch.linspace(depth_min, depth_min + depth_range, n_slices + 1)
        print(f"Min: {depth_min}, Max: {depth_max}, Range: {depth_range}")
        print(f"Thresholds: {thresholds}")

        thresholds = smart_depth_slicing(depth, n_slices)
        print(f"Thresholds: {thresholds}")

        # Initialize the output tensor
        masks = torch.zeros((n_slices, depth_map.shape[1], depth_map.shape[2], 3), dtype=torch.float32)

        # Create masks for each slice
        for i in range(n_slices):
            lower = thresholds[i]
            upper = thresholds[i+1]

            print(f"Slice {i}: lower:{lower} - upper:{upper}")

            # Create binary mask
            mask = ((depth >= lower) & (depth < upper)).float()

            # Add to masks
            masks[i, :, :, :] = mask.unsqueeze(-1).repeat(1, 1, 3)

        # invert the masks
        masks = 1 - masks

        return (masks,)
class ParallaxZoom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"masks": ("IMAGE",),
                 "image_slices": ("IMAGE",),
                 "foreground_zoom_factor": ("FLOAT", {"default": 1.1, "step": 0.01}),
                 "background_zoom_factor": ("FLOAT", {"default": 1.05, "step": 0.01}),
                 "shift_left": ("FLOAT", {"default": 0.1, "step": 0.01}),
                 "n_frames": ("INT", {"default": 25}),
                 "loop": ("BOOLEAN", {"default": False}),
                }
               }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("frames","masks")
    FUNCTION = "zoom"
    CATEGORY = "Eden 🌱/Depth"

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

    def zoom(self, masks, image_slices, foreground_zoom_factor, background_zoom_factor, shift_left, n_frames, loop):
        # Adjust for n_frames and looping
        if loop:
            n_frames = n_frames * 2 - 1  # Double the frames minus 1 for looping

        foreground_zoom_factor = foreground_zoom_factor ** (1/((n_frames + 1) // 2))
        background_zoom_factor = background_zoom_factor ** (1/((n_frames + 1) // 2))
        shift_left = shift_left / ((n_frames + 1) // 2)

        masks = masks.numpy()
        image_slices = image_slices.numpy()

        # Extract the images and masks
        foreground_image = image_slices[0].copy()
        background_image = image_slices[1].copy()
        foreground_mask = masks[0,:,:,0].copy()

        frames, foreground_masks = [], []

        for i in range(n_frames):
            print(f"Processing frame {i+1}/{n_frames}")

            # Compute zoom and shift factors
            if loop and i >= (n_frames + 1) // 2:
                # Reverse direction for the second half when looping
                frame_index = n_frames - i - 1
            else:
                frame_index = i

            fg_zoom = foreground_zoom_factor ** frame_index
            bg_zoom = background_zoom_factor ** ((n_frames + 1) // 2 - frame_index - 1)
            fg_shift = -shift_left * frame_index

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


class ParallaxZoom_old:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"masks": ("IMAGE",),
                 "image_slices": ("IMAGE",),
                 "foreground_zoom_factor": ("FLOAT", {"default": 1.1, "step": 0.01}),
                 "background_zoom_factor": ("FLOAT", {"default": 1.05, "step": 0.01}),
                 "shift_left": ("FLOAT", {"default": 0.1, "step": 0.01}),
                 "n_frames": ("INT", {"default": 25}),
                }
               }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("frames","masks")
    FUNCTION = "zoom"
    CATEGORY = "Eden 🌱/Depth"

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

    def zoom(self, masks, image_slices, foreground_zoom_factor, background_zoom_factor, shift_left, n_frames):
        # Adjust for n_frames
        foreground_zoom_factor = foreground_zoom_factor ** (1/n_frames)
        background_zoom_factor = background_zoom_factor ** (1/n_frames)
        shift_left = shift_left / n_frames

        masks = masks.numpy()
        image_slices = image_slices.numpy()

        # Extract the images and masks
        foreground_image = image_slices[0].copy()
        background_image = image_slices[1].copy()
        foreground_mask = masks[0,:,:,0].copy()

        frames, foreground_masks = [], []

        for i in range(n_frames):
            print(f"Processing frame {i+1}/{n_frames}")

            # Compute zoom and shift factors
            fg_zoom = foreground_zoom_factor ** i
            bg_zoom = background_zoom_factor ** (n_frames - i - 1)
            fg_shift = -shift_left * i

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