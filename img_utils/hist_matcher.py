"""This module implements Histogram Matching operation"""
import sys
import numpy as np
import os
import cv2
import torch

"""This module contains constants used in the command line interface"""
# color spaces
GRAY = 'gray'
HSV = 'hsv'
LAB = 'lab'
RGB = 'rgb'

# image channels
IMAGE_CHANNELS = '0,1,2'

# image matching operations
MAX_VALUE_8_BIT = 255

def read_image(path: str) -> np.ndarray:
    """ This function reads an image and transforms it to RGB color space """
    if os.path.exists(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32) / MAX_VALUE_8_BIT
        if image.ndim == 2:
            return image[:, :, np.newaxis]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError(f'Invalid image path {path}')

def write_image(image: np.ndarray, path: str) -> None:
    """ This function transforms an image to BGR color space
    and writes it to disk """
    if image.dtype == np.float32:
        image = (image * MAX_VALUE_8_BIT).astype(np.uint8)
    if image.dtype == np.uint8:
        if image.shape[-1] == 1:
            output_image = image[:, :, 0]
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, output_image):
            raise ValueError(
                f'Output directory {os.path.dirname(path)} does not exist')
    else:
        raise TypeError(
            f'Cannot write image with type {image.dtype}')




class HistogramMatching:
    """Histogram Matching operation class"""
    """ inspired by
    https://github.com/continental/image-statistics-matching/tree/master
    """
    def __init__(self):
        self.r_values = []
        self.r_counts = []
        self.r_quantiles = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "target_images":  ("IMAGE",),
                "matching_fraction": ("FLOAT", {"default": 0.75, "min": 0, "max": 1, "step": 0.01}),
                "channels": ("STRING", {"default": "0,1,2"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hist_match"
    CATEGORY = "Eden 🌱"

    def hist_match(self, ref_image, target_images, matching_fraction, channels):
        """ This function matches the histogram of the source image to the reference image """

        print(f"Input shapes:")
        print(ref_image.shape)
        print(target_images.shape)

        # Convert the input torch tensors to numpy arrays:
        ref_image     = ref_image.cpu().numpy().astype(np.float32)
        target_images = target_images.cpu().numpy().astype(np.float32)

        # clip to 0-255:
        ref_image     = np.clip(255. * ref_image, 0, 255)
        target_images = np.clip(255. * target_images, 0, 255)

        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        target_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in target_images]

        # extract the channels list:
        channels = list(map(int, channels.split(',')))

        # set the reference image:
        self.set_reference_img(ref_image, channels)

        # match the images to the reference:
        results = self.match_images_to_reference(target_images, matching_fraction, channels)

        # convert back to BGR:
        results = [cv2.cvtColor(result, cv2.COLOR_RGB2BGR) for result in results]

        # convert results back to torch tensors:
        results = [torch.from_numpy(result) * 255. for result in results]
        results = torch.stack(results)

        print(results.shape)
        return (results,)


    def set_reference_img(self, reference: np.ndarray, channels=[0, 1, 2]) -> None:
        """ This function sets the reference image used for histogram matching """
        for channel in channels:
            ref_channel = reference[:, :, channel].ravel()
            values, counts = np.unique(ref_channel, return_counts=True)
            quantiles = np.cumsum(counts).astype(float) / (ref_channel.size + sys.float_info.epsilon)
            self.r_values.append(values)
            self.r_counts.append(counts)
            self.r_quantiles.append(quantiles)

    def match_images_to_reference(self, 
                sources: list,
                match_prop: float = 1.0,
                channels = [0,1,2],
            ) -> list:

        results = []
        for source in sources:
            result = np.copy(source)
            for channel in channels:
                result[:, :, channel] = self.match_channel(source[:, :, channel], channel, match_prop)
            results.append(result.astype(np.float32))

        return results

    def match_channel(self, 
            source: np.ndarray,
            channel: int,
            match_prop: float = 1.0
            ) -> np.ndarray:

        source_shape = source.shape
        source = source.ravel()

        # get unique pixel values (sorted),
        # indices of the unique array and counts
        _, s_indices, s_counts = np.unique(source, return_counts=True, return_inverse=True)

        # compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (source.size + sys.float_info.epsilon)

        # interpolate linearly to find the pixel values in the reference
        # that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, self.r_quantiles[channel], self.r_values[channel])

        # clip the interpolated values to the valid range
        interp_values = np.clip(interp_values, 0, 1)

        # pick the interpolated pixel values using the inverted source indices
        result = interp_values[s_indices]

        # apply matching proportion
        diff   = source.astype(float) - result
        result = source.astype(float) - (diff * match_prop)

        return result.reshape(source_shape)