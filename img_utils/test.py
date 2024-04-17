"""This module implements Histogram Matching operation"""
import sys
import numpy as np
import os
import cv2

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
class HistogramMatching():
    """Histogram Matching operation class"""

    def __init__(self):
        self.r_values = []
        self.r_counts = []
        self.r_quantiles = []

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


if __name__ == "__main__":
    
    hist_matcher = HistogramMatching()

    src_dir = "frames"
    ref_img = "tree.jpg"

    sources = [read_image(os.path.join(src_dir, img)) for img in sorted(os.listdir(src_dir))]
    reference = read_image(ref_img)

    hist_matcher.set_reference_img(reference)
    results = hist_matcher.match_images_to_reference(sources, 0.5)

    for i, result in enumerate(results):
        write_image(result, f"hist_match_{i}.jpg")
