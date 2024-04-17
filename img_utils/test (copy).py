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
        pass

    def set_reference_img(self, reference: np.ndarray) -> None:
        """ This function sets the reference image used for histogram matching """
        reference = reference.ravel()
        self.r_values, r_counts = np.unique(reference, return_counts=True)
        self.r_quantiles = np.cumsum(r_counts).astype(float) / (reference.size + sys.float_info.epsilon)

    def match_image_to_reference(self, 
                source: np.ndarray,
                reference: np.ndarray,
                match_prop: float = 1.0,
                channels = [0,1,2],
            ) -> np.ndarray:

        result = np.copy(source)
        for channel in channels:
            result[:, :, channel] = self.match_channel(source[:, :, channel], reference[:, :, channel], match_prop)

        return result.astype(np.float32)

    def match_channel(self, 
            source: np.ndarray,
            reference: np.ndarray,
            match_prop: float = 1.0
            ) -> np.ndarray:

        self.set_reference_img(reference)

        source_shape = source.shape
        source = source.ravel()

        # get unique pixel values (sorted),
        # indices of the unique array and counts
        _, s_indices, s_counts = np.unique(source, return_counts=True, return_inverse=True)

        # compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (source.size + sys.float_info.epsilon)

        # interpolate linearly to find the pixel values in the reference
        # that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, self.r_quantiles, self.r_values)

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
    ref_img = "trees.jpg"

    src = [read_image(os.path.join(src_dir, img)) for img in sorted(os.listdir(src_dir))]
    ref = read_image(ref_img)

    hist_match.set_reference_img(ref)
    results = hist_matcher.match_images_to_reference(src, 1.0)

    for i, result in enumerate(results):
        write_image(result, f"hist_match_{i}.jpg")
