import torch
import comfy.model_management


def _cdf(values):
    """Sorted unique values of a 1D tensor, their quantiles, and the inverse index back into `values`."""
    uniq, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    quantiles = torch.cumsum(counts, 0).double() / values.numel()
    return uniq, quantiles, inverse


def _interp(x, xp, fp):
    """torch equivalent of np.interp for increasing xp."""
    if xp.numel() == 1:
        return fp.expand_as(x).clone()
    idx = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


class HistogramMatching:
    """Classic histogram matching (quantile mapping), based on
    https://github.com/continental/image-statistics-matching"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE", {"tooltip": "Reference image(s) whose color histogram is matched. All frames are pooled."}),
                "target_images":  ("IMAGE", {"tooltip": "Images to recolor; each frame is matched independently."}),
                "matching_fraction": ("FLOAT", {"default": 0.75, "min": 0, "max": 1, "step": 0.01, "tooltip": "Blend between the original (0) and the fully matched result (1)."}),
                "channels": ("STRING", {"default": "0,1,2", "tooltip": "Comma-separated RGB channel indices to match (0=R, 1=G, 2=B)."})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hist_match"
    CATEGORY = "Eden 🌱/Image"
    DESCRIPTION = "Matches the per-channel color histogram of the target images to a reference image."
    OUTPUT_TOOLTIPS = ("Target images with the reference's color distribution.",)

    def hist_match(self, ref_image, target_images, matching_fraction, channels):
        device = comfy.model_management.get_torch_device()
        channels = [int(c) for c in channels.split(',') if c.strip()]
        ref = ref_image.to(device, torch.float32)
        result = target_images.to(device, torch.float32).clone()

        for c in channels:
            r_values, r_quantiles, _ = _cdf(ref[..., c].reshape(-1))
            r_values = r_values.double()
            for i in range(result.shape[0]):
                source = result[i, ..., c]
                _, s_quantiles, s_inverse = _cdf(source.reshape(-1))
                matched = _interp(s_quantiles, r_quantiles, r_values).clamp(0, 1)[s_inverse].view_as(source).float()
                result[i, ..., c] = torch.lerp(source, matched, matching_fraction)

        return (result.to(comfy.model_management.intermediate_device()),)
