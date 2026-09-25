import hashlib
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import comfy.utils
import folder_paths
from comfy.cli_args import args


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

MAX_RESOLUTION = 8192
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG")


def list_valid_images(folder):
    """Image files in `folder` (os.listdir order) whose header PIL can read."""
    valid = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not (path.endswith(IMG_EXTENSIONS) and os.path.isfile(path)):
            continue
        try:
            with Image.open(path):
                valid.append(path)
        except OSError as e:
            logging.warning(f"Skipping invalid image: {path} - {e}")
    return valid


def folder_fingerprint(folder):
    entries = sorted((e.name, e.stat().st_mtime_ns) for e in os.scandir(folder) if e.is_file())
    return hashlib.sha256(repr(entries).encode()).hexdigest()


def load_image_as_array(path):
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as e:  # malformed EXIF should not prevent loading the pixels
        logging.warning(f"Error during EXIF transpose for {path}: {e}")
    if img.mode == 'I':
        img = img.point(lambda i: i * (1 / 255))
    return np.array(img.convert("RGB")).astype(np.float32) / 255.0


def arrays_to_batch(imgs):
    if len(imgs) > 1:
        imgs = get_uniformly_sized_crops(imgs, target_n_pixels=1024**2)
    return torch.from_numpy(np.stack(imgs))


class Eden_RGBA_to_RGB:
    DESCRIPTION = "Converts RGBA images to RGB by alpha-blending them over a flat gray background. RGB inputs pass through unchanged."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to convert; only 4-channel (RGBA) images are changed."}),
                "background_color": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Gray level (0 = black, 1 = white) shown through transparent pixels."}),
            }
        }

    FUNCTION = "convert_rgba_to_rgb"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("RGB images.",)
    CATEGORY = "Eden 🌱/Image"

    def convert_rgba_to_rgb(self, images, background_color=0.0):
        if images.numel() == 0 or images.shape[-1] != 4:
            return (images,)
        rgb = images[..., :3].float()
        alpha = images[..., 3:].float()
        blended = alpha * rgb + (1 - alpha) * background_color
        return (torch.clamp(blended, 0, 1).to(images.dtype),)


class Eden_MaskBoundingBox:
    DESCRIPTION = "Crops a mask (and optionally an image) to the bounding box of the mask's non-zero pixels, after removing speckle noise. The box is taken from the first mask in the batch."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask whose non-zero region defines the crop."}),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "Extra pixels added around the bounding box."}),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Gaussian blur kernel size applied to the mask before finding the box (0 = off)."}),
                "noise_threshold": ("INT", { "default": 1, "min": 0, "max": 1000, "step": 1, "tooltip": "Minimum number of non-zero pixels in a 3x3 neighbourhood for a pixel to survive denoising."}),
            },
            "optional": {
                "image_optional": ("IMAGE", {"tooltip": "Image to crop with the same box; resized to the mask if needed. Defaults to the mask itself."}),
            }
        }
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    OUTPUT_TOOLTIPS = ("Cropped mask.", "Cropped image.", "Left edge of the box.", "Top edge of the box.", "Box width.", "Box height.")
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Mask"

    def execute(self, mask, padding, blur, noise_threshold, image_optional=None):
        import torchvision.transforms.functional as T
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        mask = self.reduce_noise(mask, 3, noise_threshold)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        y_indices, x_indices = torch.where(mask[0] > 0)
        if len(y_indices) > 0:
            x1 = max(0, x_indices.min().item() - padding)
            x2 = min(mask.shape[2], x_indices.max().item() + 1 + padding)
            y1 = max(0, y_indices.min().item() - padding)
            y2 = min(mask.shape[1], y_indices.max().item() + 1 + padding)
        else:
            x1, y1, x2, y2 = 0, 0, mask.shape[2], mask.shape[1]

        mask = mask[:, y1:y2, x1:x2]
        image_optional = image_optional[:, y1:y2, x1:x2, :]

        return (mask, image_optional, x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def reduce_noise(mask, kernel_size, threshold):
        mask = mask.unsqueeze(1)
        pooled = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        neighbor_count = F.conv2d(mask.float(), kernel, padding=kernel_size//2)
        return torch.where(neighbor_count >= threshold, pooled, torch.zeros_like(pooled)).squeeze(1)


class WidthHeightPicker:
    DESCRIPTION = "Scales a width/height pair by a multiplier and rounds both to the nearest multiple of a given number."

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"width":  ("INT", {"default": 512, "min": 0, "max": sys.maxsize, "tooltip": "Input width in pixels."}),
                     "height":  ("INT", {"default": 512, "min": 0, "max": sys.maxsize, "tooltip": "Input height in pixels."}),
                     "output_multiplier":  ("FLOAT", {"default": 0.5, "tooltip": "Factor applied to both width and height."}),
                     "multiple_off":  ("INT", {"default": 64, "min": 1, "max": 264, "tooltip": "Round the result to the nearest multiple of this."}),
                     }
                }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)
    OUTPUT_TOOLTIPS = ("Scaled, rounded width.", "Scaled, rounded height.")
    FUNCTION = "compute_resolution"
    OUTPUT_NODE = False
    CATEGORY = "Eden 🌱/Logic"

    def compute_resolution(self, width, height, output_multiplier, multiple_off):
        width = int(width * output_multiplier)
        height = int(height * output_multiplier)
        width = int(round(width / multiple_off) * multiple_off)
        height = int(round(height / multiple_off) * multiple_off)
        return width, height


class SaveImageAdvanced:
    DESCRIPTION = "Saves images as PNG to the output folder, optionally with a timestamp in the filename and a sidecar JSON of the workflow metadata."

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", {"tooltip": "Images to save."}),
                     "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Filename prefix; may include a subfolder (e.g. 'renders/shot')."}),
                     "add_timestamp": ("BOOLEAN", {"default": True, "tooltip": "Insert a YYYYMMDD-HHMMSS timestamp into each filename."}),
                     "save_metadata_json": ("BOOLEAN", {"default": True, "tooltip": "Also write the prompt/workflow metadata as a .json next to each PNG."}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Eden 🌱/Loaders"

    def save_images(self, images, add_timestamp, save_metadata_json, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(full_output_folder, exist_ok = True)

        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                metadata_dict = {}
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                    metadata_dict["prompt"] = prompt
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                        metadata_dict[x] = extra_pnginfo[x]

            if add_timestamp:
                file = f"{filename}_{timestamp_str}_{counter:05}.png"
            else:
                file = f"{filename}_{counter:05}_.png"

            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })

            if save_metadata_json and not args.disable_metadata:
                json_path = os.path.join(full_output_folder, file.replace(".png", ".json"))
                with open(json_path, "w") as f:
                    json.dump(metadata_dict, f, indent=4)

            counter += 1

        return { "ui": { "images": results } }


class LatentTypeConversion:
    DESCRIPTION = "Casts latent samples between float16 (half the memory) and float32."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to convert."}),
                "output_type": (["float16", "float32"], {"tooltip": "Target dtype for the latent samples."}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "Print dtype, shape, device and free RAM before and after."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("Latent with converted samples.",)
    FUNCTION = "convert"
    CATEGORY = "Eden 🌱/Latent"

    def convert(self, latent, output_type="float16", verbose=True):
        import psutil
        samples = latent["samples"]
        if verbose:
            logging.info(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
            logging.info(f"Input latent type: {samples.dtype}, shape: {tuple(samples.shape)}, device: {samples.device}")

        if output_type == "float32" and samples.dtype == torch.float16:
            samples = samples.float()
        elif output_type == "float16" and samples.dtype == torch.float32:
            samples = samples.half()

        if verbose:
            logging.info(f"After conversion, latent type: {samples.dtype}")
            logging.info(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")

        return ({**latent, "samples": samples},)


class VAEDecode_to_folder:
    DESCRIPTION = "Decodes latents one frame at a time and writes them as JPGs into a new timestamped subfolder, so long sequences never have to fit in memory as one image batch."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "samples": ("LATENT", {"tooltip": "Latent frames to decode."}),
                 "vae": ("VAE", {"tooltip": "VAE used for decoding."}),
                 "prefix": ("STRING", {"default": "test", "tooltip": "Name of the subfolder (a timestamp is appended)."}),
                 "output_folder": ("STRING", {"default": "output/frames", "tooltip": "Parent folder, relative to the ComfyUI working directory or absolute."}),
                }
            }
    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path of the folder the frames were written to.",)
    OUTPUT_NODE = True
    FUNCTION = "decode"
    CATEGORY = "Eden 🌱/Loaders"

    def decode(self, vae, samples, prefix, output_folder):
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        output_folder = os.path.join(output_folder, f"{prefix}_{timestamp_str}")
        os.makedirs(output_folder, exist_ok=True)

        latents = samples["samples"]
        pbar = comfy.utils.ProgressBar(len(latents))
        for i, sample in enumerate(latents):
            img = vae.decode(sample.unsqueeze(0))
            img = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img.squeeze()).save(os.path.join(output_folder, f"{i:06d}.jpg"), quality=95)
            pbar.update(1)

        return (output_folder, )


class Eden_MaskCombiner:
    DESCRIPTION = "Blends up to three masks with signed relative strengths (negative = inverted), then stretches the result between two percentiles with a soft clamp."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_a": ("MASK", {"tooltip": "First mask."}),
                "rel_strength_a": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Weight of mask A; negative values use the inverted mask, 0 ignores it."}),
                "rel_strength_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Weight of mask B; negative values use the inverted mask, 0 ignores it."}),
                "rel_strength_c": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Weight of mask C; negative values use the inverted mask, 0 ignores it."}),
                "lower_clamp": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Percentile of the blend mapped to (roughly) black."}),
                "upper_clamp": ("FLOAT", {"default": 98.0, "min": 50.0, "max": 100.0, "step": 0.5, "tooltip": "Percentile of the blend mapped to (roughly) white."}),
                "gamma": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Gamma applied to each mask before blending (inverted afterwards)."})
            },
            "optional": {
                "mask_b": ("MASK", {"default": None, "tooltip": "Optional second mask."}),
                "mask_c": ("MASK", {"default": None, "tooltip": "Optional third mask."})
            }
        }

    RETURN_TYPES = ("MASK",)
    OUTPUT_TOOLTIPS = ("Combined mask.",)
    FUNCTION = "combine_masks"
    CATEGORY = "Eden 🌱/Mask"

    def soft_clamp(self, x, min_val, max_val, smoothness=0.1):
        normalized = (x - min_val) / (max_val - min_val)
        return torch.sigmoid((normalized - 0.5) / smoothness)

    def compute_quantile(self, tensor, q, max_elements = 10000):
        """torch.quantile has an input size limit, so estimate it from a random sample on large tensors."""
        if tensor.numel() > max_elements:
            indices = torch.randperm(tensor.numel(), device=tensor.device)[:max_elements]
            return torch.quantile(tensor.reshape(-1)[indices], q)
        return torch.quantile(tensor, q)

    def combine_masks(self, mask_a, rel_strength_a, lower_clamp, upper_clamp, gamma,
                     mask_b=None, mask_c=None, rel_strength_b=None, rel_strength_c=None):
        mask_a = torch.pow(mask_a, gamma)
        masks, weights = [], []
        for mask, strength, is_a in ((mask_a, rel_strength_a, True), (mask_b, rel_strength_b, False), (mask_c, rel_strength_c, False)):
            if mask is None or not strength:
                continue
            if not is_a:
                mask = torch.pow(mask, gamma)
            masks.append(1 - mask if strength < 0 else mask)
            weights.append(abs(strength))

        if not masks:
            return (torch.full_like(mask_a, 0.5),)

        weights = F.softmax(torch.tensor(weights, device=mask_a.device), dim=0)
        combined = torch.zeros_like(mask_a)
        for mask, weight in zip(masks, weights):
            combined += mask * weight

        lower_threshold = self.compute_quantile(combined, lower_clamp/100)
        upper_threshold = self.compute_quantile(combined, upper_clamp/100)
        combined = self.soft_clamp(combined, lower_threshold, upper_threshold)

        return (torch.pow(combined, 1/gamma),)


def detect_faces(img_array):
    """Returns a uint8 HxW mask that is 255 inside every face box MediaPipe finds in an RGB uint8 image."""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Eden_FaceToMask needs a mediapipe version that still ships the legacy `mediapipe.solutions` API.")

    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)

    for detection in results.detections or []:
        bbox = detection.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        width = min(int(bbox.width * w), w - x)
        height = min(int(bbox.height * h), h - y)
        mask[y:y+height, x:x+width] = 255
    return mask


class Eden_FaceToMask:
    DESCRIPTION = "Detects faces with MediaPipe and returns a mask with a filled rectangle over each detected face."

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", {"tooltip": "Images to detect faces in."})},
                }

    RETURN_TYPES = ("MASK",)
    OUTPUT_TOOLTIPS = ("Face-box mask, one per input image.",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/Mask"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def run(self, image):
        images = np.clip(255. * image[..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
        masks = [detect_faces(np.ascontiguousarray(img)) for img in images]
        return (torch.from_numpy(np.stack(masks).astype(np.float32) / 255.0),)


class Eden_Face_Crop:
    DESCRIPTION = "Finds the largest blob in a (face) mask, pads its bounding box, and crops the image to it. Also returns the crop location and masks to paste the result back."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to crop."}),
                "face_mask": ("MASK", {"tooltip": "Hard or soft face mask; the first mask in the batch picks the crop."}),
                "padding_factor": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Crop size relative to the face box (1.2 = 20% larger)."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mask values above this count as face."}),
                "min_face_ratio": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip": "Ignore blobs smaller than (ratio x longest image side)^2 pixels."}),
            }
        }

    FUNCTION = "crop_face"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "MASK", "MASK")
    RETURN_NAMES = ("cropped_face", "crop_x", "crop_y", "crop_width", "crop_height", "crop_mask", "custom_mask")
    OUTPUT_TOOLTIPS = ("Cropped face image (the full image if no face was found).", "Left edge of the crop.", "Top edge of the crop.", "Crop width.", "Crop height.", "Full-size mask that is 1 inside the crop.", "The input face mask cut to the crop, normalized to max 1.")
    CATEGORY = "Eden 🌱/Image"

    def find_main_face_bbox(self, mask_np, threshold=0.5, min_face_ratio=0.01):
        """Bounding box (x_min, y_min, x_max, y_max), inclusive, of the largest valid blob in a HxW mask, or None."""
        from scipy import ndimage

        if mask_np.max() > 1.0 + 1e-6:
            mask_np = mask_np / 255.0
        labeled, num_features = ndimage.label(mask_np > threshold)
        if num_features == 0:
            return None

        min_face_size = (max(mask_np.shape) * min_face_ratio) ** 2
        sizes = np.bincount(labeled.ravel())
        valid_regions = []
        for label, (sy, sx) in enumerate(ndimage.find_objects(labeled), start=1):
            if sizes[label] < min_face_size:
                continue
            x_min, y_min, x_max, y_max = sx.start, sy.start, sx.stop - 1, sy.stop - 1
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0:
                continue
            valid_regions.append((width * height, (x_min, y_min, x_max, y_max)))

        return max(valid_regions)[1] if valid_regions else None

    def apply_padding(self, bbox, image_shape, padding_factor):
        x_min, y_min, x_max, y_max = bbox
        H, W = image_shape

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        target_width = max(1, int((x_max - x_min) * padding_factor))
        target_height = max(1, int((y_max - y_min) * padding_factor))

        # Make square if dimensions are within 20% of each other
        if min(target_width, target_height) / max(target_width, target_height) > 0.8:
            target_width = target_height = max(target_width, target_height)

        new_x_min = max(0, center_x - target_width // 2)
        new_x_max = min(W - 1, center_x + (target_width - target_width // 2))
        new_y_min = max(0, center_y - target_height // 2)
        new_y_max = min(H - 1, center_y + (target_height - target_height // 2))

        # At an image edge, shift the box inwards to keep the target size
        if new_x_min == 0:
            new_x_max = min(W - 1, new_x_min + target_width)
        if new_x_max == W - 1:
            new_x_min = max(0, new_x_max - target_width)
        if new_y_min == 0:
            new_y_max = min(H - 1, new_y_min + target_height)
        if new_y_max == H - 1:
            new_y_min = max(0, new_y_max - target_height)

        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            return bbox

        new_x_min = max(0, min(new_x_min, W - 2))
        new_x_max = max(new_x_min + 1, min(new_x_max, W - 1))
        new_y_min = max(0, min(new_y_min, H - 2))
        new_y_max = max(new_y_min + 1, min(new_y_max, H - 1))
        return new_x_min, new_y_min, new_x_max, new_y_max

    def crop_face(self, image, face_mask, padding_factor=1.2, threshold=0.5, min_face_ratio=0.01):
        if face_mask.dim() == 2:
            face_mask = face_mask.unsqueeze(0)
        mask_hw = face_mask.shape[1:]

        bbox = self.find_main_face_bbox(face_mask[0].cpu().numpy(), threshold, min_face_ratio)
        if bbox is None:
            empty_mask = torch.zeros((1, *mask_hw))
            return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)

        x_min, y_min, x_max, y_max = self.apply_padding(bbox, mask_hw, padding_factor)

        image_height, image_width = image.shape[1:3]
        x_min = int(max(0, min(x_min, image_width - 1)))
        y_min = int(max(0, min(y_min, image_height - 1)))
        x_max = int(max(x_min + 1, min(x_max, image_width)))
        y_max = int(max(y_min + 1, min(y_max, image_height)))

        cropped = image[:, y_min:y_max, x_min:x_max, :]

        crop_mask = torch.zeros_like(face_mask)
        crop_mask[:, y_min:y_max, x_min:x_max] = 1.0

        custom_mask = face_mask[:, y_min:y_max, x_min:x_max]
        if custom_mask.shape[0] != cropped.shape[0]:
            if custom_mask.shape[0] == 1:
                custom_mask = custom_mask.expand(cropped.shape[0], -1, -1)
            else:
                custom_mask = torch.ones(cropped.shape[:3])
        if custom_mask.max() > 0:
            custom_mask = custom_mask / custom_mask.max()

        return (cropped, x_min, y_min, x_max - x_min, y_max - y_min, crop_mask, custom_mask)


def match_batch(t, n):
    """Truncate t to n items, or pad it by repeating its last item."""
    if t.shape[0] >= n:
        return t[:n]
    return torch.cat((t, t[-1:].expand(n - t.shape[0], *t.shape[1:])), dim=0)


class Eden_ImageMaskComposite:
    DESCRIPTION = "Pastes a source image onto a destination image at (x + offset_x, y + offset_y), blended through an optional mask. The mask is resized to the source and batches are matched to the source."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE", {"tooltip": "Background image."}),
                "source": ("IMAGE", {"tooltip": "Image pasted on top."}),
                "x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Left position of the source in the destination."}),
                "y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Top position of the source in the destination."}),
                "offset_x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Extra horizontal shift added to x."}),
                "offset_y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Extra vertical shift added to y."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Blend mask in source coordinates (1 = source, 0 = destination). Defaults to all ones."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Composited image.",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Image"

    def execute(self, destination, source, x, y, offset_x, offset_y, mask=None):
        batch, h_src, w_src = source.shape[:3]
        if mask is None:
            mask = torch.ones((batch, h_src, w_src), device=source.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)

        if mask.shape[1:3] != source.shape[1:3]:
            mask = F.interpolate(mask.movedim(-1, 1), size=(h_src, w_src), mode='bicubic', align_corners=False).movedim(1, -1)

        mask = match_batch(mask, batch)
        destination = match_batch(destination, batch)

        x += offset_x
        y += offset_y
        h_dst, w_dst = destination.shape[1:3]
        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(w_dst, x + w_src), min(h_dst, y + h_src)

        output = destination.clone()
        if x_end > x_start and y_end > y_start:
            src_y, src_x = slice(y_start - y, y_end - y), slice(x_start - x, x_end - x)
            m = mask[:, src_y, src_x]
            d = output[:, y_start:y_end, x_start:x_end]
            output[:, y_start:y_end, x_start:x_end] = source[:, src_y, src_x] * m + d * (1 - m)
        return (output,)


def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))


def get_centre_crop(img, aspect_ratio):
    h, w = img.shape[:2]
    if w/h > aspect_ratio:
        new_w = int(h * aspect_ratio)
        return img[:, (w - new_w) // 2:(w + new_w) // 2]
    new_h = int(w / aspect_ratio)
    return img[(h - new_h) // 2:(h + new_h) // 2, :]


def get_uniformly_sized_crops(imgs, target_n_pixels=2048**2):
    """
    Given a list of HxWx3 arrays:
        - extract the best possible centre crop of same aspect ratio for all images
        - rescale these crops to have ~target_n_pixels
    """
    import cv2

    final_aspect_ratio = np.mean([img.shape[1] / img.shape[0] for img in imgs])
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs]

    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = round_to_nearest_multiple(final_h * final_aspect_ratio, 8)
    final_h = round_to_nearest_multiple(final_h, 8)

    # cv2.INTER_CUBIC used to be passed positionally into the `dst` slot, so resizing has always been bilinear.
    return [cv2.resize(crop, (final_w, final_h), interpolation=cv2.INTER_LINEAR) for crop in crops]


class LoadRandomImage:
    DESCRIPTION = "Loads a seeded random selection of images from a folder. Multiple images are centre-cropped to a shared aspect ratio and resized to ~1 megapixel so they form one batch."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "folder": ("STRING", {"default": ".", "tooltip": "Folder to load images from (png, jpg, jpeg, bmp, webp)."}),
                    "n_images": ("INT", {"default": 1, "min": -1, "max": 100, "tooltip": "How many images to load; -1 loads all of them."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Seed for the random pick. With sort on and n_images = 1 it is used as an index into the sorted file list."}),
                    "sort": ("BOOLEAN", {"default": False, "tooltip": "Return the picked images in filename order."}),
                    "loop_sequence": ("BOOLEAN", {"default": False, "tooltip": "Append the first image again at the end (for looping animations)."}),
                }
        }

    CATEGORY = "Eden 🌱/Loaders"
    RETURN_TYPES = ("IMAGE", any_typ, any_typ, "STRING")
    RETURN_NAMES = ("Image(s)", "paths", "filenames", "filenames[0]_str")
    OUTPUT_TOOLTIPS = ("Loaded image batch.", "List of full paths.", "List of filenames.", "Filename of the first image.")
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return folder_fingerprint(folder)

    def load(self, folder, n_images, seed, sort, loop_sequence):
        valid_image_paths = list_valid_images(folder)
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")

        if sort and n_images == 1:
            valid_image_paths = [sorted(valid_image_paths)[seed % len(valid_image_paths)]]
        else:
            random.Random(seed).shuffle(valid_image_paths)
            if n_images > 0:
                valid_image_paths = valid_image_paths[:n_images]
            if sort:
                valid_image_paths = sorted(valid_image_paths)

        imgs = [load_image_as_array(p) for p in valid_image_paths]
        paths = list(valid_image_paths)
        filenames = [os.path.basename(p) for p in paths]

        if loop_sequence and len(imgs) > 1:
            imgs.append(imgs[0])
            paths.append(paths[0])
            filenames.append(filenames[0])

        return (arrays_to_batch(imgs), paths, filenames, str(filenames[0]))


class ImageFolderIterator:
    DESCRIPTION = "Loads the image at a given index from a folder (wrapping around past the end). Drive the index with a counter to step through a folder."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ".", "tooltip": "Folder to load images from (png, jpg, jpeg, bmp, webp)."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "Which image to load; wraps around modulo the number of images."}),
                "sort": ("BOOLEAN", {"default": True, "tooltip": "Order files by name (otherwise filesystem order)."}),
            }
        }

    CATEGORY = "Eden 🌱/Loaders"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    OUTPUT_TOOLTIPS = ("The loaded image.", "Its filename without extension.")
    FUNCTION = "load_image"

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return folder_fingerprint(folder)

    def load_image(self, folder, index, sort):
        valid_image_paths = list_valid_images(folder)
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")
        if sort:
            valid_image_paths = sorted(valid_image_paths)

        image_path = valid_image_paths[index % len(valid_image_paths)]
        output_image = torch.from_numpy(load_image_as_array(image_path))[None,]
        filename = os.path.splitext(os.path.basename(image_path))[0]
        return (output_image, filename)


class LoadImagesByFilename:
    DESCRIPTION = "Loads images from a list of file paths (e.g. the 'paths' output of Load Random Image), with optional seeded shuffling and a max count."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "filename": ("COMBO", {"default": [], "tooltip": "List of image file paths."}),
                    "max_num_images": ("INT", {"default": None, "min": 0, "max": sys.maxsize, "tooltip": "Load at most this many images; 0 loads all."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Seed for shuffling the list before picking."}),
                    "sort": ("BOOLEAN", {"default": False, "tooltip": "Return the picked images in path order."}),
                    "loop_sequence": ("BOOLEAN", {"default": False, "tooltip": "Append the first image again at the end (for looping animations)."}),
                }
        }

    CATEGORY = "Eden 🌱/Loaders"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Loaded image batch.",)
    FUNCTION = "load_image"

    def load_image(self, filename: list, max_num_images, seed, sort, loop_sequence):
        files = list(filename)
        random.Random(seed).shuffle(files)
        image_paths = files[:max_num_images or None]
        if sort:
            image_paths = sorted(image_paths)

        output_images = [load_image_as_array(p) for p in image_paths]
        if loop_sequence:
            output_images.append(output_images[0])
        return (arrays_to_batch(output_images),)


class GetRandomFile:
    DESCRIPTION = "Returns the path of a seeded random file from a folder."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "folder": ("STRING", {"default": ".", "tooltip": "Folder to pick a file from."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Seed for the random pick."}),
                }
        }

    CATEGORY = "Eden 🌱/Loaders"
    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path of the picked file.",)
    FUNCTION = "get_path"

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return folder_fingerprint(folder)

    def get_path(self, folder, seed):
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files if os.path.isfile(f)]
        return (random.Random(seed).choice(files),)


class IMG_resolution_multiple_of:
    DESCRIPTION = "Crops the bottom/right edge of an image so width and height become multiples of a number."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to crop."}),
                "multiple_of": ("INT", {"default": 8, "min": 2, "max": 264, "tooltip": "Width and height are cropped down to a multiple of this."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Cropped image.",)
    FUNCTION = "pad"
    CATEGORY = "Eden 🌱/Image"

    def pad(self, image, multiple_of):
        h, w = image.shape[1:3]
        return (image[:, :h - h % multiple_of, :w - w % multiple_of, :],)


class IMG_padder:
    DESCRIPTION = "Adds a flat border on one side of an image, sized as a fraction of the image and colored with the mean of the adjacent 4-pixel edge."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to pad."}),
                "pad_fraction": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01, "tooltip": "Border size as a fraction of the image height (top/bottom) or width (left/right)."}),
                "pad_location": (["bottom", "top", "left", "right"], {"tooltip": "Side to add the border to."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Padded image.",)
    FUNCTION = "pad"
    CATEGORY = "Eden 🌱/Image"

    def pad(self, image, pad_fraction, pad_location="bottom"):
        bs, h, w, c = image.shape
        color_mean_w = 4  # pixels

        if pad_location in ("bottom", "top"):
            shape = (bs, int(h * pad_fraction), w, c)
            edge = image[:, -color_mean_w:] if pad_location == "bottom" else image[:, :color_mean_w]
        else:
            shape = (bs, h, int(w * pad_fraction), c)
            edge = image[:, :, :color_mean_w] if pad_location == "left" else image[:, :, -color_mean_w:]
        border = torch.full(shape, edge.mean().item(), dtype=image.dtype, device=image.device)

        if pad_location in ("bottom", "right"):
            parts = (image, border)
        else:
            parts = (border, image)
        return (torch.cat(parts, dim=1 if pad_location in ("bottom", "top") else 2),)


class IMG_blender:
    DESCRIPTION = "Linearly blends two image batches; mismatched sizes are cropped to the smaller one."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "First image batch."}),
                "image2": ("IMAGE", {"tooltip": "Second image batch (same batch size)."}),
                "image1_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01, "tooltip": "Weight of image1; image2 gets 1 - weight."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Blended image.",)
    FUNCTION = "blend"
    CATEGORY = "Eden 🌱/Image"

    def blend(self, image1, image2, image1_weight = 0.5):
        if image1.shape[0] != image2.shape[0]:
            raise ValueError("Images must have the same batch size for blending!")

        h = min(image1.shape[1], image2.shape[1])
        w = min(image1.shape[2], image2.shape[2])
        image1 = image1[:, :h, :w, :]
        image2 = image2[:, :h, :w, :]
        return (image1 * image1_weight + image2 * (1 - image1_weight),)


class IMG_unpadder:
    DESCRIPTION = "Removes a fraction of the image from one side (the inverse of the padder), then crops width and height to multiples of 4."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to crop."}),
                "unpad_fraction": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01, "tooltip": "Fraction of the height (top/bottom) or width (left/right) to remove."}),
                "unpad_location": (["bottom", "top", "left", "right"], {"tooltip": "Side to remove from."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Cropped image.",)
    FUNCTION = "unpad"
    CATEGORY = "Eden 🌱/Image"

    def unpad(self, image, unpad_fraction, unpad_location = "bottom"):
        h, w = image.shape[1:3]

        if unpad_location == "bottom":
            image = image[:, :int(h * (1 - unpad_fraction)), :, :]
        elif unpad_location == "top":
            image = image[:, int(h * unpad_fraction):, :, :]
        elif unpad_location == "left":
            image = image[:, :, int(w * unpad_fraction):, :]
        elif unpad_location == "right":
            image = image[:, :, :int(w * (1 - unpad_fraction)), :]

        h, w = image.shape[1:3]
        return (image[:, :h - h % 4, :w - w % 4, :],)


def _math_fn(fn):
    return lambda v: fn(torch.as_tensor(v))

IMG_MATH_FUNCTIONS = {name: _math_fn(getattr(torch, name)) for name in ("sin", "cos", "exp", "sqrt", "abs", "log", "tanh")}


class IMG_scaler:
    DESCRIPTION = "Applies a math expression to every pixel value, e.g. 'x * 1.2' or '0.5 + 0.5 * sin(6.28 * x)'. The result is clamped to 0..1."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to transform."}),
                 "math_string": ("STRING", {"default": "", "tooltip": "Expression in x (the pixel value, 0..1). Available: sin, cos, exp, sqrt, abs, log, tanh. Empty = unchanged."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Transformed image.",)
    FUNCTION = "apply_math"
    CATEGORY = "Eden 🌱/Image"

    def apply_math(self, image, math_string):
        input_dtype = image.dtype
        if image.max() > 1:
            image = image.float() / 255.0

        if math_string.strip():
            result = eval(math_string, {"__builtins__": None}, {**IMG_MATH_FUNCTIONS, "x": image})  # noqa: S307
            result = torch.as_tensor(result, device=image.device).expand_as(image)
        else:
            result = image

        result = torch.clamp(result, 0, 1)
        if input_dtype == torch.uint8:
            result = result * 255
        return (result.to(input_dtype),)


def to_grayscale(images, keep_dims=True, alpha_channel_convert_to=None):
    """
    Convert a batch of RGB or RGBA images [B,H,W,C] to grayscale.
    With keep_dims the result has 3 channels; transparent pixels are then blended towards alpha_channel_convert_to.
    """
    if images.shape[-1] not in [3, 4]:
        raise ValueError("Input images must have 3 (RGB) or 4 (RGBA) channels.")

    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device)
    rgb = images[..., :3]
    alpha = images[..., 3:] if images.shape[-1] == 4 else None

    grayscale_images = torch.tensordot(rgb.permute(0, 3, 1, 2), weights, dims=([1], [0])).unsqueeze(-1)

    if keep_dims:
        grayscale_images = grayscale_images.repeat(1, 1, 1, 3)
        if alpha is not None and alpha_channel_convert_to is not None:
            grayscale_images = grayscale_images * alpha + alpha_channel_convert_to * (1 - alpha)

    return grayscale_images


class ConvertToGrayscale:
    DESCRIPTION = "Converts images to 3-channel grayscale. RGBA images are flattened onto a gray background."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to convert."}),
                "alpha_channel_convert_to": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01, "tooltip": "Gray level shown through transparent pixels of RGBA inputs."})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Grayscale image.",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "Eden 🌱/Image"

    def convert_to_grayscale(self, image, alpha_channel_convert_to):
        c = image.shape[-1]
        if c == 1:
            return (image,)
        if c not in [3, 4]:
            raise ValueError(f"Input image must have 1, 3, or 4 channels, but got {c} channels. Image shape = {image.shape}")
        return (to_grayscale(image, keep_dims=True, alpha_channel_convert_to=alpha_channel_convert_to),)


class AspectPadImageForOutpainting:
    DESCRIPTION = "Resizes an image to fit inside an SDXL-sized canvas of the chosen aspect ratio and returns the padding amounts to feed into ComfyUI's 'Pad Image for Outpainting'."

    ASPECT_RATIO_MAP = {
        "1-1_square_1024x1024": (1024, 1024),
        "4-3_landscape_1152x896": (1152, 896),
        "3-2_landscape_1216x832": (1216, 832),
        "16-9_landscape_1344x768": (1344, 768),
        "21-9_landscape_1536x640": (1536, 640),
        "3-4_portrait_896x1152": (896, 1152),
        "2-3_portrait_832x1216": (832, 1216),
        "9-16_portrait_768x1344": (768, 1344),
        "9-21_portrait_640x1536": (640, 1536),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to place on the canvas."}),
                "aspect_ratio": (list(s.ASPECT_RATIO_MAP.keys()), {"default": "16-9_landscape_1344x768", "tooltip": "Target canvas size."}),
                "justification": (["top-left", "center", "bottom-right"], {"default": "center", "tooltip": "Where the resized image sits on the canvas."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE","LEFT","TOP","RIGHT","BOTTOM")
    OUTPUT_TOOLTIPS = ("Resized image.", "Left padding.", "Top padding.", "Right padding.", "Bottom padding.")
    FUNCTION = "fit_and_calculate_padding"
    CATEGORY = "Eden 🌱/Image"

    def fit_and_calculate_padding(self, image, aspect_ratio, justification):
        h, w = image.shape[1:3]
        canvas_width, canvas_height = self.ASPECT_RATIO_MAP[aspect_ratio]

        image_aspect_ratio = w / h
        if image_aspect_ratio > canvas_width / canvas_height:
            new_width = canvas_width
            new_height = int(canvas_width / image_aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * image_aspect_ratio)

        resized_image = F.interpolate(image.permute(0, 3, 1, 2), size=(new_height, new_width), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)

        pad_w = canvas_width - new_width
        pad_h = canvas_height - new_height
        if justification == "center":
            left, top = pad_w // 2, pad_h // 2
        elif justification == "top-left":
            left, top = 0, 0
        else:
            left, top = pad_w, pad_h

        return (resized_image, left, top, pad_w - left, pad_h - top)


class Extend_Sequence:
    DESCRIPTION = "Extends (or trims) an image sequence to a target frame count by looping it or playing it back and forth."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames to extend."}),
                "target_n_frames": ("INT", {"default": 24, "min": 1, "step": 1, "max": sys.maxsize, "tooltip": "Number of output frames."}),
                "mode": (["wrap_around", "ping_pong"], {"tooltip": "wrap_around: 1 2 3 1 2 3 ... ; ping_pong: 1 2 3 3 2 1 1 2 ..."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Extended frame sequence.",)
    FUNCTION = "process_sequence"
    CATEGORY = "Eden 🌱/Video"

    def process_sequence(self, images, target_n_frames, mode="wrap_around"):
        n_frames = images.shape[0]
        indices = torch.arange(target_n_frames, device=images.device)
        if mode == "ping_pong":
            indices = indices % (2 * n_frames)
            indices = torch.where(indices >= n_frames, 2 * n_frames - indices - 1, indices)
        else:
            indices = indices % n_frames
        return (images[indices],)
