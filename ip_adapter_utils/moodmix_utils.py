import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def get_id_from_filename(filename):
    """'folder/abc.jpg' -> 'abc'"""
    return os.path.splitext(os.path.basename(str(filename)))[0]


def _folder_signature(folder, filename_filter=lambda name: True):
    """(name, mtime, size) of every matching file, used as IS_CHANGED fingerprint."""
    if not os.path.isdir(folder):
        return None
    return sorted((e.name, e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(folder) if e.is_file() and filename_filter(e.name))


def centre_crop_images(images, target_resolution):
    processed_images = []
    for img in images:
        width, height = img.size
        new_side = min(width, height)
        left = (width - new_side) / 2
        top = (height - new_side) / 2
        img_cropped = img.crop((left, top, left + new_side, top + new_side))
        processed_images.append(img_cropped.resize((target_resolution, target_resolution), Image.BICUBIC))
    return processed_images


class Get_Prefixed_Imgs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": os.path.basename(folder_paths.get_output_directory()), "tooltip": "Folder to search (relative to the ComfyUI folder or absolute)."}),
                "filename_prefix": ("STRING", {"default": "Prefered_Images", "tooltip": "Only files whose name contains this text are used."}),
                "max_n_imgs": ("INT", {"default": 1, "tooltip": "Load at most this many images, last in name order first."}),
                "seed": ("INT", {"default": 0, "tooltip": "Unused by the node; change it to force a re-run."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("anchor_images", "use_anchor")
    OUTPUT_TOOLTIPS = ("Matching images, centre-cropped to 512x512 (a random 256x256 dummy if none).", "1.0 when images were found, else 0.0.")
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/Loaders"
    DESCRIPTION = "Loads the most recent images whose filename contains a prefix (e.g. user-preferred renders) as anchor images."

    @classmethod
    def IS_CHANGED(s, folder, filename_prefix, max_n_imgs, seed=0):
        return seed, _folder_signature(folder, lambda name: filename_prefix in name)

    def run(self, folder, filename_prefix, max_n_imgs, seed=0):
        if not os.path.exists(folder):
            logging.warning(f"Get_Prefixed_Imgs: couldn't find folder {folder}, not using anchor images.")
            return (torch.rand(1, 256, 256, 3), 0.0)

        # Most recent first (assumes a counter in the filename):
        img_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if filename_prefix in f], reverse=True)[:max_n_imgs]

        if not img_paths:
            # Nothing found: a (nearly black) dummy image
            return (torch.from_numpy(np.random.rand(1, 256, 256, 3).astype(np.float32) / 255.0), 0.0)

        anchor_images = []
        for image_path in img_paths:
            img = ImageOps.exif_transpose(Image.open(image_path))
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            anchor_images.append(img.convert("RGB"))

        anchor_images = centre_crop_images(anchor_images, 512)
        anchor_images = np.stack([np.array(image).astype(np.float32) / 255.0 for image in anchor_images])
        return (torch.from_numpy(anchor_images), 1.0)


class SavePosEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "cache_dir": ("STRING", {"default": "eden_images/xander_big", "tooltip": "Folder to write <image_id>.pth files to."}),
                "non_embedded_image_filenames": ("LIST",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_dir",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Saves each IP-Adapter embedding of a batch as <image_id>.pth next to its image."

    def run(self, pos_embed, cache_dir, non_embedded_image_filenames):
        assert pos_embed.ndim == 3, f"Expected batch to have 3 dims but got: {pos_embed.ndim} dims"
        assert len(non_embedded_image_filenames) == pos_embed.shape[0], f"Expected the batch size of pos_embed ({pos_embed.shape[0]}) to be the same as the number of images found in non_embedded_images_folder: {len(non_embedded_image_filenames)}. non_embedded_image_filenames: {non_embedded_image_filenames}"

        for embed, filename in zip(pos_embed, non_embedded_image_filenames):
            torch.save(embed, os.path.join(cache_dir, f"{get_id_from_filename(filename)}.pth"))

        return (cache_dir,)


class FolderScanner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_dir": ("STRING", {"default": "eden_images/xander_big", "tooltip": "Folder with images and their <image_id>.pth embeddings."}),
                "seed": ("INT", {"default": 0, "tooltip": "Unused by the node; change it to force a re-run."}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("non_embedded_image_filenames",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Lists the images in a folder that have no <image_id>.pth embedding yet, and deletes embeddings whose image is gone."

    @classmethod
    def IS_CHANGED(s, cache_dir, seed=0):
        return seed, _folder_signature(cache_dir)

    def run(self, cache_dir, seed=0):
        assert os.path.exists(cache_dir), f"Invalid cache_dir: {cache_dir}"

        filenames = [os.path.join(cache_dir.rstrip("/"), f) for f in os.listdir(cache_dir)]
        image_filenames = [f for f in filenames if f.endswith(IMAGE_EXTENSIONS)]
        embedding_filenames = [f for f in filenames if f.endswith(".pth")]

        image_ids = {get_id_from_filename(f) for f in image_filenames}
        embedding_ids = {get_id_from_filename(f) for f in embedding_filenames}

        for embedding_filename in embedding_filenames:
            if get_id_from_filename(embedding_filename) not in image_ids:
                logging.info(f"[FolderScanner] Deleting: {embedding_filename}")
                os.remove(embedding_filename)

        return ([f for f in image_filenames if get_id_from_filename(f) not in embedding_ids],)


class Load_Embeddings_From_Folder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "eden_images/xander_big", "tooltip": "Folder with .pth embeddings (loaded in name order)."}),
            }
        }

    RETURN_TYPES = ("EMBEDS", "FLOAT")
    RETURN_NAMES = ("embeddings", "avg_embed_norm")
    OUTPUT_TOOLTIPS = ("All embeddings stacked into one batch.", "Mean L2 norm of the embeddings.")
    FUNCTION = "load"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Loads every .pth IP-Adapter embedding in a folder into one batch."

    @classmethod
    def IS_CHANGED(s, directory_path):
        return _folder_signature(directory_path, lambda name: name.endswith(".pth"))

    def load(self, directory_path):
        embeddings = []
        for p in sorted(f for f in os.listdir(directory_path) if f.endswith(".pth")):
            try:
                embeddings.append(torch.load(os.path.join(directory_path, p)))
            except Exception as e:
                logging.warning(f"Failed to load {os.path.join(directory_path, p)}: {e}")

        embeddings = torch.stack(embeddings)
        avg_norm = torch.linalg.vector_norm(embeddings.flatten(1).float(), dim=1).mean().item()
        embeddings = embeddings.squeeze()
        logging.info(f"Loaded image embeddings of shape {embeddings.shape} from {directory_path}")
        return (embeddings, avg_norm)


class Linear_Combine_IP_Embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a_embeds": ("EMBEDS", ),
                "b_embeds": ("EMBEDS", ),
                "a_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weight of a_embeds; b_embeds gets 1 - a_strength."}),
            },
            "optional": {
                "optional_target_norm": ("FLOAT", {"default": -1.0, "tooltip": "If > 0, rescale each embedding to this L2 norm."}),
            }
        }

    RETURN_TYPES = ("EMBEDS",)
    RETURN_NAMES = ("embeds",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Linearly interpolates between two IP-Adapter embeddings, optionally renormalizing the result."

    def run(self, a_embeds, b_embeds, a_strength, optional_target_norm=None):
        embeds = a_strength * a_embeds + (1 - a_strength) * b_embeds
        if optional_target_norm and optional_target_norm > 0:
            norm = torch.norm(embeds, dim=tuple(range(1, embeds.dim())), keepdim=True)
            embeds = embeds / norm * optional_target_norm
        return (embeds, )


class Random_Style_Mixture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_embeddings": ("EMBEDS", ),
                "avg_embed_norm": ("FLOAT", {"default": 300, "min": 0.0, "max": 500, "step": 0.5, "tooltip": "Each mixture is rescaled to this L2 norm."}),
                "num_samples": ("INT", {"default": 4, "min": 1, "tooltip": "Number of random mixtures to create."}),
                "num_style_components": ("INT", {"default": 4, "min": 1, "tooltip": "Embeddings mixed per sample (must be <= N)."}),
                "min_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lower bound of the random mixing weights."}),
            }
        }

    RETURN_TYPES = ("EMBEDS", "INT")
    RETURN_NAMES = ("pos_embed", "batch_size")
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Creates random weighted mixtures of style embeddings (new random mix every time the inputs change)."

    def run(self, style_embeddings, avg_embed_norm, num_samples, num_style_components, min_weight):
        assert num_style_components <= style_embeddings.size(0), "num_style_components is greater than the number of style images!"

        style_directions = []
        for _ in range(num_samples):
            indices = torch.randperm(style_embeddings.size(0))[:num_style_components]
            random_weights = np.random.uniform(min_weight, 1.0, num_style_components)
            random_weights = torch.tensor(random_weights / np.mean(random_weights)).to(style_embeddings.device)

            linear_combination = torch.sum(style_embeddings[indices] * random_weights[:, None, None], dim=0)
            style_directions.append(linear_combination / torch.norm(linear_combination) * avg_embed_norm)

        style_directions = torch.stack(style_directions).squeeze()
        return (style_directions, style_directions.shape[0])
