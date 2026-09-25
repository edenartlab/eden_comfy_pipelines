import logging
import os
import re

import numpy as np
from PIL import Image

import comfy.model_management
import folder_paths

# Interrogator shared between node instances while keep_model_alive is on.
global_interrogator_model = None


def comfy_tensor_to_pil(tensor):
    if tensor.max() > 1:
        tensor = tensor / 255
    return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))


class CLIP_Interrogator:
    def __init__(self):
        self.ci = None
        self.keep_model_alive = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["fast", "full"], ),
                "keep_model_alive": ("BOOLEAN", {"default": True, "tooltip": "Keep BLIP + CLIP in memory between runs. Off frees them after each run."}),
                "prepend_blip_caption": ("BOOLEAN", {"default": True, "tooltip": "Start the prompt with the BLIP caption."}),
                "save_prompt_to_txt_file": ("STRING", {"default": "clip_interrogator_prompt.txt", "tooltip": "Also write the prompt to this .txt file (relative to the ComfyUI folder). Empty to skip."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("full_prompt", "blip_caption")
    FUNCTION = "interrogate"
    CATEGORY = "Eden 🌱/AI"
    DESCRIPTION = "Turns the first image of a batch into a Stable Diffusion style prompt with BLIP captioning + CLIP ViT-L-14 term ranking (clip-interrogator). mode fast: caption + top-ranked terms; full: slower search for the best-matching prompt."

    def interrogate(self, image, mode="fast", keep_model_alive=True, prepend_blip_caption=True, save_prompt_to_txt_file=None):
        global global_interrogator_model
        self.keep_model_alive = keep_model_alive

        if image.shape[0] > 1:
            logging.warning("CLIP_Interrogator: got a batch of images, using only the first one.")
        pil_image = comfy_tensor_to_pil(image[0])

        clip_model_dir = os.path.join(str(folder_paths.models_dir), "clip")
        os.makedirs(clip_model_dir, exist_ok=True)
        ci = self.load_ci(clip_model_path=clip_model_dir)

        # A non-empty caption stops the interrogator from generating its own BLIP caption.
        prepend_caption = None if prepend_blip_caption else " "

        if mode == "fast":
            prompt = ci.interrogate_fast(pil_image, caption=prepend_caption)
        else:
            prompt = ci.interrogate(pil_image, caption=prepend_caption)

        blip_caption = self.clean_prompt(ci.generate_caption(pil_image))
        prompt = self.clean_prompt(prompt)
        logging.info(f"CLIP_Interrogator prompt: {prompt}")

        if save_prompt_to_txt_file:
            if not save_prompt_to_txt_file.endswith(".txt"):
                save_prompt_to_txt_file += ".txt"
            save_prompt_to_txt_file = os.path.abspath(save_prompt_to_txt_file)
            os.makedirs(os.path.dirname(save_prompt_to_txt_file), exist_ok=True)
            with open(save_prompt_to_txt_file, "w", encoding="utf-8") as f:
                f.write(prompt)

        if not keep_model_alive:
            self.ci = None
            global_interrogator_model = None
            del ci
            comfy.model_management.soft_empty_cache()

        return (prompt, blip_caption)

    def load_ci(self, clip_model_path=None):
        global global_interrogator_model

        if self.ci is None:
            if global_interrogator_model:
                self.ci = global_interrogator_model
            else:
                from .clip_interrogator import Interrogator, Config
                blip_model_dir = os.path.abspath(os.path.join(str(folder_paths.models_dir), "blip"))
                self.ci = Interrogator(Config(clip_model_path=clip_model_path, clip_model_name="ViT-L-14/openai", cache_dir=blip_model_dir))

            global_interrogator_model = self.ci if self.keep_model_alive else None

        return self.ci

    def clean_prompt(self, text):
        text = text.replace("arafed", "")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r',+', ',', text)
        text = re.sub(r'\s+,', ',', text)
        # Ensure a space after commas unless followed by punctuation
        text = re.sub(r',([^\s\.,;?!])', r', \1', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'^,', '', text)
        text = text[0].upper() + text[1:] if text else text
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        return text.strip()
