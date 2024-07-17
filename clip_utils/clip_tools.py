import time
import os
import torch
import PIL.Image
from PIL import Image
import sys, os, time, re

# custom version of clip_interrogator which downloads to the ComfyUI models dir:
from .clip_interrogator import Interrogator, Config

import torch
import numpy as np
from PIL import Image

sys.path.append('..')
import folder_paths

def comfy_tensor_to_pil(tensor):
    # Clone the tensor and detach it from the computation graph
    tensor = tensor.clone().detach()
    
    # Normalize the tensor if it's not already in [0, 1]
    if torch.max(tensor) > 1:
        print("Normalizing tensor to [0, 1]")
        tensor = torch.div(tensor, 255)
    
    # Convert to PIL Image and return
    return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

# Global variable to hold the model
global_interrogator_model = None

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
                "keep_model_alive": ("BOOLEAN", {"default": True}),
                "prepend_blip_caption": ("BOOLEAN", {"default": True}),
                "save_prompt_to_txt_file": ("STRING", {"default": "clip_interrogator_prompt.txt"}),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("full_prompt", "blip_caption")
    FUNCTION = "interrogate"
    CATEGORY = "Eden 🌱"

    def interrogate(self, image, mode="fast", keep_model_alive=True, prepend_blip_caption = True, save_prompt_to_txt_file=None):

        self.keep_model_alive = keep_model_alive

        print(f"Interrogating image with mode {mode}, keep_model_alive={keep_model_alive}")

        # ci expects a PIL image, but we get a torch tensor:
        if image.shape[0] > 1:
            print("Warning: CLIP_Interrogator expects a single image, but got a batch. Using first image in batch.")
            
        pil_image = comfy_tensor_to_pil(image[0])

        clip_model_dir = os.path.join(str(folder_paths.models_dir), "clip")
        os.makedirs(clip_model_dir, exist_ok=True)

        ci = self.load_ci(clip_model_path=clip_model_dir)

        if prepend_blip_caption:
            prepend_caption = None
        else:
            prepend_caption = " " # make sure there is a space so that the prompt is not joined with the caption

        if mode == "fast":
            prompt = ci.interrogate_fast(pil_image, caption = prepend_caption)
        else:
            prompt = ci.interrogate(pil_image, caption = prepend_caption)

        blip_caption = ci.generate_caption(pil_image)
            
        blip_caption = self.clean_prompt(blip_caption)
        prompt = self.clean_prompt(prompt)

        print(f"Interogated prompt: {prompt}")

        if save_prompt_to_txt_file:
            if not save_prompt_to_txt_file.endswith(".txt"):
                save_prompt_to_txt_file += ".txt"

            # Make sure the path is absolute:
            save_prompt_to_txt_file = os.path.abspath(save_prompt_to_txt_file)

            # Make sure the directory exists:
            os.makedirs(os.path.dirname(save_prompt_to_txt_file), exist_ok=True)
            
            with open(save_prompt_to_txt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved interrogated prompt to {save_prompt_to_txt_file}")

        return (prompt, blip_caption)
    
    def load_ci(self, clip_model_path=None):
        global global_interrogator_model

        if self.ci is None:
            if global_interrogator_model:
                self.ci = global_interrogator_model
            else:
                BLIP_MODEL_DIR = os.path.abspath(os.path.join(str(folder_paths.models_dir), "blip"))
                self.ci = Interrogator(Config(clip_model_path=clip_model_path, clip_model_name="ViT-L-14/openai", cache_dir=BLIP_MODEL_DIR))
                
            if self.keep_model_alive:
                global_interrogator_model = self.ci
            else:
                global_interrogator_model = None

        return self.ci
    
    def clean_prompt(self, text):
        text = text.replace("arafed", "")

        # Replace double spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Replace double commas with single comma
        text = re.sub(r',+', ',', text)

        # Remove spaces before commas
        text = re.sub(r'\s+,', ',', text)

        # Ensure space after commas (if not followed by another punctuation or end of string)
        text = re.sub(r',([^\s\.,;?!])', r', \1', text)

        # Trim spaces around periods and ensure one space after
        text = re.sub(r'\s*\.\s*', '. ', text)

        # Remove leading commas
        text = re.sub(r'^,', '', text)

        # Capitalize the first letter of the sentence
        text = text[0].upper() + text[1:] if text else text

        # convert to utf-8:
        text = text.encode('utf-8', 'ignore').decode('utf-8')

        return text.strip()