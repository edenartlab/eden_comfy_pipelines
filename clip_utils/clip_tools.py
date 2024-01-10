import time
import os
import torch
import PIL.Image
from PIL import Image
import sys, os, time
from clip_interrogator import Interrogator, Config

import torch
import numpy as np
from PIL import Image

def find_comfy_models_dir():
    current_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    while True:
        # Check if "ComfyUI/models" exists in the current path
        target_path = os.path.join(current_path, "ComfyUI", "models")
        if os.path.isdir(target_path):
            return os.path.abspath(target_path)

        # Move up one directory level
        new_path = os.path.dirname(current_path)
        if new_path == current_path:
            # If the new path is the same, we've reached the root and didn't find the directory
            break
        current_path = new_path

    return None

# Make sure we're always downloading BLIP captioning models to the same dir:
transformers_model_dir = os.path.join(find_comfy_models_dir(), "transformers_cache")
os.makedirs(transformers_model_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = transformers_model_dir

def comfy_tensor_to_pil(tensor):
    # Clone the tensor and detach it from the computation graph
    tensor = tensor.clone().detach()
    
    # Normalize the tensor if it's not already in [0, 1]
    if torch.max(tensor) > 1:
        print("Normalizing tensor to [0, 1]")
        tensor = torch.div(tensor, 255)
    
    # Convert to PIL Image and return
    return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

class CLIP_Interrogator:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["fast", "full"], ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "interrogate"
    CATEGORY = "Eden 🌱"

    def interrogate(self, image, mode="fast"):
        print(f"Interrogating image with mode {mode}")

        # ci expects a PIL image, but we get a torch tensor:
        if image.shape[0] > 1:
            print("Warning: CLIP_Interrogator expects a single image, but got a batch. Using first image in batch.")
            
        pil_image = comfy_tensor_to_pil(image[0])

        clip_model_dir = os.path.join(find_comfy_models_dir(), "clip")
        os.makedirs(clip_model_dir, exist_ok=True)

        ci = self.load_ci(clip_model_path=clip_model_dir)
        if mode == "fast":
            prompt = ci.interrogate_fast(pil_image)
        else:
            prompt = ci.interrogate(pil_image)
            
        prompt = prompt.replace("arafed", "")

        print(f"Interogated prompt: {prompt}")

        return (prompt,)
    
    def load_ci(self, force_reload=False, clip_model_path=None):
        if self.ci is None or force_reload:
            ci = Interrogator(Config(clip_model_path=clip_model_path, clip_model_name="ViT-L-14/openai"))
        return ci