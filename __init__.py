import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import IMG_padder, IMG_unpadder, IMG_scaler, Filepicker, VAEDecode_to_folder, SaveImageAdvanced, LatentTypeConversion

NODE_CLASS_MAPPINGS = {
    "CLIP_Interrogator": CLIP_Interrogator,
    "IMG_padder": IMG_padder,
    "IMG_unpadder": IMG_unpadder,
    "IMG_scaler": IMG_scaler,
    "Filepicker": Filepicker,
    "VAEDecode_to_folder": VAEDecode_to_folder,
    "SaveImageAdvanced": SaveImageAdvanced,
    "LatentTypeConversion": LatentTypeConversion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIP_Interrogator": "CLIP_Interrogator",
    "IMG_padder": "IMG_Padder",
    "IMG_unpadder": "IMG_Unpadder",
    "IMG_scaler": "IMG_Scaler",
    "Filepicker": "Filepicker",
    "VAEDecode_to_folder": "VAEDecode_to_folder",
    "SaveImageAdvanced": "Save Image",
    "LatentTypeConversion": "LatentTypeConversion",
}





