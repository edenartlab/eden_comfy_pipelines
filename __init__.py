import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import *
from logic.logic_nodes import *


NODE_CLASS_MAPPINGS = {
    "CLIP_Interrogator": CLIP_Interrogator,
    "IMG_padder": IMG_padder,
    "IMG_unpadder": IMG_unpadder,
    "IMG_scaler": IMG_scaler,
    "IMG_blender": IMG_blender,
    "Filepicker": Filepicker,
    "VAEDecode_to_folder": VAEDecode_to_folder,
    "SaveImageAdvanced": SaveImageAdvanced,
    "LatentTypeConversion": LatentTypeConversion,
    "Compare": Compare,
    "Int": Int,
    "Float": Float,
    "Bool": Bool,
    "String": String,
    "If ANY execute A else B": IfExecute,
    "DebugPrint": DebugPrint,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIP_Interrogator": "CLIP_Interrogator",
    "IMG_padder": "IMG_Padder",
    "IMG_unpadder": "IMG_Unpadder",
    "IMG_scaler": "IMG_Scaler",
    "IMG_blender": "IMG_blender",
    "Filepicker": "Filepicker",
    "VAEDecode_to_folder": "VAEDecode_to_folder",
    "SaveImageAdvanced": "Save Image",
    "LatentTypeConversion": "LatentTypeConversion",
    "Compare": "Compare",
    "Int": "Int",
    "Float": "Float",
    "Bool": "Bool",
    "String": "String",
    "If ANY execute A else B": "If",
    "DebugPrint": "DebugPrint",
}





