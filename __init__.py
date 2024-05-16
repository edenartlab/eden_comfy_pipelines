import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import *
from img_utils.hist_matcher import HistogramMatching
from logic.logic_nodes import *
from img_utils.animation import Animation_RGB_Mask


NODE_CLASS_MAPPINGS = {
    "CLIP_Interrogator": CLIP_Interrogator,
    "IMG_padder": IMG_padder,
    "IMG_unpadder": IMG_unpadder,
    "IMG_scaler": IMG_scaler,
    "IMG_blender": IMG_blender,
    "ConvertToGrayscale": ConvertToGrayscale,
    "LoadRandomImage": LoadRandomImage,
    "VAEDecode_to_folder": VAEDecode_to_folder,
    "HistogramMatching": HistogramMatching,
    "SaveImageAdvanced": SaveImageAdvanced,
    "LatentTypeConversion": LatentTypeConversion,
    "IMG_resolution_multiple_of": IMG_resolution_multiple_of,
    "Eden_Compare": Eden_Compare,
    "Eden_Int": Eden_Int,
    "Eden_Float": Eden_Float,
    "Eden_Bool": Eden_Bool,
    "Eden_String": Eden_String,
    "If ANY execute A else B": Eden_IfExecute,
    "Eden_DebugPrint": Eden_DebugPrint,
    "MaskFromRGB": MaskFromRGB,
    "MaskFromRGB_KMeans": MaskFromRGB_KMeans,
    "GetRandomFile": GetRandomFile,
    "Animation_RGB_Mask": Animation_RGB_Mask,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIP_Interrogator": "CLIP_Interrogator",
    "IMG_padder": "IMG_Padder",
    "IMG_unpadder": "IMG_Unpadder",
    "IMG_scaler": "IMG_Scaler",
    "IMG_blender": "IMG_blender",
    "HistogramMatching": "HistogramMatching",
    "ConvertToGrayscale": "ConvertToGrayscale",
    "LoadRandomImage": "LoadRandomImage",
    "VAEDecode_to_folder": "VAEDecode_to_folder",
    "SaveImageAdvanced": "Save Image",
    "LatentTypeConversion": "LatentTypeConversion",
    "IMG_resolution_multiple_of": "IMG_resolution_multiple_of",
    "Eden_Compare": "Compare (Eden)",
    "Eden_Int": "Int (Eden)",
    "Eden_Float": "Float (Eden)",
    "Eden_Bool": "Bool (Eden)",
    "Eden_String": "String (Eden)",
    "If ANY execute A else B": "If (Eden)",
    "Eden_DebugPrint": "DebugPrint (Eden)",
    "MaskFromRGB": "MaskFromRGB",
    "MaskFromRGB_KMeans": "MaskFromRGB_KMeans",
    "GetRandomFile": "GetRandomFile",
    "Animation_RGB_Mask": "Animation_RGB_Mask",
}





