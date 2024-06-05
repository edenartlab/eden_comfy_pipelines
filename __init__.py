import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import *
from img_utils.gpt_nodes import *
from img_utils.hist_matcher import HistogramMatching
from logic.logic_nodes import *
from img_utils.animation import Animation_RGB_Mask
from eden_utils.lora_utils import Eden_Lora_Loader
from ip_adapter_utils.random_rotate import IPAdapterRandomRotateEmbeds, SaveExplorationState, FolderScanner, SavePosEmbeds, Mix_IP_Embeddings
from video_utils.video_interpolation import VideoFrameSelector

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
    "ImageDescriptionNode": ImageDescriptionNode,
    "IPAdapterRandomRotateEmbeds": IPAdapterRandomRotateEmbeds,
    "SaveExplorationState": SaveExplorationState,
    "FolderScanner": FolderScanner,
    "SavePosEmbeds": SavePosEmbeds,
    "Eden_Lora_Loader": Eden_Lora_Loader,
    "VideoFrameSelector": VideoFrameSelector,
    "LoadImagesByFilename": LoadImagesByFilename,
    "Mix_IP_Embeddings": Mix_IP_Embeddings
}

