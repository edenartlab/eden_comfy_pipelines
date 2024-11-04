import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import *
from img_utils.depth_nodes import *
from img_utils.gpt_nodes import *
from img_utils.hist_matcher import HistogramMatching
from logic.logic_nodes import *
from img_utils.animation import Animation_RGB_Mask
from ip_adapter_utils.moodmix_utils import *
from video_utils.video_interpolation import VideoFrameSelector
from general_utils import *

NODE_CLASS_MAPPINGS = {
    "CLIP_Interrogator": CLIP_Interrogator,
    "Eden_IMG_padder": IMG_padder,
    "Eden_IMG_unpadder": IMG_unpadder,
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
    "MaskFromRGB_KMeans": MaskFromRGB_KMeans,
    "GetRandomFile": GetRandomFile,
    "Animation_RGB_Mask": Animation_RGB_Mask,
    "ImageDescriptionNode": ImageDescriptionNode,
    "Eden_gpt4_node": Eden_gpt4_node,
    "FolderScanner": FolderScanner,
    "SavePosEmbeds": SavePosEmbeds,
    "VideoFrameSelector": VideoFrameSelector,
    "LoadImagesByFilename": LoadImagesByFilename,
    "Random_Style_Mixture": Random_Style_Mixture,
    "Linear_Combine_IP_Embeds": Linear_Combine_IP_Embeds,
    "SaveImageAdvanced": SaveImageAdvanced,
    "Load_Embeddings_From_Folder": Load_Embeddings_From_Folder,
    "Get_Prefixed_Imgs": Get_Prefixed_Imgs,
    "WidthHeightPicker": WidthHeightPicker,
    "DepthSlicer": DepthSlicer,
    "ParallaxZoom": ParallaxZoom,
    "AspectPadImageForOutpainting": AspectPadImageForOutpainting,
    "Eden_MaskBoundingBox": Eden_MaskBoundingBox,
    "Eden_Seed": Eden_Seed,
    "Eden_RepeatLatentBatch": Eden_RepeatLatentBatch,
    "Extend_Sequence": Extend_Sequence,
    "Eden_DetermineFrameCount": Eden_DetermineFrameCount,
    "Eden_Math": Eden_Math,
    "Eden_Image_Math": Eden_Image_Math,
    "IP_Adapter_Settings_Distribution": IP_Adapter_Settings_Distribution,
    "Eden_StringHash": Eden_StringHash,
    "ImageFolderIterator": ImageFolderIterator,
    "Eden_MaskCombiner": Eden_MaskCombiner
}

