import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  
from img_utils.img_nodes import *
from img_utils.clustering import *
from img_utils.depth_nodes import *
from img_utils.gpt_nodes import *
from img_utils.hist_matcher import HistogramMatching
from logic.logic_nodes import *
from img_utils.animation import Animation_RGB_Mask, AnimatedShapeMaskNode
from video_utils.gradient_mask_video import KeyframeBlender, MaskedRegionVideoExport
from ip_adapter_utils.moodmix_utils import *
from video_utils.video_interpolation import VideoFrameSelector
from video_utils.fill_image_mask import OrganicFillNode
from general_utils import *
from img_utils.projection_nodes import ProjectionPreview, SurfaceRadiometricCompensation

WEB_DIRECTORY = "./js"

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
    "Eden_BoolBinaryOperation": Eden_BoolBinaryOperation,
    "Eden_String": Eden_String,
    "If ANY execute A else B": Eden_IfExecute,
    "MaskFromRGB_KMeans": MaskFromRGB_KMeans,
    "GetRandomFile": GetRandomFile,
    "Animation_RGB_Mask": Animation_RGB_Mask,
    "AnimatedShapeMaskNode": AnimatedShapeMaskNode,
    "ImageDescriptionNode": ImageDescriptionNode,
    "Eden_gpt4_node": Eden_gpt4_node,
    "Eden_GPTPromptEnhancer": Eden_GPTPromptEnhancer,
    "Eden_GPTStructuredOutput": Eden_GPTStructuredOutput,
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
    "Eden_IntToFloat": Eden_IntToFloat,
    "Eden_FloatToInt": Eden_FloatToInt,
    "Eden_Image_Math": Eden_Image_Math,
    "IP_Adapter_Settings_Distribution": IP_Adapter_Settings_Distribution,
    "Eden_StringHash": Eden_StringHash,
    "ImageFolderIterator": ImageFolderIterator,
    "Eden_MaskCombiner": Eden_MaskCombiner,
    "Eden_DepthSlice_MaskVideo": Eden_DepthSlice_MaskVideo,
    "KeyframeBlender": KeyframeBlender,
    "MaskedRegionVideoExport": MaskedRegionVideoExport,
    "Eden_RandomPromptFromFile": Eden_RandomPromptFromFile,
    "Eden_StringReplace": Eden_StringReplace,
    "Eden_randbool": Eden_randbool,
    "Eden_Face_Crop": Eden_Face_Crop,
    "SDTypeConverter": SDTypeConverter,
    "SDAnyConverter": SDAnyConverter,
    "Eden_FaceToMask": Eden_FaceToMask,
    "Eden_ImageMaskComposite": Eden_ImageMaskComposite,
    "Eden_Regex_Replace": Eden_Regex_Replace,
    "Eden_Debug_Anything": Eden_Debug_Anything,
    "Eden_RandomNumberSampler": Eden_RandomNumberSampler,
    "Eden_RandomFilepathSampler": Eden_RandomFilepathSampler,
    "Eden_AllMediaLoader": Eden_AllMediaLoader,
    "Eden_Save_Param_Dict": Eden_Save_Param_Dict,
    "OrganicFillNode": OrganicFillNode,
    "Eden_RGBA_to_RGB": Eden_RGBA_to_RGB,
    "ProjectionPreview": ProjectionPreview,
    "SurfaceRadiometricCompensation": SurfaceRadiometricCompensation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eden_RandomNumberSampler": "Random Number Sampler 🎲",
    "Eden_RandomFilepathSampler": "Random Filepath Sampler 🎲",
    "Eden_AllMediaLoader": "All Media Loader 📁",
    "Eden_Save_Param_Dict": "Save Param Dict 📁",
    "OrganicFillNode": "Organic Fill Mask Animation",
    "AnimatedShapeMaskNode": "Animated Shape Mask",
    "ProjectionPreview": "Projection Preview (Additive)",
    "SurfaceRadiometricCompensation": "Surface Radiometric Compensation"
}

try:
    from random_conditioning.random_c_utils import *
    # add keys:
    NODE_CLASS_MAPPINGS_ADD = {
        "SaveConditioning": Eden_SaveConditioning,
        "LoadConditioning": Eden_LoadConditioning,
        "Inspect_Conditioning": Eden_Inspect_Conditioning,
        "Eden_RandomConditioningSamplerNode": Eden_RandomConditioningSamplerNode,
        "Eden_Load_Legacy_Conditioning": Eden_Load_Legacy_Conditioning,
        "Eden_get_random_file_from_folder": Eden_get_random_file_from_folder
    }
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ADD)
except:
    pass


def print_eden_banner():
    """
    Prints a decorative banner for the Eden ComfyUI pack on load
    """

    green = "\033[32m"
    reset = "\033[0m"
    bold = "\033[1m"

    banner = f"""
    {green}🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱{reset}
    {bold}🌱 Eden ComfyUI Pack maintained by {green}https://eden.art/  🌱{reset}
    {green}🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱{reset}
    """
    print(banner)

# Call this function when your package loads
print_eden_banner()