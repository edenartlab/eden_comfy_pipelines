import logging
import os
import sys

from server import PromptServer

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator
from general_utils import (
    Eden_AllMediaLoader, Eden_Debug_Anything, Eden_DetermineFrameCount, Eden_FloatToInt, Eden_Image_Math,
    Eden_IntToFloat, Eden_Math, Eden_RandomFilepathSampler, Eden_RandomPromptFromFile, Eden_randbool,
    Eden_Regex_Replace, Eden_RepeatLatentBatch, Eden_Save_Param_Dict, Eden_Seed, Eden_StringHash,
    Eden_StringReplace, IP_Adapter_Settings_Distribution, SDAnyConverter, SDTypeConverter,
)
from img_utils.animation import Animation_RGB_Mask, AnimatedShapeMaskNode
from img_utils.clustering import MaskFromRGB_KMeans
from img_utils.depth_nodes import DepthSlicer, Eden_DepthSlice_MaskVideo, ParallaxZoom
from img_utils.gpt_nodes import Eden_GPTPromptEnhancer, Eden_GPTStructuredOutput, Eden_gpt4_node, ImageDescriptionNode
from img_utils.hist_matcher import HistogramMatching
from img_utils.img_nodes import (
    AspectPadImageForOutpainting, ConvertToGrayscale, Eden_Face_Crop, Eden_FaceToMask, Eden_ImageMaskComposite,
    Eden_MaskBoundingBox, Eden_MaskCombiner, Eden_RGBA_to_RGB, Extend_Sequence, GetRandomFile, ImageFolderIterator,
    IMG_blender, IMG_padder, IMG_resolution_multiple_of, IMG_scaler, IMG_unpadder, LatentTypeConversion,
    LoadImagesByFilename, LoadRandomImage, SaveImageAdvanced, VAEDecode_to_folder, WidthHeightPicker,
)
from img_utils.organic_fill_nodes import Eden_GradientBorderMask, Eden_OrganicFillAnimation, Eden_OrganicFillRandom
from img_utils.projection_nodes import ProjectionPreview, SurfaceRadiometricCompensation
from ip_adapter_utils.moodmix_utils import (
    FolderScanner, Get_Prefixed_Imgs, Linear_Combine_IP_Embeds, Load_Embeddings_From_Folder, Random_Style_Mixture,
    SavePosEmbeds,
)
from logic.logic_nodes import (
    Eden_Bool, Eden_BoolBinaryOperation, Eden_Compare, Eden_Float, Eden_IfExecute, Eden_Int, Eden_RandomNumberSampler,
    Eden_String,
)
from prompt_utils.nodes import Eden_PromptFromImageFolder
from video_utils.fill_image_mask import OrganicFillNode as OrganicFillNode_Deprecated
from video_utils.gradient_mask_video import KeyframeBlender, MaskedRegionVideoExport
from video_utils.video_interpolation import VideoFrameSelector

WEB_DIRECTORY = "./js"

# Keys are the node type ids stored in saved workflows: never rename them.
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
    "SaveImageAdvanced": SaveImageAdvanced,  # shadowed by the core node of the same id since May 2026
    "Eden_SaveImageAdvanced": SaveImageAdvanced,
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
    "OrganicFillNode": OrganicFillNode_Deprecated,
    "Eden_RGBA_to_RGB": Eden_RGBA_to_RGB,
    "ProjectionPreview": ProjectionPreview,
    "SurfaceRadiometricCompensation": SurfaceRadiometricCompensation,
    "Eden_OrganicFillAnimation": Eden_OrganicFillAnimation,
    "Eden_GradientBorderMask": Eden_GradientBorderMask,
    "Eden_OrganicFillRandom": Eden_OrganicFillRandom,
    "Eden_PromptFromImageFolder": Eden_PromptFromImageFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # AI
    "CLIP_Interrogator": "CLIP Interrogator 🔍",
    "Eden_gpt4_node": "GPT-4 Completion 🤖",
    "Eden_GPTPromptEnhancer": "GPT Prompt Enhancer 🤖",
    "ImageDescriptionNode": "GPT Image Description 🤖",
    "Eden_GPTStructuredOutput": "GPT Structured Output (JSON) 🤖",
    # Image
    "Eden_IMG_padder": "Image Padder",
    "Eden_IMG_unpadder": "Image Unpadder",
    "IMG_scaler": "Image Math (Pixel Expression)",
    "IMG_blender": "Image Blender",
    "ConvertToGrayscale": "Convert to Grayscale",
    "IMG_resolution_multiple_of": "Crop to Resolution Multiple",
    "AspectPadImageForOutpainting": "Aspect Pad Image for Outpainting",
    "Eden_RGBA_to_RGB": "RGBA to RGB",
    "Eden_Image_Math": "Image / Mask Math",
    "HistogramMatching": "Histogram Matching",
    "WidthHeightPicker": "Width/Height Picker",
    "Eden_Face_Crop": "Face Crop",
    "Eden_ImageMaskComposite": "Image Mask Composite",
    "ProjectionPreview": "Projection Preview (Additive)",
    "SurfaceRadiometricCompensation": "Surface Radiometric Compensation",
    # Mask
    "MaskFromRGB_KMeans": "Mask From RGB (KMeans) 🎨",
    "Eden_MaskBoundingBox": "Mask Bounding Box Crop",
    "Eden_MaskCombiner": "Mask Combiner",
    "Eden_FaceToMask": "Face to Mask (MediaPipe)",
    "Animation_RGB_Mask": "Animation RGB Mask",
    "AnimatedShapeMaskNode": "Animated Shape Mask",
    "Eden_GradientBorderMask": "Gradient Border Mask",
    "Eden_OrganicFillAnimation": "Organic Fill Animation 🌿",
    "Eden_OrganicFillRandom": "Organic Fill Random 🎲",
    # Depth
    "DepthSlicer": "Depth Slicer",
    "ParallaxZoom": "Parallax Zoom",
    "Eden_DepthSlice_MaskVideo": "Depth Slice Mask Video",
    # Video
    "VideoFrameSelector": "Video Frame Selector",
    "KeyframeBlender": "Keyframe Blender 🎞️",
    "MaskedRegionVideoExport": "Masked Region Video Export (Alpha)",
    "Extend_Sequence": "Extend Sequence (Loop / Ping-Pong)",
    "Eden_DetermineFrameCount": "Determine Frame Count",
    # Loaders
    "LoadRandomImage": "Load Random Image(s) 🎲",
    "ImageFolderIterator": "Image Folder Iterator",
    "LoadImagesByFilename": "Load Images by Filename",
    "GetRandomFile": "Get Random File 🎲",
    "VAEDecode_to_folder": "VAE Decode to Folder",
    "SaveImageAdvanced": "Save Image Advanced 💾",
    "Eden_SaveImageAdvanced": "Save Image Advanced (Eden) 💾",
    "Eden_AllMediaLoader": "All Media Loader 📁",
    "Eden_Save_Param_Dict": "Save Param Dict 📁",
    # Latent
    "LatentTypeConversion": "Latent Type Conversion (fp16/fp32)",
    "Eden_RepeatLatentBatch": "Repeat Latent Batch",
    # Logic
    "Eden_Compare": "Compare (a ? b)",
    "Eden_Int": "Int",
    "Eden_Float": "Float",
    "Eden_Bool": "Bool",
    "Eden_BoolBinaryOperation": "Bool Binary Operation",
    "If ANY execute A else B": "If ANY Execute A Else B 🔀",
    "Eden_Math": "Math Expression",
    "Eden_IntToFloat": "Int to Float",
    "Eden_FloatToInt": "Float to Int",
    "SDTypeConverter": "SD Type to String Converter",
    "SDAnyConverter": "SD Any-Type Converter",
    # Text
    "Eden_String": "String",
    "Eden_StringHash": "String Hash",
    "Eden_StringReplace": "String Replace",
    "Eden_Regex_Replace": "Regex Replace",
    "Eden_RandomPromptFromFile": "Prompt From File (by Seed) 🎲",
    "Eden_PromptFromImageFolder": "Prompt from Image Folder 🎲",
    # Random
    "Eden_Seed": "Seed 🎲",
    "Eden_randbool": "Random Bool 🎲",
    "Eden_RandomNumberSampler": "Random Number Sampler 🎲",
    "Eden_RandomFilepathSampler": "Random Filepath Sampler 🎲",
    # IP-Adapter
    "IP_Adapter_Settings_Distribution": "IP-Adapter Settings",
    "Get_Prefixed_Imgs": "Load Prefixed Images",
    "SavePosEmbeds": "Save IP-Adapter Embeds",
    "FolderScanner": "IP-Adapter Embed Folder Scanner",
    "Load_Embeddings_From_Folder": "Load IP-Adapter Embeds From Folder",
    "Linear_Combine_IP_Embeds": "Linear Combine IP Embeds",
    "Random_Style_Mixture": "Random Style Mixture 🎲",
    # Utils / deprecated
    "Eden_Debug_Anything": "Debug Anything 🐞",
    "OrganicFillNode": "Organic Fill Mask Animation (Deprecated)",
}

# Local-only experimental nodes, not shipped with the repo.
try:
    from random_conditioning.random_c_utils import (
        Eden_get_random_file_from_folder, Eden_Inspect_Conditioning, Eden_Load_Legacy_Conditioning,
        Eden_LoadConditioning, Eden_RandomConditioningSamplerNode, Eden_SaveConditioning,
    )
    NODE_CLASS_MAPPINGS.update({
        "SaveConditioning": Eden_SaveConditioning,
        "LoadConditioning": Eden_LoadConditioning,
        "Inspect_Conditioning": Eden_Inspect_Conditioning,
        "Eden_RandomConditioningSamplerNode": Eden_RandomConditioningSamplerNode,
        "Eden_Load_Legacy_Conditioning": Eden_Load_Legacy_Conditioning,
        "Eden_get_random_file_from_folder": Eden_get_random_file_from_folder,
    })
except ImportError:
    pass



def _route_legacy_save_image_advanced(json_data):
    # Core ComfyUI now owns the "SaveImageAdvanced" id; send old Eden API prompts to the renamed node.
    for node in json_data.get("prompt", {}).values():
        if node.get("class_type") == "SaveImageAdvanced" and "add_timestamp" in node.get("inputs", {}):
            node["class_type"] = "Eden_SaveImageAdvanced"
    return json_data


if getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.add_on_prompt_handler(_route_legacy_save_image_advanced)

logging.info(f"🌱 Eden ComfyUI Pack: {len(NODE_CLASS_MAPPINGS)} nodes loaded (https://eden.art)")
