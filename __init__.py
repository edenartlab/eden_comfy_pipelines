import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from clip_utils.clip_tools import CLIP_Interrogator  

NODE_CLASS_MAPPINGS = {
    "CLIP_Interrogator": CLIP_Interrogator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIP_Interrogator": "CLIP Interrogator",
}





