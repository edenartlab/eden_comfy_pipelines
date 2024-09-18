import sys, os, time
import folder_paths

def find_comfy_models_dir():
    return str(folder_paths.models_dir)

class Eden_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT','STRING')
    RETURN_NAMES = ('seed','seed_string')
    FUNCTION = 'output'
    CATEGORY = "Eden 🌱"

    def output(self, seed):
        seed_string = str(seed)
        return (seed, seed_string,)
