import sys, os, time, math
import hashlib
import folder_paths
from statistics import mean

def find_comfy_models_dir():
    return str(folder_paths.models_dir)

class Eden_StringHash:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
                "hash_length": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("hash_int", "hash_string")
    FUNCTION = "generate_hash"
    CATEGORY = "Eden 🌱/general"
    DESCRIPTION = "Generates a deterministic hash from an input string with configurable length"

    def generate_hash(self, input_string: str, hash_length: int = 8):
        hasher = hashlib.md5(input_string.encode('utf-8'))
        
        # Get the first N bytes of the hash and convert to integer
        hash_bytes = hasher.digest()[:min(8, hash_length)]
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        
        # Return a truncated hex string based on desired length
        hash_string = hasher.hexdigest()[:hash_length]
        
        return (hash_int, hash_string,)
    

class Eden_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT','STRING')
    RETURN_NAMES = ('seed','seed_string')
    FUNCTION = 'output'
    CATEGORY = "Eden 🌱/general"

    def output(self, seed):
        seed_string = str(seed)
        return (seed, seed_string,)
    
# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class Eden_Math:
    """Node to evaluate a simple math expression string with variables a, b, c"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "a": (any_typ, {"default": 0.0}),
                "b": (any_typ, {"default": 0.0}),
                "c": (any_typ, {"default": 0.0}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("result (float)", "result (int)", "result (float_str)")
    CATEGORY = "Eden 🌱/general"
    DESCRIPTION = (
        "evaluate a simple math expression string with variables a, b, c, supports basic math and functions"
    )

    def eval_expression(self, expression: str, a = 0.0, b = 0.0, c = 0.0):
        allowed_functions = {
            'min': min,
            'max': max,
            'mean': mean,
            'sqrt': math.sqrt,
            '^': pow,
            'pow': pow,
        }

        a = float(a)
        b = float(b)
        c = float(c)

        # Add variables a, b, c to the namespace
        local_vars = {'a': a, 'b': b, 'c': c, **allowed_functions}

        # Replace caret symbol (^) with '**' for power operation
        expression = expression.replace('^', '**')

        result = -1
        try:
            result = eval(expression, {"__builtins__": None}, local_vars)
        except SyntaxError as e:
            raise ValueError(
                f"The expression syntax is wrong '{expression}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error evaluating math expression '{expression}': {e}"
            )
        
        return (float(result), int(round(result)), str(round(result, 3)))


class IP_Adapter_Settings_Distribution:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "weight_type": (["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise'],),
            },
        }
    RETURN_TYPES = ("FLOAT", any_typ)
    RETURN_NAMES = ("weight", "weight_type")
    FUNCTION = "set"
    CATEGORY = "Eden 🌱/general"

    def set(self, weight, weight_type):
        return (weight, weight_type)
    


class Eden_RepeatLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 1024}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat"
    CATEGORY = "Eden 🌱/general"

    def repeat(self, samples, amount):
        s = samples.copy()
        s_in = samples["samples"]
        
        s["samples"] = s_in.repeat((amount, 1,1,1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount, 1,1,1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

class Eden_DetermineFrameCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "n_target_frames": ("INT", {"default": 24, "min": 0, "max": 1024}),
                "n_source_frames": ("INT", {"default": 1, "min": 0, "max": 1024}),
                "policy": (["closest", "round down", "round up"],),
            },
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "determine_frame_count"
    CATEGORY = "Eden 🌱/general"

    def determine_frame_count(self, n_target_frames, n_source_frames, policy):
        # Handle the case where n_target_frames is 0
        if n_target_frames == 0:
            return (0,)
        
        if n_source_frames <= 1:
            return (n_target_frames,)
        
        # Calculate output based on policy
        if policy == "closest":
            # Closest multiple of n_source_frames to n_target_frames
            closest_multiple = round(n_target_frames / n_source_frames) * n_source_frames
            return (closest_multiple,)
        elif policy == "round down":
            # Largest multiple of n_source_frames smaller than or equal to n_target_frames
            round_down_multiple = (n_target_frames // n_source_frames) * n_source_frames
            
            if round_down_multiple <= n_source_frames:
                round_down_multiple = n_source_frames

            return (round_down_multiple,)
        elif policy == "round up":
            # Smallest multiple of n_source_frames greater than or equal to n_target_frames
            round_up_multiple = math.ceil(n_target_frames / n_source_frames) * n_source_frames
            return (round_up_multiple,)

        return (n_target_frames,)
