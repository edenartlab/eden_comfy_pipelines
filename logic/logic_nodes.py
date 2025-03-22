"""

This code was heavily inspired by
https://github.com/theUpsider/ComfyUI-Logic/tree/fb8897351f715ea75eebf52e74515b6d07c693b8

"""

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class Eden_String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": "", "multiline": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"

    def execute(self, value):
        return (value,)


class Eden_Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"default": 0})},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"

    def execute(self, value):
        return (value,)


class Eden_Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"

    def execute(self, value):
        return (value,)


class Eden_Bool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"

    def execute(self, value):
        return (value,)


COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class Eden_Compare:
    @classmethod
    def INPUT_TYPES(s):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
                "comparison": (compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "Eden 🌱/Logic"

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)


from typing import Any, Callable, Mapping
BOOL_BINARY_OPERATIONS: Mapping[str, Callable[[bool, bool], bool]] = {
    "Nor": lambda a, b: not (a or b),
    "Xor": lambda a, b: a ^ b,
    "Nand": lambda a, b: not (a and b),
    "And": lambda a, b: a and b,
    "Xnor": lambda a, b: not (a ^ b),
    "Or": lambda a, b: a or b,
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
}

class Eden_BoolBinaryOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "op": (list(BOOL_BINARY_OPERATIONS.keys()),),
                "a": ("BOOLEAN", {"default": False}),
                "b": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/Logic"

    def op(self, op: str, a: bool, b: bool) -> tuple[bool]:
        return (BOOL_BINARY_OPERATIONS[op](a, b),)


class Eden_IfExecute:
    """
    This node executes IF_TRUE if ANY is True, otherwise it executes IF_FALSE.
    ANY can be any input, IF_TRUE and IF_FALSE can be any output.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (AlwaysEqualProxy("*"),),
                "IF_TRUE": (AlwaysEqualProxy("*"),),
                "IF_FALSE": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("?",)
    FUNCTION = "return_based_on_bool"
    CATEGORY = "Eden 🌱/Logic"
    
    def return_based_on_bool(self, ANY, IF_TRUE, IF_FALSE):
        result_str = "True" if ANY else "False"
        print(f"Evaluating {type(ANY)}, *** {ANY} *** as {result_str}")
        return (IF_TRUE if ANY else IF_FALSE,)


import torch
import random

class Eden_RandomNumberSampler:
    """Node that generates a random number from a uniform distribution with configurable min/max values"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "min_value": ("FLOAT", {"default": 0.00, "min": -1000.00, "max": 1000.00, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.00, "min": -1000.00, "max": 1000.00, "step": 0.01}),
            }
        }

    FUNCTION = "sample_random_number"
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("sampled_int", "sampled_float", "sampled_string")
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Samples a random number from a uniform distribution between min_value and max_value"
    OUTPUT_NODE = True  # Enable UI updates

    def sample_random_number(self, seed, min_value, max_value):
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Ensure min_value is not greater than max_value
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        
        # Generate random float between min and max
        sampled_float = min_value + (max_value - min_value) * torch.rand(1).item()
        
        # Round to 2 decimal places for consistency
        sampled_float = round(sampled_float, 2)
        
        # Convert to integer (rounded)
        sampled_int = int(round(sampled_float))
        
        # Create string representation with 2 decimal places
        sampled_string = f"{sampled_float:.2f}"
        
        # Return the sampled values along with special UI update
        return {
            "ui": {
                "random_number": [f"{sampled_float:.2f}"]
            }, 
            "result": (sampled_int, sampled_float, sampled_string)
        }