"""
Logic nodes, heavily inspired by
https://github.com/theUpsider/ComfyUI-Logic/tree/fb8897351f715ea75eebf52e74515b6d07c693b8
"""

import sys

import torch


class AlwaysEqualProxy(str):
    """Wildcard type: compares equal to every other type so any link can connect."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class Eden_String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to output."})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Text"
    DESCRIPTION = "Outputs a (multiline) string constant."

    def execute(self, value):
        return (value,)


class Eden_Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"min": -sys.maxsize, "max": sys.maxsize, "default": 0, "tooltip": "Integer to output."})},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Outputs an integer constant."

    def execute(self, value):
        return (value,)


class Eden_Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01, "tooltip": "Float to output."})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Outputs a float constant."

    def execute(self, value):
        return (value,)


class Eden_Bool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("BOOLEAN", {"default": False, "tooltip": "Boolean to output."})},
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Outputs a boolean constant."

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


class Eden_Compare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0, "tooltip": "Left operand (any type)."}),
                "b": (AlwaysEqualProxy("*"), {"default": 0, "tooltip": "Right operand (any type)."}),
                "comparison": (list(COMPARE_FUNCTIONS), {"default": "a == b", "tooltip": "Comparison to apply."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Compares two values of any type and outputs the boolean result."

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)


BOOL_BINARY_OPERATIONS = {
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
                "op": (list(BOOL_BINARY_OPERATIONS),),
                "a": ("BOOLEAN", {"default": False}),
                "b": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Applies a binary boolean operation (And, Or, Xor, Nand, ...) to two booleans."

    def op(self, op, a, b):
        return (BOOL_BINARY_OPERATIONS[op](a, b),)


class Eden_IfExecute:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (AlwaysEqualProxy("*"), {"tooltip": "Condition: truthy (non-zero, non-empty, True) picks IF_TRUE."}),
                "IF_TRUE": (AlwaysEqualProxy("*"), {"lazy": True, "tooltip": "Returned (and computed) only when ANY is truthy."}),
                "IF_FALSE": (AlwaysEqualProxy("*"), {"lazy": True, "tooltip": "Returned (and computed) only when ANY is falsy."}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("?",)
    FUNCTION = "return_based_on_bool"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Outputs IF_TRUE when ANY is truthy (non-zero, non-empty, True), otherwise IF_FALSE. Only the chosen branch is computed."

    def check_lazy_status(self, ANY, IF_TRUE=None, IF_FALSE=None):
        return ["IF_TRUE"] if ANY else ["IF_FALSE"]

    def return_based_on_bool(self, ANY, IF_TRUE, IF_FALSE):
        return (IF_TRUE if ANY else IF_FALSE,)


class Eden_RandomNumberSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Same seed gives the same number."}),
                "min_value": ("FLOAT", {"default": 0.00, "min": -1000.00, "max": 1000.00, "step": 0.01, "tooltip": "Lower bound (swapped with max_value if larger)."}),
                "max_value": ("FLOAT", {"default": 1.00, "min": -1000.00, "max": 1000.00, "step": 0.01, "tooltip": "Upper bound."}),
            }
        }

    FUNCTION = "sample_random_number"
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("sampled_int", "sampled_float", "sampled_string")
    OUTPUT_TOOLTIPS = ("The sampled value rounded to an integer.", "The sampled value, rounded to 2 decimals.", "The sampled value formatted with 2 decimals.")
    CATEGORY = "Eden 🌱/Random"
    DESCRIPTION = "Samples a number uniformly between min_value and max_value (2 decimals) and shows it on the node."
    OUTPUT_NODE = True

    def sample_random_number(self, seed, min_value, max_value):
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        generator = torch.Generator().manual_seed(seed)
        sampled_float = round(min_value + (max_value - min_value) * torch.rand(1, generator=generator).item(), 2)
        sampled_string = f"{sampled_float:.2f}"

        return {
            "ui": {"random_number": [sampled_string]},
            "result": (int(round(sampled_float)), sampled_float, sampled_string),
        }
