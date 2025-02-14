"""

This code was heavily inspired by
https://github.com/theUpsider/ComfyUI-Logic/tree/fb8897351f715ea75eebf52e74515b6d07c693b8

"""

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


class Eden_DebugPrint:
    """
    This node prints the input to the console.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Takes in any input.
        """
        return {"required": {"ANY": (AlwaysEqualProxy({}),)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "Eden 🌱/Logic"

    def log_input(self, ANY):
        print(ANY)
        return {}



