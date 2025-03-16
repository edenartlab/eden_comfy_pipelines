import sys, os, time, math, re
import hashlib
import folder_paths
from statistics import mean
import torch
from typing import Any, Mapping
import comfy.samplers
import re

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

def find_comfy_models_dir():
    return str(folder_paths.models_dir)

class Eden_Debug_Anything:
    """Node that prints the input to the console with detailed information based on datatype"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_typ, {"default": None}),
            }
        }

    FUNCTION = "debug_anything"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("input",)

    CATEGORY = "Eden 🌱/debug"

    def debug_anything(self, input):
        print("\n=== Eden Debug Anything ===")
        
        # Handle None case
        if input is None:
            print("Input is None")
            return input
            
        # Get input type
        input_type = type(input)
        print(f"Type: {input_type.__name__}")
        
        # Handle dictionaries
        if isinstance(input, dict):
            print("Dictionary content:")
            for key, value in input.items():
                value_type = type(value).__name__
                value_info = ""
                
                # Add shape for tensors
                if isinstance(value, torch.Tensor):
                    value_info = f" shape={value.shape}, dtype={value.dtype}"
                # Basic info for lists/tuples
                elif isinstance(value, (list, tuple)):
                    value_info = f" length={len(value)}"
                # Special handling for strings
                elif isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    value_info = f" value=\"{preview}\""
                
                print(f"  {key}: {value_type}{value_info}")
        
        # Handle lists and tuples
        elif isinstance(input, (list, tuple)):
            print(f"{input_type.__name__} content (length={len(input)}):")
            if len(input) > 0:
                # Print type of first element
                first_type = type(input[0]).__name__
                print(f"  First element type: {first_type}")
                
                # Check if all elements are same type
                same_type = all(isinstance(x, type(input[0])) for x in input)
                print(f"  All same type: {same_type}")
                
                # Show first few elements
                max_preview = min(5, len(input))
                for i in range(max_preview):
                    item = input[i]
                    item_repr = str(item)
                    if len(item_repr) > 50:
                        item_repr = item_repr[:50] + "..."
                    print(f"  [{i}]: {item_repr}")
                
                if len(input) > max_preview:
                    print(f"  ... and {len(input) - max_preview} more elements")
        
        # Handle tensors
        elif isinstance(input, torch.Tensor):
            print(f"Tensor information:")
            print(f"  Shape: {input.shape}")
            print(f"  Dtype: {input.dtype}")
            print(f"  Device: {input.device}")
            
            # Stats for numeric tensors
            if torch.is_floating_point(input) or torch.is_complex(input) or input.dtype in [torch.int32, torch.int64]:
                try:
                    print(f"  Min: {input.min().item()}")
                    print(f"  Max: {input.max().item()}")
                    print(f"  Mean: {input.mean().item()}")
                    print(f"  Std: {input.std().item()}")
                    # Show non-zero elements
                    non_zero = torch.count_nonzero(input).item()
                    total = input.numel()
                    print(f"  Non-zero elements: {non_zero}/{total} ({non_zero/total:.2%})")
                except Exception as e:
                    print(f"  Stats calculation error: {str(e)}")
            
            # Sample values for small tensors
            if input.numel() < 100:
                print(f"  Values: {input.tolist()}")
            else:
                flat = input.flatten()
                sample_idx = torch.linspace(0, flat.numel()-1, 5).long()
                samples = flat[sample_idx].tolist()
                print(f"  Sample values: {samples}")
        
        # Handle strings
        elif isinstance(input, str):
            print(f"String content (length={len(input)}):")
            if len(input) > 200:
                print(f"  Preview: \"{input[:200]}...\"")
            else:
                print(f"  Full string: \"{input}\"")
                
            # Check if it looks like a file path
            if os.path.sep in input:
                print(f"  Might be a file path. Exists: {os.path.exists(input)}")
                
            # Check if it looks like JSON
            if input.strip().startswith('{') and input.strip().endswith('}'):
                print("  Might be JSON")
                
            # Check if it contains newlines
            newline_count = input.count('\n')  # Use raw string to avoid the issue
            print(f"  Contains {newline_count} newlines")
        
        # Handle numbers
        elif isinstance(input, (int, float)):
            print(f"Numeric value: {input}")
            if isinstance(input, float):
                print(f"  As fraction: {input.as_integer_ratio()}")
        
        # Handle objects with special attributes
        else:
            # Try common attributes
            common_attrs = ['shape', 'size', 'dtype', 'name', 'mode', 'filename', 'metadata']
            for attr in common_attrs:
                if hasattr(input, attr):
                    try:
                        value = getattr(input, attr)
                        print(f"  {attr}: {value}")
                    except:
                        pass
            
            # Try dir() for all public attributes
            attrs = [a for a in dir(input) if not a.startswith('_') and not callable(getattr(input, a, None))]
            if attrs and len(attrs) < 15:  # Only show if not too many
                print("  Public attributes:")
                for attr in attrs[:10]:  # Limit to first 10
                    try:
                        value = getattr(input, attr)
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                        print(f"    {attr}: {value_str}")
                    except:
                        print(f"    {attr}: <error getting value>")
                        
                if len(attrs) > 10:
                    print(f"    ... and {len(attrs) - 10} more attributes")
        
        print("===========================\n")
        return (input, )

class Eden_Regex_Replace:
    """Node that performs regex-based string replacement operations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_string": ("STRING", {"multiline": True, "default": ""}),
                "regex_pattern": ("STRING", {"default": ""}),
                "replacement": ("STRING", {"default": ""}),
                "max_n_replacements": ("INT", {"default": -1, "min": -1, "max": 1000000, "step": 1}),
                "case_sensitive": ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "regex_replace"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_string",)

    CATEGORY = "Eden 🌱/text"

    def regex_replace(self, source_string, regex_pattern, replacement="", max_n_replacements=-1, case_sensitive=True):
        """
        Performs regex-based string replacement
        
        Args:
            source_string: The input text to process
            regex_pattern: The regex pattern to search for
            replacement: The replacement text (can include backreferences like \1, \2, etc.)
            max_n_replacements: Maximum number of replacements to make (-1 = all)
            case_sensitive: Whether the pattern matching should be case-sensitive
            
        Returns:
            The resulting string after replacement
        """
        # Handle empty or None inputs
        if not source_string or not regex_pattern:
            return source_string
            
        try:
            # Set regex flags
            flags = 0 if case_sensitive else re.IGNORECASE
            
            # Convert -1 to 0 for re.sub since it uses 0 to replace all instances
            count = 0 if max_n_replacements == -1 else max_n_replacements
            
            # Perform the replacement
            result = re.sub(
                pattern=regex_pattern,
                repl=replacement,
                string=source_string,
                count=count,
                flags=flags
            )
            
            return (result, )
            
        except Exception as e:
            # Return original string with error indication if the regex is invalid
            return f"REGEX ERROR: {str(e)}\nOriginal: {source_string}"

class Eden_FloatToInt:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"a": ("FLOAT", {"default": 0.0, "round": False})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/general"

    def op(self, a: float) -> tuple[int]:
        return (int(a),)

class Eden_IntToFloat:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"a": ("INT", {"default": 0})}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/general"

    def op(self, a: int) -> tuple[float]:
        return (float(a),)


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

class Eden_Image_Math:
    """Node to evaluate a simple math expression on image or mask tensors"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": False}),
                "conversion_mode": (["mean", "r", "g", "b"], {"default": "mean"}),
            },
            "optional": {
                "a": (any_typ, {"default": None}),
                "b": (any_typ, {"default": None}),
                "c": (any_typ, {"default": None}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "Eden 🌱/image"
    DESCRIPTION = (
        "Apply math expression to image/mask tensors a,b and c, supports basic math and functions. "
        "Example: 'sin(a*pi) * b + c' applies trigonometric operation to a, multiplies by b and adds c. "
        "Masks are automatically broadcasted to match image dimensions. "
        "Returns both IMAGE and MASK formats with configurable RGB->mask conversion behavior."
    )

    def convert_mask_to_image(self, mask_tensor):
        """Convert a mask tensor (B,H,W,1) to image tensor (B,H,W,3)"""
        return mask_tensor.repeat(1, 1, 1, 3)

    def convert_image_to_mask(self, image_tensor, mode="mean"):
        """Convert an image tensor (B,H,W,3) to mask tensor (B,H,W,1)"""
        if mode == "mean":
            return torch.mean(image_tensor, dim=-1, keepdim=True)
        elif mode in ["r", "g", "b"]:
            channel = {"r": 0, "g": 1, "b": 2}[mode]
            return image_tensor[:, :, :, channel:channel+1]
        else:
            raise ValueError(f"Unsupported conversion mode: {mode}")

    def eval_expression(self, expression: str, conversion_mode: str, a=None, b=None, c=None):
        # Updated pattern to properly match allowed characters without problematic escapes
        allowed_pattern = r'^[a-zA-Z0-9\s+\-*/(),.\^]+$'
        if not re.match(allowed_pattern, expression):
            raise ValueError(f"Expression contains invalid characters: {expression}")

        # Check which variables are used in the expression
        used_vars = {var: var in expression for var in ['a', 'b', 'c']}

        # Validate inputs against expression needs
        if used_vars['a'] and a is None:
            raise ValueError("Expression uses 'a' but no input was provided for a")
        if used_vars['b'] and b is None:
            raise ValueError("Expression uses 'b' but no input was provided for b")
        if used_vars['c'] and c is None:
            raise ValueError("Expression uses 'c' but no input was provided for c")

        # Create namespace for evaluation
        namespace = {}

        # Helper function to handle broadcasting of masks to match image dimensions
        def broadcast_input(tensor, target_shape):
            if tensor is None:
                return None
                
            # Check if input is a mask (B,H,W,1) vs image (B,H,W,3)
            is_mask = tensor.shape[-1] == 1
            target_channels = target_shape[-1]
            
            # Handle broadcasting between masks and images
            if is_mask and target_channels == 3:
                tensor = self.convert_mask_to_image(tensor)
            elif not is_mask and target_channels == 1:
                tensor = self.convert_image_to_mask(tensor, conversion_mode)
            
            # Handle batch dimension
            if tensor.shape[0] == 1 and target_shape[0] > 1:
                tensor = tensor.repeat(target_shape[0], 1, 1, 1)
            
            # Use torch's broadcast_to for final shape matching
            try:
                new_shape = list(target_shape)
                new_shape[-1] = tensor.shape[-1]  # Preserve channel dimension
                tensor = torch.broadcast_to(tensor, new_shape)
            except RuntimeError as e:
                raise ValueError(f"Cannot broadcast tensor of shape {tensor.shape} to shape {new_shape}: {str(e)}")
                
            return tensor

        # Determine target shape from first input
        target_shape = a.shape if a is not None else (b.shape if b is not None else c.shape)
        
        # Process inputs
        if a is not None:
            namespace['a'] = broadcast_input(a, target_shape)
        if b is not None:
            namespace['b'] = broadcast_input(b, target_shape)
        if c is not None:
            namespace['c'] = broadcast_input(c, target_shape)

        # Add mathematical constants
        namespace['pi'] = math.pi
        namespace['e'] = math.e

        # Add comprehensive set of torch operations
        torch_functions = {
            # Basic math
            'sqrt': torch.sqrt,
            'pow': torch.pow,
            '^': torch.pow,
            'abs': torch.abs,
            'round': torch.round,
            'ceil': torch.ceil,
            'floor': torch.floor,
            'trunc': torch.trunc,
            'sign': torch.sign,

            # Trigonometric functions
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'asin': torch.asin,
            'acos': torch.acos,
            'atan': torch.atan,
            'sinh': torch.sinh,
            'cosh': torch.cosh,
            'tanh': torch.tanh,
            'asinh': torch.asinh,
            'acosh': torch.acosh,
            'atanh': torch.atanh,
            
            # Exponential and logarithmic
            'exp': torch.exp,
            'log': torch.log,
            'log2': torch.log2,
            'log10': torch.log10,
            
            # Statistics
            'mean': torch.mean,
            'min': torch.min,
            'max': torch.max,
            'median': torch.median,
            'std': torch.std,
            'var': torch.var
        }
        namespace.update(torch_functions)

        # Replace caret symbol with power operator
        expression = expression.replace('^', '**')

        try:
            result = eval(expression, {"__builtins__": None}, namespace)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")

        # Always ensure result is in float format
        result = result.float()

        # Always return both IMAGE and MASK formats
        if result.shape[-1] == 3:
            return (result, self.convert_image_to_mask(result, conversion_mode))
        else:
            return (self.convert_mask_to_image(result), result)

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
    
import random
import torch

class Eden_StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
                "target_text": ("STRING", {"default": "", "multiline": False}),
                "replace_with": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_text"
    CATEGORY = "Eden 🌱/general"
    DESCRIPTION = "Replaces all occurrences of target text with replacement text in the input string"

    def replace_text(self, input_string: str, target_text: str, replace_with: str):
        result = input_string.replace(target_text, replace_with)
        return (result,)
    
class Eden_randbool:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0}),
                    "probability": ("FLOAT", {"default": 0.5}),
                }
            }
    RETURN_TYPES = ("BOOL",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "sample"
    CATEGORY = "Eden 🌱/random_c"

    def sample(self, seed, probability):
        torch.manual_seed(seed)
        return (torch.rand(1).item() < probability,)
    

class Eden_RandomPromptFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "prompts.txt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_random_prompt"
    CATEGORY = "Eden 🌱/general"
    DESCRIPTION = "Reads prompts from a text file (one per line) and returns a random prompt"

    def get_random_prompt(self, file_path: str, seed: int):
        
        try:
            # Read all lines from the file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            
            # Filter out empty lines
            lines = [line for line in lines if line]
            
            if not lines:
                raise ValueError(f"No valid prompts found in file: {file_path}")
            
            # Select a prompt
            index = seed % len(lines)
            selected_prompt = lines[index]
            
            return (selected_prompt,)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading prompt file: {str(e)}")

import os
import random
import torch
from typing import List, Tuple


class Eden_RandomFilepathSampler:
    """Node that samples a random filepath from a directory with filtering options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "./"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ""}),
                "include_subdirectories": ("BOOLEAN", {"default": False}),
                "filter_string": ("STRING", {"default": ""}),
                "filter_mode": (["contains", "starts_with", "ends_with", "regex"], {"default": "contains"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "sample_filepath"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    CATEGORY = "Eden 🌱/random"

    def sample_filepath(self, directory_path: str, seed: int, 
                        file_extension: str = "", include_subdirectories: bool = False,
                        filter_string: str = "", filter_mode: str = "contains",
                        case_sensitive: bool = False) -> Tuple[str]:
        """
        Sample a random filepath from the given directory matching the specified criteria.
        
        Args:
            directory_path: The directory path to sample files from
            seed: Random seed for reproducible sampling
            file_extension: File extension to filter by (e.g., ".jpg", ".png")
            include_subdirectories: Whether to include files from subdirectories
            filter_string: String to filter filenames by
            filter_mode: How to apply the filter_string (contains, starts_with, ends_with, regex)
            case_sensitive: Whether string filtering should be case-sensitive
            
        Returns:
            A tuple containing the randomly selected filepath as a string
        """
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Clean up input parameters
        directory_path = os.path.expanduser(directory_path)  # Expand ~ if present
        if file_extension and not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        # Validate directory exists
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Gather all files based on the include_subdirectories flag
        all_files = []
        
        if include_subdirectories:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                        if os.path.isfile(os.path.join(directory_path, f))]
        
        # Filter by extension if specified
        if file_extension:
            all_files = [f for f in all_files if f.lower().endswith(file_extension.lower())]
        
        # Apply additional filename filtering if specified
        if filter_string:
            filtered_files = []
            
            for filepath in all_files:
                filename = os.path.basename(filepath)
                
                # Apply case sensitivity
                compare_filename = filename if case_sensitive else filename.lower()
                compare_filter = filter_string if case_sensitive else filter_string.lower()
                
                # Apply filtering based on selected mode
                if filter_mode == "contains":
                    if compare_filter in compare_filename:
                        filtered_files.append(filepath)
                elif filter_mode == "starts_with":
                    if compare_filename.startswith(compare_filter):
                        filtered_files.append(filepath)
                elif filter_mode == "ends_with":
                    if compare_filename.endswith(compare_filter):
                        filtered_files.append(filepath)
                elif filter_mode == "regex":
                    import re
                    flags = 0 if case_sensitive else re.IGNORECASE
                    if re.search(filter_string, filename, flags):
                        filtered_files.append(filepath)
            
            all_files = filtered_files
        
        # Check if any files match the criteria
        if not all_files:
            raise ValueError(f"No files found matching the specified criteria in {directory_path}")
        
        selected_file = random.choice(all_files)
        
        return (selected_file,)

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
                "min_frames": ("INT", {"default": 1, "min": 0, "max": 1024, "step": 1}),
                "max_frames": ("INT", {"default": 1024, "min": 0, "max": 1024, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "determine_frame_count"
    CATEGORY = "Eden 🌱/general"

    def determine_frame_count(self, n_target_frames, n_source_frames, policy, min_frames, max_frames):
        # Validate min/max values
        if min_frames > max_frames:
            min_frames, max_frames = max_frames, min_frames

        # Early returns for special cases
        if n_target_frames == 0:
            return (max(min_frames, min(0, max_frames)),)
        
        if n_source_frames <= 1:
            return (max(min_frames, min(n_target_frames, max_frames)),)
        
        # Calculate output based on policy
        result = n_target_frames  # default fallback
        
        if policy == "closest":
            result = round(n_target_frames / n_source_frames) * n_source_frames
        elif policy == "round down":
            result = (n_target_frames // n_source_frames) * n_source_frames
            result = max(n_source_frames, result)  # Ensure we don't go below n_source_frames
        elif policy == "round up":
            result = math.ceil(n_target_frames / n_source_frames) * n_source_frames
        
        # Clamp the result between min and max frames
        result = max(min_frames, min(result, max_frames))
        
        return (int(result),)

class SDTypeConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"forceInput": True},
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"forceInput": True},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"forceInput": True}
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "MODEL_NAME_STR",
        "SAMPLER_NAME_STR",
        "SCHEDULER_STR",
    )

    FUNCTION = "convert_string"
    CATEGORY = "SD Prompt Reader"

    def convert_string(
        self, model_name: str = "", sampler_name: str = "", scheduler: str = ""
    ):
        return (
            model_name,
            sampler_name,
            scheduler,
        )


class SDAnyConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "any_type_input": (
                    any_typ,
                    {"forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("ANY_TYPE_OUTPUT",)
    FUNCTION = "convert_any"
    CATEGORY = "SD Prompt Reader"

    def convert_any(self, any_type_input: str = ""):
        return (any_type_input,)