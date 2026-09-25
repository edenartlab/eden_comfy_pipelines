import os
import sys
import math
import re
import json
import glob
import hashlib
import logging
import random
import tarfile
import tempfile
import zipfile
from datetime import datetime
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy.samplers
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


def _preview(value, n=50):
    s = str(value)
    return s[:n] + "..." if len(s) > n else s


class Eden_Debug_Anything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_typ, {"default": None, "tooltip": "Any value to inspect; it is passed through unchanged."}),
            }
        }

    FUNCTION = "debug_anything"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("input",)
    OUTPUT_TOOLTIPS = ("The input, unchanged.",)
    CATEGORY = "Eden 🌱/Utils"
    DESCRIPTION = "Logs type, shape, stats and a preview of any input to the console, then passes it through."

    def debug_anything(self, input):
        lines = ["=== Eden Debug Anything ==="]
        if input is None:
            lines.append("Input is None")
        else:
            lines.append(f"Type: {type(input).__name__}")

        if isinstance(input, dict):
            lines.append("Dictionary content:")
            for key, value in input.items():
                info = ""
                if isinstance(value, torch.Tensor):
                    info = f" shape={value.shape}, dtype={value.dtype}"
                elif isinstance(value, (list, tuple)):
                    info = f" length={len(value)}"
                elif isinstance(value, str):
                    info = f' value="{_preview(value)}"'
                lines.append(f"  {key}: {type(value).__name__}{info}")

        elif isinstance(input, (list, tuple)):
            lines.append(f"{type(input).__name__} content (length={len(input)}):")
            if input:
                lines.append(f"  First element type: {type(input[0]).__name__}")
                lines.append(f"  All same type: {all(isinstance(x, type(input[0])) for x in input)}")
                for i, item in enumerate(input[:5]):
                    lines.append(f"  [{i}]: {_preview(item)}")
                if len(input) > 5:
                    lines.append(f"  ... and {len(input) - 5} more elements")

        elif isinstance(input, torch.Tensor):
            lines += ["Tensor information:", f"  Shape: {input.shape}", f"  Dtype: {input.dtype}", f"  Device: {input.device}"]
            if input.numel() > 0 and (torch.is_floating_point(input) or input.dtype in (torch.int32, torch.int64)):
                t = input.float()
                non_zero = torch.count_nonzero(input).item()
                lines += [
                    f"  Min: {t.min().item()}",
                    f"  Max: {t.max().item()}",
                    f"  Mean: {t.mean().item()}",
                    f"  Std: {t.std().item()}",
                    f"  Non-zero elements: {non_zero}/{input.numel()} ({non_zero / input.numel():.2%})",
                ]
            if input.numel() < 20:
                lines.append(f"  Values: {input.tolist()}")
            else:
                flat = input.flatten()
                idx = torch.linspace(0, flat.numel() - 1, 5, device=flat.device).long()
                lines.append(f"  Sample values: {flat[idx].tolist()}")

        elif isinstance(input, str):
            lines.append(f"String content (length={len(input)}):")
            lines.append(f'  Preview: "{input[:200]}..."' if len(input) > 200 else f'  Full string: "{input}"')
            if os.path.sep in input:
                lines.append(f"  Might be a file path. Exists: {os.path.exists(input)}")
            if input.strip().startswith('{') and input.strip().endswith('}'):
                lines.append("  Might be JSON")
            lines.append(f"  Contains {input.count(chr(10))} newlines")

        elif isinstance(input, (int, float)):
            lines.append(f"Numeric value: {input}")
            if isinstance(input, float) and math.isfinite(input):
                lines.append(f"  As fraction: {input.as_integer_ratio()}")

        elif input is not None:
            for attr in ('shape', 'size', 'dtype', 'name', 'mode', 'filename', 'metadata'):
                if hasattr(input, attr):
                    lines.append(f"  {attr}: {getattr(input, attr)}")
            attrs = [a for a in dir(input) if not a.startswith('_') and not callable(getattr(input, a, None))]
            if 0 < len(attrs) < 15:
                lines.append("  Public attributes:")
                for attr in attrs[:10]:
                    lines.append(f"    {attr}: {_preview(getattr(input, attr, '<error getting value>'))}")
                if len(attrs) > 10:
                    lines.append(f"    ... and {len(attrs) - 10} more attributes")

        lines.append("===========================")
        logger.info("\n" + "\n".join(lines))
        return (input,)


class Eden_Regex_Replace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_string": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to search in."}),
                "regex_pattern": ("STRING", {"default": "", "tooltip": "Python regular expression to find. Empty = return the source unchanged."}),
                "replacement": ("STRING", {"default": "", "tooltip": "Replacement text; may use backreferences like \\1."}),
                "max_n_replacements": ("INT", {"default": -1, "min": -1, "max": 1000000, "step": 1, "tooltip": "Maximum number of replacements. -1 (or 0) replaces all matches."}),
                "case_sensitive": ("BOOLEAN", {"default": True, "tooltip": "Match case exactly."}),
            }
        }

    FUNCTION = "regex_replace"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_string",)
    CATEGORY = "Eden 🌱/Text"
    DESCRIPTION = "Replaces regex matches in a string (Python re.sub syntax)."

    def regex_replace(self, source_string, regex_pattern, replacement="", max_n_replacements=-1, case_sensitive=True):
        if not source_string or not regex_pattern:
            return (source_string,)
        flags = 0 if case_sensitive else re.IGNORECASE
        count = 0 if max_n_replacements == -1 else max_n_replacements
        try:
            return (re.sub(regex_pattern, replacement, source_string, count=count, flags=flags),)
        except re.error as e:
            raise ValueError(f"Invalid regex '{regex_pattern}' or replacement '{replacement}': {e}") from e


class Eden_FloatToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("FLOAT", {"default": 0.0, "round": False, "tooltip": "Float to convert (truncated toward zero)."})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Converts a float to an int by truncating toward zero."

    def op(self, a: float) -> tuple[int]:
        return (int(a),)


class Eden_IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 0, "max": sys.maxsize, "tooltip": "Int to convert."})}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "op"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Converts an int to a float."

    def op(self, a: int) -> tuple[float]:
        return (float(a),)


class Eden_StringHash:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to hash."}),
                "hash_length": ("INT", {"default": 8, "min": 4, "max": 64, "step": 1, "tooltip": "Length of the hex hash string (the int uses at most the first 8 bytes)."}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("hash_int", "hash_string")
    OUTPUT_TOOLTIPS = ("Integer from the first min(8, hash_length) bytes of the MD5 digest.", "Hex MD5 digest truncated to hash_length.")
    FUNCTION = "generate_hash"
    CATEGORY = "Eden 🌱/Text"
    DESCRIPTION = "Generates a deterministic MD5-based hash from an input string with configurable length."

    def generate_hash(self, input_string: str, hash_length: int = 8):
        hasher = hashlib.md5(input_string.encode('utf-8'))
        hash_int = int.from_bytes(hasher.digest()[:min(8, hash_length)], byteorder='big')
        return (hash_int, hasher.hexdigest()[:hash_length])


class Eden_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, "tooltip": "Seed value."})}}

    RETURN_TYPES = ('INT', 'STRING')
    RETURN_NAMES = ('seed', 'seed_string')
    FUNCTION = 'output'
    CATEGORY = "Eden 🌱/Random"
    DESCRIPTION = "Outputs a seed as both INT and STRING."

    def output(self, seed):
        return (seed, str(seed))


class Eden_Math:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": False, "tooltip": "Math expression using a, b, c, e.g. 'max(a, 16) * b ^ 2'. Functions: min, max, mean, sqrt, pow."}),
            },
            "optional": {
                "a": (any_typ, {"default": 0.0, "tooltip": "Variable a (any number-like value)."}),
                "b": (any_typ, {"default": 0.0, "tooltip": "Variable b (any number-like value)."}),
                "c": (any_typ, {"default": 0.0, "tooltip": "Variable c (any number-like value)."}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("result (float)", "result (int)", "result (float_str)")
    OUTPUT_TOOLTIPS = ("Result as float.", "Result rounded to the nearest int.", "Result rounded to 3 decimals, as text.")
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Evaluates a simple math expression with variables a, b, c; '^' means power."

    def eval_expression(self, expression: str, a=0.0, b=0.0, c=0.0):
        local_vars = {
            'a': float(a), 'b': float(b), 'c': float(c),
            'min': min, 'max': max, 'mean': mean, 'sqrt': math.sqrt, 'pow': pow,
        }
        expression = expression.replace('^', '**')
        try:
            result = eval(expression, {"__builtins__": None}, local_vars)  # noqa: S307 - restricted namespace
        except SyntaxError as e:
            raise ValueError(f"The expression syntax is wrong '{expression}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error evaluating math expression '{expression}': {e}") from e
        return (float(result), int(round(result)), str(round(result, 3)))


class Eden_Image_Math:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": False, "tooltip": "Expression on a, b, c, e.g. 'sin(a*pi) * b + c'. Supports torch math (sqrt, exp, log, sin, abs, clamp-free min/max...), pi and e."}),
                "conversion_mode": (["mean", "r", "g", "b"], {"default": "mean", "tooltip": "How an RGB image is turned into a mask: channel mean or a single channel."}),
            },
            "optional": {
                "a": (any_typ, {"default": None, "tooltip": "IMAGE or MASK. The first connected input sets the output shape."}),
                "b": (any_typ, {"default": None, "tooltip": "IMAGE or MASK."}),
                "c": (any_typ, {"default": None, "tooltip": "IMAGE or MASK."}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_TOOLTIPS = ("Result as an RGB image.", "Result as a mask.")
    CATEGORY = "Eden 🌱/Image"
    DESCRIPTION = (
        "Applies a math expression to image/mask tensors a, b, c, e.g. 'sin(a*pi) * b + c'. "
        "Masks and images are broadcast to the shape of the first connected input. "
        "Returns the result as both IMAGE and MASK."
    )

    FUNCTIONS = {
        'sqrt': torch.sqrt, 'pow': torch.pow, 'abs': torch.abs, 'round': torch.round, 'ceil': torch.ceil,
        'floor': torch.floor, 'trunc': torch.trunc, 'sign': torch.sign,
        'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan, 'asin': torch.asin, 'acos': torch.acos,
        'atan': torch.atan, 'sinh': torch.sinh, 'cosh': torch.cosh, 'tanh': torch.tanh,
        'asinh': torch.asinh, 'acosh': torch.acosh, 'atanh': torch.atanh,
        'exp': torch.exp, 'log': torch.log, 'log2': torch.log2, 'log10': torch.log10,
        'mean': torch.mean, 'min': torch.min, 'max': torch.max, 'median': torch.median,
        'std': torch.std, 'var': torch.var,
        'pi': math.pi, 'e': math.e,
    }

    @staticmethod
    def to_mask(image, mode):
        if mode == "mean":
            return image.mean(dim=-1, keepdim=True)
        channel = {"r": 0, "g": 1, "b": 2}[mode]
        return image[..., channel:channel + 1]

    def eval_expression(self, expression: str, conversion_mode: str, a=None, b=None, c=None):
        if not re.match(r'^[a-zA-Z0-9\s+\-*/(),.\^]+$', expression):
            raise ValueError(f"Expression contains invalid characters: {expression}")

        inputs = {'a': a, 'b': b, 'c': c}
        for name, value in inputs.items():
            if value is None and re.search(rf'\b{name}\b', expression):
                raise ValueError(f"Expression uses '{name}' but no input was provided for {name}")

        # MASK is [B,H,W]; give it a channel dim so it lines up with IMAGE [B,H,W,C]
        inputs = {k: (v.unsqueeze(-1) if v.ndim == 3 else v) for k, v in inputs.items() if v is not None}
        target_shape = next(iter(inputs.values())).shape

        namespace = dict(self.FUNCTIONS)
        for name, t in inputs.items():
            if t.shape[-1] == 1 and target_shape[-1] == 3:
                t = t.expand(-1, -1, -1, 3)
            elif t.shape[-1] != 1 and target_shape[-1] == 1:
                t = self.to_mask(t, conversion_mode)
            new_shape = (*target_shape[:-1], t.shape[-1])
            try:
                namespace[name] = t.expand(new_shape)
            except RuntimeError as e:
                raise ValueError(f"Cannot broadcast tensor of shape {tuple(t.shape)} to shape {new_shape}: {e}") from e

        expression = expression.replace('^', '**')
        try:
            result = eval(expression, {"__builtins__": None}, namespace)  # noqa: S307 - restricted namespace
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}") from e

        result = result.float().contiguous()
        if result.shape[-1] == 3:
            return (result, self.to_mask(result, conversion_mode).squeeze(-1))
        return (result.expand(*result.shape[:-1], 3).contiguous(), result.squeeze(-1))


class IP_Adapter_Settings_Distribution:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip": "IP-Adapter weight."}),
                "weight_type": (["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise'], {"tooltip": "IP-Adapter weight type, passed through to IPAdapter nodes."}),
            },
        }
    RETURN_TYPES = ("FLOAT", any_typ)
    RETURN_NAMES = ("weight", "weight_type")
    FUNCTION = "set"
    CATEGORY = "Eden 🌱/IP-Adapter"
    DESCRIPTION = "Outputs an IP-Adapter weight and weight type so several IPAdapter nodes can share them."

    def set(self, weight, weight_type):
        return (weight, weight_type)


class Eden_StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to search in."}),
                "target_text": ("STRING", {"default": "", "multiline": False, "tooltip": "Literal text to find."}),
                "replace_with": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to insert instead."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_text"
    CATEGORY = "Eden 🌱/Text"
    DESCRIPTION = "Replaces all occurrences of target text with replacement text in the input string."

    def replace_text(self, input_string: str, target_text: str, replace_with: str):
        return (input_string.replace(target_text, replace_with),)


class Eden_randbool:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "max": sys.maxsize, "tooltip": "Seed; the same seed always gives the same result."}),
                    "probability": ("FLOAT", {"default": 0.5, "tooltip": "Probability of returning True."}),
                }
            }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "sample"
    CATEGORY = "Eden 🌱/Random"
    DESCRIPTION = "Returns True with the given probability, deterministically per seed."

    def sample(self, seed, probability):
        return (torch.rand(1, generator=torch.Generator().manual_seed(seed)).item() < probability,)


class Eden_RandomPromptFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "prompts.txt", "tooltip": "Text file with one prompt per line (empty lines are ignored)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Picks line number seed % line_count."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_random_prompt"
    CATEGORY = "Eden 🌱/Random"
    DESCRIPTION = "Reads prompts from a text file (one per line) and returns the one at index seed % line_count."

    @classmethod
    def IS_CHANGED(cls, file_path, seed):
        return os.path.getmtime(file_path) if os.path.isfile(file_path) else ""

    def get_random_prompt(self, file_path: str, seed: int):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError(f"No valid prompts found in file: {file_path}")
        return (lines[seed % len(lines)],)


class Eden_RandomFilepathSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "./", "tooltip": "Folder to sample from (~ is expanded)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the random pick."}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": "", "tooltip": "Only files with this extension, e.g. '.png'. Empty = any."}),
                "include_subdirectories": ("BOOLEAN", {"default": False, "tooltip": "Also sample from subfolders."}),
                "filter_string": ("STRING", {"default": "", "tooltip": "Only filenames matching this text (see filter_mode). Empty = no filter."}),
                "filter_mode": (["contains", "starts_with", "ends_with", "regex"], {"default": "contains", "tooltip": "How filter_string is matched against the filename."}),
                "case_sensitive": ("BOOLEAN", {"default": False, "tooltip": "Case-sensitive filename filtering."}),
            }
        }

    FUNCTION = "sample_filepath"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    CATEGORY = "Eden 🌱/Random"
    DESCRIPTION = "Picks a random file path from a folder, with optional extension and filename filters. Hidden files are skipped."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # the output only depends on the set of matching files, so re-run exactly when the pick would change
        return cls().sample_filepath(**kwargs)[0]

    def sample_filepath(self, directory_path: str, seed: int,
                        file_extension: str = "", include_subdirectories: bool = False,
                        filter_string: str = "", filter_mode: str = "contains",
                        case_sensitive: bool = False):
        directory_path = os.path.expanduser(directory_path)
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        if file_extension and not file_extension.startswith('.'):
            file_extension = '.' + file_extension

        if include_subdirectories:
            all_files = [os.path.join(root, f) for root, _, files in os.walk(directory_path) for f in files]
        else:
            all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                         if os.path.isfile(os.path.join(directory_path, f))]

        if file_extension:
            all_files = [f for f in all_files if f.lower().endswith(file_extension.lower())]

        if filter_string:
            if filter_mode == "regex":
                pattern = re.compile(filter_string, 0 if case_sensitive else re.IGNORECASE)
                match = lambda name: pattern.search(name) is not None  # noqa: E731
            else:
                needle = filter_string if case_sensitive else filter_string.lower()
                test = {"contains": lambda n: needle in n, "starts_with": lambda n: n.startswith(needle),
                        "ends_with": lambda n: n.endswith(needle)}[filter_mode]
                match = lambda name: test(name if case_sensitive else name.lower())  # noqa: E731
            all_files = [f for f in all_files if match(os.path.basename(f))]

        all_files = sorted(f for f in all_files if not os.path.basename(f).startswith('.'))
        if not all_files:
            raise ValueError(f"No files found matching the specified criteria in {directory_path}")

        return (random.Random(seed).choice(all_files),)


class Eden_RepeatLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", {"tooltip": "Latent batch to repeat."}),
                             "amount": ("INT", {"default": 1, "min": 1, "max": 1024, "tooltip": "How many times to repeat the whole batch."}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat"
    CATEGORY = "Eden 🌱/Latent"
    DESCRIPTION = "Repeats a latent batch (and its noise mask / batch indices) N times."

    def repeat(self, samples, amount):
        s = samples.copy()
        s_in = samples["samples"]
        s["samples"] = s_in.repeat((amount,) + (1,) * (s_in.ndim - 1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat((math.ceil(s_in.shape[0] / masks.shape[0]),) + (1,) * (masks.ndim - 1))[:s_in.shape[0]]
            s["noise_mask"] = masks.repeat((amount,) + (1,) * (masks.ndim - 1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)


class Eden_DetermineFrameCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "n_target_frames": ("INT", {"default": 24, "min": 0, "max": 1024, "tooltip": "Desired number of frames."}),
                "n_source_frames": ("INT", {"default": 1, "min": 0, "max": 1024, "tooltip": "The result is snapped to a multiple of this (e.g. number of keyframes)."}),
                "policy": (["closest", "round down", "round up"], {"tooltip": "How to snap n_target_frames to a multiple of n_source_frames."}),
                "min_frames": ("INT", {"default": 1, "min": 0, "max": 1024, "step": 1, "tooltip": "Lower clamp for the result."}),
                "max_frames": ("INT", {"default": 1024, "min": 0, "max": 1024, "step": 1, "tooltip": "Upper clamp for the result."}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "determine_frame_count"
    CATEGORY = "Eden 🌱/Video"
    DESCRIPTION = "Snaps a target frame count to a multiple of the source frame count, then clamps it to [min_frames, max_frames]."

    def determine_frame_count(self, n_target_frames, n_source_frames, policy, min_frames, max_frames):
        min_frames, max_frames = min(min_frames, max_frames), max(min_frames, max_frames)

        if n_target_frames == 0:
            result = min(0, max_frames)
        elif n_source_frames <= 1:
            result = n_target_frames
        elif policy == "closest":
            result = round(n_target_frames / n_source_frames) * n_source_frames
        elif policy == "round down":
            result = max(n_source_frames, (n_target_frames // n_source_frames) * n_source_frames)
        elif policy == "round up":
            result = math.ceil(n_target_frames / n_source_frames) * n_source_frames
        else:
            result = n_target_frames

        return (int(max(min_frames, min(result, max_frames))),)


class SDTypeConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {"forceInput": True, "tooltip": "Checkpoint name to convert to text."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "tooltip": "Sampler name to convert to text."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "tooltip": "Scheduler name to convert to text."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL_NAME_STR", "SAMPLER_NAME_STR", "SCHEDULER_STR")
    FUNCTION = "convert_string"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Converts checkpoint / sampler / scheduler combo values into plain strings."

    def convert_string(self, model_name: str = "", sampler_name: str = "", scheduler: str = ""):
        return (model_name, sampler_name, scheduler)


class SDAnyConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "any_type_input": (any_typ, {"forceInput": True, "tooltip": "Any value."}),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("ANY_TYPE_OUTPUT",)
    FUNCTION = "convert_any"
    CATEGORY = "Eden 🌱/Logic"
    DESCRIPTION = "Passes any value through as a wildcard type, so it can connect to inputs of any type."

    def convert_any(self, any_type_input: str = ""):
        return (any_type_input,)


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v')
ARCHIVE_EXTENSIONS = ('.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z')


def resize_image_max_size(image: Image.Image, max_res: int) -> Image.Image:
    """Downscale so neither side exceeds max_res, preserving aspect ratio."""
    w, h = image.size
    if max_res <= 0 or (w <= max_res and h <= max_res):
        return image
    if w > h:
        new_w, new_h = max_res, int(h * (max_res / w))
    else:
        new_w, new_h = int(w * (max_res / h)), max_res
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _rgb_array(image: Image.Image, max_res: int) -> np.ndarray:
    return np.asarray(resize_image_max_size(image.convert('RGB'), max_res))


def _frames_to_tensor(frames: list) -> torch.Tensor:
    return torch.from_numpy(np.stack(frames)).float().div_(255.0)


def _resolve_media_path(full_path: str):
    if ('*' in full_path or '?' in full_path) and full_path.lower().endswith(IMAGE_EXTENSIONS):
        return [os.path.abspath(p) for p in glob.glob(full_path, recursive=True)] or None
    if os.path.exists(full_path):
        return os.path.abspath(full_path)
    return None


def load_path(path: str):
    """
    Resolves an absolute path, a path relative to the input directory, a path with
    [input]/[output]/[temp] annotation, or an image wildcard pattern (returns a list).
    """
    path = path.strip('"').strip("'").replace("\\", "/")

    if "[" in path:
        name, base_dir = folder_paths.annotated_filepath(path)
        if base_dir is not None:
            resolved = _resolve_media_path(os.path.join(base_dir, name))
            if resolved:
                return resolved

    input_path = os.path.join(folder_paths.get_input_directory(), path)
    for candidate in (path, input_path):
        resolved = _resolve_media_path(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(f"Could not find file or directory at {path} or {input_path}")


def list_image_files(directory: str) -> list:
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(directory, f)))


def load_video_frames(video_path: str, force_rate: float = 0.0, image_load_cap: int = 0, max_res: int = 0):
    """Reads frames sequentially with OpenCV. Returns (list of uint8 RGB arrays, fps)."""
    import cv2  # lazy: only this loader path needs OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(original_fps / force_rate)) if force_rate > 0 and original_fps > 0 else 1
    expected = math.ceil(max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1) / interval)
    if image_load_cap > 0:
        expected = min(expected, image_load_cap)
    pbar = comfy.utils.ProgressBar(expected)

    frames = []
    index = 0
    while image_load_cap <= 0 or len(frames) < image_load_cap:
        if index % interval:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if max_res > 0 and max(w, h) > max_res:
                frame = np.asarray(resize_image_max_size(Image.fromarray(frame), max_res))
            frames.append(frame)
            pbar.update(1)
        index += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    return frames, (force_rate if force_rate > 0 else original_fps)


def load_gif_frames(gif_path: str, force_rate: float = 0.0, image_load_cap: int = 0, max_res: int = 0):
    """Returns (list of uint8 RGB arrays, fps). A non-animated GIF gives one frame at fps 0."""
    gif = Image.open(gif_path)
    if not getattr(gif, "is_animated", False):
        return [np.asarray(resize_image_max_size(gif, max_res).convert('RGB'))], 0.0

    duration = gif.info.get('duration', 100)
    original_fps = 1000 / duration if duration else 10.0
    interval = max(1, round(original_fps / force_rate)) if force_rate > 0 else 1

    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        if image_load_cap > 0 and len(frames) >= image_load_cap:
            break
        if i % interval == 0:
            frames.append(_rgb_array(frame, max_res))

    if not frames:
        raise ValueError(f"No frames extracted from GIF: {gif_path}")
    return frames, (force_rate if force_rate > 0 else original_fps)


def load_image_array(image_path: str, max_res: int = 0) -> np.ndarray:
    return np.asarray(resize_image_max_size(ImageOps.exif_transpose(Image.open(image_path)), max_res).convert('RGB'))


class Eden_AllMediaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {
                    "image_upload": True,
                    "tooltip": "Image, folder of images, image wildcard (e.g. frames/*.png), video, GIF or archive (zip/tar/7z). Absolute, relative to input/, or [input]/[output]/[temp] annotated."
                }),
                "image_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": sys.maxsize,
                    "step": 1,
                    "tooltip": "Maximum number of images/frames to load. 0 means load all."
                }),
                "force_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "step": 0.1,
                    "tooltip": "Videos/GIFs: sample frames at roughly this FPS. 0 means use the original rate."
                }),
                "max_res": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": sys.maxsize,
                    "step": 1,
                    "tooltip": "Downscale so width and height stay at or below this. 0 means no resize."
                }),
                "sort": (["None", "alphabetical", "date_created", "date_modified", "random"], {
                    "default": "None",
                    "tooltip": "Order for multiple images. None = by filename; random uses a fixed seed."
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "WIDTH", "HEIGHT", "COUNT", "FILE_NAME", "FILE_PATH", "FPS")
    OUTPUT_TOOLTIPS = (
        "Loaded images/frames [B,H,W,3].",
        "Width in pixels.",
        "Height in pixels.",
        "Number of images/frames.",
        "File name without extension (first three names joined with | for multiple images).",
        "Path of the file, archive or containing folder.",
        "Frame rate for videos/GIFs, 0 for images.",
    )
    FUNCTION = "load_media"
    CATEGORY = "Eden 🌱/Loaders"
    DESCRIPTION = (
        "Loads an image, a folder of images, an image wildcard, a video, a GIF or an archive of images. "
        "Can subsample video frames with force_rate and downscale anything above max_res."
    )

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        resolved = load_path(path)
        if isinstance(resolved, str) and os.path.isdir(resolved):
            resolved = list_image_files(resolved)
        paths = resolved if isinstance(resolved, list) else [resolved]
        return hashlib.sha256(str([(p, os.path.getmtime(p)) for p in paths]).encode()).hexdigest()

    def load_media(self, path: str, image_load_cap: int = 0, force_rate: float = 0.0,
                   max_res: int = 0, sort: str = "None"):
        resolved = load_path(path)
        if isinstance(resolved, list):
            return self.process_image_list(resolved, image_load_cap, max_res, sort)

        path = resolved
        lower = path.lower()
        if os.path.isdir(path):
            return self.load_from_directory(path, image_load_cap, max_res, sort)
        if lower.endswith(ARCHIVE_EXTENSIONS):
            return self.load_from_archive(path, image_load_cap, max_res, sort)

        if lower.endswith(VIDEO_EXTENSIONS):
            frames, fps = load_video_frames(path, force_rate, image_load_cap, max_res)
        elif lower.endswith('.gif'):
            frames, fps = load_gif_frames(path, force_rate, image_load_cap, max_res)
        else:
            frames, fps = [load_image_array(path, max_res)], 0.0

        images = _frames_to_tensor(frames)
        b, h, w, _ = images.shape
        return (images, w, h, b, os.path.basename(path).rsplit('.', 1)[0], path, fps)

    def process_image_list(self, image_paths: list, image_load_cap: int = 0,
                           max_res: int = 0, sort: str = "None"):
        if not image_paths:
            raise ValueError("No image paths provided")

        if sort == "alphabetical":
            image_paths = sorted(image_paths, key=lambda x: x.lower())
        elif sort == "date_created":
            image_paths = sorted(image_paths, key=os.path.getctime)
        elif sort == "date_modified":
            image_paths = sorted(image_paths, key=os.path.getmtime)
        elif sort == "random":
            image_paths = list(image_paths)
            random.Random(0).shuffle(image_paths)
        else:
            image_paths = sorted(image_paths)

        if image_load_cap > 0:
            image_paths = image_paths[:image_load_cap]

        frames = []
        loaded_paths = []
        pbar = comfy.utils.ProgressBar(len(image_paths))
        for img_path in image_paths:
            try:
                frame = load_image_array(img_path, max_res)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
            if frames and frame.shape != frames[0].shape:
                h, w = frames[0].shape[:2]
                frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS))
            frames.append(frame)
            loaded_paths.append(img_path)
            pbar.update(1)

        if not frames:
            raise ValueError("No valid images found in the provided paths")

        images = _frames_to_tensor(frames)
        _, h, w, _ = images.shape
        file_names = [os.path.basename(p).rsplit('.', 1)[0] for p in image_paths]
        file_name = "|".join(file_names[:3]) + ("..." if len(file_names) > 3 else "")
        return (images, w, h, len(frames), file_name, os.path.dirname(image_paths[0]), 0.0)

    def load_from_directory(self, directory: str, image_load_cap: int = 0,
                            max_res: int = 0, sort: str = "None"):
        image_files = list_image_files(directory)
        if not image_files:
            raise ValueError(f"No valid image files found in directory {directory}")
        return self.process_image_list(image_files, image_load_cap, max_res, sort)

    def load_from_archive(self, archive_path: str, image_load_cap: int = 0,
                          max_res: int = 0, sort: str = "None"):
        with tempfile.TemporaryDirectory() as temp_dir:
            lower = archive_path.lower()
            if lower.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as z:
                    z.extractall(temp_dir)
            elif lower.endswith('.7z'):
                import py7zr  # optional dependency, only needed for .7z
                with py7zr.SevenZipFile(archive_path, mode='r') as z:
                    z.extractall(path=temp_dir)
            else:
                with tarfile.open(archive_path, 'r') as t:
                    t.extractall(temp_dir, filter='data')

            contents = os.listdir(temp_dir)
            root = temp_dir
            if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
                root = os.path.join(temp_dir, contents[0])

            images, w, h, count, _, _, fps = self.load_from_directory(root, image_load_cap, max_res, sort)
            return (images, w, h, count, os.path.basename(archive_path).rsplit('.', 1)[0], archive_path, fps)


class Eden_Save_Param_Dict:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"key_{i}"] = ("STRING", {"default": f"param{i}", "tooltip": f"JSON key for value_{i}. Empty = skip this pair."})
            optional[f"value_{i}"] = (any_typ,)
        return {
            "required": {
                "save_path": ("STRING", {"default": "params.json", "multiline": False, "tooltip": "File name (or subfolder/name) inside the output folder; a _00001 counter is appended. Supports %date:FORMAT%."}),
            },
            "optional": optional,
        }

    FUNCTION = "make_dict"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "json_string")
    OUTPUT_TOOLTIPS = ("Full path of the saved JSON file.", "The JSON content.")
    CATEGORY = "Eden 🌱/Loaders"
    DESCRIPTION = "Collects up to 10 key/value inputs and saves them as a JSON file in the output directory."

    LIST_KEYWORDS = ("images", "files", "paths", "styles", "inputs", "sources")

    def _make_serializable(self, obj, key=""):
        """Convert to JSON-friendly values, keeping numbers as numbers and path-like string lists as lists."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            n = obj.numel() if isinstance(obj, torch.Tensor) else obj.size
            if n == 1:
                return obj.item()
            if n < 1000:
                return obj.tolist()
            return f"{'Tensor' if isinstance(obj, torch.Tensor) else 'Array'} with shape {obj.shape}"
        if isinstance(obj, (list, tuple)):
            if all(isinstance(item, str) for item in obj):
                if any(k in key.lower() for k in self.LIST_KEYWORDS) or any(re.search(r'\.\w+$', item) for item in obj):
                    return list(obj)
                return "\n".join(obj)
            return [self._make_serializable(item, key) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v, str(k)) for k, v in obj.items()}
        try:
            props = {attr: self._make_serializable(getattr(obj, attr), attr)
                     for attr in dir(obj) if not attr.startswith('_') and not callable(getattr(obj, attr))}
        except Exception:
            return str(obj)
        return {"_type": type(obj).__name__, "properties": props} if props else str(obj)

    def _get_unique_filename(self, filename_prefix):
        if not filename_prefix.lower().endswith('.json'):
            filename_prefix += '.json'
        filename_prefix = re.sub(r"%date:([^%]+)%", lambda m: datetime.now().strftime(m.group(1)), filename_prefix)

        base_name, ext = os.path.splitext(filename_prefix)
        output_dir = os.path.join(self.output_dir, os.path.dirname(base_name))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(base_name)

        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{counter:05}{ext}")):
            counter += 1
        return os.path.join(output_dir, f"{base_name}_{counter:05}{ext}")

    def make_dict(self, save_path, **kwargs):
        result_dict = {}
        try:
            for i in range(1, 11):
                key = kwargs.get(f"key_{i}")
                value = kwargs.get(f"value_{i}")
                if key and value is not None:
                    try:
                        result_dict[key] = self._make_serializable(value, key)
                    except Exception as e:
                        logger.warning(f"Eden_Save_Param_Dict: error processing {key}: {e}")
                        result_dict[key] = f"ERROR: {e}"

            full_path = self._get_unique_filename(save_path)
            json_string = json.dumps(result_dict, indent=2, ensure_ascii=False)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            logger.info(f"Saved parameter dictionary to {full_path}")
            return (full_path, json_string)
        except Exception as e:
            logger.error(f"Error in Eden_Save_Param_Dict: {e}")
            return (f"ERROR: {e}", "{}")
