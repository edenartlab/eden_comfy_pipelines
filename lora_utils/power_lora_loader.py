"""
Multi-LoRA loader with per-LoRA random strength ranges, adapted from rgthree's Power Lora Loader
(https://github.com/rgthree/rgthree-comfy). The LoRA rows are dynamic frontend widgets
(js/eden_power_lora_loader.js); each one arrives as a `lora_N` dict in this node's prompt inputs:
    {"on": bool, "lora": str, "strength": float, "random": bool, "min": float, "max": float, "out": int}
"""

import json
import os
import random

from nodes import LoraLoader

MAX_LORAS = 16


def _lora_rows(node_inputs):
    return [v for k, v in node_inputs.items() if k.startswith("lora_") and isinstance(v, dict) and v.get("lora")]


class Eden_PowerLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seeds the strength sampling. Each LoRA draws from its own stream (seed + LoRA name), so adding, removing or reordering LoRAs doesn't change the others' samples."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Diffusion model to apply the LoRAs to. Leave empty to only sample strengths."}),
                "clip": ("CLIP", {"tooltip": "CLIP model to apply the LoRAs to (same strength as the model)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING") + ("FLOAT",) * MAX_LORAS
    RETURN_NAMES = ("MODEL", "CLIP", "settings") + tuple(f"strength_{i + 1}" for i in range(MAX_LORAS))
    OUTPUT_TOOLTIPS = (
        "Model with all enabled LoRAs applied.",
        "CLIP with all enabled LoRAs applied.",
        'JSON with the seed and the applied strength per LoRA (by file basename), e.g. {"seed": 3, "loras": {"style_a": 0.72}}. Disabled LoRAs are listed with 0.0.',
    ) + ("Strength applied to this LoRA (0.0 when disabled).",) * MAX_LORAS
    FUNCTION = "load_loras"
    CATEGORY = "Eden 🌱/Loaders"
    DESCRIPTION = "Loads any number of LoRAs. Each row either uses a fixed strength or (🎲 on) samples one uniformly in [min, max] from the seed. Outputs one strength per LoRA and a JSON of all applied strengths for dataset logging."

    def load_loras(self, seed, model=None, clip=None, prompt=None, unique_id=None):
        rows = _lora_rows(prompt[unique_id]["inputs"])
        if len(rows) > MAX_LORAS:
            raise ValueError(f"Power Lora Loader supports at most {MAX_LORAS} LoRAs, got {len(rows)}.")

        strengths = [0.0] * MAX_LORAS
        settings = {}
        loader = LoraLoader()
        seen = {}
        for i, row in enumerate(rows):
            name = row["lora"]
            seen[name] = seen.get(name, 0) + 1
            if row.get("random"):
                lo, hi = sorted((row["min"], row["max"]))
                rng = random.Random(f"{seed}:{name}:{seen[name]}")
                strength = round(rng.uniform(lo, hi), 2)
            else:
                strength = round(row["strength"], 2)
            if not row["on"]:
                strength = 0.0

            if model is not None or clip is not None:
                model, clip = loader.load_lora(model, clip, name, strength, strength)

            strengths[row.get("out", i)] = strength
            key = os.path.splitext(os.path.basename(name))[0]
            base, n = key, 2
            while key in settings:
                key, n = f"{base} ({n})", n + 1
            settings[key] = strength

        settings_json = json.dumps({"seed": seed, "loras": settings})
        return {
            "ui": {"sampled_strengths": strengths[:len(rows)]},
            "result": (model, clip, settings_json, *strengths),
        }
