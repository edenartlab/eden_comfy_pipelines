import os
import json
import torch
import numpy as np
import re

def replace_in_string(s, replacements):
    while True:
        replaced = False
        for target, replacement in replacements.items():
            new_s = re.sub(target, replacement, s, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s
                replaced = True
        if not replaced:
            break
    return s

def prepare_prompt_for_lora(prompt, lora_path, verbose=True):
    """
    This function is rather ugly, but implements a custom token-replacement policy we adopted at Eden:
    Basically you trigger the lora with a token "TOK" or "<concept>", and then this token gets replaced with the actual learned tokens
    """

    orig_prompt = prompt

    # Helper function to read JSON
    def read_json_from_path(path):
        with open(path, "r") as f:
            return json.load(f)

    # Check existence of "special_params.json"
    if not os.path.exists(os.path.join(lora_path, "special_params.json")):
        raise ValueError(
            "This concept is prob wasnt trained through Eden and is missing necessary information..."
        )

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))
    trigger_text = training_args["training_attributes"]["trigger_text"]
    lora_name = str(training_args["name"])
    lora_name_encapsulated = "<" + lora_name + ">"

    try:
        mode = training_args["concept_mode"]
    except KeyError:
        try:
            mode = training_args["mode"]
        except KeyError:
            mode = "object"

    # Handle different modes
    if mode != "style":
        replacements = {
            "<concept>": trigger_text,
            "<concepts>": trigger_text + "'s",
            lora_name_encapsulated: trigger_text,
            lora_name_encapsulated.lower(): trigger_text,
            lora_name: trigger_text,
            lora_name.lower(): trigger_text,
        }
        prompt = replace_in_string(prompt, replacements)
        if trigger_text not in prompt:
            prompt = trigger_text + ", " + prompt
    else:
        style_replacements = {
            "in the style of <concept>": "in the style of TOK",
            f"in the style of {lora_name_encapsulated}": "in the style of TOK",
            f"in the style of {lora_name_encapsulated.lower()}": "in the style of TOK",
            f"in the style of {lora_name}": "in the style of TOK",
            f"in the style of {lora_name.lower()}": "in the style of TOK",
        }
        prompt = replace_in_string(prompt, style_replacements)
        if "in the style of TOK" not in prompt:
            prompt = "in the style of TOK, " + prompt

    # Final cleanup
    prompt = replace_in_string(
        prompt, {"<concept>": "TOK", lora_name_encapsulated: "TOK"}
    )

    # Replace tokens based on token map
    prompt = replace_in_string(prompt, token_map)
    prompt = fix_prompt(prompt)

    if verbose:
        print("-------------------------")
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print("-- to:")
        print(prompt)
        print("-------------------------")

    return prompt


class Eden_Lora_Loader:
    def __init__(self):

        self.lora_scale  = 1.0
        self.token_scale = 1.0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "lora_folder": ("STRING", {"default": ""}),
                "concept_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_prompt"
    CATEGORY = "Eden 🌱"

    def prep_prompt(self, prompt, lora_folder, concept_strength):
        # find the concept-mode:



        return (prompt, lora_scale, positive_conditioning, negative_conditioning)