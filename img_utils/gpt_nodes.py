import base64
import io
import logging
import os
import sys

import numpy as np
from PIL import Image

NO_KEY_MESSAGE = "An OpenAI API key is required for {}. Put OPENAI_API_KEY in a .env file in the ComfyUI root (or in eden_comfy_pipelines) and never share it."

_openai_client = None


def _get_openai_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
        except ImportError:
            logging.warning("Eden_Comfy_Pipelines: python-dotenv is not installed, reading OPENAI_API_KEY from the environment only.")
        else:
            load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def _get_openai_client():
    """Shared OpenAI client, created on first use (retried until a key is found)."""
    global _openai_client
    if _openai_client is None:
        api_key = _get_openai_api_key()
        if not api_key:
            logging.warning("Eden_Comfy_Pipelines: OPENAI_API_KEY not found in the environment or .env, GPT nodes are disabled.")
            return None
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


class Eden_gpt4_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_token": ("INT", {"default": 100, "min": 1, "max": sys.maxsize, "tooltip": "Maximum number of tokens in the reply."}),
                "model": (["gpt-4o", "gpt-4-turbo"], {"default": "gpt-4o"}),
                "prompt": ("STRING", {"multiline": True, "default": "Write a poem about ComfyUI"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "tooltip": "Change to get a new completion; OpenAI sampling is best-effort deterministic per seed."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "gpt4_completion"
    CATEGORY = "Eden 🌱/AI"
    DESCRIPTION = "Sends a prompt to an OpenAI chat model and returns the reply. Needs OPENAI_API_KEY; errors are returned as text."

    def gpt4_completion(self, max_token, model, prompt, seed):
        try:
            client = _get_openai_client()
            if not client:
                return (NO_KEY_MESSAGE.format("the GPT node"),)

            response = client.chat.completions.create(
                model=model,
                seed=seed,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_token
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class Eden_GPTPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basic_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "The prompt to enhance."
                }),
                "enhancement_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Augment this visual description by adding specific details about lighting, scene elements, composition, and artistic style. Make it more descriptive and specific. Be bold and creative! Limit the final prompt to 100 words.",
                    "tooltip": "How GPT should rewrite the prompt."
                }),
                "max_token": ("INT", {"default": 500, "min": 1, "max": sys.maxsize, "tooltip": "Maximum number of tokens in the reply."}),
                "model": ([
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                ], {"default": "gpt-4o"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "tooltip": "Change to get a new enhancement."}),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Higher is more creative."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "Eden 🌱/AI"
    DESCRIPTION = "Rewrites a prompt with GPT following your enhancement instructions. Needs OPENAI_API_KEY; errors are returned as text."

    def enhance_prompt(self, basic_prompt, enhancement_instructions, max_token, model, seed, temperature=0.7):
        try:
            client = _get_openai_client()
            if not client:
                return (NO_KEY_MESSAGE.format("GPT Prompt Enhancer"),)

            system_message = """You are a prompt engineering expert. Your task is to enhance and improve the given prompt according to the provided instructions.
            Keep the enhanced prompt focused and coherent. Maintain the original intent while adding valuable details and improvements."""

            user_message = f"""Original prompt: {basic_prompt}

Enhancement instructions: {enhancement_instructions}

Please enhance this prompt according to the instructions. Provide only the enhanced prompt without any explanations or additional text."""

            response = client.chat.completions.create(
                model=model,
                seed=seed,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_token
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error in prompt enhancement: {str(e)}",)


class ImageDescriptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_token": ("INT", {"default": 100, "min": 1, "max": sys.maxsize, "tooltip": "Maximum number of tokens in the reply."}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1", "tooltip": "OpenAI-compatible API base URL."}),
                "model": (["gpt-4-vision Low", "gpt-4-vision High"], {"default": "gpt-4-vision Low", "tooltip": "Image detail level sent to gpt-4o (low is cheaper)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Consicely describe the content of the images. Respond with a single description per line (ending with a period and a newline character)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "Eden 🌱/AI"
    DESCRIPTION = "Describes the first image of a batch with GPT-4o vision. Needs OPENAI_API_KEY; errors are returned as text."

    def describe_image(self, image, max_token, endpoint, model, prompt):
        try:
            api_key = _get_openai_api_key()
            if not api_key:
                return (NO_KEY_MESSAGE.format("GPT-4 Vision"),)

            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=endpoint)

            img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            detail = "low" if model == "gpt-4-vision Low" else "high"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}", "detail": detail}
                        }]
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_token
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class Eden_GPTStructuredOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate a structured response about the given topic"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant that generates structured JSON responses."
                }),
                "json_schema": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "title": {"type": "string"},\n    "description": {"type": "string"},\n    "key_points": {"type": "array", "items": {"type": "string"}}\n  },\n  "required": ["title", "description", "key_points"]\n}',
                    "tooltip": "JSON schema the reply must follow (included in the system prompt)."
                }),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": sys.maxsize, "tooltip": "Maximum number of tokens in the reply."}),
                "model": (["gpt-4o", "gpt-4-turbo"], {"default": "gpt-4o"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "tooltip": "Change to get a new response."}),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Higher is more creative."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_structured_output"
    CATEGORY = "Eden 🌱/AI"
    DESCRIPTION = "Asks GPT for a JSON object following the given schema and returns it as a string. Needs OPENAI_API_KEY; errors are returned as text."

    def generate_structured_output(self, prompt, system_prompt, json_schema, max_tokens, model, seed, temperature=0.7):
        try:
            client = _get_openai_client()
            if not client:
                return (NO_KEY_MESSAGE.format("GPT Structured Output"),)

            system_message = f"""{system_prompt}

You must respond with a valid JSON object that strictly follows this schema:
{json_schema}

Do not include any explanations or text outside the JSON object."""

            response = client.chat.completions.create(
                model=model,
                seed=seed,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error in structured output generation: {str(e)}",)
