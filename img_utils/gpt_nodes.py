import base64
import io, os
from PIL import Image
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI API key loaded")
except:
    OPENAI_API_KEY = None
    client = None
    print("Eden_Comfy_Pipelines: WARNING: Could not find OPENAI_API_KEY in .env, disabling gpt prompt generation.")


class Eden_gpt4_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_token": ("INT", {"default": 100}),
                "model": (["gpt-4o", "gpt-4-turbo"], {"default": "gpt-4o"}),
                "prompt": ("STRING", {"multiline": True, "default": "Write a poem about ComfyUI"}),
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "gpt4_completion"
    CATEGORY = "Eden 🌱"

    def gpt4_completion(self, max_token, model, prompt, seed):
        try:

            if not client:
                print("An OpenAI API key is required for GPT node, put a .env file with your key in the comfyui root directory!")
                return ("An OpenAI API key is required for GPT-4 Vision. Make sure to place a .env file in the root directory of eden_comfy_pipelines with your secret API key. Make sure to never share your API key with anyone.", )

            response = client.chat.completions.create(
                    model=model,
                    seed=seed,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_token
            )

            gpt_completion = response.choices[0].message.content
            print(f"GPT4 completion:\n{gpt_completion}")
            return (gpt_completion,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class Eden_GPTPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basic_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape"
                }),
                "enhancement_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Augment this visual description by adding specific details about lighting, scene elements, composition, and artistic style. Make it more descriptive and specific. Be bold and creative! Limit the final prompt to 100 words."
                }),
                "max_token": ("INT", {"default": 500}),
                "model": ([
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                ], {"default": "gpt-4o"}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "Eden 🌱"

    def enhance_prompt(self, basic_prompt, enhancement_instructions, max_token, model, seed, temperature=0.7):
        try:
            if not client:
                return ("An OpenAI API key is required for GPT Prompt Enhancer. Make sure to place a .env file in the root directory with your OpenAI API key.",)

            # Construct the system message to guide GPT's behavior
            system_message = """You are a prompt engineering expert. Your task is to enhance and improve the given prompt according to the provided instructions. 
            Keep the enhanced prompt focused and coherent. Maintain the original intent while adding valuable details and improvements."""

            # Construct the user message combining the prompt and instructions
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

            enhanced_prompt = response.choices[0].message.content
            return (enhanced_prompt,)
            
        except Exception as e:
            return (f"Error in prompt enhancement: {str(e)}",)

class ImageDescriptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_token": ("INT", {"default": 100}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
                "model": (["gpt-4-vision Low", "gpt-4-vision High"], {"default": "gpt-4-vision Low"}),
                "prompt": ("STRING", {"multiline": True, "default": "Consicely describe the content of the images. Respond with a single description per line (ending with a period and a newline character)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "Eden 🌱"

    def image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def set_system_message(self, sysmsg):
        return [{
            "role": "system",
            "content": sysmsg
        }]

    def describe_image(self, image, max_token, endpoint, model, prompt):
        try:
            image = image[0]
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if not OPENAI_API_KEY:
                return "An OpenAI API key is required for GPT-4 Vision. Make sure to place a .env file in the root directory of eden_comfy_pipelines with your secret API key. Make sure to never share your API key with anyone."
            
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=endpoint)
            processed_image = self.image_to_base64(img)
            detail = "low" if model == "gpt-4-vision Low" else "high"
            system_message = self.set_system_message("You are a helpful assistant.")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=system_message + [
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{processed_image}", "detail": detail}
                        }]
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_token
            )
            description = response.choices[0].message.content
            print(f"GPT4-v Description:\n{description}")
            return (description,)
        except Exception as e:
            return (f"Error: {str(e)}",)
