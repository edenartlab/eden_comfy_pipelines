import torch
from .exploration_state import ExplorationState
import os
import time

def small_random_rotation(x, epsilon=1e-4):
    input_dtype = x.dtype
    # x shape is expected to be [1, 1, 1280]
    dim = x.shape[-1]

    seed = int(time.time() * 1e6) % 2**32
    torch.manual_seed(seed)

    # Generate a random skew-symmetric matrix
    A = torch.randn((dim, dim), device=x.device)
    A = (A - A.t()) / 2  # Making A skew-symmetric

    # Small rotation approximation
    # R = I + epsilon * A
    I = torch.eye(dim, device=x.device)
    R = I + epsilon * A

    # Applying the rotation to x
    x_rotated = torch.matmul(x.float(), R)
    return x_rotated.to(input_dtype)

def random_rotate_embeds(
    embeds, 
    noise_scale=1e-4, 
    num_samples: int = 4
):
    new_embeds = torch.cat(
        [
            small_random_rotation(
                x=embeds,
                epsilon=noise_scale
            )
            for i in range(num_samples)
        ]
    )
    return new_embeds

class IPAdapterRandomRotateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "latent": ("LATENT", ),
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "seed": ("INT",{"default": 4}),
                "noise_scale": ("FLOAT", {"default": 1e-2, "min": 0.0, "max": 0.5, "step": 0.01}),
                "exploration_state_filename": ("STRING", {"default": "eden_exploration_state.pth"})
            }
        }

    RETURN_TYPES = ("EMBEDS","INT",)
    RETURN_NAMES = ("pos_embeds","batch_size",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"

    def run(
        self, 
        pos_embed: torch.tensor,
        latent: torch.tensor,
        seed: int,
        num_samples: int = 4, 
        noise_scale: float = 1e-2,
        exploration_state_filename: torch.tensor = None
    ):
        print(f"Fake seed to make this node run every time: {seed}")
        if os.path.exists(exploration_state_filename):
            print(f"Loading ExplorationState: {exploration_state_filename}")
            pos_embed = ExplorationState.from_file(
                filename = exploration_state_filename
            ).sample_embed
        else:
            print("No ExplorationState found. Using the input pos_embeds")
        
        new_pos_embeds = random_rotate_embeds(
            embeds = pos_embed,
            num_samples=num_samples,
            noise_scale=noise_scale
        )

        return (new_pos_embeds, num_samples,)


class SaveExplorationState:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "filename": ("STRING", {"default": "eden_exploration_state.pth"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"

    def run(
        self, 
        pos_embed: str,
        filename: str,
    ):

        exploration_state = ExplorationState(
            sample_embed=pos_embed,
        )
        exploration_state.save(filename)
        print("-----------------------------------")
        print(f"Saved ExplorationState: {filename}")
        return (filename,)