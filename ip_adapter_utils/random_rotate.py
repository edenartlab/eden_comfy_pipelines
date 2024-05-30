import torch
from .exploration_state import ExplorationState
import os
import torch
import numpy as np
import torch

def generate_random_rotation_matrix(dim, max_angle):
    """
    Generate a random rotation matrix for a given dimension and maximum angle.
    
    Args:
        dim (int): The dimension of the rotation matrix.
        max_angle (float): The maximum rotation angle in degrees.
    
    Returns:
        np.ndarray: A rotation matrix of shape (dim, dim).
    """
    max_angle_rad = np.deg2rad(max_angle)
    angle = np.random.uniform(-max_angle_rad, max_angle_rad)
    
    rotation_matrix = np.eye(dim)
    
    for i in range(dim):
        angle = np.random.uniform(-max_angle_rad, max_angle_rad)
        rotation_matrix[i, i] = np.cos(angle)
        if i < dim - 1:
            rotation_matrix[i, i + 1] = -np.sin(angle)
            rotation_matrix[i + 1, i] = np.sin(angle)
            rotation_matrix[i + 1, i + 1] = np.cos(angle)
    
    return torch.tensor(rotation_matrix)

def small_random_rotation(tensor, max_angle):
    """
    Apply a random rotation to each slice along the last dimension of the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [1, 257, 1280].
        max_angle (float): Maximum rotation angle in degrees.
    
    Returns:
        torch.Tensor: Rotated tensor.
    """
    # Get the shape of the tensor
    _, num_slices, num_features = tensor.shape
    
    # Generate a random rotation matrix for the last dimension
    rotation_matrix = generate_random_rotation_matrix(num_features, max_angle)
    print(f"Applying a random rotation of upto {max_angle} degrees")
    # Apply the rotation matrix to each slice
    rotated_tensor = torch.matmul(tensor, rotation_matrix.to(dtype = tensor.dtype))
    
    return rotated_tensor

def random_rotate_embeds(
    embeds, 
    max_angle=1e-4, 
    num_samples: int = 4
):
    new_embeds = torch.cat(
        [
            small_random_rotation(
                tensor=embeds,
                max_angle=max_angle
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
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "seed": ("INT",{"default": 4}),
                "max_angle": ("FLOAT", {"default": 20}),
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
        seed: int,
        num_samples: int = 4, 
        max_angle: float = 1e-2,
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
            max_angle=max_angle
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