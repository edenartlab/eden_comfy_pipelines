import torch
from .exploration_state import ExplorationState
import os
import torch
import numpy as np
import torch

def generate_random_rotation_matrix(dim, max_angle, min_angle = 10):
    """
    Generate a random rotation matrix for a given dimension and minimum and maximum angle.
    
    Args:
        dim (int): The dimension of the rotation matrix.
        min_angle (float): The minimum absolute rotation angle in degrees.
        max_angle (float): The maximum rotation angle in degrees.
    
    Returns:
        np.ndarray: A rotation matrix of shape (dim, dim).
    """
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    
    rotation_matrix = np.eye(dim)
    
    for i in range(dim - 1):
        angle = np.random.uniform(min_angle_rad, max_angle_rad)
        sign = np.random.choice([-1, 1])
        angle *= sign
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Create a temporary 2x2 rotation matrix
        temp_matrix = np.array([[cos_angle, -sin_angle],
                                [sin_angle, cos_angle]])
        
        # Embed this 2x2 rotation matrix into the larger rotation matrix
        rotation_matrix[i:i+2, i:i+2] = temp_matrix
    
    return torch.tensor(rotation_matrix)

def small_random_rotation(tensor, max_angle, min_angle):
    """
    Apply a random rotation to each slice along the last dimension of the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [1, 257, 1280].
        max_angle (float): Maximum rotation angle in degrees.
        min_angle (float): Minimum rotation angle in degrees.
    
    Returns:
        torch.Tensor: Rotated tensor.
    """
    # Get the shape of the tensor
    _, num_slices, num_features = tensor.shape
    
    # Generate a random rotation matrix for the last dimension
    rotation_matrix = generate_random_rotation_matrix(num_features, max_angle=max_angle, min_angle=min_angle)
    print(f"Applying a random rotation min_angle: {min_angle} max_angle: {max_angle}")
    # Apply the rotation matrix to each slice
    rotated_tensor = torch.matmul(tensor, rotation_matrix.to(dtype = tensor.dtype))
    
    return rotated_tensor

def random_rotate_embeds(
    embeds, 
    max_angle = 1.0,
    min_angle = 0.1,
    num_samples: int = 4,
):
    new_embeds = torch.cat(
        [
            small_random_rotation(
                tensor=embeds,
                max_angle=max_angle,
                min_angle=min_angle
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
                "min_angle": ("FLOAT", {"default": 5}),
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
        max_angle: float = 1.0,
        min_angle: float = 0.1,
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
            max_angle=max_angle,
            min_angle=min_angle
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