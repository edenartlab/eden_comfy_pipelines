import torch
from .exploration_state import ExplorationState
import os
import torch
import numpy as np
import torch
import time

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
        sign  = np.random.choice([-1, 1])
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



def load_random_image_embeddings():
    embed_dir = "custom_nodes/eden_comfy_pipelines/ip_adapter_utils/img_embeds"
    embeddings = []

    for p in [f for f in os.listdir(embed_dir) if f.endswith(".pth")]:
        embedding = ExplorationState.from_file(filename = os.path.join(embed_dir,p)).sample_embed
        embeddings.append(embedding)

    embeddings = torch.stack(embeddings).squeeze()
    print(f"Loaded image embeddings of shape {embeddings.shape} from {embed_dir}")

    norms = []
    for embed in embeddings:
        norm = torch.norm(embed).item()
        norms.append(norm)

    return embeddings, np.mean(norms)


def random_linear_combination(strength, embeds, num_samples, num_elements = 2):
    print("Applying random linear combination")
    random_embeddings, avg_norm = load_random_image_embeddings()
    new_embeds = []
    for i in range(num_samples):
        random_weights = np.random.uniform(0.3, 0.7, num_elements)
        random_weights = random_weights / np.mean(random_weights)

        indices = np.random.choice(range(len(random_embeddings)), num_elements, replace=False)

        print(f"Sampled indices: {indices} with weights: {random_weights}")

        weights = np.zeros(len(random_embeddings))
        weights[indices] = random_weights

        weights = torch.tensor(weights).to(embeds.device)

        # create a new random IP embedding by using the random weights (linear combination) to sum along the first axis of the random embeddings
        linear_combination = torch.sum(random_embeddings * weights.unsqueeze(1).unsqueeze(1), axis=0)

        # re-normalize:
        linear_combination = linear_combination / torch.norm(linear_combination) * avg_norm
        new_embed = (1-strength) * embeds + strength * linear_combination

        # renormalize:
        new_embed = new_embed / torch.norm(new_embed) * avg_norm
        new_embeds.append(new_embed)

    return torch.stack(new_embeds).squeeze()


class IPAdapterRandomRotateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "mode": (['random_rotation', 'random_linear_combination'], ),
                "seed": ("INT",{"default": 4}),
                "strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        mode: str = "random_rotation",
        strength: float = 0.65,
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

        if mode == "random rotation":
            new_pos_embeds = random_rotate_embeds(
                embeds = pos_embed,
                num_samples=num_samples,
                max_angle=max_angle,
                min_angle=min_angle
            )
        elif mode == "random_linear_combination":
            assert strength <= 1.0, "strength should be less than or equal to 1.0 when using random_linear_combination"
            assert strength >= 0.0, "strength should be greater than or equal to 0.0 when using random_linear_combination"

            new_pos_embeds = random_linear_combination(
                strength = strength,
                embeds = pos_embed,
                num_samples=num_samples
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