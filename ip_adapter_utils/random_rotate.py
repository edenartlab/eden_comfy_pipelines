import torch
from .exploration_state import ExplorationState
import os
import torch
import numpy as np
import torch
import time
from typing import List


def get_id_from_filename(filename):
    ## image_id of images/scoobydoo.jpg = scoobydoo
    image_id = os.path.basename(filename).split(".")[0]
    return image_id

def find_all_filenames_with_extension(filenames: List[str], extensions: List[str]) -> List[str]:
    result = []
    for filename in filenames:
        if any(filename.endswith(ext) for ext in extensions):
            result.append(filename)
    return result

def get_filenames_in_a_folder(folder: str):
    """
    returns the list of paths to all the files in a given folder
    """
    
    if folder[-1] == '/':
        folder = folder[:-1]
        
    files =  os.listdir(folder)
    files = [f'{folder}/' + x for x in files]
    return files

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



def load_random_image_embeddings(embed_dir):
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


def random_linear_combination(embed_dir, strength, embeds, num_samples, num_elements = 2):
    print("Applying random linear combination")
    random_embeddings, avg_norm = load_random_image_embeddings(embed_dir)
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


import urllib.request
import zipfile
import os

EMBEDDINGS_DIR = "custom_nodes/eden_comfy_pipelines/ip_adapter_utils/img_embeds"

if not os.path.exists(EMBEDDINGS_DIR):
    # download the folder zip:
    url = "https://storage.googleapis.com/public-assets-xander/A_workbox/img_embeds.zip"
    
    # download the .zipfile:

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, "img_embeds.zip")

    # unzip the folder:
    with zipfile.ZipFile("img_embeds.zip", 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(EMBEDDINGS_DIR))

    # remove the .zip file:
    os.remove("img_embeds.zip")


class IPAdapterRandomRotateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "mode": ("STRING", {"default": 'random_rotation'}),
                "seed": ("INT",{"default": 4}),
                "embed_dir": (os.listdir(EMBEDDINGS_DIR), ),
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
        embed_dir: str = None,
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

        if mode == "random_rotation":
            new_pos_embeds = random_rotate_embeds(
                embeds = pos_embed,
                num_samples=num_samples,
                max_angle=max_angle,
                min_angle=min_angle
            )
        elif mode == "random_linear_combination":
            assert strength <= 1.0, "strength should be less than or equal to 1.0 when using random_linear_combination"
            assert strength >= 0.0, "strength should be greater than or equal to 0.0 when using random_linear_combination"
            assert embed_dir is not None, "embed_dir should be provided when using random_linear_combination"

            embed_dir = os.path.join(EMBEDDINGS_DIR, embed_dir)

            new_pos_embeds = random_linear_combination(
                embed_dir=embed_dir,
                strength = strength,
                embeds = pos_embed,
                num_samples=num_samples
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

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

class SavePosEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "cache_dir": ("STRING", {"default": "eden_images/xander_big"}),
                "non_embedded_images_folder": ("STRING", {"default": "eden_images/non_embedded_images"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_dir",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"

    def run(
        self, 
        pos_embed,
        cache_dir: str,
        non_embedded_images_folder: str
    ):
        assert pos_embed.ndim == 3, f"Expected batch to have 3 dims (batch, 257, 1280) but got: {pos_embed.ndim} dims"
        non_embedded_image_filenames = get_filenames_in_a_folder(
            folder = non_embedded_images_folder,
        )
        assert len(non_embedded_image_filenames) == pos_embed.shape[0], f"Expected the batch size of pos_embed ({pos_embed.shape[0]}) to be the same as the number of images found in non_embedded_images_folder: {len(non_embedded_image_filenames)}. non_embedded_image_filenames: {non_embedded_image_filenames}"

        all_image_ids = [
            get_id_from_filename(filename = f)
            for f in non_embedded_image_filenames
        ]

        for batch_idx, image_id in enumerate(all_image_ids):
            save_filename = os.path.join(
                cache_dir,
                f"{image_id}.pth"
            )
            torch.save(
                pos_embed[batch_idx],
                f = save_filename
            )
            print(f"[SavePosEmbeds] Saved: {save_filename}")
        
        return (cache_dir,)

class FolderScanner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_dir": ("STRING", {"default": "eden_images/xander_big"}),
                "non_embedded_images_folder": ("STRING", {"default": "eden_images/non_embedded_images"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("non_embedded_images_folder",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"
    
    def run(self, cache_dir: str, non_embedded_images_folder: str):
        """
        expects image_folder to be a folder containing both images and embeddings.
        ideally, it should contain pairs of image and their corresponding IP adapter embeddings as:
        - x.jpg, x.pth
        - y.jpg, y.pth

        {id}.jpg should have an {id}.pth associated to it.

        if image and no embedding:
            generate embedding for image and save {id}.pth (this is done in a subsequent node, not on this one)
        if embedding and no image:
            delete {id}.pth

        run this scan every time the node is run
        """
        assert os.path.exists(cache_dir), f"Invalid cache_dir: {cache_dir}"
        assert os.path.exists(non_embedded_images_folder), f"Invalid non_embedded_images_folder: {non_embedded_images_folder}"

        filenames = get_filenames_in_a_folder(folder = cache_dir)
        all_image_filenames = find_all_filenames_with_extension(
            filenames = filenames,
            extensions = [".jpg"]
        )

        all_embedding_filenames = find_all_filenames_with_extension(
            filenames = filenames,
            extensions = [".pth"]
        )

        image_filenames_without_embeddings = []
        for image_filename in all_image_filenames:
            image_id = get_id_from_filename(filename = image_filename)
            corresponding_embedding_filename = os.path.join(cache_dir, f"{image_id}.jpg")

            if corresponding_embedding_filename not in all_embedding_filenames:
                image_filenames_without_embeddings.append(
                    image_filename
                )
            else:
                pass

        embedding_filenames_to_be_deleted = []
        for embedding_filename in all_embedding_filenames:
            image_id = os.path.basename(embedding_filename).split(".")[0]
            corresponding_image_filename = os.path.join(cache_dir, f"{image_id}.jpg")
            if corresponding_image_filename not in all_image_filenames:
                embedding_filenames_to_be_deleted.append(
                    embedding_filename
                )

        for image_filename in image_filenames_without_embeddings:
            print(f"[FolderScanner] Copying: {image_filename} to {non_embedded_images_folder}")
            os.system(
                f"cp {image_filename} {non_embedded_images_folder}"
            )

        for embedding_filename in embedding_filenames_to_be_deleted:
            print(f"[FolderScanner] Deleting: {embedding_filename}")
            os.system(
                f"rm {embedding_filename}"
            )

        return (non_embedded_images_folder,)    