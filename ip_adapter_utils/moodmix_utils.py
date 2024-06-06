import torch
import folder_paths
from .exploration_state import ExplorationState
import os
import numpy as np
import time
from typing import List
from PIL import ImageOps, Image


def get_id_from_filename(filename):
    """
    Extract the image ID from a given filename.
    Args:
        filename (str): The path to the file.
    Returns:
        str: The extracted image ID.
    
    """
    filename = str(filename)
    basename = os.path.basename(filename)
    # Handle case where there might be no extension or multiple dots
    image_id, _ = os.path.splitext(basename)
    
    if not image_id:
        return ""

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


def centre_crop_images(images, target_resolution):
    processed_images = []

    for img in images:
        width, height = img.size

        # Calculate the center crop box
        new_side = min(width, height)
        left = (width - new_side) / 2
        top = (height - new_side) / 2
        right = (width + new_side) / 2
        bottom = (height + new_side) / 2

        # Crop the image to the calculated box
        img_cropped = img.crop((left, top, right, bottom))

        # Resize the image to the target resolution
        img_resized = img_cropped.resize((target_resolution, target_resolution), Image.BICUBIC)

        # Append the processed image to the list
        processed_images.append(img_resized)

    return processed_images

##########################################################################
################################## NODES #################################
##########################################################################

class Get_Prefixed_Imgs:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": os.path.basename(folder_paths.get_output_directory())}),
                "filename_prefix": ("STRING", {"default": "Prefered_Images"}),
                "max_n_imgs": ("INT", {"default": 1}),
                "seed": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("anchor_images", "use_anchor")
    FUNCTION = "run"
    CATEGORY = "Eden 🌱"
    
    def run(self, folder, filename_prefix, max_n_imgs, seed = 0):
        # Load the most recent preference img:
        img_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if filename_prefix in f])
        img_paths = img_paths[::-1]

        print(f"Found {len(img_paths)} prefixed img paths")

        # For now lets just use the last user-preference image:
        last_preference_img_paths = img_paths[:max_n_imgs]

        if len(last_preference_img_paths) > 0:
            anchor_images = []
            for image_path in last_preference_img_paths:
                print(image_path)
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                image = img.convert("RGB")
                anchor_images.append(image)

            anchor_images = centre_crop_images(anchor_images, 512)
            anchor_images = [np.array(image).astype(np.float32) / 255.0 for image in anchor_images]
            use_anchor = 1
        else:
            # There were no prior prefered imgs
            dummy_img = np.random.rand(256, 256, 3)
            anchor_images = [dummy_img.astype(np.float32) / 255.0]
            use_anchor = 0

        if anchor_images:
            if len(anchor_images) > 1:
                anchor_images = [torch.from_numpy(img)[None,] for img in anchor_images]
                anchor_images = torch.cat(anchor_images, dim=0)
            else:
                anchor_images = torch.from_numpy(anchor_images[0])[None,]

        print(anchor_images.shape)

        return (anchor_images, use_anchor,)


class SavePosEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "cache_dir": ("STRING", {"default": "eden_images/xander_big"}),
                "non_embedded_image_filenames": ("LIST",),
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
        non_embedded_image_filenames: str,
    ):
        assert pos_embed.ndim == 3, f"Expected batch to have 3 dims but got: {pos_embed.ndim} dims"
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
        
        return (cache_dir,)



class FolderScanner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_dir": ("STRING", {"default": "eden_images/xander_big"}),
                "seed": ("INT",{"default": 0}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("non_embedded_image_filenames",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"
    
    def run(self, cache_dir, seed = 0):
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

        filenames = get_filenames_in_a_folder(folder = cache_dir)
        all_image_filenames = find_all_filenames_with_extension(
            filenames = filenames,
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
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

        for embedding_filename in embedding_filenames_to_be_deleted:
            print(f"[FolderScanner] Deleting: {embedding_filename}")
            os.system(
                f"rm {embedding_filename}"
            )

        return (image_filenames_without_embeddings,)

class Load_Embeddings_From_Folder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "eden_images/xander_big"}),
            }
        }
    RETURN_TYPES = ("EMBEDS", "FLOAT")
    RETURN_NAMES = ("embeddings", "avg_embed_norm")
    FUNCTION = "load"
    CATEGORY = "Eden 🌱"

    def load(self, directory_path):
        embeddings = []

        for p in [f for f in sorted(os.listdir(directory_path)) if f.endswith(".pth")]:
            try:
                embedding = torch.load(os.path.join(directory_path, p))
                embeddings.append(embedding)
            except:
                print(f"Failed to load {os.path.join(directory_path, p)}")

        embeddings = torch.stack(embeddings).squeeze()
        print(f"Loaded image embeddings of shape {embeddings.shape} from {directory_path}")

        norms = []
        for embed in embeddings:
            norm = torch.norm(embed).item()
            norms.append(norm)

        return (embeddings, np.mean(norms),)



class Linear_Combine_IP_Embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a_embeds": ("EMBEDS", ),
                "b_embeds": ("EMBEDS", ),
                "a_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "optional_target_norm": ("FLOAT", {"default": -1.0}),
            }
        }

    RETURN_TYPES = ("EMBEDS",)
    RETURN_NAMES = ("embeds",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱"

    def run(
        self,
        a_embeds,
        b_embeds,
        a_strength: float,
        optional_target_norm = None
    ):

        embeds = a_strength * a_embeds + (1 - a_strength) * b_embeds

        # re-normalize when target norm is provided:
        if optional_target_norm and optional_target_norm > 0:
            norm = torch.norm(embeds, dim=tuple(range(1, embeds.dim())), keepdim=True)
            embeds = embeds / norm * optional_target_norm
        
        return (embeds, )


class Random_Style_Mixture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_embeddings": ("EMBEDS", ),
                "avg_embed_norm": ("FLOAT",  {"default": 300, "min": 0.0, "max": 500, "step": 0.5}),
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "num_style_components":  ("INT", {"default": 4, "min": 1}),
                "min_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("EMBEDS", "INT")
    RETURN_NAMES = ("pos_embed", "batch_size")
    FUNCTION = "run"
    CATEGORY = "Eden 🌱"

    def run(
        self,
        style_embeddings,
        avg_embed_norm,
        num_samples: int,
        num_style_components: int,
        min_weight: float
    ):

        style_directions = []

        # Ensure that num_style_components is not greater than the size of the first dimension
        assert num_style_components <= style_embeddings.size(0), "num_style_components is greater than the number of style images!"

        for i in range(num_samples):
            # Sample the style images to use:
            indices = torch.randperm(style_embeddings.size(0))[:num_style_components]
            selected_style_embeddings = style_embeddings[indices]

            # Sample the style image weights to use:
            random_weights = np.random.uniform(min_weight, 1.0, num_style_components)
            random_weights = random_weights / np.mean(random_weights)
            random_weights = torch.tensor(random_weights).to(style_embeddings.device)

            print(f"Sampled indices: {indices} with weights: {random_weights}")

            # create a new random IP embedding by using the random weights (linear combination) to sum along the first axis of the random embeddings
            linear_combination = torch.sum(selected_style_embeddings * random_weights.unsqueeze(1).unsqueeze(1), axis=0)

            # re-normalize:
            linear_combination = linear_combination / torch.norm(linear_combination) * avg_embed_norm
            style_directions.append(linear_combination)

        style_directions = torch.stack(style_directions).squeeze()

        return (style_directions, style_directions.shape[0])

#########################################################
#################### DEPRECATED #########################
#########################################################



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


class IPAdapterRandomRotateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos_embed": ("EMBEDS", ),
                "num_samples": ("INT", {"default": 4, "min": 1}),
                "mode": ("STRING", {"default": 'random_rotation'}),
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

        if mode == "random_rotation":
            new_pos_embeds = random_rotate_embeds(
                embeds = pos_embed,
                num_samples=num_samples,
                max_angle=max_angle,
                min_angle=min_angle
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return (new_pos_embeds, num_samples,)


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