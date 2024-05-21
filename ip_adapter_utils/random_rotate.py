import torch

def small_random_rotation(x, epsilon=1e-4):
    input_dtype = x.dtype
    # x shape is expected to be [1, 1, 1280]
    dim = x.shape[-1]
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
                "noise_scale": ("FLOAT", {"default": 1e-2}),
            }
        }

    RETURN_TYPES = ("EMBEDS","LATENT")
    RETURN_NAMES = ("pos_embeds",)
    FUNCTION = "run"

    CATEGORY = "Eden 🌱"

    def run(
        self, 
        pos_embed: torch.tensor,
        latent: torch.tensor,
        num_samples: int = 4, 
        noise_scale: float = 1e-2,
    ):
        pos_embed = random_rotate_embeds(
            embeds = pos_embed,
            num_samples=num_samples,
            noise_scale=noise_scale
        )

        """
        The caveat right now is that it supports a latent batch size of 1 only
        """
        assert latent["samples"].shape[0] == 1, f"Expected batch size of latents to be 1 but got: {latent['samples'].shape[0]}"

        latent_tensor = latent["samples"]
        latent_tensor = torch.cat(
            [
                latent_tensor
                for i in range(num_samples)
            ],
            dim = 0
        )
        latent = {
            "samples": latent_tensor
        }
        return (pos_embed, latent)

# NODE_CLASS_MAPPINGS = {
#     "IPAdapterRandomRotateEmbeds": IPAdapterRandomRotateEmbeds
# }