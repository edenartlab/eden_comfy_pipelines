import sys, os, time, math
import folder_paths

def find_comfy_models_dir():
    return str(folder_paths.models_dir)

class Eden_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT','STRING')
    RETURN_NAMES = ('seed','seed_string')
    FUNCTION = 'output'
    CATEGORY = "Eden 🌱/general"

    def output(self, seed):
        seed_string = str(seed)
        return (seed, seed_string,)


class Eden_RepeatLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 1024}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat"
    CATEGORY = "Eden 🌱/general"

    def repeat(self, samples, amount):
        s = samples.copy()
        s_in = samples["samples"]
        
        s["samples"] = s_in.repeat((amount, 1,1,1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount, 1,1,1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

class Eden_DetermineFrameCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "n_target_frames": ("INT", {"default": 24, "min": 0, "max": 1024}),
                "n_source_frames": ("INT", {"default": 1, "min": 0, "max": 1024}),
                "policy": (["closest", "round down", "round up"],),
            },
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "determine_frame_count"
    CATEGORY = "Eden 🌱/general"

    def determine_frame_count(self, n_target_frames, n_source_frames, policy):
        # Handle the case where n_target_frames is 0
        if n_target_frames == 0:
            return (0,)
        
        if n_source_frames <= 1:
            return (n_target_frames,)
        
        # Calculate output based on policy
        if policy == "closest":
            # Closest multiple of n_source_frames to n_target_frames
            closest_multiple = round(n_target_frames / n_source_frames) * n_source_frames
            return (closest_multiple,)
        elif policy == "round down":
            # Largest multiple of n_source_frames smaller than or equal to n_target_frames
            round_down_multiple = (n_target_frames // n_source_frames) * n_source_frames
            
            if round_down_multiple <= n_source_frames:
                round_down_multiple = n_source_frames

            return (round_down_multiple,)
        elif policy == "round up":
            # Smallest multiple of n_source_frames greater than or equal to n_target_frames
            round_up_multiple = math.ceil(n_target_frames / n_source_frames) * n_source_frames
            return (round_up_multiple,)

        return (n_target_frames,)
