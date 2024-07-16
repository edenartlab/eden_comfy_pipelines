import numpy as np
import matplotlib.pyplot as plt

def compute_sampling_indices(total_n_frames, target_n_frames, verbose=0):
    """
    Computes an optimal subset of frames to sample from a video. It calculates a cost that represents the
    temporal (visual) distortion in the output video. The cost combines standard deviation of frame index
    differences with a penalty for abrupt changes, providing a more accurate measure of visual continuity.
    """

    # Generate target_n_frames evenly spaced frame indices
    target_indices = np.linspace(0, total_n_frames - 1, target_n_frames)
    target_indices_rounded = target_indices.round().astype(int)

    # Calculate the differences between consecutive indices
    index_diffs = np.diff(target_indices_rounded)

    # Calculate standard deviation of index differences
    std_diff  = np.std(index_diffs)
    mean_diff = np.mean(index_diffs)

    visual_cost = float(std_diff / mean_diff)

    if verbose:
        print("---------------------------")
        print("Target indices:")
        print(target_indices_rounded)
        print(f"Total frames: {total_n_frames}")
        print(f"Target frames: {target_n_frames}")
        print(f"Standard Deviation of Differences: {std_diff:.3f}")
        print(f"Visual Cost: {visual_cost:.3f}")

        # plot the index differences:
        plt.figure(figsize=(10, 5))
        plt.plot(index_diffs, marker='o')
        plt.title(f"diffs @{target_n_frames}, cost = {visual_cost:.3f}, std_diff = {std_diff:.3f}")
        plt.savefig(f"index_diffs_{target_n_frames}.png")
        plt.close()

    return list(target_indices_rounded), visual_cost


def compute_frame_parameters(video_info, target_video_speedup_factor, output_fps, source_sampling_fps_range = [7,12], n_tests = 20):
    # Extract relevant data from video_info dictionary
    source_fps     = video_info['source_fps']
    total_n_frames = video_info['loaded_frame_count']

    if source_fps < source_sampling_fps_range[0]:
        select_frame_indices = list(range(total_n_frames))
        best_source_sampling_fps = source_fps
        best_cost = 0
    else:
        # Step 1: Pick the optimal subset of frames to sample from the source video:
        best_cost, best_source_sampling_fps = np.inf, source_fps
        max_sampling_rate = min(source_fps, source_sampling_fps_range[1])

        for source_sampling_fps in list(np.linspace(source_sampling_fps_range[0], max_sampling_rate + 1, 100)):
            n_target_frames = round(total_n_frames * (source_sampling_fps / source_fps))
            target_indices, rounding_cost = compute_sampling_indices(total_n_frames, n_target_frames)

            if rounding_cost < best_cost:
                best_cost = rounding_cost
                select_frame_indices = target_indices
                best_source_sampling_fps = source_sampling_fps

    # Step 2: Compute the output frame multiplier such that the output video has the desired output_fps and appropriate speedup
    original_duration = total_n_frames / source_fps
    target_duration   = original_duration / target_video_speedup_factor
    required_output_frames = target_duration * output_fps

    # to achieve target_video_speedup_factor at output_fps, we need to create final video frames at a rate of 
    # the frame_multiplier will make sure that the select_frame_indices frames will get interpolated to produce frame_multiplier * len(select_frame_indices) frames
    # those will then finally be played back at output_fps
    frame_multiplier = int(round(required_output_frames / len(select_frame_indices)))

    # Compute how much the video will be sped up visually compared to the source:
    output_duration = len(select_frame_indices) * frame_multiplier / output_fps
    actual_video_speedup_factor = original_duration / output_duration

    print(f"Selected source_video sampling FPS: {best_source_sampling_fps:.3f} with visual cost {best_cost:.5f}")
    print(f"Selecting {len(select_frame_indices)} frames from the source video (originally {total_n_frames} frames).")
    print(f"Output frame multiplier: {frame_multiplier}")
    print(f"Actual achieved visual speedup: {actual_video_speedup_factor:.3f}")

    return select_frame_indices, output_fps, frame_multiplier


class VideoFrameSelector:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "input_frames": ("IMAGE",),
                "video_info": ("VHS_VIDEOINFO",),
                "output_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "target_video_speedup_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "min_source_sampling_fps": ("INT", {"default": 8, "min": 1, "max": 24}),
                "max_source_sampling_fps": ("INT", {"default": 12, "min": 1, "max": 24}),
                "frame_load_cap": ("INT", {"default": 0, "min": 1, "max": 1000}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE","INT","FLOAT",)
    RETURN_NAMES = ("Selected_frames","multiplier","frame_rate",)
    FUNCTION = "select_frames"

    def select_frames(self, input_frames, video_info, output_fps, target_video_speedup_factor, min_source_sampling_fps, max_source_sampling_fps, frame_load_cap):
        
        # Compute the optimal subset of frames to sample from the source video:
        select_frame_indices, output_fps, frame_multiplier = compute_frame_parameters(video_info, target_video_speedup_factor, output_fps, source_sampling_fps_range=[min_source_sampling_fps, max_source_sampling_fps])

        # Select the frames from the input_frames:
        selected_frames = input_frames[select_frame_indices]

        if frame_load_cap > 0:
            # Limit the number of frames to be loaded:
            selected_frames = selected_frames[:frame_load_cap]

        return (selected_frames, frame_multiplier, output_fps,)


if __name__ == "__main__":    

    video_info_dict = {"source_fps": 100.0, "source_frame_count": 150, "source_duration": 9.375, "source_width": 896, "source_height": 512, "loaded_fps": 16.0, "loaded_frame_count": 150, "loaded_duration": 9.375, "loaded_width": 896, "loaded_height": 512}
    target_video_speedup_factor = 0.75
    output_fps = 24

    compute_frame_parameters(video_info_dict, target_video_speedup_factor, output_fps)
