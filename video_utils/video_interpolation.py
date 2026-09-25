import logging

import numpy as np


def compute_sampling_indices(total_n_frames, target_n_frames):
    """
    Evenly spaced (rounded) indices of target_n_frames frames out of total_n_frames, plus a visual cost:
    the std / mean of the gaps between consecutive indices (0 = perfectly regular sampling, no stutter).
    """
    target_indices = np.linspace(0, total_n_frames - 1, target_n_frames).round().astype(int)
    index_diffs = np.diff(target_indices)
    visual_cost = float(np.std(index_diffs) / np.mean(index_diffs))
    return list(target_indices), visual_cost


def compute_frame_parameters(video_info, target_video_speedup_factor, output_fps, source_sampling_fps_range=(7, 12)):
    source_fps = video_info['source_fps']
    total_n_frames = video_info['loaded_frame_count']

    # Pick the source sampling fps whose frame subset has the most regular spacing
    select_frame_indices = list(range(total_n_frames))
    best_cost, best_source_sampling_fps = 0, source_fps
    if source_fps >= source_sampling_fps_range[0]:
        best_cost = np.inf
        max_sampling_rate = min(source_fps, source_sampling_fps_range[1])
        for source_sampling_fps in np.linspace(source_sampling_fps_range[0], max_sampling_rate + 1, 100):
            n_target_frames = round(total_n_frames * (source_sampling_fps / source_fps))
            if n_target_frames < 2:
                continue
            target_indices, rounding_cost = compute_sampling_indices(total_n_frames, n_target_frames)
            if rounding_cost < best_cost:
                best_cost = rounding_cost
                select_frame_indices = target_indices
                best_source_sampling_fps = source_sampling_fps

    # Frame multiplier: how many frames the interpolator must make per selected frame so that playback at
    # output_fps gives the requested speedup
    original_duration = total_n_frames / source_fps
    required_output_frames = original_duration / target_video_speedup_factor * output_fps
    frame_multiplier = max(int(round(required_output_frames / len(select_frame_indices))), 1)

    output_duration = len(select_frame_indices) * frame_multiplier / output_fps
    logging.info(f"VideoFrameSelector: sampling source at {best_source_sampling_fps:.3f} fps (visual cost {best_cost:.5f}): "
          f"{len(select_frame_indices)}/{total_n_frames} frames, multiplier {frame_multiplier}, "
          f"speedup {original_duration / output_duration:.3f}")

    return select_frame_indices, output_fps, frame_multiplier


class VideoFrameSelector:
    DESCRIPTION = "Picks the most evenly spaced subset of video frames for frame interpolation, and computes the interpolation multiplier that gives the requested output fps and speedup."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_frames": ("IMAGE", {"tooltip": "All loaded video frames."}),
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Video info from a VideoHelperSuite loader (source fps, frame count)."}),
                "output_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "tooltip": "Frame rate the final video will be played at."}),
                "target_video_speedup_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "tooltip": "Desired playback speed relative to the source (<1 = slow motion)."}),
                "min_source_sampling_fps": ("INT", {"default": 8, "min": 1, "max": 24, "tooltip": "Lowest fps to sample the source at. Sources below this keep all frames."}),
                "max_source_sampling_fps": ("INT", {"default": 12, "min": 1, "max": 24, "tooltip": "Highest fps to sample the source at."}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Keep at most this many selected frames (0 = no limit)."}),
                }
        }

    CATEGORY = "Eden 🌱/Video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT",)
    RETURN_NAMES = ("Selected_frames", "multiplier", "frame_rate",)
    OUTPUT_TOOLTIPS = ("Selected frames.", "Frame interpolation multiplier to use downstream.", "Output frame rate.")
    FUNCTION = "select_frames"

    def select_frames(self, input_frames, video_info, output_fps, target_video_speedup_factor, min_source_sampling_fps, max_source_sampling_fps, frame_load_cap):
        select_frame_indices, output_fps, frame_multiplier = compute_frame_parameters(
            video_info, target_video_speedup_factor, output_fps,
            source_sampling_fps_range=(min_source_sampling_fps, max_source_sampling_fps))

        selected_frames = input_frames[select_frame_indices]
        if frame_load_cap > 0:
            selected_frames = selected_frames[:frame_load_cap]

        return (selected_frames, frame_multiplier, output_fps,)
