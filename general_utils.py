import sys, os, time

def find_comfy_models_dir():
    current_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    while True:
        # Check if "ComfyUI/models" exists in the current path
        target_path = os.path.join(current_path, "ComfyUI", "models")
        if os.path.isdir(target_path):
            return os.path.abspath(target_path)

        # Move up one directory level
        new_path = os.path.dirname(current_path)
        if new_path == current_path:
            # If the new path is the same, we've reached the root and didn't find the directory
            break
        current_path = new_path

    return None