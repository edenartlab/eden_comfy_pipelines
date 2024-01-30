# eden_comfy_pipelines
A collection of custom nodes and workflows for ComfyUI

# Current nodes:
- **CLIP_interrogator** node (based off [clip_interrogator](https://github.com/pharmapsychotic/clip-interrogator)).
  This is a simple CLIP_interrogator node that has a few handy options:
    - "keep_model_alive" will not remove the CLIP/BLIP models from the gpu after the node is executed, this avoids having to reload the entire model every time you run a new pipeline (but will use more gpu-memory)
    - "prepend_BLIP_caption" can be turned off to only get the matching modifier tags but not use a BLIP-interrogation. This can be useful if you're using an image with IP_adapter and are mainly looking to copy textures, but not global image contents.
    - "save_prompt_to_txt_file" specify a path where the prompt is saved to disk
- **VAEDecode_to_folder** node (Decodes VAE latents to imgs, but saves them directly to a folder instead of keeping them in memory, allowing you to render much longer videos w eg AnimateDiff) (you will have to ffmpeg the video manually in post)
- **SaveImage**: a basic Image saver with the option to add timestamps and the option to also save the entire pipeline as a .json (so you can read prompts and settings directly from that .json file without loading the entire pipe)

NOTE:
some of the included nodes arent finished yet!
