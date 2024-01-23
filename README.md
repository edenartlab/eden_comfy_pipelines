# eden_comfy_pipelines
A collection of custom nodes and workflows for ComfyUI

# Current nodes:
- CLIP_interrogator node (based off [clip_interrogator](https://github.com/pharmapsychotic/clip-interrogator)) to automatically get a prompt from an image
- VAEDecode_to_folder node (Decodes VAE latents to imgs, but saves them directly to a folder instead of keeping them in memory, allowing you to render much longer videos w eg AnimateDiff) (you will have to ffmpeg the video manually in post)

NOTE:
some of the included nodes arent finished yet!
