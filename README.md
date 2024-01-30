<h1>eden_comfy_pipelines</h1>
<p>A collection of custom nodes and workflows for ComfyUI</p>

<h2>Current nodes:</h2>

<h3>CLIP_interrogator node</h3>
<p>Based off <a href="https://github.com/pharmapsychotic/clip-interrogator">clip_interrogator</a>.</p>
<img src="assets/CLIP_interrogator.png" alt="CLIP Interrogator Node Image" style="display: block; margin: auto; width: 50%;">
<p>This is a simple CLIP_interrogator node that has a few handy options:</p>
<ul>
    <li>"keep_model_alive" will not remove the CLIP/BLIP models from the GPU after the node is executed, avoiding the need to reload the entire model every time you run a new pipeline (but will use more GPU memory).</li>
    <li>"prepend_BLIP_caption" can be turned off to only get the matching modifier tags but not use a BLIP-interrogation. Useful if you're using an image with IP_adapter and are mainly looking to copy textures, but not global image contents.</li>
    <li>"save_prompt_to_txt_file" to specify a path where the prompt is saved to disk.</li>
</ul>

<h3>VAEDecode_to_folder node</h3>
<img src="assets/VAEDecode_to_folder.png" alt="VAE Decode to Folder Node Image" style="display: block; margin: auto; width: 50%;">
<p>Decodes VAE latents to imgs, but saves them directly to a folder. This allows rendering much longer videos with, for example, AnimateDiff (manual video compilation with ffmpeg required in post).</p>

<h3>SaveImage node</h3>
<img src="assets/SaveImage.png" alt="Save Image Node Image" style="display: block; margin: auto; width: 50%;">
<p>A basic Image saver with the option to add timestamps and to also save the entire pipeline as a .json file (so you can read prompts and settings directly from that .json file without loading the entire pipe).</p>

<p><strong>NOTE:</strong> Some of the included nodes aren't finished yet!</p>

