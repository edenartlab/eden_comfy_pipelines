<div align="center">
  <img src="assets/eden.png" alt="Eden.art Logo" width="300">
  <h1>Eden.art Custom Node Suite for ComfyUI</h1>
  <p>A collection of powerful custom nodes and workflows developed by <a href="https://www.eden.art/">Eden</a></p>
  <br>
</div>

This repository contains a comprehensive suite of specialized ComfyUI nodes designed to enhance our generative workflows. While some nodes may not be fully documented here yet, they are actively used in [our workflows repository](https://github.com/edenartlab/workflows) which powers many of the creative AI tools on **https://www.eden.art/**

## 🤝 Contributing

We welcome contributions! If you want to contribute custom ComfyUI workflows as tools into our creative AI agent ecosystem, please checkout our in-production workflows repository:
https://github.com/edenartlab/workflows

## 🌟 Featured Nodes

### GPT4 Node
<img src="assets/eden_gpt4_node.jpg" alt="GPT4 Node" width="70%">

**Call GPT4 for Text Completion**
- Versatile node that provides seamless integration with the OpenAI API
- Configuration: Place a `.env` file containing your API key in your root ComfyUI folder

---

### GPT4 Vision Node
<img src="assets/imagedescriptionnode.jpg" alt="GPT4 Vision Node" width="70%">

**Image Captioning and Visual Understanding**
- Interprets images and generates detailed, contextual descriptions
- Configuration: Requires an `.env` file with your OpenAI API key in the root ComfyUI folder

---

### Load Random Images
<img src="assets/loadrandomimage.jpg" alt="LoadRandomImage Node" width="70%">

**Dynamic Random Image Selection for Automated Workflows**
- Queue your prompt multiple times to process different inputs through the same workflow
- Automatically processes multiple images to maintain consistent aspect ratio and resolution

---

### Color Clustering Mask Generator
<img src="assets/maskfromrgb_kmeans.jpg" alt="maskfromrgb_kmeans Node" width="70%">

**Generate Precise Masks Using Color Clustering**
- Implements KMeans clustering algorithm to produce accurate segmentation masks
- Particularly effective for creating AnimateDiff masks directly from source videos

---

### DepthSlicer
<img src="assets/depthslicer.jpg" alt="DepthSlicer Node" width="70%">

**Create Targeted Masks from Depth Maps**
- Creates segmentation along the z-axis to produce specific "depth slices"
- Ideal for creative animations and precise inpainting applications

---

### 3D Parallax Zoom
<img src="assets/parallaxzoom.jpg" alt="Parallax Zoom Node" width="70%">

**Create Immersive Three-Dimensional Effects**
- Uses depth map and image data to generate Deforum-style 3D-zoom parallax videos
- Adds spatial dimension and dynamic movement to static images

---

### CLIP Interrogator
<img src="assets/CLIP_interrogator.png" alt="CLIP Interrogator Node" width="70%">

**Extract Detailed Text Descriptions from Images**
- Based on the [clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) project
- Key features:
  - Optional model persistence in GPU memory with `keep_model_alive` parameter
  - Configurable BLIP caption inclusion via `prepend_BLIP_caption` parameter
  - Automatic prompt saving with `save_prompt_to_txt_file` option
  - Installation note: If auto-download fails, manually clone https://huggingface.co/Salesforce/blip-image-captioning-large into ComfyUI/models/blip

---

### VAEDecode to Folder
<img src="assets/VAEDecode_to_folder.png" alt="VAE Decode to Folder Node" width="70%">

**Direct-to-Disk VAE Decoding**
- Saves decoded images directly to a specified directory
- Enables rendering of extended video sequences with AnimateDiff
- Note: Requires manual video compilation using ffmpeg afterward

---

### Enhanced SaveImage
<img src="assets/SaveImage.png" alt="Save Image Node" width="70%">

**Advanced Image Preservation Options**
- Add timestamps to filenames for better organization
- Automatically save the complete pipeline as a JSON file
- Easily reference prompts and settings without loading the entire pipeline

---

> **Note:** Some nodes in this collection are still under active development and may not be fully functional or documented.

## 🔧 Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/edenartlab/eden_comfy_pipelines.git
cd eden_comfy_pipelines
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
