"""Eden_PromptFromImageFolder: pull the prompt out of a ComfyUI image picked from a folder."""

from __future__ import annotations

import os
import re

from comfy_api.latest import io

from .comfy_prompt_reader import ExtractedPrompt, extract_prompt

# path -> (mtime_ns, size, result); re-parsed only when the file changes on disk.
_PROMPT_CACHE: dict[str, tuple[int, int, ExtractedPrompt | None]] = {}

ORDER_OPTIONS = ["name", "date modified (oldest first)", "date modified (newest first)"]


def _natural_key(path: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path)]


def _parse_extensions(extensions: str) -> tuple[str, ...]:
    exts = [e.strip().lower() for e in re.split(r"[,\s;]+", extensions) if e.strip()]
    return tuple(e if e.startswith(".") else f".{e}" for e in exts) or (".png",)


def _list_files(folder: str, include_subfolders: bool, extensions: str,
                name_filter: str, order: str) -> list[tuple[str, int, int]]:
    """Matching files as (path, mtime_ns, size), sorted by `order`."""
    exts = _parse_extensions(extensions)
    needle = name_filter.strip().lower()
    files = []

    def scan(directory: str):
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if include_subfolders and not entry.name.startswith("."):
                    scan(entry.path)
            elif entry.name.lower().endswith(exts) and (not needle or needle in entry.name.lower()):
                stat = entry.stat()
                files.append((entry.path, stat.st_mtime_ns, stat.st_size))

    scan(folder)
    if order == ORDER_OPTIONS[0]:
        files.sort(key=lambda f: _natural_key(os.path.relpath(f[0], folder)))
    else:
        files.sort(key=lambda f: (f[1], f[0]), reverse=order == ORDER_OPTIONS[2])
    return files


def _cached_prompt(path: str, mtime_ns: int, size: int) -> ExtractedPrompt | None:
    cached = _PROMPT_CACHE.get(path)
    if cached and cached[:2] == (mtime_ns, size):
        return cached[2]
    try:
        result = extract_prompt(path)
    except Exception:  # unreadable / truncated file: treat as "no prompt"
        result = None
    _PROMPT_CACHE[path] = (mtime_ns, size, result)
    return result


class Eden_PromptFromImageFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Eden_PromptFromImageFolder",
            display_name="Prompt from Image Folder 🎲",
            category="Eden 🌱/Text",
            description=(
                "Picks an image from a folder of ComfyUI renders and returns the prompt that "
                "generated it, found by tracing the embedded graph from the sampler back to its "
                "text encoder (Note nodes and negative prompts are ignored).\n\n"
                "Use the control widget under `seed` to choose the behaviour:\n"
                "• randomize: a random image every run\n"
                "• increment: walk the folder one image at a time\n"
                "• fixed: repeat the current pick"
            ),
            search_aliases=["prompt from image", "png prompt", "extract prompt", "metadata prompt",
                            "random prompt folder", "prompt reader"],
            inputs=[
                io.String.Input("folder", default="", placeholder="/path/to/comfyui/output",
                                tooltip="Folder of ComfyUI-generated images."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=io.ControlAfterGenerate.randomize,
                             tooltip="Picks file number seed % file_count. randomize = random image "
                                     "each run, increment = next image each run, fixed = repeat."),
                io.Boolean.Input("include_subfolders", default=False),
                io.String.Input("extensions", default=".png",
                                tooltip="Comma-separated, e.g. '.png, .webp'."),
                io.Combo.Input("order", options=ORDER_OPTIONS, default=ORDER_OPTIONS[0],
                               tooltip="File order used by increment mode (and by the seed → file mapping)."),
                io.String.Input("name_filter", default="", optional=True, advanced=True,
                                tooltip="Only use files whose name contains this text (case-insensitive)."),
                io.Boolean.Input("skip_files_without_prompt", default=True, optional=True, advanced=True,
                                 tooltip="If the picked file has no readable prompt, move on to the next "
                                         "file instead of raising an error."),
                io.String.Input("fallback_prompt", default="", multiline=True, optional=True, advanced=True,
                                tooltip="Returned when no file in the folder has a prompt. "
                                        "Leave empty to raise an error instead."),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("negative_prompt"),
                io.String.Output("filepath"),
                io.Int.Output("index", tooltip="Position of the picked file in the sorted list."),
                io.Int.Output("file_count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder, seed, include_subfolders, extensions, order,
                           name_filter="", **_):
        # Re-run when the seed or the folder contents change; a fixed seed on an
        # unchanged folder is served from ComfyUI's cache.
        files = _list_files(os.path.expanduser(folder.strip()), include_subfolders, extensions, name_filter, order)
        return seed, hash(tuple(files))

    @classmethod
    def execute(cls, folder, seed, include_subfolders, extensions, order, name_filter="",
                skip_files_without_prompt=True, fallback_prompt="") -> io.NodeOutput:
        folder = os.path.expanduser(folder.strip())
        if not os.path.isdir(folder):
            raise ValueError(f"Prompt from Image Folder: folder not found: {folder!r}")

        files = _list_files(folder, include_subfolders, extensions, name_filter, order)
        if not files:
            hint = "" if include_subfolders else " (include_subfolders is off)"
            raise ValueError(f"Prompt from Image Folder: no {extensions} files in {folder!r}{hint}")

        start = seed % len(files)
        attempts = len(files) if skip_files_without_prompt else 1
        for offset in range(attempts):
            index = (start + offset) % len(files)
            path, mtime_ns, size = files[index]
            result = _cached_prompt(path, mtime_ns, size)
            if result and result.positive:
                return io.NodeOutput(result.positive, result.negative, path, index, len(files))

        if fallback_prompt:
            return io.NodeOutput(fallback_prompt, "", "", start, len(files))
        if skip_files_without_prompt:
            raise ValueError(f"Prompt from Image Folder: none of the {len(files)} files in "
                             f"{folder!r} contain a ComfyUI prompt")
        raise ValueError(f"Prompt from Image Folder: no ComfyUI prompt found in {files[start][0]!r}")
