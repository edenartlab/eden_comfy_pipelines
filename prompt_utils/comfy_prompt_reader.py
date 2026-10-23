"""Extract the positive / negative prompt from a ComfyUI-generated image.

ComfyUI embeds two JSON blobs in every image it saves:
  - "workflow": the editor graph (layout, Note nodes, reroutes, bypassed nodes)
  - "prompt":   the API graph that was actually executed: {node_id: {class_type, inputs}}

We only read "prompt": frontend-only nodes (Notes, Reroutes, Primitive widgets) are
already stripped or inlined there. Links are encoded as ["<node_id>", <output_slot>].

Strategy: keep only the nodes ComfyUI actually executed (everything upstream of an
output node such as SaveImage), find the samplers/guiders among them, walk their `positive` (resp. `negative`)
conditioning upstream until a text-encoder node (anything with a `clip` input), then
resolve that encoder's text input, following links through primitive / concat nodes.
Node types are recognised by input names rather than class names, so custom nodes
work as long as they use the conventional input names. A returned prompt is never a
file path and always at least MIN_PROMPT_LENGTH characters; when the graph walk finds
no such prompt the image counts as having none. Images saved by other tools fall back
to A1111-style "parameters" text.

No ComfyUI imports: this module is usable (and testable) outside ComfyUI.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from PIL import Image

MAX_DEPTH = 64
MIN_PROMPT_LENGTH = 50

# Input names that never carry prompt text, even when they hold a string.
NON_TEXT_KEYS = {
    "delimiter", "separator", "join_with", "clean_whitespace", "mode", "method",
    "sampler_name", "scheduler", "type", "weight_interpretation", "token_normalization",
    "llama_template", "template", "filename_prefix", "format", "device", "dtype", "weight_dtype",
    "clip", "model", "vae", "system_prompt", "system_message", "system", "instruction", "instructions",
}
# A trailing " [output]" / " [input]" is how LoadImage-style nodes annotate their files.
FILE_LIKE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx|png|jpe?g|webp|gif|mp4|webm|mov|wav|mp3|flac"
                       r"|json|txt|csv|yaml|npy|pkl)(\s*\[\w+\])?$", re.I)
# Absolute / relative / home / UNC paths and URLs.
PATH_START = re.compile(r"^([a-z]:[\\/]|\\\\|~?\.{0,2}/|[a-z][\w+.-]*://)", re.I)
# Output-node guess by class name, used when ComfyUI doesn't know the class (not installed).
OUTPUT_HINT = re.compile(r"save|preview|output", re.I)
CONCAT_HINT = re.compile(r"concat|join|combine|append|merge", re.I)
# LLM / captioner nodes: their output is never stored in the graph, so we read their input instead.
GENERATOR_HINT = re.compile(r"generat|llm|ollama|gpt|gemini|caption|florence", re.I)
GENERATOR_KEYS = {"max_length", "max_new_tokens", "max_tokens"}
# Nodes (by title or class) holding an LLM system prompt rather than the user's prompt.
SYSTEM_HINT = re.compile(r"system|instruction", re.I)
DELIMITER_KEYS = ("delimiter", "separator", "join_with")

# EXIF tags ComfyUI's WEBP/JPEG savers use to stash "prompt:{...}" / "workflow:{...}".
EXIF_TAGS = (0x0110, 0x010F, 0x010E, 0x9286)


@dataclass
class ExtractedPrompt:
    positive: str
    negative: str
    source: str  # "graph" | "a1111" — how the prompt was found


# ---------------------------------------------------------------- metadata reading

def read_metadata(path: str) -> dict[str, str]:
    """Return the text metadata of an image (PNG text chunks, or ComfyUI EXIF tags)."""
    with Image.open(path) as img:
        meta = {k: v for k, v in img.info.items() if isinstance(v, str)}
        if "prompt" not in meta:
            try:
                exif = img.getexif()
            except Exception:
                exif = {}
            for tag in EXIF_TAGS:
                value = exif.get(tag)
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "ignore")
                if isinstance(value, str) and ":" in value:
                    key, _, payload = value.partition(":")
                    if key in ("prompt", "workflow") and key not in meta:
                        meta[key] = payload
    return meta


def default_is_output_node(class_type: str) -> bool:
    return bool(OUTPUT_HINT.search(class_type))


def extract_prompt(path: str, is_output_node=default_is_output_node) -> ExtractedPrompt | None:
    """Extract the prompt from an image file; None when it has no usable metadata.

    is_output_node(class_type) says which nodes ComfyUI runs as outputs (SaveImage, ...).
    """
    meta = read_metadata(path)
    if "prompt" in meta:
        try:
            graph = json.loads(meta["prompt"])
        except (json.JSONDecodeError, TypeError):
            graph = None
        if isinstance(graph, dict):
            result = extract_from_graph(graph, is_output_node)
            if result is not None:
                return result
    if "parameters" in meta:
        return extract_from_a1111(meta["parameters"])
    return None


def explain_missing_prompt(path: str, is_output_node=default_is_output_node) -> str:
    """Why extract_prompt() returned None for this file, for log messages."""
    meta = read_metadata(path)
    if "prompt" not in meta and "parameters" not in meta:
        return f"no ComfyUI 'prompt' or A1111 'parameters' metadata (metadata keys: {', '.join(meta) or 'none'})"
    reasons = []
    if "prompt" in meta:
        try:
            graph = json.loads(meta["prompt"])
        except (json.JSONDecodeError, TypeError):
            graph = None
        if not isinstance(graph, dict):
            reasons.append("'prompt' metadata is not a JSON graph")
        else:
            g = _Graph(graph)
            active = g.active_nodes(is_output_node)
            roots = g.sampler_roots(active)
            encoders = [e for root in roots for e in g.encoders_upstream(root, "positive")]
            texts = [t for e in encoders if (t := g.encoder_text(e))]
            if not active:
                classes = sorted({g.class_type(nid) for nid in g.nodes})
                reasons.append(f"graph has no output node (Save/Preview) among its {len(classes)} node types: {', '.join(classes)}")
            elif not roots:
                reasons.append(f"no sampler/guider upstream of an output node ({len(active)} executed nodes)")
            elif not encoders:
                reasons.append(f"sampler(s) {', '.join(g.class_type(r) for r in roots)} have no text encoder on their positive input")
            elif not texts:
                reasons.append(f"positive text encoder(s) {', '.join(g.class_type(e) for e in encoders)} have no resolvable "
                               "text (fed from a file, LLM or unsupported custom node?)")
            else:
                found = "; ".join(repr(t[:100]) for t in texts)
                reasons.append(f"positive encoder text rejected (under {MIN_PROMPT_LENGTH} chars or a file path): {found}")
    if "parameters" in meta:
        reasons.append(f"A1111 'parameters' has no positive prompt of {MIN_PROMPT_LENGTH}+ chars")
    return "; ".join(reasons)


# ---------------------------------------------------------------- graph walking

class _Graph:
    def __init__(self, nodes: dict):
        self.nodes = {str(k): v for k, v in nodes.items() if isinstance(v, dict)}

    def inputs(self, node_id: str) -> dict:
        inputs = self.nodes.get(node_id, {}).get("inputs", {})
        return inputs if isinstance(inputs, dict) else {}

    def class_type(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("class_type", ""))

    def is_system_prompt(self, node_id: str) -> bool:
        meta = self.nodes[node_id].get("_meta")
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        return bool(SYSTEM_HINT.search(f"{title} {self.class_type(node_id)}"))

    def link(self, value) -> str | None:
        """Return the source node id if `value` is a link, else None."""
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
            src = str(value[0])
            if src in self.nodes:
                return src
        return None

    def is_encoder(self, node_id: str) -> bool:
        inputs = self.inputs(node_id)
        return ("clip" in inputs or "TextEncode" in self.class_type(node_id)) and not self.is_generator(node_id)

    def is_generator(self, node_id: str) -> bool:
        return bool(GENERATOR_HINT.search(self.class_type(node_id)) or GENERATOR_KEYS & self.inputs(node_id).keys())

    def active_nodes(self, is_output_node) -> set[str]:
        """Nodes ComfyUI executed: output nodes and everything upstream of them.

        The saved graph also holds nodes that were never run (dangling branches),
        whose text must not be mistaken for the prompt.
        """
        active: set[str] = set()
        stack = [nid for nid in self.nodes if is_output_node(self.class_type(nid))]
        while stack:
            nid = stack.pop()
            if nid not in active:
                active.add(nid)
                stack.extend(src for value in self.inputs(nid).values() if (src := self.link(value)))
        return active

    # -- strings ------------------------------------------------------------

    def resolve_string(self, value, depth: int = 0) -> str | None:
        """Resolve a literal or linked input to the text it carries."""
        if isinstance(value, str):
            return value
        src = self.link(value)
        if src is None or depth > MAX_DEPTH or self.is_system_prompt(src):
            return None
        return self.node_text(src, depth + 1)

    def resolve_bool(self, value, depth: int = 0) -> bool | None:
        """Resolve a literal or linked (primitive) boolean input."""
        if isinstance(value, bool):
            return value
        src = self.link(value)
        if src is None or depth > MAX_DEPTH:
            return None
        return self.resolve_bool(self.inputs(src).get("value"), depth + 1)

    def node_text(self, node_id: str, depth: int) -> str | None:
        """Best guess at the string a node outputs (primitive, concat, switch, or passthrough).

        LLM prompt enhancers count as passthrough: we return the user's pre-enhancement prompt.
        """
        inputs = self.inputs(node_id)
        # If/else switches: follow only the branch that was actually taken.
        if "on_true" in inputs and "on_false" in inputs:
            switch = self.resolve_bool(inputs.get("switch"))
            if switch is not None:
                return self.resolve_string(inputs["on_true" if switch else "on_false"], depth)
        concat = CONCAT_HINT.search(self.class_type(node_id)) or any(k in inputs for k in DELIMITER_KEYS)
        # A node fed a file/folder path (prompt-from-file, random line picker, this very node...)
        # outputs text read from disk, which the graph doesn't contain.
        if not concat and any(isinstance(v, str) and _is_path(v) for k, v in inputs.items() if k not in NON_TEXT_KEYS):
            return None
        texts = [
            (key, text) for key, value in inputs.items()
            if key not in NON_TEXT_KEYS
            and (text := self.resolve_string(value, depth)) is not None
            and _looks_like_text(text)
        ]
        if not texts:
            return None
        if concat:
            delimiter = next((inputs[k] for k in DELIMITER_KEYS if isinstance(inputs.get(k), str)), "")
            delimiter = delimiter.replace("\\n", "\n")
            return delimiter.join(text for _, text in texts if text)
        return max((text for _, text in texts), key=len)

    def encoder_text(self, node_id: str) -> str | None:
        """The prompt fed into a text-encoder node (longest of its text inputs)."""
        texts = [
            text for key, value in self.inputs(node_id).items()
            if key not in NON_TEXT_KEYS
            and (text := self.resolve_string(value)) is not None
            and _looks_like_text(text)
        ]
        if not texts:
            return None
        # Lumina 2 prepends a system prompt; the user prompt follows this marker.
        return max(texts, key=len).split("<Prompt Start>")[-1]

    # -- conditioning -------------------------------------------------------

    def encoders_upstream(self, node_id: str, polarity: str) -> list[str]:
        """Text-encoder node ids feeding the `positive`/`negative` side of node_id."""
        found: list[str] = []
        seen: set[str] = set()

        def walk(nid: str, side: str, depth: int):
            if (nid, side) in seen or depth > MAX_DEPTH:
                return
            seen.add((nid, side))
            if "ZeroOut" in self.class_type(nid):  # e.g. Flux "negative" = zeroed positive
                return
            if self.is_encoder(nid):
                found.append(nid)
                return
            inputs = self.inputs(nid)
            # Only nodes with an explicit negative input split into sides; others (ZeroOut,
            # Combine, SetArea, FluxGuidance...) just pass their conditioning through.
            dual = any("neg" in key.lower() for key in inputs)
            if side == "negative" and not dual and depth == 0:  # e.g. BasicGuider: no negative at all
                return
            for key, value in inputs.items():
                src = self.link(value)
                if src is None or not _is_conditioning_key(key, side if dual else "positive"):
                    continue
                # Nodes like ControlNetApplyAdvanced carry both sides: slot 0 = positive, 1 = negative.
                src_inputs = self.inputs(src)
                both = "positive" in src_inputs and "negative" in src_inputs
                walk(src, "negative" if both and value[1] == 1 else "positive", depth + 1)

        walk(node_id, polarity, 0)
        return found

    def sampler_roots(self, active: set[str]) -> list[str]:
        """Executed nodes where conditioning is consumed: samplers and guiders."""
        with_cond = [
            nid for nid in active
            if any(k in ("positive", "cond1") and self.link(v) for k, v in self.inputs(nid).items())
        ]
        # BasicGuider-style nodes take a single `conditioning` input.
        guiders = [
            nid for nid in active
            if "Guider" in self.class_type(nid) and self.link(self.inputs(nid).get("conditioning"))
        ]
        roots = [nid for nid in with_cond + guiders
                 if re.search(r"sampl|guider", self.class_type(nid), re.I)]
        return roots or with_cond + guiders


def _is_conditioning_key(key: str, polarity: str) -> bool:
    key = key.lower()
    if polarity == "negative":
        return "neg" in key
    if "neg" in key or key == "cond2":  # DualCFGGuider: cond1 is the prompt, cond2 an auxiliary cond
        return False
    return "cond" in key or "positive" in key or key == "guider"


def _is_path(text: str) -> bool:
    text = text.strip()
    return bool(PATH_START.match(text) or FILE_LIKE.search(text)
                or (re.search(r"[\\/]", text) and not re.search(r"\s", text)))


def _looks_like_text(text: str) -> bool:
    return bool(text.strip()) and not _is_path(text)


def _is_prompt(text: str) -> bool:
    return len(text.strip()) >= MIN_PROMPT_LENGTH and not _is_path(text)


def _pick(texts: list[str]) -> str:
    """Most common text across samplers (hires-fix / refiner repeat it); ties → longest."""
    if not texts:
        return ""
    counts = Counter(texts)
    return max(counts, key=lambda t: (counts[t], len(t)))


def extract_from_graph(nodes: dict, is_output_node=default_is_output_node) -> ExtractedPrompt | None:
    graph = _Graph(nodes)
    positives, negatives = [], []
    for root in graph.sampler_roots(graph.active_nodes(is_output_node)):
        pos = [t for e in graph.encoders_upstream(root, "positive") if (t := graph.encoder_text(e)) and _is_prompt(t)]
        neg = [t for e in graph.encoders_upstream(root, "negative") if (t := graph.encoder_text(e))]
        if pos:
            positives.append(max(pos, key=len))  # regional/combined prompts: keep the main one
        if neg:
            negatives.append(max(neg, key=len))
    if positives:
        return ExtractedPrompt(_pick(positives).strip(), _pick(negatives).strip(), "graph")
    return None


# ---------------------------------------------------------------- A1111-style metadata

def extract_from_a1111(parameters: str) -> ExtractedPrompt | None:
    """Parse "prompt\\nNegative prompt: ...\\nSteps: ..." text (A1111, Forge, many savers)."""
    body = re.split(r"\n(?=Steps: )", parameters, maxsplit=1)[0]
    positive, _, negative = body.partition("Negative prompt:")
    positive = positive.strip()
    return ExtractedPrompt(positive, negative.strip(), "a1111") if _is_prompt(positive) else None
