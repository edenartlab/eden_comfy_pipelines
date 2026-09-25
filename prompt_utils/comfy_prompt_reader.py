"""Extract the positive / negative prompt from a ComfyUI-generated image.

ComfyUI embeds two JSON blobs in every image it saves:
  - "workflow": the editor graph (layout, Note nodes, reroutes, bypassed nodes)
  - "prompt":   the API graph that was actually executed: {node_id: {class_type, inputs}}

We only read "prompt": frontend-only nodes (Notes, Reroutes, Primitive widgets) are
already stripped or inlined there. Links are encoded as ["<node_id>", <output_slot>].

Strategy: find the samplers/guiders, walk their `positive` (resp. `negative`)
conditioning upstream until a text-encoder node (anything with a `clip` input), then
resolve that encoder's text input, following links through primitive / concat nodes.
Node types are recognised by input names rather than class names, so custom nodes
work as long as they use the conventional input names. When the graph walk finds
nothing we fall back to the longest prompt-like string in the executed graph, and to
A1111-style "parameters" text for images saved by other tools.

No ComfyUI imports: this module is usable (and testable) outside ComfyUI.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from PIL import Image

MAX_DEPTH = 64

# Input names that never carry prompt text, even when they hold a string.
NON_TEXT_KEYS = {
    "delimiter", "separator", "join_with", "clean_whitespace", "mode", "method",
    "sampler_name", "scheduler", "type", "weight_interpretation", "token_normalization",
    "llama_template", "template", "filename_prefix", "format", "device", "dtype",
}
FILE_LIKE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx|png|jpe?g|webp|json|txt|yaml)$", re.I)
CONCAT_HINT = re.compile(r"concat|join|combine|append|merge", re.I)
DELIMITER_KEYS = ("delimiter", "separator", "join_with")

# EXIF tags ComfyUI's WEBP/JPEG savers use to stash "prompt:{...}" / "workflow:{...}".
EXIF_TAGS = (0x0110, 0x010F, 0x010E, 0x9286)


@dataclass
class ExtractedPrompt:
    positive: str
    negative: str
    source: str  # "graph" | "graph-fallback" | "a1111" — how the prompt was found


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


def extract_prompt(path: str) -> ExtractedPrompt | None:
    """Extract the prompt from an image file; None when it has no usable metadata."""
    meta = read_metadata(path)
    if "prompt" in meta:
        try:
            graph = json.loads(meta["prompt"])
        except (json.JSONDecodeError, TypeError):
            graph = None
        if isinstance(graph, dict):
            result = extract_from_graph(graph)
            if result is not None:
                return result
    if "parameters" in meta:
        return extract_from_a1111(meta["parameters"])
    return None


# ---------------------------------------------------------------- graph walking

class _Graph:
    def __init__(self, nodes: dict):
        self.nodes = {str(k): v for k, v in nodes.items() if isinstance(v, dict)}

    def inputs(self, node_id: str) -> dict:
        inputs = self.nodes.get(node_id, {}).get("inputs", {})
        return inputs if isinstance(inputs, dict) else {}

    def class_type(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("class_type", ""))

    def link(self, value) -> str | None:
        """Return the source node id if `value` is a link, else None."""
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
            src = str(value[0])
            if src in self.nodes:
                return src
        return None

    def is_encoder(self, node_id: str) -> bool:
        inputs = self.inputs(node_id)
        return "clip" in inputs or "TextEncode" in self.class_type(node_id)

    # -- strings ------------------------------------------------------------

    def resolve_string(self, value, depth: int = 0) -> str | None:
        """Resolve a literal or linked input to the text it carries."""
        if isinstance(value, str):
            return value
        src = self.link(value)
        if src is None or depth > MAX_DEPTH:
            return None
        return self.node_text(src, depth + 1)

    def node_text(self, node_id: str, depth: int) -> str | None:
        """Best guess at the string a node outputs (primitive, concat, or passthrough)."""
        inputs = self.inputs(node_id)
        texts = [
            (key, text) for key, value in inputs.items()
            if key not in NON_TEXT_KEYS
            and (text := self.resolve_string(value, depth)) is not None
            and _looks_like_text(text)
        ]
        if not texts:
            return None
        if CONCAT_HINT.search(self.class_type(node_id)) or any(k in inputs for k in DELIMITER_KEYS):
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

    def sampler_roots(self) -> list[str]:
        """Nodes where conditioning is consumed: samplers and guiders."""
        with_cond = [
            nid for nid in self.nodes
            if any(k in ("positive", "cond1") and self.link(v) for k, v in self.inputs(nid).items())
        ]
        # BasicGuider-style nodes take a single `conditioning` input.
        guiders = [
            nid for nid in self.nodes
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


def _looks_like_text(text: str) -> bool:
    text = text.strip()
    return bool(text) and not FILE_LIKE.search(text)


def _pick(texts: list[str]) -> str:
    """Most common text across samplers (hires-fix / refiner repeat it); ties → longest."""
    if not texts:
        return ""
    counts = Counter(texts)
    return max(counts, key=lambda t: (counts[t], len(t)))


def extract_from_graph(nodes: dict) -> ExtractedPrompt | None:
    graph = _Graph(nodes)
    positives, negatives = [], []
    for root in graph.sampler_roots():
        pos = [t for e in graph.encoders_upstream(root, "positive") if (t := graph.encoder_text(e))]
        neg = [t for e in graph.encoders_upstream(root, "negative") if (t := graph.encoder_text(e))]
        if pos:
            positives.append(max(pos, key=len))  # regional/combined prompts: keep the main one
        if neg:
            negatives.append(max(neg, key=len))
    if positives:
        return ExtractedPrompt(_pick(positives).strip(), _pick(negatives).strip(), "graph")

    # Fallback: any text encoder not on a negative branch, else the longest string input.
    negative_encoders = {
        e for root in graph.sampler_roots() for e in graph.encoders_upstream(root, "negative")
    }
    candidates = [
        t for nid in graph.nodes
        if graph.is_encoder(nid) and nid not in negative_encoders and (t := graph.encoder_text(nid))
    ]
    if not candidates:
        candidates = [
            v for nid in graph.nodes for k, v in graph.inputs(nid).items()
            if isinstance(v, str) and k not in NON_TEXT_KEYS and _looks_like_text(v) and " " in v.strip()
        ]
    if candidates:
        return ExtractedPrompt(max(candidates, key=len).strip(), "", "graph-fallback")
    return None


# ---------------------------------------------------------------- A1111-style metadata

def extract_from_a1111(parameters: str) -> ExtractedPrompt | None:
    """Parse "prompt\\nNegative prompt: ...\\nSteps: ..." text (A1111, Forge, many savers)."""
    body = re.split(r"\n(?=Steps: )", parameters, maxsplit=1)[0]
    positive, _, negative = body.partition("Negative prompt:")
    positive = positive.strip()
    return ExtractedPrompt(positive, negative.strip(), "a1111") if positive else None
