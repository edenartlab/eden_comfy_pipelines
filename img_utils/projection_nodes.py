# -----------------------------------------------------------------------------
# Eden 🌱 Projection — Simple, Robust Baseline (MVP fixed & optimized)
# -----------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

def save_debug_image(outdir, name, tensor):
    arr = (linear_to_srgb(tensor).cpu().numpy()[0] * 255).astype(np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outdir, name), rgb)

# =============================
# sRGB <-> linear
# =============================
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)).pow(2.4))

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1 / 2.4) - a
    ).clamp(0.0, 1.0)

def tone_map_log(x: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
    """
    Logarithmic tone mapping to approximate human visual adaptation.
    Input: linear-light tensor (can exceed 1.0).
    Output: sRGB-like [0,1].
    gamma: adaptation strength (higher = more compression).
    """
    return (torch.log1p(gamma * x) / math.log1p(gamma)).clamp(0.0, 1.0)

# =============================
# Helpers
# =============================
def _ensure_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    return x.clamp(0.0, 1.0)

def _match_spatial(tgt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, Hr, Wr, _ = ref.shape
    B, Ht, Wt, C = tgt.shape
    if (Ht, Wt) == (Hr, Wr):
        return tgt
    t = tgt.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(Hr, Wr), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1)

def _luma(linear_rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=linear_rgb.device, dtype=linear_rgb.dtype).view(1, 1, 1, 3)
    return (linear_rgb * w).sum(dim=-1, keepdim=True)

def _downsample_for_stats(t: torch.Tensor, max_samples: int = 750_000) -> torch.Tensor:
    B, H, W, C = t.shape
    n = B * H * W
    if n <= max_samples:
        return t
    stride = int(math.ceil((n / max_samples) ** 0.5))
    stride = max(1, stride)
    return t[:, ::stride, ::stride, :]

def _safe_quantile(v: torch.Tensor, q: float, max_samples: int = 200_000) -> torch.Tensor:
    v = v.reshape(-1)
    if v.numel() > max_samples:
        stride = v.numel() // max_samples
        v = v[::stride]
    return torch.quantile(v.to("cpu", non_blocking=True), q).to(v.device)

# =============================
# Robust white estimation
# =============================
def _robust_white_per_channel(S_lin, max_samples=500_000):
    L = _luma(S_lin).reshape(-1)
    S = S_lin.reshape(-1, 3)

    # sample down
    if S.shape[0] > max_samples:
        stride = S.shape[0] // max_samples
        S = S[::stride]
        L = L[::stride]

    # sort by luminance
    idx = torch.argsort(L, descending=True)
    top = S[idx[: int(0.05 * len(idx))]]  # top 5%

    med = torch.median(top, dim=0).values
    return med.clamp(1e-3, 0.7).view(1,1,1,3)  # cap reflectance white at 70%


# =============================
# Low-frequency reflectance
# =============================
def _lowfreq_reflectance_scalar(S_lin: torch.Tensor,
                                white_luma: float,
                                r_min: float,
                                kernel_frac: float = 1 / 24):
    L = _luma(S_lin)
    denom = max(white_luma, 1e-6)
    R = (L / denom).clamp(r_min, 1.0)

    B, H, W, _ = R.shape
    k = max(3, int(round(max(H, W) * kernel_frac)) | 1)
    Rn = R.permute(0, 3, 1, 2)
    Rn = F.avg_pool2d(Rn, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
    return Rn.permute(0, 2, 3, 1)

# =============================
# Core simplified compensation
# =============================
def save_debug_image(outdir, name, tensor):
    arr = (linear_to_srgb(tensor).cpu().numpy()[0] * 255).astype(np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outdir, name), rgb)

def surface_compensate(
    target_srgb: torch.Tensor,
    surface_srgb: torch.Tensor,
    projector_gain: float = 2.0,
    reflectance_floor_percent: float = 5.0,
    color_compensation: bool = True,
    gamma: float = 4.0,   # tone-mapping strength
    eps: float = 1e-6,
    debug_outdir: str = None,
):
    """
    Radiometric surface compensation with perceptual tone mapping.
    No alpha compression, instead uses log tone-mapping to approximate adaptation.
    """

    import os, cv2, numpy as np

    # --- prep
    T = _ensure_bhwc(target_srgb)
    S = _ensure_bhwc(surface_srgb)
    if S.shape[0] == 1 and T.shape[0] > 1:
        S = S.expand(T.shape[0], -1, -1, -1).contiguous()
    S = _match_spatial(S, T)

    T_lin = srgb_to_linear(T)
    S_lin = srgb_to_linear(S)

    # --- optional color compensation (simple white balance)
    if color_compensation:
        W = _robust_white_per_channel(S_lin)
        W_mean = W.mean(dim=-1, keepdim=True)
        gain_c = (W_mean / (W + eps)).clamp(0.7, 1.3)
        T_lin = (T_lin * gain_c).clamp(0.0, 1.0)

    # --- reflectance floor to avoid divide-by-zero
    r_min = max(0.0, min(1.0, reflectance_floor_percent / 100.0))
    white_luma = _luma(srgb_to_linear(torch.ones_like(S))).mean().item()
    R_s = _lowfreq_reflectance_scalar(S_lin, white_luma=white_luma, r_min=r_min)

    # --- projector command (linear)
    k = float(projector_gain)
    denom = (k * R_s + eps)
    P_lin = ((T_lin - S_lin) / denom).clamp(0.0, 1.0)

    # --- physical predicted view (unbounded linear light)
    V_lin = S_lin + k * R_s * P_lin

    # --- perceptual tone-mapped preview
    V_tone = (torch.log1p(gamma * V_lin) / math.log1p(gamma)).clamp(0.0, 1.0)

    # --- debug saves
    if debug_outdir:
        def save_dbg(name, X, linear=True):
            if linear:
                arr = (linear_to_srgb(X).cpu().numpy()[0] * 255).astype(np.uint8)
            else:
                arr = (X.cpu().numpy()[0] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_outdir, name),
                        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        save_dbg("dbg_tone_target.png", T_lin)
        save_dbg("dbg_tone_surface.png", S_lin)
        save_dbg("dbg_tone_proj.png", P_lin)
        save_dbg("dbg_tone_linearview.png", V_lin)
        save_dbg("dbg_tone_preview.png", V_tone, linear=False)

    return linear_to_srgb(P_lin), V_tone, f"tone-mapped γ={gamma}"

# =============================
# ComfyUI Nodes
# =============================
class ProjectionPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "surface_photo": ("IMAGE",),
                "projection_output": ("IMAGE",),
                "beamer_gain": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "white_band_low_pct": ("FLOAT", {"default": 90.0, "min": 70.0, "max": 99.0, "step": 0.5}),
                "white_pct": ("FLOAT", {"default": 99.0, "min": 90.0, "max": 99.9, "step": 0.1}),
                "reflectance_floor_pct": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Perceived View",)
    FUNCTION = "render"
    CATEGORY = "Eden 🌱/projection"

    def render(self, surface_photo, projection_output,
               beamer_gain, white_band_low_pct, white_pct, reflectance_floor_pct):
        S = _ensure_bhwc(surface_photo)
        P = _ensure_bhwc(projection_output)
        if S.shape[0] == 1 and P.shape[0] > 1:
            S = S.expand(P.shape[0], -1, -1, -1).contiguous()
        S = _match_spatial(S, P)

        S_lin = srgb_to_linear(S)
        P_lin = srgb_to_linear(P)

        W = _robust_white_per_channel(
            S_lin
        )
        r_min = max(0.0, min(1.0, reflectance_floor_pct / 100.0))
        white_luma = _luma(W).mean().item()
        R_s = _lowfreq_reflectance_scalar(S_lin, white_luma=white_luma, r_min=r_min)

        V_lin = (S_lin + float(beamer_gain) * R_s * P_lin).clamp(0.0, 1.0)
        return (linear_to_srgb(V_lin),)

class SurfaceRadiometricCompensation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_texture": ("IMAGE",),
                "surface_photo": ("IMAGE",),
                "beamer_gain": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "white_band_low_pct": ("FLOAT", {"default": 90.0, "min": 70.0, "max": 99.0, "step": 0.5}),
                "white_pct": ("FLOAT", {"default": 98.0, "min": 90.0, "max": 99.9, "step": 0.1}),
                "reflectance_floor_pct": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "safety_pctile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "color_compensation": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Projector RGB (sRGB)", "Predicted View", "Applied Compression α")
    FUNCTION = "compensate"
    CATEGORY = "Eden 🌱/projection"

    def compensate(self,
                   target_texture, surface_photo, beamer_gain,
                   white_band_low_pct, white_pct, reflectance_floor_pct,
                   safety_pctile, color_compensation):

        P_srgb, V_srgb, alpha = surface_compensate(
            target_srgb=target_texture,
            surface_srgb=surface_photo,
            projector_gain=float(beamer_gain),
            reflectance_floor_percent=float(reflectance_floor_pct),
            color_compensation=bool(color_compensation),
            eps=1e-6
        )
        return (P_srgb, V_srgb, f"{alpha:.3f}")

# =============================
# CLI Test Harness
# =============================
if __name__ == "__main__":

    # --- Hardcoded paths ---
    target_path  = "texture_frame.png"   # or frame extracted from video
    surface_path = "surface_photo.png"
    outdir = "./proj_test_outputs"
    os.makedirs(outdir, exist_ok=True)

    # --- Load inputs ---
    def load_image(path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]  # [1,H,W,3]

    target_srgb = load_image(target_path)
    surface_srgb = load_image(surface_path)

    # --- Run compensation ---
    P_srgb, V_srgb, _ = surface_compensate(
        target_srgb=target_srgb,
        surface_srgb=surface_srgb,
        projector_gain=5.0,
        debug_outdir=outdir
    )

    # --- Save results ---
    def save_image(path, tensor):
        arr = (tensor[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, rgb)

    save_image(os.path.join(outdir, "projector_rgb.png"), P_srgb)
    save_image(os.path.join(outdir, "predicted_view.png"), V_srgb)
    print(f"[INFO] Saved results to {outdir}")
