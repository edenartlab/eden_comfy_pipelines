import math

import torch
import torch.nn.functional as F
import comfy.model_management


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1 / 2.4) - a).clamp(0.0, 1.0)


def _prep(x: torch.Tensor) -> torch.Tensor:
    return x.to(comfy.model_management.get_torch_device(), torch.float32).clamp(0.0, 1.0)


def _match_surface(S: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a single surface photo over the batch and resize it to ref's resolution."""
    if S.shape[0] == 1 and ref.shape[0] > 1:
        S = S.expand(ref.shape[0], -1, -1, -1).contiguous()
    if S.shape[1:3] == ref.shape[1:3]:
        return S
    S = F.interpolate(S.movedim(-1, 1), size=ref.shape[1:3], mode="bilinear", align_corners=False)
    return S.movedim(1, -1)


def _luma(linear_rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=linear_rgb.device, dtype=linear_rgb.dtype)
    return (linear_rgb * w).sum(dim=-1, keepdim=True)


def _robust_white_per_channel(S_lin, max_samples=500_000):
    """Median color of the brightest 5% of surface pixels, capped at 70% reflectance."""
    L = _luma(S_lin).reshape(-1)
    S = S_lin.reshape(-1, 3)
    if S.shape[0] > max_samples:
        stride = S.shape[0] // max_samples
        S = S[::stride]
        L = L[::stride]

    idx = torch.argsort(L, descending=True)
    top = S[idx[: int(0.05 * len(idx))]]
    return torch.median(top, dim=0).values.clamp(1e-3, 0.7).view(1, 1, 1, 3)


def _lowfreq_reflectance_scalar(S_lin: torch.Tensor, white_luma: float, r_min: float, kernel_frac: float = 1 / 24):
    R = (_luma(S_lin) / max(white_luma, 1e-6)).clamp(r_min, 1.0)
    _, H, W, _ = R.shape
    k = max(3, int(round(max(H, W) * kernel_frac)) | 1)
    R = F.avg_pool2d(R.movedim(-1, 1), kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
    return R.movedim(1, -1)


def surface_compensate(target_srgb, surface_srgb, projector_gain=2.0, reflectance_floor_percent=5.0,
                       color_compensation=True, gamma=4.0, eps=1e-6):
    """
    Radiometric surface compensation with perceptual (log) tone mapping.
    Returns (projector image in sRGB, tone-mapped predicted view).
    """
    T = _prep(target_srgb)
    S = _match_surface(_prep(surface_srgb), T)

    T_lin = srgb_to_linear(T)
    S_lin = srgb_to_linear(S)

    if color_compensation:
        W = _robust_white_per_channel(S_lin)
        gain_c = (W.mean(dim=-1, keepdim=True) / (W + eps)).clamp(0.7, 1.3)
        T_lin = (T_lin * gain_c).clamp(0.0, 1.0)

    r_min = max(0.0, min(1.0, reflectance_floor_percent / 100.0))
    white_luma = _luma(srgb_to_linear(torch.ones_like(S))).mean().item()
    R_s = _lowfreq_reflectance_scalar(S_lin, white_luma=white_luma, r_min=r_min)

    k = float(projector_gain)
    P_lin = ((T_lin - S_lin) / (k * R_s + eps)).clamp(0.0, 1.0)
    V_lin = S_lin + k * R_s * P_lin
    V_tone = (torch.log1p(gamma * V_lin) / math.log1p(gamma)).clamp(0.0, 1.0)
    return linear_to_srgb(P_lin), V_tone


class ProjectionPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "surface_photo": ("IMAGE", {"tooltip": "Photo of the projection surface under ambient light (single frame or one per output frame)."}),
                "projection_output": ("IMAGE", {"tooltip": "The image(s) the projector will show."}),
                "beamer_gain": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.05, "tooltip": "Projector brightness relative to the ambient-lit surface."}),
                "white_band_low_pct": ("FLOAT", {"default": 90.0, "min": 70.0, "max": 99.0, "step": 0.5, "tooltip": "Unused (kept for workflow compatibility)."}),
                "white_pct": ("FLOAT", {"default": 99.0, "min": 90.0, "max": 99.9, "step": 0.1, "tooltip": "Unused (kept for workflow compatibility)."}),
                "reflectance_floor_pct": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Minimum surface reflectance (%) so dark areas still receive some light."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Perceived View",)
    FUNCTION = "render"
    CATEGORY = "Eden 🌱/Image"
    DESCRIPTION = "Simulates what a projection will look like on a textured surface (additive light model)."
    OUTPUT_TOOLTIPS = ("Predicted view of the surface with the projection on it.",)

    def render(self, surface_photo, projection_output,
               beamer_gain, white_band_low_pct, white_pct, reflectance_floor_pct):
        P = _prep(projection_output)
        S = _match_surface(_prep(surface_photo), P)

        S_lin = srgb_to_linear(S)
        P_lin = srgb_to_linear(P)

        W = _robust_white_per_channel(S_lin)
        r_min = max(0.0, min(1.0, reflectance_floor_pct / 100.0))
        R_s = _lowfreq_reflectance_scalar(S_lin, white_luma=_luma(W).mean().item(), r_min=r_min)

        V_lin = (S_lin + float(beamer_gain) * R_s * P_lin).clamp(0.0, 1.0)
        return (linear_to_srgb(V_lin).to(comfy.model_management.intermediate_device()),)


class SurfaceRadiometricCompensation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_texture": ("IMAGE", {"tooltip": "The image(s) you want the audience to see on the surface."}),
                "surface_photo": ("IMAGE", {"tooltip": "Photo of the projection surface under ambient light."}),
                "beamer_gain": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.05, "tooltip": "Projector brightness relative to the ambient-lit surface."}),
                "white_band_low_pct": ("FLOAT", {"default": 90.0, "min": 70.0, "max": 99.0, "step": 0.5, "tooltip": "Unused (kept for workflow compatibility)."}),
                "white_pct": ("FLOAT", {"default": 98.0, "min": 90.0, "max": 99.9, "step": 0.1, "tooltip": "Unused (kept for workflow compatibility)."}),
                "reflectance_floor_pct": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Minimum surface reflectance (%), avoids blowing out dark areas."}),
                "safety_pctile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Unused (kept for workflow compatibility)."}),
                "color_compensation": ("BOOLEAN", {"default": False, "tooltip": "White-balance the target against the surface's color cast."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Projector RGB (sRGB)", "Predicted View", "Applied Compression α")
    FUNCTION = "compensate"
    CATEGORY = "Eden 🌱/Image"
    DESCRIPTION = "Computes the projector image that compensates for a textured/colored surface so the projection looks like the target."
    OUTPUT_TOOLTIPS = ("Image to send to the projector.", "Tone-mapped prediction of what the audience will see.", "Short description of the tone mapping applied.")

    def compensate(self, target_texture, surface_photo, beamer_gain,
                   white_band_low_pct, white_pct, reflectance_floor_pct,
                   safety_pctile, color_compensation):
        gamma = 4.0
        P_srgb, V_srgb = surface_compensate(
            target_srgb=target_texture,
            surface_srgb=surface_photo,
            projector_gain=float(beamer_gain),
            reflectance_floor_percent=float(reflectance_floor_pct),
            color_compensation=bool(color_compensation),
            gamma=gamma,
        )
        out = comfy.model_management.intermediate_device()
        return (P_srgb.to(out), V_srgb.to(out), f"tone-mapped γ={gamma}")
