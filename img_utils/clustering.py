import torch
import torch.nn.functional as F
import hashlib
from .img_utils import rgb_to_lab


# ============================================================
# KMeans (GPU-friendly, deterministic)
# ============================================================

def _sq_dists(a, b):
    """
    Squared Euclidean distances between rows of a (N, D) and b (M, D).
    Returns (N, M) tensor.  Uses matmul — much faster than torch.cdist on MPS/CPU.
    """
    a_sq = (a * a).sum(dim=1, keepdim=True)   # (N, 1)
    b_sq = (b * b).sum(dim=1).unsqueeze(0)     # (1, M)
    return (a_sq + b_sq - 2.0 * a @ b.t()).clamp_min(0.0)


def kmeans_torch(x, k, iters=25, tol=1e-4, seed=12345):
    """
    x: (N, D) float tensor on any device.
    Returns centers (k, D).
    """
    N, D = x.shape
    device = x.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # KMeans++ initialization
    centers = x[torch.randint(0, N, (1,), device=device, generator=g)]
    for _ in range(1, k):
        dist2 = _sq_dists(x, centers).min(dim=1).values
        probs = dist2 / (dist2.sum() + 1e-12)
        centers = torch.cat([centers, x[torch.multinomial(probs, 1, generator=g)]], dim=0)

    # Iterative refinement
    prev_inertia = None
    for _ in range(iters):
        dist2 = _sq_dists(x, centers)
        labels = dist2.argmin(dim=1)
        inertia = dist2.gather(1, labels.unsqueeze(1)).sum()

        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=k).clamp_min(1).unsqueeze(1).to(centers.dtype)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
        new_centers /= counts

        if prev_inertia is not None and (prev_inertia - inertia).abs() < tol * (prev_inertia + 1e-12):
            return new_centers
        centers = new_centers
        prev_inertia = inertia

    return centers


def _separable_gaussian_blur(masks, sigma):
    """
    Blur (B, K, H, W) masks with a separable Gaussian.
    Two 1D convolutions (horizontal + vertical) — O(radius) per pixel, not O(radius²).
    """
    if sigma < 0.5:
        return masks
    B, K, H, W = masks.shape
    radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=masks.device, dtype=masks.dtype)
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()

    # Batch all masks as independent channels: (B*K, 1, H, W)
    m = masks.reshape(B * K, 1, H, W)
    kh = k1d.reshape(1, 1, 1, -1)
    kv = k1d.reshape(1, 1, -1, 1)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode='replicate'), kh)
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode='replicate'), kv)
    return m.reshape(B, K, H, W)


# ============================================================
# ComfyUI Node
# ============================================================

class MaskFromRGB_KMeans:
    MAX_KMEANS_SAMPLES = 200_000

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "n_color_clusters": ("INT", {"default": 6, "min": 2, "max": 10}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.01}),
                "equalize_areas": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",) * 9
    RETURN_NAMES = ("1", "2", "3", "4", "5", "6", "7", "8", "combined")
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, n_color_clusters, softness=0.15, equalize_areas=0.0, **kwargs):
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cpu'
        )
        original_device = image.device
        image = image.to(device)
        B, H, W, _ = image.shape
        K = n_color_clusters

        # Convert to LAB
        lab = torch.stack([rgb_to_lab(img) for img in image])  # (B, H, W, 3)
        lab_flat = lab.reshape(-1, 3).float()
        N = lab_flat.shape[0]

        # Deterministic content-dependent seed
        seed = int.from_bytes(
            hashlib.blake2b(lab_flat.mean(dim=0).cpu().numpy().tobytes(), digest_size=8).digest(),
            'little'
        ) & 0x7FFFFFFF

        # Subsample for KMeans fitting
        if N > self.MAX_KMEANS_SAMPLES:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            idx = torch.randperm(N, device=device, generator=g)[:self.MAX_KMEANS_SAMPLES]
            fit_data = lab_flat[idx]
        else:
            fit_data = lab_flat

        # Fit KMeans and sort centers by luminance
        centers = kmeans_torch(fit_data, K, iters=25, seed=seed)
        centers = centers[centers[:, 0].argsort()]

        # Area equalization bias (estimated from KMeans subsample)
        pairwise_dist = _sq_dists(centers, centers).sqrt()
        off_diag = pairwise_dist[~torch.eye(K, dtype=torch.bool, device=device)]
        scale = off_diag.mean().clamp_min(1e-3)

        area_bias = torch.zeros(K, device=device)
        if equalize_areas > 0:
            fit_dist = _sq_dists(fit_data, centers).sqrt()
            fit_labels = fit_dist.argmin(dim=1)
            counts = torch.bincount(fit_labels, minlength=K).float()
            target = fit_data.shape[0] / K
            ratio = (counts + 1) / (target + 1)
            area_bias = equalize_areas * scale * torch.log(ratio)

        # Hard assign pixels (frame-by-frame for memory efficiency)
        masks = torch.empty(B, K, H, W, device=device)
        combined = torch.empty(B, H, W, device=device)
        cluster_vals = torch.arange(K, device=device, dtype=torch.float32) / max(K - 1, 1)
        for b in range(B):
            frame_lab = lab[b].reshape(-1, 3).float()      # (H*W, 3)
            dist = _sq_dists(frame_lab, centers)            # (H*W, K)
            if equalize_areas > 0:
                dist = dist + area_bias.unsqueeze(0)
            labels = dist.argmin(dim=1)                     # (H*W,)
            masks[b] = F.one_hot(labels, K).float().T.reshape(K, H, W)
            combined[b] = cluster_vals[labels].reshape(H, W)

        # Smooth mask edges with separable Gaussian blur
        sigma = softness * min(H, W) * 0.02
        masks = _separable_gaussian_blur(masks, sigma)
        masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Pad to 8 mask outputs
        if K < 8:
            masks = F.pad(masks, (0, 0, 0, 0, 0, 8 - K))
        masks = masks[:, :8].to(original_device)
        combined = combined.to(original_device)

        return (masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3],
                masks[:, 4], masks[:, 5], masks[:, 6], masks[:, 7], combined)
