import torch
import torch.nn.functional as F
import hashlib
from functools import lru_cache
from contextlib import nullcontext
from .img_utils import lab_to_rgb, rgb_to_lab

# ============================================================
# Gaussian kernels (separable, cached)
# ============================================================
@lru_cache(maxsize=16)
def gaussian_kernel_1d(kernel_size, sigma=None, device="cpu", dtype=torch.float32):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    r = torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device, dtype=dtype)
    k = torch.exp(-(r**2) / (2 * sigma**2))
    k = k / k.sum()
    return k

def separable_gaussian_blur(x, kernel_size, device):
    # x: (B,1,H,W) float
    dtype = x.dtype
    k1 = gaussian_kernel_1d(kernel_size, device=device, dtype=dtype)
    kx = k1.view(1,1,1,-1)
    ky = k1.view(1,1,-1,1)
    pad = kernel_size // 2
    x = F.pad(x, (pad,pad,pad,pad), mode='reflect')
    x = F.conv2d(x, kx)
    x = F.conv2d(x, ky)
    return x

# ---------- KMeans (CUDA-friendly) ----------
def _seed_from_tensor(t: torch.Tensor) -> int:
    """Content-dependent deterministic seed based on small stats of t (float32 CPU)."""
    # t expected shape (N,3) Lab flat subset
    with torch.no_grad():
        m = t.mean(dim=0).float().cpu().numpy().tobytes()
    return int.from_bytes(hashlib.blake2b(m, digest_size=8).digest(), 'little') & 0x7FFFFFFF

def kmeans_torch(x, k, iters=20, tol=1e-4, seeding='kmeans++', device=None, use_amp=True, seed=None):
    """
    x: (N, D) float on CUDA
    returns centers (k, D), labels (N,)
    """
    assert x.dim() == 2
    N, D = x.shape
    device = device or x.device
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and x.is_cuda, dtype=torch.float16) if x.is_cuda else nullcontext()

    # deterministic RNG (per call)
    if seed is None:
        seed = 12345
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # init
    if seeding == 'kmeans++':
        # first center
        idx0 = torch.randint(0, N, (1,), device=device, generator=g)
        centers = x[idx0]
        for _ in range(1, k):
            with amp_ctx:
                # float32 for stability in distance aggregation
                dist2 = torch.cdist(x.float(), centers.float(), p=2).pow(2)  # (N,c)
                dmin = dist2.min(dim=1).values
            probs = dmin / (dmin.sum() + 1e-12)
            idx = torch.multinomial(probs, 1, generator=g)
            centers = torch.cat([centers, x[idx]], dim=0)
    else:
        perm = torch.randperm(N, generator=g, device=device)[:k]
        centers = x[perm].clone()

    prev_inertia = None
    for _ in range(iters):
        with amp_ctx:
            dist2 = torch.cdist(x.float(), centers.float(), p=2).pow(2)  # (N,k) in fp32 for stability
            labels = torch.argmin(dist2, dim=1)
            inertia = dist2.gather(1, labels.view(-1,1)).sum()

        # update
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=k).clamp_min(1).view(-1,1).to(new_centers.dtype)
        new_centers.scatter_add_(0, labels.view(-1,1).expand(-1, D), x)
        new_centers = new_centers / counts

        if prev_inertia is not None and torch.abs(prev_inertia - inertia) < tol * (prev_inertia + 1e-12):
            centers = new_centers
            break
        centers = new_centers
        prev_inertia = inertia

    return centers, labels

def _deterministic_stride_indices(N, max_points, device):
    if N <= max_points:
        return torch.arange(N, device=device)
    step = float(N) / float(max_points)
    # evenly-spaced (round to nearest)
    idx = torch.clamp(torch.round(torch.arange(0, max_points, device=device) * step).long(), max=N-1)
    return idx.unique()

def fit_kmeans_gpu(lab_flat, n_color_clusters, max_fit_points=200_000, iters=25, device=None):
    # lab_flat: (N,3) float on CUDA
    N = lab_flat.shape[0]
    device = device or lab_flat.device
    # deterministic, content-dependent subset (prevents frame-to-frame jitter)
    idx = _deterministic_stride_indices(N, max_fit_points, device=device)
    x_fit = lab_flat[idx]

    # content-dependent seed for k-means++
    seed = _seed_from_tensor(x_fit.detach())

    centers, _ = kmeans_torch(
        x_fit, n_color_clusters, iters=iters, device=device, use_amp=True, seed=seed
    )

    # Final assignment in **float32** for stability (no autocast)
    dist2_full = torch.cdist(lab_flat.float(), centers.float(), p=2).pow(2)
    labels_full = torch.argmin(dist2_full, dim=1)
    return centers, labels_full

def equalize_cluster_areas(lab_flat, labels, centers, strength, max_iters=5):
    """
    Reassign pixels to equalize cluster areas while preserving color similarity.

    Uses an iterative algorithm that adds a size-based penalty to cluster distances.
    This naturally causes pixels near cluster boundaries to redistribute, growing small
    clusters and shrinking large ones, without creating spatial discontinuities.

    Works purely in LAB color space (like KMeans), ensuring temporal consistency when
    clustering video frames. Pixels only switch clusters if an alternative cluster
    center is similarly close in color space.

    Args:
        lab_flat: (N, 3) LAB color values for all pixels across all frames
        labels: (N,) initial cluster assignments from KMeans
        centers: (K, 3) cluster centers in LAB space
        strength: 0-1 value controlling the weight of area equalization
                  0 = keep original KMeans assignments
                  1 = strongly enforce equal areas
        max_iters: maximum number of reassignment iterations (default: 5)

    Returns:
        labels: (N,) adjusted cluster assignments with more balanced areas
    """
    if strength <= 0:
        return labels

    K = centers.shape[0]
    N = lab_flat.shape[0]
    target_size = N / K
    device = lab_flat.device

    labels = labels.clone()

    # Compute base distances once (in float32 for stability)
    base_distances = torch.cdist(lab_flat.float(), centers.float(), p=2)  # (N, K)

    # Scale strength to a reasonable range for the penalty
    # We want the penalty to be comparable to distance variations
    distance_scale = base_distances.std().item()
    scaled_strength = strength * distance_scale

    for iteration in range(max_iters):
        # Current cluster sizes
        counts = torch.bincount(labels, minlength=K).float()

        # Compute size penalty: log(count / target) makes larger clusters less attractive
        # The logarithm ensures scale-invariance: a cluster 2x too large has the same
        # penalty magnitude as a cluster 0.5x too small
        size_ratio = (counts + 1.0) / (target_size + 1.0)  # +1 to avoid log(0)
        size_penalty = scaled_strength * torch.log(size_ratio)  # (K,)

        # Adjust distances by adding size penalty
        adjusted_distances = base_distances + size_penalty.unsqueeze(0)  # (N, K)

        # Reassign based on adjusted distances
        new_labels = torch.argmin(adjusted_distances, dim=1)

        # Check convergence: stop if no pixels changed assignment
        if (new_labels == labels).all():
            break

        labels = new_labels

    return labels

# ============================================================
# ComfyUI Node
# ============================================================
class MaskFromRGB_KMeans:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "n_color_clusters": ("INT", {"default": 6, "min": 2, "max": 10}),
                "clustering_resolution": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "feathering_fraction": ("FLOAT", { "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01 }),
                "equalize_areas": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("1","2","3","4","5","6","7","8","combined",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, n_color_clusters, clustering_resolution, feathering_fraction, equalize_areas):
        # pick device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("--- Warning: Using MPS backend")
            device = torch.device('mps')
        else:
            print("--- Warning: Using CPU")
            device = torch.device('cpu')

        original_device = image.device
        image = image.to(device)

        # convert to LAB, keep channels-first
        lab_images = torch.stack([rgb_to_lab(img) for img in image])  # (N,H,W,3)
        lab_images = lab_images.permute(0,3,1,2)                      # (N,3,H,W)

        # resize to clustering_resolution
        n, c, h, w = lab_images.shape
        h_target = int(clustering_resolution)
        w_target = int(round(clustering_resolution * w / h))
        lab_images = F.interpolate(lab_images, size=[h_target, w_target], mode='bicubic', align_corners=False)

        # flatten across batch (stable centers across all frames in this batch)
        n, c, h, w = lab_images.shape
        lab_flat = lab_images.permute(0,2,3,1).reshape(-1, 3).contiguous()

        # KMeans (deterministic sampling & seeding)
        centers, cluster_labels_flat = fit_kmeans_gpu(lab_flat, n_color_clusters, device=device)

        # luminance ordering (stable label remap)
        cluster_luminance = centers[:, 0]
        sorted_indices = torch.argsort(cluster_luminance)
        index_map = torch.empty_like(sorted_indices)
        index_map[sorted_indices] = torch.arange(n_color_clusters, device=device)
        cluster_labels_flat = index_map[cluster_labels_flat]

        # Apply area equalization if requested
        if equalize_areas > 0:
            cluster_labels_flat = equalize_cluster_areas(
                lab_flat,
                cluster_labels_flat,
                centers[sorted_indices],  # Use luminance-sorted centers
                strength=float(equalize_areas)
            )

        # reshape labels back to (N, H, W)
        cluster_labels = cluster_labels_flat.view(n, h, w)

        # ========== TEMPORAL CONSISTENCY ==========
        # K-means is fit on ALL frames together and luminance sorting provides
        # deterministic ordering, ensuring labels are temporally consistent without
        # additional matching. Area equalization (if enabled) also works on all frames
        # together, maintaining this temporal consistency.

        # ========== Generate individual masks directly from cluster_labels ==========
        # cluster_labels uses luminance-sorted indices that are consistent across frames
        # because k-means was fit on all frames together. We create binary masks for
        # each cluster and apply feathering for smooth boundaries.

        # Generate binary masks for each cluster: 1.0 where cluster == k, 0.0 elsewhere
        masks_list = []
        for k in range(n_color_clusters):
            mask_k = (cluster_labels == k).float()  # (N, H, W)
            masks_list.append(mask_k)

        masks = torch.stack(masks_list, dim=1)  # (N, K, H, W)

        # Keep only up to 8 channels in outputs
        K = min(n_color_clusters, 8)
        masks = masks[:, :K, :, :]

        # Combined mask: grayscale representation (0 to 1) based on cluster index
        if n_color_clusters > 1:
            combined_mask = (cluster_labels.float() / (n_color_clusters - 1)).clamp(0, 1)
        else:
            combined_mask = torch.zeros_like(cluster_labels, dtype=torch.float)

        # Apply feathering (Gaussian blur) to smooth mask boundaries
        if feathering_fraction > 0:
            feather = int(feathering_fraction * (w + h) / 2.0)
            # Ensure kernel size is odd and at least 3
            if feather < 3:
                feather = 3
            if feather % 2 == 0:
                feather += 1

            # Feather individual cluster masks
            masks_b = masks.reshape(-1, 1, h, w)
            masks_b = separable_gaussian_blur(masks_b, feather, device)
            masks = masks_b.view(n, K, h, w)

            # Feather combined mask
            cmb_b = combined_mask.unsqueeze(1)
            cmb_b = separable_gaussian_blur(cmb_b, feather, device)
            combined_mask = cmb_b.squeeze(1)

        # upscale back to original
        H0, W0 = image.shape[1], image.shape[2]
        masks = F.interpolate(masks, size=(H0,W0), mode='bicubic', align_corners=False).clamp(0, 1)
        combined_mask = F.interpolate(combined_mask.unsqueeze(1), size=(H0,W0), mode='bicubic', align_corners=False).squeeze(1).clamp(0, 1)

        # back to original device
        masks = masks.to(original_device, non_blocking=True)
        combined_mask = combined_mask.to(original_device, non_blocking=True)

        # Ensure 8 outputs
        N, Kcur, H, W = masks.shape
        if Kcur < 8:
            pad = torch.zeros((N, 8-Kcur, H, W), device=masks.device, dtype=masks.dtype)
            masks = torch.cat([masks, pad], dim=1)

        return masks[:,0], masks[:,1], masks[:,2], masks[:,3], masks[:,4], masks[:,5], masks[:,6], masks[:,7], combined_mask


