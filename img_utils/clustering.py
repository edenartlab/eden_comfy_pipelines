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

def _lab_sort_indices(centers_lab: torch.Tensor) -> torch.Tensor:
    """
    Deterministic lexicographic sort by (L, a, b) to avoid ties that reshuffle channels.
    centers_lab: (K,3) in Lab
    """
    L = centers_lab[:, 0]
    a = centers_lab[:, 1]
    b = centers_lab[:, 2]
    # Compose a single key with enough dynamic range to preserve lexicographic order
    key = L * 1e6 + a * 1e3 + b
    return torch.argsort(key)


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
    return centers, labels_full, dist2_full  # return distances so we can build soft masks

def _soft_assignments_from_dist2(dist2, tau):
    """
    dist2: (Npix, K) squared distances
    tau: temperature (>= 1e-6). Lower = sharper (harder) assignments.
    returns probs: (Npix, K), sum=1 across K
    """
    # use softmin over dist2/tau
    # subtract min for numerical stability
    m = dist2.min(dim=1, keepdim=True).values
    logits = -(dist2 - m) / max(tau, 1e-6)
    probs = torch.softmax(logits, dim=1)
    return probs

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
                # New: stability controls
                "soft_labels_tau": ("FLOAT", { "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01 }),  # 0.0 = hard labels
                "min_confidence": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),     # 0 => no binarization gate
                "temporal_ema": ("FLOAT", { "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.01 }),       # 0 = no smoothing, higher = more smoothing
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("1","2","3","4","5","6","7","8","combined",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, n_color_clusters, clustering_resolution, feathering_fraction, equalize_areas, soft_labels_tau, min_confidence, temporal_ema):
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

        # KMeans (deterministic sampling & seeding); distances in fp32
        centers, cluster_labels_flat, dist2_full = fit_kmeans_gpu(lab_flat, n_color_clusters, device=device)

        # Deterministic lexicographic sort by (L, a, b) to avoid luminance-tie shuffles
        sorted_indices = _lab_sort_indices(centers)
        index_map = torch.empty_like(sorted_indices)
        index_map[sorted_indices] = torch.arange(n_color_clusters, device=device)
        cluster_labels_flat = index_map[cluster_labels_flat]

        # shape labels & distances back
        cluster_labels = cluster_labels_flat.view(n, h, w)
        dist2_full = dist2_full[:, sorted_indices]  # keep distances aligned with sorted label order
        dist2_full = dist2_full.view(n, h, w, n_color_clusters).reshape(-1, n_color_clusters)  # (N*h*w, K)

        # ========== TEMPORAL CONSISTENCY: Match clusters across frames ==========
        # Build a robust, bijective permutation per frame that maps current clusters to previous.
        # This ensures stable channel semantics over time and prevents glitchy artifacts.
        if n > 1:
            def _build_bijective_permutation(prev_labels, curr_labels, prev_centers, curr_centers, K, device, lambda_de=0.05):
                """
                Build a permutation P where P[curr_idx] = prev_idx.
                Uses a hybrid score: overlap - lambda * deltaE(Lab centers).
                Guarantees: bijective (no -1, no duplicates), deterministic.

                Args:
                    prev_labels: (H,W) previous frame labels
                    curr_labels: (H,W) current frame labels
                    prev_centers: (K,3) Lab centers from previous frame
                    curr_centers: (K,3) Lab centers from current frame
                    K: number of clusters
                    device: torch device
                    lambda_de: weight for deltaE penalty (higher = more color-continuity bias)
                """
                # Compute overlap matrix: overlap[i,j] = # pixels where prev==i AND curr==j
                overlap = torch.zeros((K, K), device=device, dtype=torch.int64)
                for i in range(K):
                    prev_mask = (prev_labels == i)
                    if prev_mask.any():
                        for j in range(K):
                            overlap[i, j] = (prev_mask & (curr_labels == j)).sum()

                # Normalize overlap to [0,1] by total pixels
                total_pixels = prev_labels.numel()
                overlap_norm = overlap.float() / max(total_pixels, 1.0)

                # Compute deltaE matrix: de[i,j] = deltaE(prev_center_i, curr_center_j)
                # Using Euclidean distance in Lab space as a deltaE proxy
                de = torch.cdist(prev_centers.float(), curr_centers.float(), p=2)  # (K,K)

                # Normalize deltaE to roughly [0,1] range (Lab deltaE typically < 100, but can be larger)
                de_norm = de / 100.0

                # Hybrid score: higher is better
                # score[i,j] = overlap_norm[i,j] - lambda_de * de_norm[i,j]
                score = overlap_norm - lambda_de * de_norm

                # Greedy matching with deterministic tie-breaks
                perm = torch.full((K,), -1, dtype=torch.long, device=device)
                used_prev = torch.zeros(K, dtype=torch.bool, device=device)
                used_curr = torch.zeros(K, dtype=torch.bool, device=device)
                work = score.clone()

                # Match pairs with highest score first
                for _ in range(K):
                    # Mask already-used indices with large negative value
                    work[used_prev, :] = -1e9
                    work[:, used_curr] = -1e9

                    # Find best remaining match
                    # Deterministic tie-break: use argmax (takes first occurrence)
                    flat_idx = work.argmax()
                    i = (flat_idx // K).item()
                    j = (flat_idx % K).item()

                    # Stop if score is too negative (no good matches left)
                    if work[i, j] < -1e8:
                        break

                    # Safety: skip if already used (shouldn't happen, but be defensive)
                    if used_prev[i] or used_curr[j]:
                        work[i, j] = -1e9
                        continue

                    # Assign mapping
                    perm[j] = i
                    used_prev[i] = True
                    used_curr[j] = True

                # Fill any remaining unmapped indices (ensures bijection)
                if (perm == -1).any():
                    remaining_curr = torch.where(perm == -1)[0]
                    remaining_prev = torch.where(~used_prev)[0]

                    # Map remaining by best available score
                    for j in remaining_curr.tolist():
                        if remaining_prev.numel() == 0:
                            break
                        # Pick prev with max score for this curr (deterministic via argmax)
                        best_idx_in_remaining = score[remaining_prev, j].argmax()
                        best_idx = remaining_prev[best_idx_in_remaining].item()
                        perm[j] = best_idx
                        # Remove from remaining
                        remaining_prev = remaining_prev[remaining_prev != best_idx]

                    # Final fallback: identity mapping for any still unassigned
                    if (perm == -1).any():
                        for j in torch.where(perm == -1)[0].tolist():
                            # Find any unused prev index (deterministic order)
                            unused = torch.where(~torch.isin(torch.arange(K, device=device), perm))[0]
                            perm[j] = unused[0].item() if unused.numel() > 0 else j

                # Sanity check: ensure valid permutation (no -1, no duplicates)
                assert (perm >= 0).all() and (perm < K).all(), "Invalid permutation indices"
                assert perm.unique().numel() == K, "Permutation has duplicates"

                return perm

            # We need to track the sorted Lab centers per frame for temporal matching
            # First, compute the sorted centers (after the initial Lab sort)
            sorted_centers = centers[sorted_indices]  # (K,3) in Lab, now in sorted order

            # Store centers per frame (we'll recompute them per frame from labels)
            frame_centers = []
            for frame_idx in range(n):
                frame_start = frame_idx * h * w
                frame_end = (frame_idx + 1) * h * w
                frame_lab = lab_flat[frame_start:frame_end]  # (h*w, 3)
                frame_lbl = cluster_labels[frame_idx].view(-1)  # (h*w,)

                # Recompute centers for this frame
                curr_centers = torch.zeros((n_color_clusters, 3), device=device, dtype=frame_lab.dtype)
                for k in range(n_color_clusters):
                    mask = (frame_lbl == k)
                    if mask.any():
                        curr_centers[k] = frame_lab[mask].mean(dim=0)
                    else:
                        # Fallback to sorted center if no pixels
                        curr_centers[k] = sorted_centers[k]
                frame_centers.append(curr_centers)

            # Apply temporal matching frame by frame
            for frame_idx in range(1, n):
                prev_labels = cluster_labels[frame_idx - 1]
                curr_labels = cluster_labels[frame_idx]
                prev_centers = frame_centers[frame_idx - 1]
                curr_centers = frame_centers[frame_idx]

                # Build permutation: perm[curr_idx] = prev_idx
                perm = _build_bijective_permutation(
                    prev_labels, curr_labels, prev_centers, curr_centers,
                    n_color_clusters, device, lambda_de=0.05
                )

                # Remap labels: pixels labeled curr_idx become prev_idx
                remapped_labels = perm[curr_labels]
                cluster_labels[frame_idx] = remapped_labels

                # Update frame_centers to reflect the remapping
                frame_centers[frame_idx] = curr_centers[perm]

                # Remap distance columns to maintain alignment with labels
                frame_start = frame_idx * h * w
                frame_end = (frame_idx + 1) * h * w
                frame_dist2 = dist2_full[frame_start:frame_end, :]  # (h*w, K)

                # Vectorized column reordering: column curr_idx goes to position perm[curr_idx]
                remapped_dist2 = torch.empty_like(frame_dist2)
                remapped_dist2[:, perm] = frame_dist2
                dist2_full[frame_start:frame_end, :] = remapped_dist2

        # optional equalize (kept off by default)
        if equalize_areas > 0:
            print("equalize_areas is not implemented yet!!")
            #flat_eq = equalize_areas_fast(
            #    lab_flat, cluster_labels.view(-1), centers[sorted_indices], strength=float(equalize_areas)
            #)
            #cluster_labels = flat_eq.view(n, h, w)

        # ---------- Build soft or confidence-gated masks ----------
        if soft_labels_tau > 0.0:
            probs_flat = _soft_assignments_from_dist2(dist2_full, tau=float(soft_labels_tau))  # (N*h*w,K)
            probs = probs_flat.view(n, h, w, n_color_clusters).permute(0,3,1,2)  # (N,K,H,W)
            masks = probs
            # Optional confidence gate to push near-binary only where margin is large
            if min_confidence > 0.0:
                # confidence = softmax margin between top1 and top2
                top2 = torch.topk(probs_flat, k=2, dim=1).values
                margin = (top2[:,0] - top2[:,1]).view(n, h, w, 1)  # (N,H,W,1)
                confident = (margin >= float(min_confidence)).float()
                hard = F.one_hot(cluster_labels, num_classes=n_color_clusters).permute(0,3,1,2).float()  # (N,K,H,W)
                masks = confident.permute(0,3,1,2) * hard + (1.0 - confident.permute(0,3,1,2)) * masks
        else:
            # Old hard labeling path (kept for compatibility)
            labels_exp = cluster_labels.unsqueeze(1)  # (N,1,H,W)
            choices = torch.arange(n_color_clusters, device=device).view(1, n_color_clusters, 1, 1)
            masks = (labels_exp == choices).float()

        # Keep only up to 8 channels in outputs
        K = min(n_color_clusters, 8)
        masks = masks[:, :K, :, :]

        # combined mask grayscale (unchanged)
        if n_color_clusters > 1:
            combined_mask = (cluster_labels.float() / (n_color_clusters - 1)).clamp(0,1)
        else:
            combined_mask = torch.zeros_like(cluster_labels, dtype=torch.float)

        # upscale back to original BEFORE feathering (so feathering is resolution-independent)
        H0, W0 = image.shape[1], image.shape[2]
        masks = F.interpolate(masks, size=(H0,W0), mode='bicubic', align_corners=False)
        combined_mask = F.interpolate(combined_mask.unsqueeze(1), size=(H0,W0), mode='bicubic', align_corners=False).squeeze(1)

        # feather at full resolution, with kernel size derived from feather_frac * max(H0, W0)
        if feathering_fraction > 0:
            feather = int(feathering_fraction * max(H0, W0))
            # ensure odd and at least 3
            if feather < 3: feather = 3
            if feather % 2 == 0: feather += 1
            masks_b = masks.reshape(-1,1,H0,W0)
            masks_b = separable_gaussian_blur(masks_b, feather, device)
            masks = masks_b.view(n, K, H0, W0)
            cmb_b = combined_mask.unsqueeze(1)
            cmb_b = separable_gaussian_blur(cmb_b, feather, device)
            combined_mask = cmb_b.squeeze(1)

        # Optional temporal EMA on aligned masks to suppress one-frame twitches
        # Applied after alignment and feathering
        if temporal_ema > 0.0 and n > 1:
            alpha = float(temporal_ema)
            # Apply EMA: mask[t] = alpha * mask[t-1] + (1 - alpha) * mask[t]
            for frame_idx in range(1, n):
                masks[frame_idx] = alpha * masks[frame_idx - 1] + (1.0 - alpha) * masks[frame_idx]
                combined_mask[frame_idx] = alpha * combined_mask[frame_idx - 1] + (1.0 - alpha) * combined_mask[frame_idx]

        # back to original device
        masks = masks.to(original_device, non_blocking=True)
        combined_mask = combined_mask.to(original_device, non_blocking=True)

        # Ensure 8 outputs
        N, Kcur, H, W = masks.shape
        if Kcur < 8:
            pad = torch.zeros((N, 8-Kcur, H, W), device=masks.device, dtype=masks.dtype)
            masks = torch.cat([masks, pad], dim=1)

        return masks[:,0], masks[:,1], masks[:,2], masks[:,3], masks[:,4], masks[:,5], masks[:,6], masks[:,7], combined_mask


