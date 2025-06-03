# Organic Fill Algorithm Fixes & Usage Guide

## Summary of Fixes

The organic fill algorithm was experiencing issues where the final mask would always be completely black (no growth). The following critical issues were identified and fixed:

### 1. Main Issue: Mask Normalization Bug

**Problem**: The `base_mask` (which should be all 1s for full image filling) was being passed through `normalize_tensor()`, which incorrectly converted it to all zeros when all values were the same.

**Fix**: Removed the unnecessary `normalize_tensor()` call for the mask in the `OrganicFillBatch.__init__()` method.

```python
# BEFORE (broken):
self.mask = normalize_tensor(self.mask)  # This would zero out uniform masks!

# AFTER (fixed):
# Don't normalize the mask - it's already a valid binary mask (0/1)
```

### 2. Improved Parameter Defaults

**Problem**: The default parameters were too restrictive, making growth difficult even when the mask was working.

**Fixes**:
- Reduced `growth_threshold` from 0.6 to 0.3 (more permissive)
- Increased `barrier_jump_power` from 0.25 to 0.5 (allows more barrier jumping)
- Increased `weight_lab` from 0.01 to 0.5 (gives LAB gradient more influence)

### 3. Fallback Mechanisms Added

- **Active Mask Reactivation**: If the active mask becomes empty but unfilled pixels remain, the algorithm now reactivates all unfilled areas
- **Growth Probability Boost**: If growth probability is too low everywhere (< 0.1), the algorithm automatically boosts all values by +0.3

## Recommended Parameter Settings

### For Easy/Permissive Growth
```python
growth_threshold = 0.1      # Very low threshold
barrier_jump_power = 0.8    # High barrier jumping
weight_lab = 1.0           # Rely mainly on LAB gradient
weight_sam = 0.0           # Disable other maps
weight_depth = 0.0
weight_canny = 0.0
weight_hed = 0.0
```

### For Conservative/Edge-Aware Growth
```python
growth_threshold = 0.5      # Higher threshold
barrier_jump_power = 0.2    # Low barrier jumping
weight_lab = 0.3           # Balanced weights
weight_sam = 1.0
weight_depth = 1.2
weight_canny = 0.1
weight_hed = 0.4
```

### For Balanced Growth (Default)
```python
growth_threshold = 0.3      # Medium threshold
barrier_jump_power = 0.5    # Medium barrier jumping
weight_lab = 0.5           # Balanced weights
weight_sam = 1.0
weight_depth = 1.5
weight_canny = 0.05
weight_hed = 0.5
```

## Parameter Explanations

### Core Parameters
- **`growth_threshold`** (0.0-1.0): Minimum growth probability required for a pixel to grow. Lower = more permissive.
- **`barrier_jump_power`** (0.0-1.0): How easily the algorithm can "jump over" barriers (low probability regions). Higher = more barrier jumping.

### Weight Parameters
- **`weight_lab`**: Influence of LAB color gradient (always available)
- **`weight_sam`**: Influence of SAM segmentation boundaries
- **`weight_depth`**: Influence of depth map gradients
- **`weight_canny`**: Influence of Canny edge detection
- **`weight_hed`**: Influence of HED edge detection

Set weights to 0.0 to disable specific maps. Higher weights give more influence.

### Advanced Parameters
- **`seed_radius`**: Radius of initial seed placement (pixels)
- **`max_steps`**: Maximum number of growth iterations
- **`stability_threshold`**: Steps before a region becomes "stable" (inactive)
- **`processing_resolution`**: Internal processing resolution (images scaled to this size)

## Testing

A test script is provided (`test_organic_fill.py`) that validates the algorithm with synthetic data. Run it to verify the fixes are working:

```bash
python test_organic_fill.py
```

Expected output should show successful growth (e.g., "SUCCESS: 12295 / 16384 pixels filled (75.04%)"). 