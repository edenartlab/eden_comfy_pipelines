#!/usr/bin/env python3
"""
Simple test script for OrganicFillNode to debug growth issues.
Creates synthetic test data to isolate the problem.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the current directory to Python path so we can import the module
sys.path.append(os.path.dirname(__file__))

from fill_image_mask import OrganicFillNode, FillNodeConfig

def create_test_data(size=256):
    """Create simple synthetic test data for debugging."""
    
    # Create a simple gradient image (BHWC format)
    H, W = size, size
    input_image = torch.zeros((1, H, W, 3), dtype=torch.float32)
    
    # Create a horizontal gradient from black to white
    for x in range(W):
        input_image[0, :, x, :] = x / W
    
    # Create seed locations mask - single point in center
    seed_locations = torch.zeros((1, H, W), dtype=torch.float32)
    seed_locations[0, H//2, W//2] = 1.0
    
    print(f"Created test data: input_image shape {input_image.shape}, seed_locations shape {seed_locations.shape}")
    print(f"Seed location: ({H//2}, {W//2})")
    
    return input_image, seed_locations

def run_debug_test():
    """Run a simple test to debug the organic fill algorithm."""
    
    print("=" * 60)
    print("ORGANIC FILL DEBUG TEST")
    print("=" * 60)
    
    # Create test data
    input_image, seed_locations = create_test_data(size=128)  # Smaller size for faster testing
    
    # Create node with debug-friendly configuration
    node = OrganicFillNode()
    
    # Test with very permissive settings
    params = {
        "input_image": input_image,
        "seed_locations": seed_locations,
        "fill_mask": None,  # Will use full image
        "SAM_map": None,
        "depth_map": None, 
        "canny_map": None,
        "hed_map": None,
        "max_steps": 50,  # Limit steps for quick test
        "growth_threshold": 0.1,  # Very low threshold
        "barrier_jump_power": 0.8,  # High barrier jumping
        "weight_lab": 1.0,  # Only use LAB gradient
        "weight_sam": 0.0,
        "weight_depth": 0.0,
        "weight_canny": 0.0,
        "weight_hed": 0.0,
        "seed": 42,
        "processing_resolution": 128,  # No downsampling
        "n_frames": 10,  # Fewer frames for testing
    }
    
    print("\nRunning organic fill with debug parameters...")
    print(f"Parameters: max_steps={params['max_steps']}, growth_threshold={params['growth_threshold']}")
    print(f"barrier_jump_power={params['barrier_jump_power']}, processing_resolution={params['processing_resolution']}")
    
    try:
        # Run the organic fill
        final_mask, frames_preview, grow_prob_map, overlayed_preview = node.execute(**params)
        
        print(f"\nResults:")
        print(f"  Final mask shape: {final_mask.shape}")
        print(f"  Final mask fill ratio: {final_mask.mean().item():.4f}")
        print(f"  Frames preview shape: {frames_preview.shape}")
        print(f"  Growth prob map range: [{grow_prob_map.min().item():.4f}, {grow_prob_map.max().item():.4f}]")
        
        # Check if any growth occurred
        if final_mask.max().item() > 0:
            filled_pixels = torch.sum(final_mask > 0.5).item()
            total_pixels = final_mask.numel()
            print(f"  SUCCESS: {filled_pixels} / {total_pixels} pixels filled ({100*filled_pixels/total_pixels:.2f}%)")
            
            # Save debug images
            os.makedirs("debug_output", exist_ok=True)
            
            # Save final mask
            mask_pil = Image.fromarray((final_mask[0].cpu().numpy() * 255).astype(np.uint8), mode='L')
            mask_pil.save("debug_output/final_mask.png")
            
            # Save growth probability map
            prob_pil = Image.fromarray((grow_prob_map[0].cpu().numpy() * 255).astype(np.uint8), mode='L')
            prob_pil.save("debug_output/growth_prob.png")
            
            print(f"  Debug images saved to debug_output/")
            
        else:
            print(f"  FAILURE: No growth occurred (final mask is all zeros)")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_debug_test()
    
    if success:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("Check the console output above for detailed debugging information.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Test failed - see error details above.")
        print("=" * 60)
        sys.exit(1) 