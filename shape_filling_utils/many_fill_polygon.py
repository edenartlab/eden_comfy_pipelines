#!/usr/bin/env python3
"""
Generate multiple variations of organic fill animations with randomized parameters.

This script takes an input directory of images and generates n variations by:
1. Randomly selecting an image from the directory
2. Sampling random parameters from predefined ranges
3. Running the organic fill algorithm with those parameters
"""

import random
import sys
from pathlib import Path
from typing import List
import numpy as np

from fill_polygon import FillConfig, StartPosition, create_animation


class ParameterRanges:
    """Define parameter ranges for random sampling"""

    # Target size range - favor 512 and 768 based on successful examples
    TARGET_SIZE_RANGE = [512, 512, 768]

    # Threshold value range - nudge toward successful 112-181 range
    THRESHOLD_VALUE_RANGE = (80, 192)

    # Seed radius range - favor mid-high values (6-13 in successful examples)
    SEED_RADIUS_RANGE = (3, 10)

    # Growth threshold range - nudge toward successful 0.44-0.79 range
    GROWTH_THRESHOLD_RANGE = (0.3, 0.8)

    # Noise range variations - favor ranges with 0.0 start like successful examples
    NOISE_RANGES = [
        (0.0, 0.8),
        (0.0, 1.2),
        (0.0, 2.0),
        (0.0, 1.0),
        (0.2, 1.0),
        (0.1, 1.5)
    ]

    # Position randomness range - successful examples show 0.34-0.93, nudge away from extremes
    POSITION_RANDOMNESS_RANGE = (0.0, 0.75)

    # Island connection radius range
    ISLAND_CONNECTION_RANGE = (0.0, 0.01)

    # Stability threshold range - successful examples show 11-28, slight nudge
    STABILITY_THRESHOLD_RANGE = (11, 30)

    # Saturation window range - successful examples show 13-24, slight nudge upward
    SATURATION_WINDOW_RANGE = (12, 25)

    # Saturation threshold range
    SATURATION_THRESHOLD_RANGE = (0.0001, 0.01)

    # Active region padding range
    ACTIVE_REGION_PADDING_RANGE = (1, 8)

    # FPS range
    FPS_RANGE = (24.0, 24.0)

    # All starting positions
    STARTING_POSITIONS = list(StartPosition)

    # Image inversion options - successful examples all use invert=1, weight toward True
    INVERT_INPUT_OPTIONS = [True, True, True, False]

    # Number of seeds range - successful examples show 2-3, exclude 1
    NUM_SEEDS_RANGE = (1, 2)

    # Color channel options - favor successful channels: red, random, luminance
    COLOR_CHANNEL_OPTIONS = ["red", "random", "luminance", "random", "green", "blue"]


def sample_random_config() -> FillConfig:
    """Sample a random FillConfig from the parameter ranges"""

    return FillConfig(
        target_size=random.choice(ParameterRanges.TARGET_SIZE_RANGE),
        threshold_value=random.randint(*ParameterRanges.THRESHOLD_VALUE_RANGE),
        seed_radius=random.randint(*ParameterRanges.SEED_RADIUS_RANGE),
        noise_range=random.choice(ParameterRanges.NOISE_RANGES),
        growth_threshold=random.uniform(*ParameterRanges.GROWTH_THRESHOLD_RANGE),
        stability_threshold=random.randint(*ParameterRanges.STABILITY_THRESHOLD_RANGE),
        saturation_window=random.randint(*ParameterRanges.SATURATION_WINDOW_RANGE),
        saturation_threshold=random.uniform(*ParameterRanges.SATURATION_THRESHOLD_RANGE),
        fps=random.uniform(*ParameterRanges.FPS_RANGE),
        starting_position=random.choice(ParameterRanges.STARTING_POSITIONS),
        position_randomness=random.uniform(*ParameterRanges.POSITION_RANDOMNESS_RANGE),
        island_connection_radius=random.uniform(*ParameterRanges.ISLAND_CONNECTION_RANGE),
        active_region_padding=random.randint(*ParameterRanges.ACTIVE_REGION_PADDING_RANGE),
        invert_input=random.choice(ParameterRanges.INVERT_INPUT_OPTIONS),
        num_seeds=random.randint(*ParameterRanges.NUM_SEEDS_RANGE),
        color_channel=random.choice(ParameterRanges.COLOR_CHANNEL_OPTIONS),
    )


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from the directory"""
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for ext in image_extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))

    return image_files


def generate_param_filename(config: FillConfig, base_name: str, output_dir: Path = None) -> str:
    """Generate filename with sampled parameters instead of variation number"""
    # Create parameter string with key parameters
    param_parts = [
        f"sz{config.target_size}",
        f"gt{config.growth_threshold:.2f}",
        f"pr{config.position_randomness:.2f}",
        f"sr{config.seed_radius}",
        f"pos{config.starting_position.value}",
        f"thr{config.threshold_value}",
        f"nr{config.noise_range[0]:.1f}-{config.noise_range[1]:.1f}",
        f"st{config.stability_threshold}",
        f"sw{config.saturation_window}",
        f"inv{1 if config.invert_input else 0}",
        f"ns{config.num_seeds}",
        f"ch{config.color_channel}"
    ]
    param_str = "_".join(param_parts)

    filename = f"{base_name}_{param_str}.mp4"

    if output_dir:
        return str(output_dir / filename)
    else:
        return filename


def generate_variations(input_dir: Path, n_variations: int = 100, output_dir: Path = None):
    """Generate n variations of organic fill animations"""

    # Get all image files
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Generating {n_variations} variations...")

    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(exist_ok=True)

    successful_generations = 0

    for i in range(n_variations):
        try:
            # Randomly select an image
            image_path = random.choice(image_files)

            # Sample random configuration
            config = sample_random_config()

            # Generate output path with parameters instead of variation number
            output_path = None
            if output_dir:
                base_name = image_path.stem
                output_path = generate_param_filename(config, base_name, output_dir)

            print(f"\nVariation {i+1}/{n_variations}:")
            print(f"  Image: {image_path.name}")
            print(f"  Target size: {config.target_size}")
            print(f"  Growth threshold: {config.growth_threshold:.2f}")
            print(f"  Position randomness: {config.position_randomness:.2f}")
            print(f"  Starting position: {config.starting_position.value}")
            print(f"  Seed radius: {config.seed_radius}")
            print(f"  Invert input: {config.invert_input}")
            print(f"  Number of seeds: {config.num_seeds}")
            print(f"  Color channel: {config.color_channel}")

            # Create the animation
            create_animation(image_path, output_path, config)
            successful_generations += 1

        except Exception as e:
            print(f"Error generating variation {i+1}: {e}")
            continue

    print(f"\nCompleted! Successfully generated {successful_generations}/{n_variations} variations.")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python many_fill_polygon.py <input_directory> [n_variations] [output_directory]")
        print("  input_directory: Directory containing input images")
        print("  n_variations: Number of variations to generate (default: 100)")
        print("  output_directory: Optional output directory (default: same as input)")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    # Parse number of variations
    n_variations = 100
    if len(sys.argv) >= 3:
        try:
            n_variations = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid number of variations: {sys.argv[2]}")
            sys.exit(1)

    # Parse output directory
    output_dir = None
    if len(sys.argv) >= 4:
        output_dir = Path(sys.argv[3])
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True)
            except Exception as e:
                print(f"Error creating output directory {output_dir}: {e}")
                sys.exit(1)

    # Set random seed for reproducibility (optional)
    import time
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    generate_variations(input_dir, n_variations, output_dir)


if __name__ == "__main__":
    main()