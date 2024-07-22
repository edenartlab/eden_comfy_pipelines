import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import segmentation, filters, morphology, measure
from scipy import ndimage


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_images(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess RGB and depth images.
    
    Args:
        rgb_path (str): Path to the RGB image file.
        depth_path (str): Path to the depth image file.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized RGB and depth images.
    """
    try:
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if rgb_image is None or depth_image is None:
            raise ValueError("Failed to load images")
        
        if rgb_image.shape[:2] != depth_image.shape:
            raise ValueError("RGB and depth images must have the same dimensions")
        
        normalized_rgb = normalize_image(rgb_image)
        normalized_depth = normalize_image(depth_image)
        
        return normalized_rgb, normalized_depth
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_images: {str(e)}")
        raise

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Normalized image.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def initial_depth_segmentation(depth_map: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Perform initial segmentation based on depth information.
    
    Args:
        depth_map (np.ndarray): Normalized depth map.
        n_clusters (int): Number of clusters for K-means.
    
    Returns:
        np.ndarray: Initial depth-based segmentation.
    """
    try:
        depth_flat = depth_map.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        depth_clusters = kmeans.fit_predict(depth_flat)
        return depth_clusters.reshape(depth_map.shape)
    except Exception as e:
        logger.error(f"Error in initial_depth_segmentation: {str(e)}")
        raise


def visualize_intermediate(image, title):
    """
    Visualize and save an intermediate result.
    
    Args:
        image (np.ndarray): Image to visualize.
        title (str): Title for the plot.
    """

    n_unique_values = len(np.unique(image))

    plt.figure(figsize=(10, 8))
    plt.imshow(image) #, cmap='nipy_spectral')
    plt.title(title + f" ({n_unique_values} segments)")
    plt.axis('off')
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def auto_segment(rgb_image: np.ndarray, depth_map: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Main function to perform automatic segmentation.
    
    Args:
        rgb_image (np.ndarray): Normalized RGB image.
        depth_map (np.ndarray): Normalized depth map.
        n_segments (int): Desired number of segments.
    
    Returns:
        np.ndarray: Final segmentation.
    """
    try:
        logger.info("Starting automatic segmentation")
        logger.info(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        logger.info(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
        logger.info(f"Desired number of segments: {n_segments}")
        
        # Initial depth segmentation
        depth_segments = initial_depth_segmentation(depth_map, n_clusters=min(n_segments, 10))
        logger.info("Initial depth segmentation completed")
        visualize_intermediate(depth_segments, "01 - Initial Depth Segmentation")
        
        # Color refinement
        color_refined = refine_with_color(rgb_image, depth_segments, n_segments=n_segments)
        logger.info("Color refinement completed")
        visualize_intermediate(color_refined, "02 - Color Refined Segmentation")
        
        # Edge-based refinement
        edge_refined = edge_based_refinement(rgb_image, color_refined)
        logger.info("Edge-based refinement completed")
        visualize_intermediate(edge_refined, "03 - Edge Refined Segmentation")
        
        # Region growing
        region_grown = region_growing(rgb_image, depth_map, edge_refined)
        logger.info("Region growing completed")
        visualize_intermediate(region_grown, "04 - Region Grown Segmentation")
        
        # Post-processing
        final_segments = post_processing(region_grown, n_segments=n_segments)
        logger.info("Post-processing completed")
        visualize_intermediate(final_segments, "05 - Final Segmentation")
        
        evaluation_score = evaluate_segmentation(final_segments)
        logger.info(f"Segmentation completed. Evaluation score: {evaluation_score:.4f}")
        
        return final_segments
    except Exception as e:
        logger.error(f"Error in auto_segment: {str(e)}")
        raise

def edge_based_refinement(rgb_image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Refine segmentation using edge detection.
    
    Args:
        rgb_image (np.ndarray): Normalized RGB image.
        segments (np.ndarray): Input segmentation.
    
    Returns:
        np.ndarray: Edge-refined segmentation.
    """
    try:
        # Convert the image to 8-bit unsigned integer
        rgb_image_8bit = (rgb_image * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(rgb_image_8bit, cv2.COLOR_RGB2GRAY)
        
        # Compute edges
        edges = filters.sobel(gray_image)
        
        # Normalize edges to 0-1 range
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        
        return segmentation.watershed(edges, markers=segments, mask=edges < 0.1)
    except Exception as e:
        logger.error(f"Error in edge_based_refinement: {str(e)}")
        raise

def region_growing(rgb_image: np.ndarray, depth_map: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Perform region growing based on color and depth similarity.
    
    Args:
        rgb_image (np.ndarray): Normalized RGB image.
        depth_map (np.ndarray): Normalized depth map.
        segments (np.ndarray): Input segmentation.
    
    Returns:
        np.ndarray: Region-grown segmentation.
    """
    try:
        def color_distance(c1, c2):
            return np.sqrt(np.sum((c1 - c2) ** 2))
        
        def depth_distance(d1, d2):
            return abs(d1 - d2)
        
        grown_segments = segments.copy()
        height, width = segments.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for y in range(height):
            for x in range(width):
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if segments[y, x] != segments[ny, nx]:
                            color_dist = color_distance(rgb_image[y, x], rgb_image[ny, nx])
                            depth_dist = depth_distance(depth_map[y, x], depth_map[ny, nx])
                            if color_dist < 0.1 and depth_dist < 0.05:
                                grown_segments[ny, nx] = grown_segments[y, x]
        
        return grown_segments
    except Exception as e:
        logger.error(f"Error in region_growing: {str(e)}")
        raise

def refine_with_color(rgb_image: np.ndarray, depth_segments: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Refine segmentation using color information.
    
    Args:
        rgb_image (np.ndarray): Normalized RGB image.
        depth_segments (np.ndarray): Initial depth-based segmentation.
        n_segments (int): Desired number of segments.
    
    Returns:
        np.ndarray: Refined segmentation.
    """
    try:
        refined_segments = segmentation.slic(rgb_image, n_segments=n_segments, compactness=10, start_label=1)
        return refined_segments
    except Exception as e:
        logger.error(f"Error in refine_with_color: {str(e)}")
        raise

import traceback

def segment_merging(segments: np.ndarray) -> np.ndarray:
    """
    Merge the two most similar adjacent segments based on size and centroid proximity.
    """
    try:
        unique_labels = np.unique(segments)
        logger.info(f"Unique labels: {unique_labels}")
        
        # Compute region properties
        props = measure.regionprops(segments)
        logger.info(f"Number of regions: {len(props)}")
        
        # Create dictionaries for centroids and areas
        centroids = {prop.label: prop.centroid for prop in props if prop.label != 0}
        areas = {prop.label: prop.area for prop in props if prop.label != 0}
        
        logger.info(f"Centroids: {centroids}")
        logger.info(f"Areas: {areas}")
        
        # Find adjacent segments
        def are_adjacent(label1, label2):
            region1 = segments == label1
            dilated = ndimage.binary_dilation(region1)
            return np.any(dilated & (segments == label2))
        
        # Find the pair of adjacent segments with the most similar size and closest centroids
        min_score = float('inf')
        merge_pair = None
        for i, label1 in enumerate(unique_labels):
            if label1 == 0:
                continue
            for label2 in unique_labels[i+1:]:
                if label2 == 0:
                    continue
                if are_adjacent(label1, label2):
                    size_diff = abs(areas[label1] - areas[label2]) / max(areas[label1], areas[label2])
                    distance = np.linalg.norm(np.array(centroids[label1]) - np.array(centroids[label2]))
                    score = size_diff + distance  # You can adjust the weighting of these factors
                    logger.debug(f"Labels {label1} and {label2}: score = {score}")
                    if score < min_score:
                        min_score = score
                        merge_pair = (label1, label2)
        
        logger.info(f"Merge pair: {merge_pair}")
        
        # Merge the selected pair
        if merge_pair:
            label1, label2 = merge_pair
            segments[segments == label2] = label1
            logger.info(f"Merged label {label2} into {label1}")
        else:
            logger.warning("No suitable merge pair found")
        
        return segments
    except Exception as e:
        logger.error(f"Error in segment_merging: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def post_processing(segments: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Apply post-processing to clean up the segmentation and ensure the desired number of segments.
    
    Args:
        segments (np.ndarray): Input segmentation.
        n_segments (int): Desired number of segments.
    
    Returns:
        np.ndarray: Post-processed segmentation.
    """
    try:
        # Remove small objects
        cleaned = morphology.remove_small_objects(segments, min_size=100)
        
        current_segments = len(np.unique(cleaned))
        logger.info(f"Number of segments after removing small objects: {current_segments}")
        
        # Merge segments if there are too many
        merge_count = 0
        while current_segments > n_segments:
            logger.info(f"Merging segments: current {current_segments}, target {n_segments}")
            cleaned = segment_merging(cleaned)
            new_current_segments = len(np.unique(cleaned))
            logger.info(f"After merging: {new_current_segments} segments")
            if new_current_segments == current_segments:
                logger.warning("No segments were merged in this iteration. Breaking loop.")
                break
            current_segments = new_current_segments
            merge_count += 1
            if merge_count > 100:  # Safeguard against infinite loops
                logger.warning("Reached maximum number of merge iterations. Breaking loop.")
                break
        
        # Split segments if there are too few
        split_count = 0
        while current_segments < n_segments:
            logger.info(f"Splitting segments: current {current_segments}, target {n_segments}")
            cleaned = segment_splitting(cleaned)
            new_current_segments = len(np.unique(cleaned))
            logger.info(f"After splitting: {new_current_segments} segments")
            if new_current_segments == current_segments:
                logger.warning("No segments were split in this iteration. Breaking loop.")
                break
            current_segments = new_current_segments
            split_count += 1
            if split_count > 100:  # Safeguard against infinite loops
                logger.warning("Reached maximum number of split iterations. Breaking loop.")
                break
        
        logger.info(f"Final number of segments: {current_segments}")
        return cleaned
    except Exception as e:
        logger.error(f"Error in post_processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

import numpy as np
from scipy import ndimage
from skimage import measure


def segment_splitting(segments: np.ndarray) -> np.ndarray:
    """
    Split the largest segment using K-means clustering.
    """
    from sklearn.cluster import KMeans
    
    unique_labels = np.unique(segments)
    
    # Find the largest segment
    largest_label = max(unique_labels, key=lambda l: np.sum(segments == l))
    
    # Get the coordinates of pixels in the largest segment
    y, x = np.where(segments == largest_label)
    coords = np.column_stack((y, x))
    
    # Apply K-means clustering to split the segment into two
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(coords)
    
    # Assign new labels to the split segments
    new_label = segments.max() + 1
    mask = segments == largest_label
    segments[mask] = np.where(cluster_labels == 0, largest_label, new_label)[mask[y, x]]
    
    return segments
if __name__ == "__main__":
    try:
        rgb_path = "image.png"
        depth_path = "depth.png"
        n_segments = 3  # Desired number of segments
        
        rgb_image, depth_map = load_and_preprocess_images(rgb_path, depth_path)
        segmentation_result = auto_segment(rgb_image, depth_map, n_segments)
        
        # Visualize the final result
        plt.figure(figsize=(12, 10))
        plt.imshow(segmentation_result, cmap='nipy_spectral')
        plt.title("Final Segmentation Result")
        plt.axis('off')
        plt.savefig("Final_Segmentation_Result.png")
        plt.savefig('plot.jpg')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")