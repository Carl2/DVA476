#!/usr/bin/env python
from pathlib import Path
from pdb import set_trace

import cv2
import imageio.v3 as iw
import numpy as np
from pymonad.maybe import Just, Maybe, Nothing


def read_image(img: Path) -> Maybe[dict]:
    """
    Read an image from the specified path and return it wrapped in a Maybe monad.

    Args:
        img: Path object pointing to the image file to be read

    Returns:
        Maybe[dict]: Just containing a dict with 'raw' key and numpy array value if successful,
                     or Maybe with error message if reading fails
    """
    try:
        np_img = iw.imread(img)
        return Just({"raw": np_img,"file": img})
    except Exception as e:
        return Maybe(value=f"Error reading img {img}: {e}", monoid=False)


def transform_grayscale(feature_key:str)->callable:
    def do_transform_grayscale(img_features: dict)->Maybe:
        img = img_features.get(feature_key, None)
        if img is not None:
            img_features['gray'] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return Just(img_features)
        else:
            return Maybe(value=f"No image with key: {feature_key} exists in feature map")
    return do_transform_grayscale


def edge_detection(feature_key: str, low: int, high: int):
    def do_edge_detection(img_features: dict):
        img = img_features.get(feature_key, None)
        if img is not None:
            img_features['edges'] = cv2.Canny(img, low, high)
            return Just(img_features)
        else:
            return Maybe(value=f"Unable to find image key {feature_key}")
    return do_edge_detection






def color_histogram(raw_img_key: str, gray_img_key:str,  number_of_bins: int) -> callable:
    """
    Creates a function that calculates color histograms for BGR channels.

    Args:
        img_key: The key to retrieve the image from img_features dictionary

    Returns:
        A function that processes img_features and adds BGR histograms to it
    """
    def do_color_histogram(img_features: dict) -> Maybe:
        """
        Calculates BGR color histograms and adds them to img_features.

        Args:
            img_features: Dictionary containing image data and features

        Returns:
            Maybe object containing updated img_features or error message
        """
        img = img_features.get(raw_img_key, None)
        gray = img_features.get(gray_img_key, None)
        if img is not None and gray is not None:
            b, g, r = cv2.split(img)
            # hist_b = cv2.calcHist([b], [0], None, [number_of_bins], [0, 256])
            # hist_g = cv2.calcHist([g], [0], None, [number_of_bins], [0, 256])
            # hist_r = cv2.calcHist([r], [0], None, [number_of_bins], [0, 256])
            # hist_gray = cv2.calcHist(gray, [0], None, [number_of_bins], [0, 256])

            img_features["meta"] = {
                'Blue': cv2.calcHist([b], [0], None, [number_of_bins], [0, 256]),
                'Green': cv2.calcHist([g], [0], None, [number_of_bins], [0, 256]),
                'Red': cv2.calcHist([r], [0], None, [number_of_bins], [0, 256]),
                'Gray': cv2.calcHist(gray, [0], None, [number_of_bins], [0, 256])
            }

            # img_features["meta"]["Blue"] = hist_b
            # img_features["meta"]["Green"] = hist_g
            # img_features["meta"]["Red"] = hist_r
            # img_features["meta"]["Gray"] = hist_gray
            return Just(img_features)
        else:
            return Maybe(value=f"Unable to find image key {raw_img_key} or {gray_img_key}"
                         , monoid=False)

    return do_color_histogram

def extract_hog_descriptor(image_key: str,*,
                           win_size: tuple = (128, 128),
                           block_size: tuple = (16, 16),
                           block_stride: tuple = (8, 8),
                           cell_size: tuple = (8, 8),
                           nbins: int = 9) -> callable:
    """
    Creates a HOG (Histogram of Oriented Gradients) descriptor extractor function.

    Args:
        win_size: tuple (width, height) - Detection window size
        block_size: tuple (width, height) - Block size in pixels
        block_stride: tuple (width, height) - Block stride (overlap) in pixels
        cell_size: tuple (width, height) - Cell size in pixels
        nbins: int - Number of histogram bins for gradient orientation

    Returns:
        callable: A function that extracts HOG descriptors from image features
    """

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    def do_extract_hog_descriptor(image_features)->Maybe:
        """
        Extract HOG descriptor from image features.

        Args:
            image_features: Input image or features

        Returns:
            Maybe: Just(descriptor) if successful, Nothing() otherwise
        """
        gray_image = image_features.get(image_key,None)
        if gray_image is not None:

            descriptor = hog.compute(gray_image)
            image_features["HOG"] = descriptor
            return Just(image_features)
        else:
            return Maybe(value=f"Unable to find image key {image_key}"
                         , monoid=False)


    return do_extract_hog_descriptor


def extract_statistical_features(image_key: str = "gray") -> callable:
    """
    Extract statistical features from grayscale image.

    Features extracted:
    - Mean, Std Dev, Variance
    - Min, Max, Range
    - Percentiles (25th, 50th, 75th)
    """
    def do_extract_statistical_features(img_features: dict) -> Maybe:

        img = img_features.get(image_key, None)
        if img is None:
            return Maybe(value=f"Unable to find image key {image_key}", monoid=False)

        # Flatten image for statistical calculations
        pixels = img.flatten().astype(np.float32)

        img_features["stats"] = {
            # Central tendency
            'mean': np.mean(pixels),
            'median': np.median(pixels),

            # Dispersion
            'std': np.std(pixels),
            'variance': np.var(pixels),
            'range': np.ptp(pixels),  # max - min

            # Shape of distribution Lets see if this is necessary
            #'skewness': skew(pixels),
            #'kurtosis': kurtosis(pixels),

            # Extremes
            'min': np.min(pixels),
            'max': np.max(pixels),

            # Percentiles
            'percentile_25': np.percentile(pixels, 25),
            'percentile_75': np.percentile(pixels, 75),
            'iqr': np.percentile(pixels, 75) - np.percentile(pixels, 25),

            # Information theory
            #'entropy': shannon_entropy(img),

            # Coefficient of variation (normalized std)
            'cv': np.std(pixels) / (np.mean(pixels) + 1e-7)
        }

        return Just(img_features)

    return do_extract_statistical_features


def create_feature_vector(*,
             stats:bool = True,
             histogram:bool = True,
             hog:bool = True,
             edges:bool = True) -> callable:

    """Create a feature vector and convert to float32
    Total Feature Dimensions:
    - Stats: 11
    - Histograms: 120 (30 bins × 4 channels)
    - HOG: 8,100
    - Edge histogram: 30
    - Total: 8,261 features """
    def do_create_feature_vector(img_features: dict):
        feature_list = []

        if stats:
            stats_vals = img_features['stats']
            stat_values = [
                stats_vals['mean'], stats_vals['median'], stats_vals['std'],
                stats_vals['variance'], stats_vals['range'], stats_vals['min'],
                stats_vals['max'], stats_vals['percentile_25'],
                stats_vals['percentile_75'], stats_vals['iqr'], stats_vals['cv']
            ]
            feature_list.append(np.array(stat_values))
        if histogram:
            meta = img_features['meta']
            for channel in ['Blue', 'Green', 'Red', 'Gray']:
                feature_list.append(meta[channel].flatten())
        if hog:
            feature_list.append(img_features['HOG'].flatten())

        if edges:
            edges_vals = img_features['edges']
            edge_hist, _ = np.histogram(edges_vals, bins=30, range=(0, 256))
            feature_list.append(edge_hist)

        feature_vector = np.concatenate(feature_list)
        return feature_vector.astype(np.float32)
    return do_create_feature_vector
