#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')    # 'TkAgg', 'Qt5Agg', or 'Gtk3Agg' are common
from pdb import set_trace

import cv2
import numpy as np


def display_images(images, titles=None, cmap='gray', figsize=(15, 5)):
    """
    Displays a list or a single image using matplotlib.

    Args:
        images (list or np.ndarray): A list of images (e.g., [img1, img2])
                                     or a single image (Height x Width, or Height x Width x Channels).
        titles (list or str, optional): A list of titles for each image,
                                        or a single title if 'images' is a single image.
                                        Defaults to None.
        cmap (str, optional): Colormap to use. Useful for grayscale images.
                              Common values: 'gray', 'viridis', 'jet'. Defaults to 'gray'.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (15, 5).
    """
    if not isinstance(images, list):
        images = [images]  # Make it a list even if it's a single image

    num_images = len(images)
    # Determine number of rows and columns for optimal display
    # Aim for a roughly square layout, or a single row if few images
    if num_images == 1:
        num_cols = 1
        num_rows = 1
        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes_list = [ax] # Ensure 'ax' is iterable for consistent loop
    else:
        num_cols = min(num_images, 4) # Max 4 columns, adjust as needed
        num_rows = int(np.ceil(num_images / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes_list = axes.flatten() # Flatten 2D array of axes for easy iteration

    if titles is not None and not isinstance(titles, list):
        titles = [titles] * num_images # Duplicate single title for all images

    for i, img in enumerate(images):
        ax = axes_list[i]

        # Handle different image dimensions gracefully
        if img.ndim == 2:  # Grayscale image (H x W)
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3:
            if img.shape[2] in [1, 3, 4]:  # Grayscale (H x W x 1) or RGB/RGBA (H x W x C)
                # Squeeze the last dimension if it's 1 (e.g., convert (H,W,1) to (H,W))
                if img.shape[2] == 1:
                    ax.imshow(img.squeeze(), cmap=cmap)
                else:
                    ax.imshow(img) # Matplotlib handles RGB/RGBA directly
            else: # Probably a feature map with many channels, like (H x W x N_features)
                  # In this case, we can only show the first channel or give a warning
                print(f"Warning: Image {i} has shape {img.shape}. Displaying first channel.")
                ax.imshow(img[:, :, 0], cmap=cmap) # Display first channel as grayscale
        else:
            print(f"Warning: Image {i} has unsupported dimensions: {img.shape}. Skipping.")
            continue

        ax.axis('off') # Hide axes for cleaner image display
        if titles and i < len(titles):
            ax.set_title(titles[i])

    # Turn off any unused subplots
    for j in range(num_images, len(axes_list)):
        axes_list[j].axis('off')

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()



def histogram_display(feature_dic: dict):
    """Displays the b,g,r histogram
    Each of the histogram is in the dictionary with key "Blue","Green","Red" respective.
    Make plot for each of them in the same plot window with the linecolor representing the name
    """
    plt.figure()

    color_map = {"Blue": "b", "Green": "g", "Red": "r", "Gray": "k"}
    #color_map = {"Blue": "b", "Green": "g", "Red": "r"}

    for color_name, color_code in color_map.items():
        if color_name in feature_dic:
            plt.plot(feature_dic[color_name], color=color_code, label=color_name)

    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("BGR Histogram")
    # plt.legend()
    # plt.show()


def visualize_hog(image_features: dict):
    """Visualize HOG descriptor with orientation arrows"""

    gray = image_features["gray"]
    hog_features = image_features["HOG"]

    # Extract cell_size from hog_features if available, otherwise use default
    if isinstance(hog_features, dict) and "cell_size" in hog_features:
        cell_size = hog_features["cell_size"]
    else:
        cell_size = (8, 8)  # Default cell size

    # Compute gradients for visualization
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)  # Horizontal gradient
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)  # Vertical gradient
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)  # Magnitude & angle

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Gradient magnitude
    axes[1].imshow(mag, cmap='jet')
    axes[1].set_title('Gradient Magnitude')
    axes[1].axis('off')

    # HOG visualization with arrows
    cell_w, cell_h = cell_size
    n_cells_x = gray.shape[1] // cell_w  # Number of cells horizontally
    n_cells_y = gray.shape[0] // cell_h  # Number of cells vertically

    # Create blank canvas
    hog_image = np.zeros_like(gray)

    # Draw dominant gradient direction in each cell
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Extract cell region
            cell_mag = mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_angle = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

            # Get dominant orientation (weighted by magnitude)
            dominant_angle = np.average(cell_angle, weights=cell_mag + 1e-5)
            dominant_mag = np.mean(cell_mag)

            # Draw arrow from center of cell
            center_x = j * cell_w + cell_w // 2
            center_y = i * cell_h + cell_h // 2

            # Calculate arrow endpoint
            rad = np.deg2rad(dominant_angle)
            length = dominant_mag * 0.5  # Scale for visibility
            end_x = int(center_x + length * np.cos(rad))
            end_y = int(center_y + length * np.sin(rad))

            # Draw line representing gradient direction
            cv2.arrowedLine(hog_image, (center_x, center_y),
                           (end_x, end_y), 255, 1, tipLength=0.3)

    axes[2].imshow(hog_image, cmap='gray')
    axes[2].set_title('HOG Visualization (Gradient Directions)')
    axes[2].axis('off')






def display_features(feature_dic:dict):
    images = []
    images.append(feature_dic['raw'])
    images.append(feature_dic['gray'])
    images.append(feature_dic['edges'])
    #set_trace()
    visualize_hog(feature_dic)
    histogram_display(feature_dic["meta"])
    display_images(images ,titles=["Raw image","Gray", "edges"])
