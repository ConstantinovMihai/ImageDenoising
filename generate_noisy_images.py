
"""
This file takes a random image from the BDSS300 dataset 
and generates noisy images to be used by the denoising algorithms
"""

import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
import os
import random
from utils import add_gaussian_noise

def generate_noisy_images(name, std_devs=[5, 10, 25], image_folder="BSDS300-images/BSDS300/images/train/", plot=False):
    """Generate noisy images by adding Gaussian noise to random images from a folder.

    Args:
        name (str): The name of the generated images.
        std_devs (list, optional): A list of standard deviations for the Gaussian noise. Defaults to [5, 10, 25].
        image_folder (str, optional): The folder path containing the images to add noise to. Defaults to "BSDS300-images/BSDS300/images/train/".
        plot (bool, optional): Whether to display the noisy images. Defaults to False.
    """
    # select a random image from the BSDS300-images/BSDS300/images/train folder
    # and add std_devs noise to it using add_gaussian_noise
    # and place them in the Data/ folder
    
    image_files = os.listdir(image_folder)
    random_image_name = random.choice(image_files)
    image_path = os.path.join(image_folder, random_image_name)
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Create a folder if it doesn't exist
    output_folder = "Data/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # add the clean image
    output_filename = f"{name}_original.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, original_image)

    # Iterate over standard deviations
    for std_dev in std_devs:
        # Add Gaussian noise to the image
        noisy_image = add_gaussian_noise(original_image, std_dev)
        
        # Save the noisy image
        output_filename = f"{name}_std_{std_dev}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, noisy_image)
        
        # Optionally, display the noisy image
        if plot:
            plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Noisy Image (Std Dev: {std_dev})")
            plt.axis("off")
            plt.show()


# Function to divide image into blocks and form measurement vectors
def extract_blocks(image, block_size=(8, 8)):
    """
    Extracts blocks from an image.

    Args:
        image (ndarray): The input image.
        block_size (tuple, optional): The size of the blocks. Defaults to (8, 8).

    Returns:
        ndarray: An array of blocks extracted from the image.
    """
    patches = extract_patches_2d(image, block_size)
    return patches.reshape(patches.shape[0], -1)


if __name__ == "__main__":
    # load images
    generate_noisy_images("example_image", std_devs=[5, 10, 25])
    
    # image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    # image3 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

    # # List of images
    # images = [image2]

    # # Standard deviations for Gaussian noise
    # std_devs = [5, 10, 25]

    # # Iterate over images
    # for idx, image in enumerate(images):
    #     print(f"Processing image {idx+1}")
        
    #     # Iterate over standard deviations
    #     for std_dev in std_devs:
    #         print(f"Adding Gaussian noise with std_dev={std_dev}")
            
    #         # Add Gaussian noise to the image
    #         noisy_image = add_gaussian_noise(image, std_dev)
    #         cv2.imwrite(filename=f'noisy_image{idx}_noise_lvl{std_dev}.jpg', img=noisy_image)
            # plt.imshow(noisy_image, cmap='gray')
            # plt.show()
            # Extract blocks from the noisy image
            # blocks = extract_blocks(noisy_image)
            # print(len(blocks))
            # # Plot 60 random blocks
            # num_blocks = blocks.shape[0]
            # random_indices = np.random.choice(num_blocks, 60, replace=False)
            # plt.figure(figsize=(10, 10))
            # for i, idx in enumerate(random_indices):
            #     plt.subplot(6, 10, i + 1)
            #     plt.imshow(blocks[idx].reshape(size_patch, size_patch), cmap='gray')  # Reshape the block to (8, 8)
            #     plt.axis('off')
            # plt.suptitle(f"Noisy Image {idx+1} - Std Dev {std_dev}")
            # plt.show()
