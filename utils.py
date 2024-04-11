import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.fftpack import dct


def reconstruct_image_from_patches(patches, image_size, patch_size):
    """
    Reconstructs an image from patches.

    Args:

        patches (numpy.ndarray): The patches of the image.
        image_size (tuple): The size of the original image.
        patch_size (tuple): The size of each patch.

    Returns:
        numpy.ndarray: The reconstructed image.
    """

    num_patches_row = image_size[0] // patch_size[0]
    num_patches_col = image_size[1] // patch_size[1]
    # Reshape patches array to 3D array
    patches_3d = patches.reshape((num_patches_row, num_patches_col, patch_size[0], patch_size[1]))

    # Concatenate patches along rows and columns to reconstruct the image

    reconstructed_image = np.concatenate(
        [np.concatenate(patches_3d[i], axis=1) for i in range(num_patches_row)], axis=0
    )
    return reconstructed_image

def show_with_diff(image, reference, title=None):
    """
    Helper function to display denoising.

    Parameters:
    image (numpy.ndarray): The image to be displayed.
    reference (numpy.ndarray): The reference image for comparison.
    title (str): The title of the plot.

    Returns:
    None
    """
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image,cmap='gray')
    plt.subplot(1, 2, 2)
    difference = image - reference
    norm = np.sqrt(np.sum(difference**2))/image.shape[0] / image.shape[1]
    plt.title(f"Difference (norm/pixel): {np.round(norm,3)})")
    plt.imshow(difference,cmap='gray')
    if title is not None: 
        plt.suptitle(title, size=16)
    

# Function to divide image into blocks and form measurement vectors
def extract_blocks(image, block_size=(8, 8)):
    """
    Extracts blocks from an image.

    Args:
        image (ndarray): The input image.
        block_size (tuple, optional): The size of the blocks. Defaults to (8, 8).

    Returns:
        ndarray: An array of extracted blocks.
    """
    patches = extract_patches_2d(image, block_size)
    return patches.reshape(patches.shape[0], -1)


def generate_data(image, K=6000, block_size=(8,8)):
    """
    Generates a random subset of data from the given image.

    Parameters:
    image (numpy.ndarray): The input image.
    K (int): The number of blocks to select randomly from the image. Default is 6000.
    block_size (tuple): The size of each block. Default is (8, 8).

    Returns:
    numpy.ndarray: The randomly selected subset of data.

    """
    blocks = extract_blocks(image, block_size)
    data = blocks[np.random.choice(blocks.shape[0], K, replace=False), :]
    return data


def SNR(noisy_img, img):
    """Calculate the Signal-to-Noise Ratio (SNR) between two images.
    
        Args:
        noisy_img (numpy.ndarray): The noisy image.
        img (numpy.ndarray): The original image.
        
        Returns:
        float: The SNR value in decibels.    
    """
    # Calculate the noise
    noise = noisy_img - img
    
    # Calculate signal to noise ratio (in dB)
    snr = 20 * np.log10(np.sqrt(np.sum(img**2)) / np.sqrt(np.sum(noise**2)))
    
    return snr

import numpy as np

def RMSE(X, Y):
    """Calculate the Root Mean Squared Error (RMSE) between two matrices.

    Args:
        X (numpy.ndarray): The first image.
        Y (numpy.ndarray): The second image.

    Returns:
        float: The RMSE value.
    """
    # Get the size of X
    M, N = X.shape
    
    # Calculate RMSE
    rmse = np.sqrt(np.sum((X - Y)**2) / (M * N))
    
    return rmse


def PSNR(ground_truth, noisy_img):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        ground_truth (numpy.ndarray): The ground truth image.
        noisy_img (numpy.ndarray): The noisy image.

    Returns:
        float: The PSNR value in decibels.
    """
    # Calculate the MSE
    mse = np.mean((ground_truth - noisy_img) ** 2)
    
    # Calculate the PSNR
    psnr = 10 * np.log10(255**2 / mse)
    
    return psnr


def get_model_size(model):
    """
    Calculates the size of a PyTorch model in megabytes.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        float: The size of the model in megabytes.
    """
    total_params = 0
    for param_name, param in model.named_parameters():
        total_params += torch.numel(param)
    total_size = total_params * 4 / (1024 ** 2)  # Convert to megabytes
    return total_size


def normalize_image(image):
    """
    Normalize an image array to have values between 0 and 1.

    Parameters:
        image (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Normalized image array.
    """
    # Normalize the values between 0 and 1
    normalized_img = (image - image.min()) / (image.max() - image.min()) * 255
    
    return normalized_img


def compute_metrics(original_image, denoised_image):
    """Compute RMSE and PSNR metrics between original and denoised images.

    Args:
        original_image (numpy.ndarray): The original image.
        denoised_image (numpy.ndarray): The denoised image.

    Returns:
        rmse_value (float): The Root Mean Squared Error (RMSE) value. Lower is better
        psnr_value (float): The Peak Signal-to-Noise Ratio (PSNR) value. Higher is better
    """
    # Ensure both images have the same dimensions
    original_image = cv2.resize(original_image, (denoised_image.shape[1], denoised_image.shape[0]))

    # Compute SSIM
    rmse_value = RMSE(original_image, denoised_image)

    # Compute PSNR
    psnr_value = PSNR(original_image, denoised_image)

    return rmse_value, psnr_value

def estimate_noise_level(image, block_size):
    """Estimates the noise level in an image using block-based discrete cosine transform (DCT).

    Args:
        image (ndarray): The input image.
        block_size (int): The size of the blocks used for the DCT.

    Returns:
        ndarray: A 2D array containing the estimated noise levels for each block in the image.
    """
    image = np.array(image)
    rows, cols = image.shape
    noise_levels = np.zeros((rows//block_size, cols//block_size))
    
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            high_freq_coeffs = np.abs(dct_block) > 1
            noise_levels[i//block_size, j//block_size] = np.mean(dct_block[high_freq_coeffs])
    
    return noise_levels

def add_gaussian_noise(image, std_dev):
    """
    Add Gaussian noise to an image.

    Args:
        image (numpy.ndarray): The input image.
        std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: The noisy image.
    """
    noise = np.random.normal(scale=std_dev, size=image.shape).astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image