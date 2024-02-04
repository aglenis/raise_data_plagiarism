from typing import Union

import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: Union[int, float] = 0,
                       std: Union[int, float] = 25):
    """
    A method that adds gaussian noise to an image
    Args:
        image (np.ndaray): The array of the image
        mean (int, float): The mean of the noise
        std (int, float): The std of the noise

    Returns:
        np.ndarray
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def add_salt_and_pepper_noise(image: np.ndarray, noise_ratio: float=0.02):
    """
    A method that adds 'salt and pepper' noise
    Args:
        image (np.ndaray): The array of the image
        noise_ratio (float): ratio of noisy pixels

    Returns:
        np.ndarray
    """
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)

    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0]
        else:
            noisy_image[row, col] = [255, 255, 255]

    return noisy_image
