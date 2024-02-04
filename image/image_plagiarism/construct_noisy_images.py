import os

import cv2

from add_noise_images import add_gaussian_noise, add_salt_and_pepper_noise

def create_noisy_images(folder_path: str):
    original_images = {}
    noisy_images = {}
    for i, image in enumerate(os.listdir(folder_path)):
        original_image = cv2.imread(image)
        original_images[i] = image
        gaussian_image = add_gaussian_noise(image)
        salt_pepper_image = add_salt_and_pepper_noise(image)
        noisy_images[i]['gaussian_noise'] = gaussian_image
        noisy_images[i]['salt_pepper_noise'] = salt_pepper_image