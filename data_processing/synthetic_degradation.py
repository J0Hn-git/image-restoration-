import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import cv2

# basic image handling funtions (loading the image).
def load_image(image_path):
    """Loads an image, converts to RGB, float32, and normalize to [0,1]."""
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        return img_np 
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None    

# saving the image after downscaling and stroing it in save_path.
def save_image(image_np, save_path):
    """Saves a float32 Numpy array as an image."""
    try:
        img_np_scaled = (image_np * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(img_np_scaled, 'RGB')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        img_pil.save(save_path)
        print(f"Image saved to: {save_path}")
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")   

# adding gaussian noise to image to make it noisy.
def add_gaussian_noise(image_np, mean = 0.0, std_dev = 0.02):
    noise = np.random.normal(mean, std_dev, image_np.shape).astype(np.float32)
    noisy_image = image_np + noise
    return np.clip(noisy_image, 0.0, 1.0)

# 
    


