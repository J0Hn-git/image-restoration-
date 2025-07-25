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

# adding salt and pepper noise to the image.
def add_salt_pepper_noise(image_np, amount=0.01):
    s_p_image = np.copy(image_np)
    num_pixels = int(image_np.size * amount)
    coords_salt = [np.random.randint(0, i - 1, num_pixels) for i in image_np.shape]
    s_p_image[tuple(coords_salt)] = 1.0
    coords_pepper = [np.random.randint(0, i - 1, num_pixels) for i in image_np.shape]
    s_p_image[tuple(coords_pepper)] = 0.0
    return s_p_image

# applying gaussian blur.
def apply_gaussian_blur(image_np, kernal_size=3):
    if kernal_size % 2 == 0:
        kernal_size += 1              # ensures the kernal_size is odd.
    blurred_image = cv2.GaussianBlur(image_np, (kernal_size, kernal_size), 0)
    return blurred_image

# aplying motion blurring.
def apply_motion_blur(image_np, kernal_size=5, angle=45):
    kernal = np.zeros((kernal_size, kernal_size), dtype=np.float32)
    center = kernal_size // 2
    
    if angle == 0:
        kernal[center, :] = 1
    elif angle == 90:
        kernal[:, center] = 1
    else:
        for i in range(kernal_size):
            if angle == 45:
                kernal[i, i] = 1 # top left to bottom right.
            elif angle == 135:
                kernal[i, kernal_size - 1 - i] # top right to bottom left.
            else:
                x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
                y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
                if 0 <= x < kernal_size and 0 <= y < kernal_size:
                    kernal[x, y] = 1
    # Normalize kernal values so they sum to 1, preventing brightness changes.
    kernal = kernal / np.sum(kernal)
    
    # apply the kernal using 2d convilotion.
    motion_blurred_image = cv2.filter2D(image_np, -1, kernal)
    return motion_blurred_image     

# apply color fading.
def add_color_fading(image_np, factor = 0.7):
    # converts RGB to HSV color space.
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1 ] = np.clip(hsv_image[:, :, 1] * factor, 0.0, 1.0)     
    faded_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return faded_image   

# applying color cast.
def apply_color_cast(image_np, r_bias=0.0, g_bias=0.0, b_bias=0.0):
    bias = np.array([b_bias, g_bias, r_bias]).astype(np.float32)
    cast_image = image_np + bias
    return np.clip(cast_image, 0.0, 1.0)

# apply gamma correction.
def apply_gamma_correction(image_np, gamma=1.2):
    gamma_corrected_image = np.power(image_np, (1.0 / gamma))
    return np.clip(gamma_corrected_image, 0.0, 1.0)

# applying scratches to the image.
def add_scratches(image_np, num_scratches=3, line_width=1, intensity=0.1):
    """
    Adds simplified line scratches to the image.
    Note: Highly realistic scratches are very complex to generate procedurally.
    This is a basic simulation.
    """
    scratch_image = np.copy(image_np)
    h, w, _ = scratch_image.shape # Get image dimensions

    for _ in range(num_scratches):
        # Random start and end points for the line
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)

        # Random color for the scratch (black or white)
        scratch_color = random.choice([0.0, 1.0])
        color_tuple = (scratch_color, scratch_color, scratch_color) # RGB tuple for line color

        # Draw line on a temporary black image (mask)
        temp_mask = np.zeros_like(image_np) # Create a black image of the same size
        cv2.line(temp_mask, (x1, y1), (x2, y2), color_tuple, thickness=line_width) # Draw line on the mask

        # Blend the scratch mask with the image
        # This blending applies the scratch based on its color and intensity
        scratch_image = scratch_image * (1 - temp_mask * intensity) + temp_mask * scratch_color * intensity
        scratch_image = np.clip(scratch_image, 0.0, 1.0)
    return scratch_image

# add dust spots.
def add_dust_spots(image_np, num_spots=10, max_spot_size=2, intensity=0.1):
    """
    Adds simplified circular dust spots to the image.
    Note: Highly realistic dust is very complex to generate procedurally.
    This is a basic simulation.
    """
    dust_image = np.copy(image_np)
    h, w, _ = dust_image.shape

    for _ in range(num_spots):
        center_x, center_y = random.randint(0, w), random.randint(0, h) # Random center for the spot
        radius = random.randint(1, max_spot_size) # Random radius for the spot
        
        spot_color = random.choice([0.0, 1.0]) # Random color (black or white)
        color_tuple = (spot_color, spot_color, spot_color)

        temp_mask = np.zeros_like(image_np)
        cv2.circle(temp_mask, (center_x, center_y), radius, color_tuple, -1) # Draw a filled circle on the mask

        # Blend the spot mask with the image (same blending logic as scratches)
        dust_image = dust_image * (1 - temp_mask * intensity) + temp_mask * spot_color * intensity
        dust_image = np.clip(dust_image, 0.0, 1.0)
    return dust_image
         
# combine the random degradations.
def combine_random_degradations(image_np):
    """
    Applies a random combination of degradations to the image.
    You can customize which degradations are applied and their parameters.
    """
    degraded_image = np.copy(image_np)
    h, w, _ = degraded_image.shape

    # Define a list of degradation functions and their parameter ranges
    degradation_options = [
        (add_gaussian_noise, {'std_dev': random.uniform(0.01, 0.1)}),
        (add_salt_pepper_noise, {'amount': random.uniform(0.001, 0.01)}),
        (apply_gaussian_blur, {'kernel_size': random.choice([3, 5, 7])}),
        (apply_motion_blur, {'kernel_size': random.choice([5, 7, 9]), 'angle': random.choice([0, 45, 90, 135, 180])}),
        (add_color_fading, {'factor': random.uniform(0.5, 0.9)}),
        (apply_color_cast, {'r_bias': random.uniform(-0.1, 0.1), 'g_bias': random.uniform(-0.1, 0.1), 'b_bias': random.uniform(-0.1, 0.1)}),
        (apply_gamma_correction, {'gamma': random.uniform(0.8, 1.5)}),
        (add_scratches, {'num_scratches': random.randint(1, 5), 'line_width': random.randint(1, 2), 'intensity': random.uniform(0.05, 0.2)}),
        (add_dust_spots, {'num_spots': random.randint(5, 20), 'max_spot_size': random.randint(1, 3), 'intensity': random.uniform(0.05, 0.2)})
    ]

    # Randomly select a subset of degradations to apply (e.g., 3 to 7 degradations)
    num_degradations_to_apply = random.randint(3, len(degradation_options))
    selected_degradations = random.sample(degradation_options, num_degradations_to_apply)

    # Apply selected degradations in a random order
    random.shuffle(selected_degradations)
    for func, params in selected_degradations:
        degraded_image = func(degraded_image, **params) # Call the function with its random parameters
    
    return degraded_image      
        
    


