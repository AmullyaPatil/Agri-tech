import os
from PIL import Image
import numpy as np

def validate_image_dir(base_path):
    """Verify soil image directory structure"""
    required_folders = ['Black_Soil', 'Cinder_Soil', 'Laterite_Soil', 'Peat_Soil', 'Yellow_Soil']
    valid = True
    
    for folder in required_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder}")
            valid = False
        else:
            num_images = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
            if num_images < 10:
                print(f"Warning: {folder} has only {num_images} images (recommend 30+)")
    
    return valid

def load_sample_images(base_path, num_samples=3):
    """Load sample images for demo purposes"""
    samples = {}
    for soil_type in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, soil_type)):
            images = []
            for img_name in sorted(os.listdir(os.path.join(base_path, soil_type)))[:num_samples]:
                if img_name.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(base_path, soil_type, img_name)
                    images.append(img_path)
            samples[soil_type] = images
    return samples