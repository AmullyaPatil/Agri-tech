import os

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Soil image paths
SOIL_IMAGE_DIR = os.path.join(DATA_DIR, 'soil_images')
SOIL_CSV_PATH = os.path.join(DATA_DIR, 'soil_parameters.csv')

# Model paths
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.h5')
CROP_MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')

# Soil type mapping
SOIL_TYPES = {
    'Black_Soil': {'N': 90, 'P': 42, 'K': 43, 'ph': 6.5},
    'Cinder_Soil': {'N': 50, 'P': 30, 'K': 35, 'ph': 7.2},
    'Laterite_Soil': {'N': 60, 'P': 55, 'K': 44, 'ph': 7.8},
    'Peat_Soil': {'N': 85, 'P': 38, 'K': 39, 'ph': 5.2},
    'Yellow_Soil': {'N': 70, 'P': 45, 'K': 40, 'ph': 6.8}
}