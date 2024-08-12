import os
import numpy as np
from PIL import Image

OPTIONS = ['normal', 'noisy', 'bright']
SEED = 42
SAMPLE_SIZE = 1000
SHOW_IMAGES = False
SAVE_IMAGES = True
BRIGHTNESS_INCREASE = 50
NOISE_MEAN = 0
NOISE_STD = 75  # Standard deviation of noise

def add_noise(image):
    image_array = np.array(image).astype(np.int16)
    noise = np.random.normal(NOISE_MEAN, NOISE_STD, image_array.shape)
    
    noisy_image_array = image_array + noise
    # Ensure we clip values to stay within [0, 255] and convert back to uint8
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)

    # Convert array to Image
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image

def brighten_image(image):
    image_array = np.array(image).astype(np.int16)
    bright_image_array = image_array + BRIGHTNESS_INCREASE

    bright_image_array = np.clip(bright_image_array, 0, 255).astype(np.uint8)
    # Convert array to Image
    bright_image = Image.fromarray(bright_image_array)
    return bright_image


# Define the directory path
mnist_path = "./MNIST"
dir_path = f"{mnist_path}/images/t10k-images-idx3-ubyte"
save_path = f"{mnist_path}/images/test-sample-{SAMPLE_SIZE}-seed-{SEED}"
for option in OPTIONS:
    os.makedirs(f"{save_path}/{option}", exist_ok=True)

# Set the random seed
np.random.seed(SEED)

# Grab image files
image_files = os.listdir(dir_path)
image_files_sample = np.random.choice(image_files, size=SAMPLE_SIZE)

for image_file in image_files_sample:
    image_path = os.path.join(dir_path, image_file)
    image = Image.open(image_path)
    
    for option in OPTIONS:
        if option == 'noisy':
            processed_image = add_noise(image)
        elif option == 'bright':
            processed_image = brighten_image(image)
        else:
            processed_image = image
            image.show() if SHOW_IMAGES else None
        
        processed_image.show() if SHOW_IMAGES else None
        processed_image.save(
            os.path.join(save_path, f"{option}/{image_file}")
            ) if SAVE_IMAGES else None

