import os
import idx2numpy
from PIL import Image

# Define the directory path
mnist_path = "./MNIST"
dir_path = f"{mnist_path}/raw"

# Define the file paths
image_files = ["t10k-images-idx3-ubyte", "train-images-idx3-ubyte"]
label_files = ["t10k-labels-idx1-ubyte", "train-labels-idx1-ubyte"]

# Process the image and label files
for image_file, label_file in zip(image_files, label_files):
    # Convert the IDX files to numpy arrays
    images = idx2numpy.convert_from_file(os.path.join(dir_path, image_file))
    labels = idx2numpy.convert_from_file(os.path.join(dir_path, label_file))

    # Create a directory to save the images
    save_dir = os.path.join(f"{mnist_path}/images", image_file)
    os.makedirs(save_dir, exist_ok=True)

    # Save the numpy arrays as image files
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert the numpy array to a PIL image
        image = Image.fromarray(image)

        # Save the image
        image.save(os.path.join(save_dir, f"{label}_{i}.png"))