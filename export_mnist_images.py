"""Export a subset of MNIST images into data/train/<digit> and data/test/<digit> folders."""

import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

def save_images(images, labels, folder, n_per_class=25):
    """
    Save a subset of MNIST images into subfolders per class (0–9).

    Args:
        images (ndarray): Array of images.
        labels (ndarray): Corresponding digit labels.
        folder (str): Target folder path.
        n_per_class (int): Number of images to save per digit.
    """
    os.makedirs(folder, exist_ok=True)
    for digit in range(10):
        digit_folder = os.path.join(folder, str(digit))
        os.makedirs(digit_folder, exist_ok=True)

        # Select the first n_per_class samples for this digit
        digit_idx = np.where(labels == digit)[0][:n_per_class]
        for i, idx in enumerate(digit_idx):
            img = Image.fromarray(images[idx])
            img.save(os.path.join(digit_folder, f"{digit}_{i}.png"))

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Save 25 sample images per class for training and testing
    save_images(X_train, y_train, "data/train", n_per_class=25)
    save_images(X_test, y_test, "data/test", n_per_class=25)
    print("Export complete: 25 images per class saved in data/train and data/test.")
