import os
import cv2
import numpy as np
from PIL import Image

def load_digits_dataset(flatten=True):
    """
    Load the sklearn digits dataset (8x8 images, 1797 samples).
    Optionally flatten each image to a 1D vector.
    """
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.images  # shape (n_samples, 8, 8)
    y = d.target
    if flatten:
        X = X.reshape((X.shape[0], -1))  # (n_samples, 64)
    return X, y

def preprocess_image(path, size=(28,28), invert=False):
    """
    Load an image, convert to grayscale, resize, normalize (0..1), 
    and return as a 1D vector.
    """
    im = Image.open(path).convert('L')  # grayscale
    im = im.resize(size, Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    if invert:
        arr = 255 - arr
    arr = arr / 255.0
    return arr.flatten()

def load_images_from_folder(folder, size=(28,28), invert=False):
    """
    Load images from a folder structure:
    folder/0/*.png, folder/1/*.png, ...
    Returns X (data) and y (labels).
    """
    X, y = [], []
    for label in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                vec = preprocess_image(os.path.join(label_path,fname), size=size, invert=invert)
                X.append(vec)
                y.append(int(label))
    return np.array(X), np.array(y)
