"""
Utility functions for handwriting/digit recognition datasets.

- Load sklearn digits dataset
- Load custom images from folder
- Preprocess images (grayscale, resize, normalize, flatten)
"""

import os
import numpy as np
from PIL import Image

def load_digits_dataset(flatten=True):
    """
    Load the sklearn digits dataset (8x8 images, 1797 samples).

    Parameters
    ----------
    flatten : bool, default=True
        If True, flattens each image to a 1D vector of length 64.

    Returns
    -------
    X : np.ndarray
        Array of images. Shape (n_samples, 64) if flattened, else (n_samples, 8, 8)
    y : np.ndarray
        Array of labels (digits 0-9), shape (n_samples,)
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
    Load an image, convert to grayscale, resize, normalize (0..1), and flatten.

    Parameters
    ----------
    path : str
        Path to the image file.
    size : tuple[int,int], default=(28,28)
        Target size for the image (width, height).
    invert : bool, default=False
        If True, invert colors (black ↔ white).

    Returns
    -------
    vec : np.ndarray
        Flattened image vector with values in [0,1].
    """
    im = Image.open(path).convert('L')  # grayscale
    im = im.resize(size, Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    if invert:
        arr = 255 - arr  # invert colors if needed (black ↔ white)
    arr = arr / 255.0
    return arr.flatten()

def load_images_from_folder(folder, size=(28,28), invert=False):
    """
    Load images from a folder structure: folder/<digit>/*.png

    Parameters
    ----------
    folder : str
        Path to the root folder containing subfolders for each digit class.
    size : tuple[int,int], default=(28,28)
        Target size for images.
    invert : bool, default=False
        Whether to invert colors.

    Returns
    -------
    X : np.ndarray
        Array of image vectors (n_samples, n_features)
    y : np.ndarray
        Array of labels (n_samples,)
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
