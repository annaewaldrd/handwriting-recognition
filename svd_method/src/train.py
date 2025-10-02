"""
Train SVD model for handwriting/digit recognition.

- Compute class-wise means and SVD subspaces
- Save model as .npz file
- Supports sklearn digits dataset or custom image folder
"""

import os
import numpy as np
import argparse
from utils import load_digits_dataset, load_images_from_folder
from numpy.linalg import svd

def compute_class_bases(X, y, n_components=20):
    """
    Compute class-wise mean and SVD subspace.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training data.
    y : np.ndarray, shape (n_samples,)
        Labels.
    n_components : int
        Number of principal components to keep per class.

    Returns
    -------
    model : dict
        Dictionary mapping label -> {"mean": ..., "U": ...}
    """
    classes = np.unique(y)
    model = {}
    for c in classes:
        Xc = X[y == c]
        # center
        mean = Xc.mean(axis=0)
        Xc_centered = (Xc - mean).T  # shape (n_features, n_samples)
        # Compute SVD and keep the top n_components basis vectors for this class
        U, S, Vt = svd(Xc_centered, full_matrices=False)
        max_comp = min(n_components, U.shape[1])
        if n_components > max_comp:
            print(f"Warning: class {c} has only {U.shape[1]} components available. "
                  f"Requested {n_components}, using {max_comp}.")
        U_r = U[:, :max_comp]  # (n_features, n_components)
        model[int(c)] = {"mean": mean, "U": U_r}
    return model

def save_model(model, path):
    # Save as npz
    np.savez(path, **{f"{k}_mean": v["mean"] for k,v in model.items()},
                      **{f"{k}_U": v["U"] for k,v in model.items()})

def main(args):
    if args.dataset == 'digits':
        X, y = load_digits_dataset(flatten=True)
    else:
        X, y = load_images_from_folder(args.dataset_folder, size=(args.size,args.size), invert=args.invert)
    model = compute_class_bases(X, y, n_components=args.n_components)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    save_model(model, out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='digits', choices=['digits','folder'])
    parser.add_argument('--dataset_folder', default='data/train')
    parser.add_argument('--size', type=int, default=8)  # 8 for sklearn digits, or 28 for MNIST-like
    parser.add_argument('--n_components', type=int, default=20)
    parser.add_argument('--out_dir', default='models')
    parser.add_argument('--out_name', default='svd_model.npz')
    parser.add_argument('--invert', action='store_true')
    args = parser.parse_args()
    main(args)
