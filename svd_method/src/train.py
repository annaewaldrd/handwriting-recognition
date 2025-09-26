import os
import numpy as np
import argparse
from utils import load_digits_dataset, load_images_from_folder
from numpy.linalg import svd

def compute_class_bases(X, y, n_components=20):
    """X: (n_samples, n_features). Return dict: label -> (mean, U)"""
    classes = np.unique(y)
    model = {}
    for c in classes:
        Xc = X[y == c]
        # center
        mean = Xc.mean(axis=0)
        Xc_centered = (Xc - mean).T  # shape (n_features, n_samples)
        # SVD
        U, S, Vt = svd(Xc_centered, full_matrices=False)
        U_r = U[:, :n_components]  # (n_features, n_components)
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
