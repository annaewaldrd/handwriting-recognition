"""
Test an SVD model on MNIST or folder dataset.

Computes overall and per-digit accuracy, reconstruction errors,
and optionally saves plots (digit accuracy, confusion matrix, error histogram).
"""

import numpy as np
import argparse
import os
from utils import load_digits_dataset, load_images_from_folder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_model(path):
    """Load an SVD model (means + subspace U) from a .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    npz = np.load(path)
    model = {}
    labels = sorted({k.split('_')[0] for k in npz.files})
    for lab in labels:
        mean_key = f"{lab}_mean"
        U_key = f"{lab}_U"
        if mean_key not in npz or U_key not in npz:
            raise KeyError(f"Missing '{mean_key}' or '{U_key}' in model file")
        
        mean = npz[mean_key]
        U = npz[U_key]
        model[int(lab)] = {"mean": mean, "U": U}
    return model

def classify_vector(x, model):
    """
    Classify a vector by projecting it into each digit's subspace
    and choosing the one with the smallest reconstruction error.
    """
    best_label, best_err = None, 1e9
    for label, v in model.items():
        mean = v["mean"]
        U = v["U"]

        if x.shape != mean.shape:
            raise ValueError(f"Shape mismatch: x={x.shape}, mean={mean.shape}")
        
        xc = x - mean
        proj = U @ (U.T @ xc)  # reconstruction from subspace
        err = np.linalg.norm(xc - proj)
        if err < best_err:
            best_err = err
            best_label = label
    return best_label, best_err

def plot_digit_accuracy(y_true, y_pred, method_name="SVD"):
    """Bar chart: accuracy per digit class."""
    digits = np.unique(y_true)
    accuracies = []
    for d in digits:
        mask = y_true == d
        acc_d = np.mean(y_pred[mask] == y_true[mask]) * 100
        accuracies.append(acc_d)

    plt.figure(figsize=(8,5))
    plt.bar(digits, accuracies, tick_label=digits)
    plt.ylim(0, 100)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Digit-specific Accuracy – {method_name}")

    os.makedirs("../../assets", exist_ok=True)
    out_path = f"../../assets/{method_name.lower()}_digit_accuracy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def plot_confusion_matrix(y_true, y_pred, method_name="SVD"):
    """Heatmap of true vs. predicted digits."""

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix – {method_name}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    os.makedirs("../../assets", exist_ok=True)
    out_path = f"../../assets/{method_name.lower()}_confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def plot_svd_errors(errors, y_true, method_name="SVD"):
    """Histogram of reconstruction errors across test samples."""
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=20, color='orange', edgecolor='k')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title(f"SVD Reconstruction Error Histogram – {method_name}")
    os.makedirs("../../assets", exist_ok=True)
    out_path = f"../../assets/{method_name.lower()}_error_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main(args):
    # Load test data depending on source
    model = load_model(args.model)
    if args.dataset == 'digits':
        X_test, y_test = load_digits_dataset(flatten=True)
    else:
        X_test, y_test = load_images_from_folder(
            args.test_folder, size=(args.size,args.size), invert=args.invert
        )
    
    if len(X_test) == 0:
        raise ValueError(f"No test samples found in dataset '{args.dataset}'")
    
    # Classify all samples and collect errors
    y_true, y_pred, errors = [], [], []
    for i, x in enumerate(X_test):
        pred, err = classify_vector(x, model)
        y_true.append(y_test[i])
        y_pred.append(pred)
        errors.append(err)
        if i < 10:  # print first 10 predictions for inspection
            print(f"idx {i} true={y_test[i]} pred={pred} err={err:.4f}")

    y_true, y_pred, errors = np.array(y_true), np.array(y_pred), np.array(errors)
    acc = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")

    print("\nDigit-specific performance:")
    print(classification_report(y_true, y_pred, labels=np.unique(y_true)))

    # Plot results if requested
    if args.plot:
        plot_digit_accuracy(y_true, y_pred, method_name="SVD")
        plot_confusion_matrix(y_true, y_pred, method_name="SVD")
        plot_svd_errors(errors, y_true, method_name="SVD")

    # Print per-digit accuracy to identify difficult digits
    digits = np.unique(y_true)
    print("\nTop difficult digits:")
    for d in digits:
        mask = y_true == d
        acc_d = np.mean(y_pred[mask] == y_true[mask]) * 100
        print(f"Digit {d}: {acc_d:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/svd_model.npz')
    parser.add_argument('--dataset', default='digits', choices=['digits','folder'])
    parser.add_argument('--test_folder', default='data/test')
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--plot', action='store_true', help="Save plots")
    args = parser.parse_args()
    main(args)
