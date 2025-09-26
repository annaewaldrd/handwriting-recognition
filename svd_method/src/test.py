import numpy as np
import argparse
import os
from utils import load_digits_dataset, load_images_from_folder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_model(path):
    npz = np.load(path)
    model = {}
    labels = sorted({k.split('_')[0] for k in npz.files})
    for lab in labels:
        mean = npz[f"{lab}_mean"]
        U = npz[f"{lab}_U"]
        model[int(lab)] = {"mean": mean, "U": U}
    return model

def classify_vector(x, model):
    best_label, best_err = None, 1e9
    for label, v in model.items():
        mean = v["mean"]
        U = v["U"]
        xc = x - mean
        proj = U @ (U.T @ xc)
        err = np.linalg.norm(xc - proj)
        if err < best_err:
            best_err = err
            best_label = label
    return best_label, err

def plot_digit_accuracy(y_true, y_pred, method_name="SVD"):
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
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=20, color='orange', edgecolor='k')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title(f"SVD Reconstruction Error Histogram – {method_name}")
    os.makedirs("../../assets", exist_ok=True)
    out_path = f"../../assets/{method_name.lower()}_svd_error_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main(args):
    model = load_model(args.model)
    if args.dataset == 'digits':
        X_test, y_test = load_digits_dataset(flatten=True)
    else:
        X_test, y_test = load_images_from_folder(
            args.test_folder, size=(args.size,args.size), invert=args.invert
        )

    y_true, y_pred, errors = [], [], []
    for i, x in enumerate(X_test):
        pred, err = classify_vector(x, model)
        y_true.append(y_test[i])
        y_pred.append(pred)
        errors.append(err)
        if i < 10:
            print(f"idx {i} true={y_test[i]} pred={pred} err={err:.4f}")

    y_true, y_pred, errors = np.array(y_true), np.array(y_pred), np.array(errors)
    acc = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")

    print("\nDigit-specific performance:")
    print(classification_report(y_true, y_pred, labels=np.unique(y_true)))

    # Plotting
    if args.plot:
        plot_digit_accuracy(y_true, y_pred, method_name="SVD")
        plot_confusion_matrix(y_true, y_pred, method_name="SVD")
        plot_svd_errors(errors, y_true, method_name="SVD")

    # Top difficult digits
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
