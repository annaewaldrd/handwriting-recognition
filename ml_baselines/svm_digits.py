"""
Train and evaluate an SVM classifier on the sklearn digits dataset.

- Splits data into train/test sets
- Standardizes features for SVM
- Computes overall and per-digit accuracy
- Generates digit-specific accuracy and confusion matrix plots
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os

def plot_digit_accuracy(y_true, y_pred, method_name="SVM"):
    """Plot and save a bar chart of accuracy per digit class."""
    digits = np.unique(y_true)
    accuracies = [np.mean(y_pred[y_true==d]==d)*100 for d in digits]

    plt.figure(figsize=(8,5))
    plt.bar(digits, accuracies, tick_label=digits)
    plt.ylim(0,100)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Digit-specific Accuracy – {method_name}")
    os.makedirs("../assets", exist_ok=True)
    plt.savefig(f"../assets/{method_name.lower()}_digit_accuracy.png")
    plt.close()
    print(f"Saved plot to ../assets/{method_name.lower()}_digit_accuracy.png")

def plot_confusion_matrix(y_true, y_pred, method_name="SVM"):
    """Plot and save the confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.arange(len(np.unique(y_true)))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {method_name}")
    os.makedirs("../assets", exist_ok=True)
    plt.savefig(f"../assets/{method_name.lower()}_confusion_matrix.png")
    plt.close()
    print(f"Saved plot to ../assets/{method_name.lower()}_confusion_matrix.png")

def main():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()  # Standardize features (zero mean, unit variance) → important for SVM performance
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf', gamma=0.001, C=10)  # RBF kernel, gamma and C tuned
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plot_digit_accuracy(y_test, y_pred, method_name="SVM")
    plot_confusion_matrix(y_test, y_pred, method_name="SVM")

    # Print per-digit accuracy to identify difficult digits
    digits_unique = np.unique(y_test)
    print("\nTop difficult digits:")
    for d in digits_unique:
        mask = y_test == d
        acc_d = np.mean(y_pred[mask] == y_test[mask])*100
        print(f"Digit {d}: {acc_d:.2f}%")

if __name__ == "__main__":
    main()
