"""
Train and evaluate a simple CNN on MNIST.

- Trains a 2-conv-layer CNN on MNIST
- Evaluates test accuracy and prints classification report
- Generates digit-specific accuracy and confusion matrix plots
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

def plot_digit_accuracy(y_true, y_pred, method_name="CNN"):
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

def plot_confusion_matrix(y_true, y_pred, method_name="CNN"):
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
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    # Reshape to (samples, height, width, channels) and normalize pixel values to [0,1]
    X_train = X_train.reshape(-1,28,28,1).astype("float32")/255.0
    X_test = X_test.reshape(-1,28,28,1).astype("float32")/255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Define a simple CNN: 2 conv+pooling layers → dense layer → softmax classifier
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation="relu"),
        layers.Dense(10,activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_cat, epochs=3, batch_size=128, validation_split=0.1)
    
    _, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    plot_digit_accuracy(y_test, y_pred_classes, method_name="CNN")
    plot_confusion_matrix(y_test, y_pred_classes, method_name="CNN")

    # Print per-digit accuracy to identify difficult digits
    digits_unique = np.unique(y_test)
    print("\nTop difficult digits:")
    for d in digits_unique:
        mask = y_test == d
        acc_d = np.mean(y_pred_classes[mask] == y_test[mask])*100
        print(f"Digit {d}: {acc_d:.2f}%")

if __name__ == "__main__":
    main()
