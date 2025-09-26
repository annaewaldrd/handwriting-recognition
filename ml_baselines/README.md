# ML Baselines: SVM & CNN

Modern machine learning baselines for handwritten digit recognition, as a comparison to SVD.

**Quick Test:** See root README.

---

## Files
- `svm_digits.py` - Support Vector Machine on sklearn digits dataset (8x8), ~97-99% accuracy  
- `cnn_mnist.py` - Convolutional Neural Network on MNIST dataset (28x28), >98% accuracy after 3 epochs  

---

## How to Run

### 1. Run Support Vector Machine (SVM) on sklearn digits (8x8)
```bash
python svm_digits.py
```

### 2. Run Convolutional Neural Network (CNN) on MNIST (28x28)
```bash
python cnn_mnist.py
```

---

**Notes:**
- The scripts run standalone, no pre-trained models needed.
- SVM automatically trains on sklearn digits (8x8).
- CNN automatically downloads MNIST dataset (28x28) via TensorFlow/Keras.
- Accuracy may vary depending on the dataset and preprocessing.

---

## SVM (sklearn digits)

![Confusion Matrix](../assets/svm_confusion_matrix.png)

**Most challenging digits (MNIST subset):**
| Digit | Accuracy |
|-------|----------|
| 4     | 98.15%   |
| 8     | 92.31%   |
| 9     | 98.15%   |

---

## CNN (MNIST full)

![Confusion Matrix](../assets/cnn_confusion_matrix.png)

**Most challenging digits (MNIST full):**
| Digit | Accuracy |
|-------|----------|
| 8     | 97.13%   |
| 9     | 98.02%   |

---

## Conclusion

- **SVM:** Reliable performance on small datasets.  
- **CNN:** Achieves top performance on large datasets.  
- Choice of method depends on dataset size and interpretability requirements.
