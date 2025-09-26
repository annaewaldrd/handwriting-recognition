# Handwriting Recognition in Python ✍️

![Python](https://img.shields.io/badge/python-3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This project is a Python reimplementation and extension of my university handwriting recognition project (2019/2020), originally implemented in MATLAB using SVD-based subspace methods.  
It now includes modern machine learning baselines (SVM and CNN) for comparison.

---

## Quick Test

After cloning the repository and installing requirements, you can quickly train and test the models:

```bash
# 1. Train SVD model on sklearn digits dataset (required before testing)
python svd_method/src/train.py --dataset digits --n_components 20

# 2. Test the trained SVD model
python svd_method/src/test.py --dataset digits

# 3. Run SVM baseline (no separate training needed)
python ml_baselines/svm_digits.py

# 4. Run CNN baseline on MNIST (downloads dataset automatically)
python ml_baselines/cnn_mnist.py
```

**Notes:**
- --n_components specifies the number of SVD components per class; defaults are suggested in the SVD README.
- SVD must be trained before testing; SVM and CNN scripts handle training internally.
- For custom digits, see svd_method/predict_single.py and use the --invert flag if necessary (black-on-white digits).

---

## Methods

- **SVD Method (`svd_method/`)**  
  Classical approach using Singular Value Decomposition (SVD) to classify digits.  
  Each digit class is represented as a subspace, and classification is based on projection and reconstruction error.  
  You can test a single custom image using `predict_single.py`.



- **Modern Baselines (`ml_baselines/`)**  
  Practical machine learning methods for digit recognition:  
  - Support Vector Machine (SVM) with scikit-learn on the `digits` dataset (8x8), ~97-99% accuracy  
  - Convolutional Neural Network (CNN) with TensorFlow/Keras on MNIST (28x28), >98% accuracy after 3 epochs  

Accuracy may vary depending on the dataset and preprocessing. See the scripts for the exact commands.

---

## Data

- [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets/load_digits.html) (8×8 images, digits 0–9, ~1800 samples)  
- [`MNIST dataset`](http://yann.lecun.com/exdb/mnist) via TensorFlow/Keras (28×28 images, digits 0–9, 70,000 samples)  
- For reproducibility, a small sample of MNIST images (25 per digit) is included in `data/train` and `data/test`.  
- You can regenerate them with: 
  ```bash
  python export_mnist_images.py
  ```

---

## Methods Overview

| Method | Dataset | Overall Accuracy | Notes |
|--------|---------|----------------|-------|
| SVD | sklearn digits (8×8, ~1800 samples) | **99.83%** | Near-perfect on small, clean data |
| SVD | MNIST sample (28×28, 25 samples/class) | **84.00%** | Lower due to limited training data |
| SVM | sklearn digits (8×8) | **98.52%** | Consistent high performer |
| CNN | MNIST (28×28, 70k samples) | **98.76%** | State-of-the-art on full dataset |

> Accuracy may slightly vary due to floating-point operations or random initialization.

See `svd_method/README.md` and `ml_baselines/README.md` for detailed outputs and screenshots.

---

## Project structure

```text
handwriting-recognition/
│
├── LICENSE
├── assets/               # Plots and visual outputs
├── data/                 # MNIST samples for train/test
├── ml_baselines/         # Modern ML methods (SVM, CNN)
│   ├── cnn_mnist.py
│   ├── svm_digits.py
│   └── README.md
├── svd_method/           # Classical SVD subspace method
│   ├── src/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── predict_single.py
│   │   └── utils.py
│   └── README.md
├── export_mnist_images.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/annaewaldrd/handwriting-recognition.git
cd handwriting-recognition
pip install -r requirements.txt
```

(Optional) If you want to export your own MNIST samples:

```bash
python export_mnist_images.py
```

---

## Testing Your Own Digits

You can test the SVD method on your own handwritten digits (see `svd_method/README.md` for details, e.g., using the `--invert` flag for black-on-white digits).

---

## Evaluation / Results

Key visualizations for model performance are saved in the `assets/` folder.  
Only one representative plot per method is shown here for clarity.

- **SVD (MNIST sample)**:
  ![SVD Confusion Matrix](assets/svd_confusion_matrix.png)

- **SVM (sklearn digits)**:
  ![SVM Confusion Matrix](assets/svm_confusion_matrix.png)

- **CNN (MNIST full)**:
  ![CNN Confusion Matrix](assets/cnn_confusion_matrix.png)

### SVD on MNIST sample with varying training size

| Samples per digit | Accuracy |
|------------------|----------|
| 25               | 84.00%   |
| 60               | 90.66%   |
| 200              | 90.95%   |

> Accuracy increases quickly with small sample sizes and plateaus around 200 samples per digit, illustrating diminishing returns with more data.

### Summary

- **SVD:** Performs extremely well on small, clean datasets but shows lower accuracy on limited MNIST samples due to high variability. Increasing the training set from 25 → 60 → 200 samples per digit shows rapid initial improvement and then plateaus. Highly interpretable but sensitive to dataset size.  
- **SVM:** Provides consistent high accuracy on small datasets. Less interpretable but stable.  
- **CNN:** Achieves near state-of-the-art performance on the full MNIST dataset. Less interpretable, best suited for large datasets.  

This concise overview highlights comparative performance without overloading with plots. Detailed visualizations and per-class accuracy are available in the method-specific READMEs.
