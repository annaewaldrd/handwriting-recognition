"""
Run SVD learning curve experiments on MNIST with varying training set sizes.

- Saves temporary train/test folders
- Trains SVD models and evaluates accuracy
- Collects results into CSVs for further plotting
"""

import os
import shutil
import subprocess
import sys
import csv
import re
import time
from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_TEMP = PROJECT_ROOT / "data" / "train_temp"
TEST_REFERENCE = PROJECT_ROOT / "data" / "test_reference"
ASSETS = PROJECT_ROOT / "assets"
SVD_SRC = PROJECT_ROOT / "svd_method" / "src"
PY = sys.executable

SAMPLE_SIZES = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,60,100,150,200]  # Number of training images per class to test
SEEDS = [0, 1, 2, 3, 4]  # Random seeds for multiple runs to get robust statistics
N_COMPONENTS_DEFAULT = 24  # Default max number of SVD components per class

ASSETS.mkdir(parents=True, exist_ok=True)

# Regex to extract accuracy line from test.py output
ACC_RE = re.compile(r"Overall Accuracy:\s*([0-9]+\.[0-9]+)%")

def save_images_per_class(X, y, out_dir, n_per_class, seed=0):
    """
    Save a subset of images per digit into folder.
    
    Parameters
    ----------
    X : np.ndarray
        Image data, shape (n_samples, height, width)
    y : np.ndarray
        Labels
    out_dir : str or Path
        Target folder
    n_per_class : int
        Number of images to save per digit
    seed : int
        Random seed for selection
    """
    rng = np.random.RandomState(seed)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for d in range(10):
        (out_dir / str(d)).mkdir(parents=True, exist_ok=True)

    for d in range(10):
        idxs = np.where(y == d)[0]
        if len(idxs) < n_per_class:
            raise ValueError(f"Not enough images for digit {d}: need {n_per_class}, have {len(idxs)}")
        chosen = rng.choice(idxs, size=n_per_class, replace=False)
        for i, idx in enumerate(chosen):
            arr = X[idx]
            img = Image.fromarray(arr)
            img_path = out_dir / str(d) / f"{d}_{i}.png"
            img.save(img_path)

def save_reference_testset(X_test, y_test, out_dir):
    """Save reference test set with min available images per class"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # determine minimum number of images across all digits
    min_per_class = min([np.sum(y_test == d) for d in range(10)])
    print(f"Using {min_per_class} images per class for reference test set")
    for d in range(10):
        (out_dir / str(d)).mkdir(parents=True, exist_ok=True)

    counters = {d: 0 for d in range(10)}
    for arr, label in zip(X_test, y_test):
        lbl = int(label)
        if counters[lbl] >= min_per_class:
            continue
        img = Image.fromarray(arr)
        img_path = out_dir / str(lbl) / f"{lbl}_{counters[lbl]}.png"
        img.save(img_path)
        counters[lbl] += 1

def run_train(n_components):
    """Run train.py with n_components and return training time"""
    cmd = [PY, "train.py", "--dataset", "folder",
           "--dataset_folder", str(TRAIN_TEMP),
           "--n_components", str(n_components),
           "--size", "28"]
    start = time.time()
    p = subprocess.run(cmd, cwd=str(SVD_SRC), capture_output=True, text=True)
    elapsed = time.time() - start
    if p.returncode != 0:
        print("train.py failed:")
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("train.py failed")
    print(p.stdout)
    return elapsed

def run_test():
    """Run test.py on reference set and parse overall accuracy"""
    cmd = [PY, "test.py", "--dataset", "folder",
           "--test_folder", str(TEST_REFERENCE),
           "--size", "28"]
    start = time.time()
    p = subprocess.run(cmd, cwd=str(SVD_SRC), capture_output=True, text=True)
    elapsed = time.time() - start
    if p.returncode != 0:
        print("test.py failed:")
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("test.py failed")
    out = p.stdout + "\n" + p.stderr
    print(out)
    m = ACC_RE.search(out)
    if not m:
        print("Output from test.py:")
        print(out)
        raise RuntimeError("Couldn't parse accuracy from test.py output (no 'Overall Accuracy' line found).")
    acc = float(m.group(1))
    return acc, elapsed

def main():
    """Run the full MNIST SVD experiment: save temp data, train models, evaluate, and save CSVs."""
    print("Loading MNIST...")
    (X_train_all, y_train_all), (X_test_all, y_test_all) = mnist.load_data()
    X_train_all = X_train_all.astype("uint8")
    X_test_all = X_test_all.astype("uint8")

    print("Saving reference test set to", TEST_REFERENCE)
    save_reference_testset(X_test_all, y_test_all, TEST_REFERENCE)

    raw_csv = ASSETS / "svd_learning_raw.csv"
    summary_csv = ASSETS / "svd_learning_summary.csv"
    with open(raw_csv, "w", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(["samples_per_class", "seed", "n_components",
                         "accuracy", "train_time_sec", "test_time_sec"])
        results = []
        for N in SAMPLE_SIZES:
            print("\n=== SAMPLE SIZE:", N, "per class ===")
            for seed in SEEDS:
                print(" Seed:", seed)
                save_images_per_class(X_train_all, y_train_all,
                                      TRAIN_TEMP, n_per_class=N, seed=seed)
                n_components = min(N-1, N_COMPONENTS_DEFAULT)  # Choose number of SVD components (cannot exceed N-1, max default)
                if n_components < 1:
                    n_components = 1
                train_time = run_train(n_components)
                acc, test_time = run_test()
                print(f" -> Acc {acc:.2f}% | N={N}, seed={seed}, n_comp={n_components}, "
                      f"train={train_time:.2f}s, test={test_time:.2f}s")
                writer.writerow([N, seed, n_components, acc, train_time, test_time])
                results.append((N, seed, n_components, acc, train_time, test_time))

    df = pd.read_csv(raw_csv)
    summary = df.groupby("samples_per_class").agg(
        # Aggregate raw results by sample size for summary
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        runs=("accuracy", "count"),
        train_time_mean=("train_time_sec", "mean"),
        test_time_mean=("test_time_sec", "mean"),
    ).reset_index()
    summary.to_csv(summary_csv, index=False)
    print("\nSaved raw results to", raw_csv)
    print("Saved summary to", summary_csv)
    print(summary)

if __name__ == "__main__":
    main()
