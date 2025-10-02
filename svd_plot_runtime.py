"""
Plot SVD training and testing runtime versus samples per class.
Uses summary CSV from learning experiments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

CSV_PATH = os.path.join("assets", "svd_learning_summary.csv")
OUT_PATH = os.path.join("assets", "svd_runtime_curve.png")

def main():
    # Ensure CSV exists
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run learning experiments first.")
    
    # Load data and sort by sample size
    df = pd.read_csv(CSV_PATH).sort_values("samples_per_class")

    required_cols = {"samples_per_class", "train_time_mean", "test_time_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    xs = df["samples_per_class"].values
    train_times = df["train_time_mean"].values
    test_times = df["test_time_mean"].values

    # Plot train/test times
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, train_times, "o-", markersize=6, linewidth=1.2, label="Train Time (mean sec)")
    ax.plot(xs, test_times, "o-", markersize=6, linewidth=1.2, label="Test Time (mean sec)")

    ax.set_xlabel("Samples per class", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("SVD Runtime vs. Samples per Class", fontsize=14)

    # Log-scale x-axis for better readability
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 40, 60, 100, 150, 200])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    ax.grid(alpha=0.3, which="both", linestyle="--")
    ax.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print("Saved runtime plot to", OUT_PATH)
    plt.show()

if __name__ == "__main__":
    main()
