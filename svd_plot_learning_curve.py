"""Plot SVD learning curve with main and inset (zoom 1–10 samples per class)."""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

summary_file = "assets/svd_learning_summary.csv"
try:
    df = pd.read_csv(summary_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Summary file not found: {summary_file}. "
                            "Run the SVD experiment script first to generate it.")

fig, ax = plt.subplots(figsize=(8, 6))

# Plot mean accuracy
ax.plot(
    df["samples_per_class"],
    df["accuracy_mean"],
    "o-",
    label="SVD classifier"
)

ax.set_xlabel("Training samples per digit", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Learning Curve (SVD)", fontsize=14)

xticks = [1, 10, 20, 40, 60, 100, 150, 200]
ax.set_xticks(xticks)

ax.grid(True, linestyle="--", alpha=0.6)

# Inset plot for 1–10 range
ax_inset = inset_axes(
    ax,
    width="45%",
    height="45%",
    loc="lower right",
    bbox_to_anchor=(0, 0.04, 1, 1),
    bbox_transform=ax.transAxes
)
ax_inset.errorbar(
    df["samples_per_class"],
    df["accuracy_mean"],
    yerr=df["accuracy_std"],
    fmt="o-",
    capsize=3
)
ax_inset.set_xlim(0.8, 10.2)
subset = df[df["samples_per_class"] <= 10]
ax_inset.set_ylim(subset["accuracy_mean"].min() - 5,
                  subset["accuracy_mean"].max() + 5)
ax_inset.set_xticks(list(range(1, 11)))
ax_inset.grid(True, linestyle="--", alpha=0.5)
ax_inset.set_title("Zoom: 1–10", fontsize=10)

# Save and show
plt.savefig("assets/svd_learning_curve.png", dpi=300)
plt.show()
