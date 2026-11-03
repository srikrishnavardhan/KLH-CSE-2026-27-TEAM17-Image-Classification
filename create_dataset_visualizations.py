"""Create CIFAR-10 visualizations for the project presentation."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_BATCH = PROJECT_ROOT / "cifar-10-python" / "cifar-10-batches-py" / "test_batch"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_test_images():
    with TEST_BATCH.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"])
    return images, labels


def save_pixel_histogram(images):
    pixels = images.reshape(-1)
    normalized_pixels = pixels / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("CIFAR-10 Pixel Intensity Distribution", fontsize=16, fontweight="bold")

    axes[0].hist(pixels, bins=32, color="#d95f39", edgecolor="white", linewidth=0.25)
    axes[0].set_title("Original pixel values")
    axes[0].set_xlabel("Pixel intensity (0-255)")
    axes[0].set_ylabel("Number of pixels")
    axes[0].grid(alpha=0.2)

    axes[1].hist(normalized_pixels, bins=32, color="#2f7d64", edgecolor="white", linewidth=0.25)
    axes[1].set_title("After normalization")
    axes[1].set_xlabel("Normalized intensity (0-1)")
    axes[1].set_ylabel("Number of pixels")
    axes[1].grid(alpha=0.2)

    fig.text(
        0.5,
        0.01,
        "The CNN receives normalized RGB values in the range 0-1.",
        ha="center",
        fontsize=10,
        color="#68736d",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output = RESULTS_DIR / "cifar10_pixel_histogram.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_average_heatmap(images):
    # Average the RGB channels and all test images to show the spatial intensity pattern.
    grayscale = images.mean(axis=3)
    average_image = grayscale.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    sns.heatmap(
        average_image,
        ax=ax,
        cmap="magma",
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Mean normalized intensity"},
    )
    ax.set_title("Average CIFAR-10 Test Image Heatmap", fontsize=15, fontweight="bold")
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.text(
        0.5,
        0.01,
        "Each cell represents the mean grayscale intensity at that pixel location across 10,000 test images.",
        ha="center",
        fontsize=9,
        color="#68736d",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output = RESULTS_DIR / "cifar10_average_heatmap.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_model_comparison_chart():
    model_names = ["Baseline CNN", "Mixup CNN", "CutMix CNN"]
    accuracies = [79.24, 81.48, 80.30]
    colors = ["#486fa8", "#2f7d64", "#d95f39"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(model_names, accuracies, color=colors, width=0.58)
    ax.set_title("CIFAR-10 Test Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 1.5,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.01,
        "Mixup achieved the highest test accuracy on the 10,000-image CIFAR-10 test set.",
        ha="center",
        fontsize=10,
        color="#68736d",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output = RESULTS_DIR / "model_accuracy_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_test_images()
    histogram_path = save_pixel_histogram(images)
    heatmap_path = save_average_heatmap(images)
    comparison_path = save_model_comparison_chart()
    print(f"Loaded {len(images):,} test images across {len(CLASS_NAMES)} classes.")
    print(f"Created: {histogram_path}")
    print(f"Created: {heatmap_path}")
    print(f"Created: {comparison_path}")


if __name__ == "__main__":
    main()
