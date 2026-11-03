"""
Demo script to visualize Mixup and CutMix augmentations
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
from utils import mixup_data, cutmix_data, get_cifar10_class_names
import os

def load_sample_data(num_samples=8):
    """Load and normalize a small sample of CIFAR-10 data"""
    (x_train, y_train), _ = datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0

    # Take a small sample
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_sample = x_train[indices]
    y_sample = y_train[indices]

    return x_sample, y_sample

def plot_image_grid(images, labels, title, class_names, save_path=None):
    """Plot a grid of images with their labels"""
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols  # Ceiling division

    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=16)

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        # Extract label from array format
        label_idx = labels[i][0] if len(labels[i]) > 0 else labels[i]
        plt.title(class_names[label_idx], fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def demo_mixup():
    """Demonstrate Mixup augmentation"""
    print("Loading sample data for Mixup demonstration...")
    x_sample, y_sample = load_sample_data(8)

    # Apply Mixup
    x_mixed, y_mixed, lam = mixup_data(x_sample, y_sample, alpha=0.2)

    print(f"Mixup lambda (λ): {lam:.3f}")

    # Plot original and mixed images
    class_names = get_cifar10_class_names()

    plot_image_grid(
        x_sample, y_sample,
        "Original Images",
        class_names,
        save_path='results/mixup_original.png'
    )

    plot_image_grid(
        x_mixed, y_mixed,
        f"Mixup Augmented Images (λ={lam:.3f})",
        class_names,
        save_path='results/mixup_augmented.png'
    )

def demo_cutmix():
    """Demonstrate CutMix augmentation"""
    print("Loading sample data for CutMix demonstration...")
    x_sample, y_sample = load_sample_data(8)

    # Apply CutMix
    x_mixed, y_mixed = cutmix_data(x_sample, y_sample, alpha=1.0)

    # Plot original and mixed images
    class_names = get_cifar10_class_names()

    plot_image_grid(
        x_sample, y_sample,
        "Original Images",
        class_names,
        save_path='results/cutmix_original.png'
    )

    plot_image_grid(
        x_mixed, y_mixed,
        "CutMix Augmented Images",
        class_names,
        save_path='results/cutmix_augmented.png'
    )

def create_results_directory():
    """Create results directory if it doesn't exist"""
    os.makedirs('results', exist_ok=True)

if __name__ == "__main__":
    # Create results directory
    create_results_directory()

    print("Mixup and CutMix Augmentation Demo")
    print("=" * 40)

    # Run demonstrations
    print("\n1. Demonstrating Mixup augmentation...")
    demo_mixup()

    print("\n2. Demonstrating CutMix augmentation...")
    demo_cutmix()

    print("\nDemo completed! Check the 'results' directory for visualizations.")