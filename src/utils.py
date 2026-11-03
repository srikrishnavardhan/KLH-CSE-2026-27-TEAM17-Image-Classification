"""
Utility functions for CNN CIFAR-10 image classification project
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
import json


def load_and_preprocess_cifar10():
    """
    Load and preprocess CIFAR-10 dataset from extracted files

    Returns:
        tuple: (x_train, y_train), (x_test, y_test) normalized and ready for training
    """
    import pickle
    import os

    # Path to the extracted CIFAR-10 dataset
    cifar10_path = os.path.join(os.getcwd(), 'cifar-10-python', 'cifar-10-batches-py')

    # Load training batches
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch_file = os.path.join(cifar10_path, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            train_data.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])

    # Reshape and normalize training data
    x_train = np.concatenate(train_data)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    y_train = np.array(train_labels)

    # Load test batch
    test_file = os.path.join(cifar10_path, 'test_batch')
    with open(test_file, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')

    x_test = test_batch[b'data']
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    y_test = np.array(test_batch[b'labels'])

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def create_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a baseline CNN model for CIFAR-10 classification

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes

    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation to a batch of data

    Args:
        x: Input images batch
        y: Labels batch
        alpha: Parameter for Beta distribution (default: 0.2)

    Returns:
        tuple: (mixed_x, mixed_y, lam) where lam is the mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch of data

    Args:
        x: Input images batch
        y: Labels batch
        alpha: Parameter for Beta distribution (default: 1.0)

    Returns:
        tuple: (mixed_x, mixed_y) where images and labels are mixed
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)

    # Apply CutMix
    x[:, bbx1:bbx2, bby1:bby2, :] = x[index, bbx1:bbx2, bby1:bby2, :]

    # Adjust lambda based on actual box ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-2] * x.shape[-1]))

    # Create mixed labels
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    return x, mixed_y


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix

    Args:
        size: Size of input images (batch, height, width, channels)
        lam: Lambda parameter from Beta distribution

    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) bounding box coordinates
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def plot_training_history(history, model_name, save_path=None):
    """
    Plot training and validation accuracy/loss

    Args:
        history: History object from model.fit()
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix from sklearn.metrics.confusion_matrix
        class_names: List of class names
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_model_and_results(model, history, metrics, model_name, models_dir='models', results_dir='results'):
    """
    Save trained model, training history, and evaluation metrics

    Args:
        model: Trained Keras model
        history: History object from model.fit()
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        models_dir: Directory to save model
        results_dir: Directory to save results
    """
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f'{model_name}.h5')
    model.save(model_path)

    # Save training history
    history_path = os.path.join(results_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f, indent=2)

    # Save metrics
    metrics_path = os.path.join(results_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {model_name} model and results to {models_dir}/ and {results_dir}/")


def load_model_and_results(model_name, models_dir='models', results_dir='results'):
    """
    Load trained model, training history, and evaluation metrics

    Args:
        model_name: Name of the model
        models_dir: Directory containing saved models
        results_dir: Directory containing saved results

    Returns:
        tuple: (model, history, metrics)
    """
    # Load model
    model_path = os.path.join(models_dir, f'{model_name}.h5')
    model = tf.keras.models.load_model(model_path)

    # Load training history
    history_path = os.path.join(results_dir, f'{model_name}_history.json')
    with open(history_path, 'r') as f:
        history_dict = json.load(f)

    # Create a mock history object
    class MockHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    history = MockHistory(history_dict)

    # Load metrics
    metrics_path = os.path.join(results_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return model, history, metrics


def get_cifar10_class_names():
    """
    Get CIFAR-10 class names

    Returns:
        list: List of class names for CIFAR-10 dataset
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    print(f"Random seeds set to {seed}")