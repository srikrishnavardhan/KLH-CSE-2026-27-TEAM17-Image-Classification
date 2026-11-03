"""
CNN with CutMix augmentation for CIFAR-10 image classification
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from utils import (
    load_and_preprocess_cifar10,
    create_baseline_cnn,
    cutmix_data,
    rand_bbox,
    plot_training_history,
    plot_confusion_matrix,
    save_model_and_results,
    get_cifar10_class_names,
    set_random_seeds
)
from sklearn.metrics import classification_report, confusion_matrix


def train_cutmix_cnn(
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    alpha=1.0,
    model_name='cutmix_cnn',
    save_results=True
):
    """
    Train and evaluate CNN with CutMix augmentation on CIFAR-10 dataset

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        alpha: Parameter for Beta distribution in CutMix
        model_name: Name for saving model and results
        save_results: Whether to save model and results

    Returns:
        tuple: (model, history, metrics)
    """
    # Set random seeds for reproducibility
    set_random_seeds(42)

    print("Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Create baseline CNN model
    print("Creating CNN model...")
    model = create_baseline_cnn()

    # Display model architecture
    model.summary()

    # Custom training loop with CutMix
    print(f"Training CNN with CutMix (alpha={alpha}) for {epochs} epochs...")
    start_time = time.time()

    # Calculate number of batches per epoch
    num_train_samples = x_train.shape[0]
    num_batches_per_epoch = num_train_samples // batch_size

    # Lists to store history
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Shuffle training data at the beginning of each epoch
        indices = np.random.permutation(num_train_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        # Initialize metrics for this epoch
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        # Batch training
        for batch_idx in range(num_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Get batch
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Apply CutMix augmentation
            x_batch_mixed, y_batch_mixed = cutmix_data(x_batch, y_batch, alpha)

            # Train on batch
            loss, accuracy = model.train_on_batch(x_batch_mixed, y_batch_mixed)

            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches_per_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Calculate average metrics for the epoch
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_accuracy)

        # Validation
        val_loss, val_accuracy = model.evaluate(
            x_train[int(num_train_samples * (1 - validation_split)):],
            y_train[int(num_train_samples * (1 - validation_split)):],
            verbose=0
        )

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Generate predictions for detailed metrics
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate detailed metrics
    class_names = get_cifar10_class_names()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Compile metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_time': float(training_time),
        'final_train_accuracy': float(history['accuracy'][-1]),
        'final_val_accuracy': float(history['val_accuracy'][-1]),
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'classification_report': report
    }

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save results if requested
    if save_results:
        print("Saving model and results...")
        save_model_and_results(model, type('History', (), {'history': history})(), metrics, model_name)

        # Plot training history
        plot_training_history(type('History', (), {'history': history})(), model_name,
                            save_path=f'results/{model_name}_training_history.png')

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, model_name,
                            save_path=f'results/{model_name}_confusion_matrix.png')

    return model, history, metrics


if __name__ == "__main__":
    # Train CNN with CutMix
    model, history, metrics = train_cutmix_cnn(
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        alpha=1.0,
        model_name='cutmix_cnn'
    )

    print("CNN with CutMix training completed successfully!")