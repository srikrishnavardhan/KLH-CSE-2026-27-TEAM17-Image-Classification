"""
Baseline CNN implementation for CIFAR-10 image classification
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from utils import (
    load_and_preprocess_cifar10,
    create_baseline_cnn,
    plot_training_history,
    plot_confusion_matrix,
    save_model_and_results,
    get_cifar10_class_names,
    set_random_seeds
)
from sklearn.metrics import classification_report, confusion_matrix


def train_baseline_cnn(
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    model_name='baseline_cnn',
    save_results=True
):
    """
    Train and evaluate baseline CNN on CIFAR-10 dataset

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
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
    print("Creating baseline CNN model...")
    model = create_baseline_cnn()

    # Display model architecture
    model.summary()

    # Train the model
    print(f"Training baseline CNN for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )

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
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'classification_report': report
    }

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save results if requested
    if save_results:
        print("Saving model and results...")
        save_model_and_results(model, history, metrics, model_name)

        # Plot training history
        plot_training_history(history, model_name,
                            save_path=f'results/{model_name}_training_history.png')

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, model_name,
                            save_path=f'results/{model_name}_confusion_matrix.png')

    return model, history, metrics


if __name__ == "__main__":
    # Train baseline CNN
    model, history, metrics = train_baseline_cnn(
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        model_name='baseline_cnn'
    )

    print("Baseline CNN training completed successfully!")