"""
Main training script for comparing Baseline CNN, CNN with Mixup, and CNN with CutMix
"""

import argparse
import time
import os
from utils import set_random_seeds
from baseline_cnn import train_baseline_cnn
from mixup_cnn import train_mixup_cnn
from cutmix_cnn import train_cutmix_cnn


def compare_models(
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    mixup_alpha=0.2,
    cutmix_alpha=1.0
):
    """
    Train and compare all three models: Baseline, Mixup, and CutMix

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        mixup_alpha: Alpha parameter for Mixup
        cutmix_alpha: Alpha parameter for CutMix
    """
    print("="*60)
    print("COMPARING CNN APPROACHES FOR CIFAR-10 CLASSIFICATION")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print(f"Mixup alpha: {mixup_alpha}")
    print(f"CutMix alpha: {cutmix_alpha}")
    print("="*60)

    # Dictionary to store results
    results = {}

    # Train Baseline CNN
    print("\n1. TRAINING BASELINE CNN")
    print("-" * 30)
    start_time = time.time()
    baseline_model, baseline_history, baseline_metrics = train_baseline_cnn(
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        model_name='baseline_cnn',
        save_results=True
    )
    baseline_time = time.time() - start_time
    baseline_metrics['total_time'] = baseline_time
    results['baseline'] = baseline_metrics

    # Train CNN with Mixup
    print("\n\n2. TRAINING CNN WITH MIXUP")
    print("-" * 30)
    start_time = time.time()
    mixup_model, mixup_history, mixup_metrics = train_mixup_cnn(
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        alpha=mixup_alpha,
        model_name='mixup_cnn',
        save_results=True
    )
    mixup_time = time.time() - start_time
    mixup_metrics['total_time'] = mixup_time
    results['mixup'] = mixup_metrics

    # Train CNN with CutMix
    print("\n\n3. TRAINING CNN WITH CUTMIX")
    print("-" * 30)
    start_time = time.time()
    cutmix_model, cutmix_history, cutmix_metrics = train_cutmix_cnn(
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        alpha=cutmix_alpha,
        model_name='cutmix_cnn',
        save_results=True
    )
    cutmix_time = time.time() - start_time
    cutmix_metrics['total_time'] = cutmix_time
    results['cutmix'] = cutmix_metrics

    # Print comparison summary
    print("\n\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)

    print(f"{'Model':<15} {'Test Acc':<10} {'Val Acc':<10} {'Train Time':<12} {'Total Time':<12}")
    print("-" * 65)

    for model_name, metrics in results.items():
        test_acc = metrics['test_accuracy']
        val_acc = metrics['final_val_accuracy']
        train_time = metrics['training_time']
        total_time = metrics['total_time']
        print(f"{model_name:<15} {test_acc:<10.4f} {val_acc:<10.4f} {train_time:<12.2f} {total_time:<12.2f}")

    # Save comparison results
    import json
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed comparison results saved to 'results/comparison_results.json'")
    print("Individual model results, plots, and saved models are in their respective directories.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and compare CNN approaches for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Alpha parameter for Mixup')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, help='Alpha parameter for CutMix')
    parser.add_argument('--baseline-only', action='store_true', help='Train only baseline CNN')
    parser.add_argument('--mixup-only', action='store_true', help='Train only CNN with Mixup')
    parser.add_argument('--cutmix-only', action='store_true', help='Train only CNN with CutMix')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_random_seeds(42)

    if args.baseline_only:
        print("Training Baseline CNN only...")
        train_baseline_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            model_name='baseline_cnn',
            save_results=True
        )
    elif args.mixup_only:
        print("Training CNN with Mixup only...")
        train_mixup_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            alpha=args.mixup_alpha,
            model_name='mixup_cnn',
            save_results=True
        )
    elif args.cutmix_only:
        print("Training CNN with CutMix only...")
        train_cutmix_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            alpha=args.cutmix_alpha,
            model_name='cutmix_cnn',
            save_results=True
        )
    else:
        # Train and compare all models
        compare_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha
        )