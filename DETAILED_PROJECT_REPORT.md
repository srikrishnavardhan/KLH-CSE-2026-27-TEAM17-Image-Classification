# Deep Learning-Based Object Classification Using CNNs on CIFAR-10: Detailed Project Report

## Abstract

This study evaluates whether Mixup and CutMix data augmentation improve image-classification performance on CIFAR-10 compared with a standard convolutional neural network (CNN). Three models share the same CNN architecture and are evaluated on the held-out 10,000-image CIFAR-10 test set. The baseline CNN achieves 79.24% test accuracy, Mixup achieves 81.48%, and CutMix achieves 80.30%. Mixup therefore provides the strongest result, improving test accuracy by 2.24 percentage points over the baseline and producing the lowest test loss. The study also reports class-level precision, recall, and F1-score, confusion matrices, training time, and an interactive prediction explorer. Because this is a single-run study and the custom Mixup and CutMix loops reuse the reported validation subset during training, the validation values are treated as diagnostic rather than independent evidence. The test-set comparison remains the primary result.

**Keywords:** CIFAR-10, convolutional neural network, image classification, Mixup, CutMix, data augmentation, deep learning

## Project Overview

This project implements and compares three convolutional neural network approaches for image classification on the CIFAR-10 dataset:
1. Baseline CNN (standard architecture)
2. CNN with Mixup augmentation
3. CNN with CutMix augmentation

The goal is to evaluate whether research-backed data augmentation techniques can improve classification performance and generalization compared to a standard CNN.

## Project Structure

```
CNN_CIFAR10_Project/
├── cifar-10-python/                 # CIFAR-10 dataset (extracted)
├── src/                             # Source code
│   ├── baseline_cnn.py              # Baseline CNN implementation
│   ├── mixup_cnn.py                 # CNN with Mixup augmentation
│   ├── cutmix_cnn.py                # CNN with CutMix augmentation
│   ├── utils.py                     # Utility functions (data loading, preprocessing, etc.)
│   ├── train.py                     # Training script with experiment configuration
│   ├── demo_augmentations.py        # Demo to visualize augmentations
│   └── README.md                    # Source directory documentation
├── notebooks/                       # Jupyter notebooks for exploration
│   └── exploration.ipynb            # Data exploration notebook
├── models/                          # Saved trained models
├── results/                         # Evaluation results, metrics, plots
├── requirements.txt                 # Required Python packages
├── README.md                        # Main project documentation
├── PROJECT_SUMMARY.md               # Project summary
├── PROJECT_REPORT.md                # Final report
├── DETAILED_PROJECT_REPORT.md       # This detailed report
├── evaluate_mixup.py                # Saved-model test-set evaluation
├── web_app.py                        # Interactive prediction explorer
├── create_dataset_visualizations.py  # Presentation visualization generator
└── test_installation.py             # Installation verification script
```

## Key Features

1. **Baseline CNN**: Standard CNN architecture with three convolutional blocks
2. **Mixup Implementation**: Linear interpolation of image pairs and labels
3. **CutMix Implementation**: Patch-based augmentation replacing regions between image pairs
4. **Unified Training Interface**: Single script to train and compare all approaches
5. **Comprehensive Evaluation**: Accuracy, loss, training time, classification reports, confusion matrices
6. **Visualization Tools**: Training history plots, confusion matrices, sample augmentation visualizations
7. **Reproducibility**: Random seeds are set to improve repeatability, although complete determinism is not guaranteed
8. **Modular Design**: Easy to extend with additional techniques or experiments

## Technical Implementation Details

### Baseline CNN Architecture
- Three convolutional blocks with increasing filters (32→64→128)
- Each block: two Conv2D layers → MaxPooling → Dropout
- Fully connected layers: Dense(512) → Dropout → Dense(10, softmax)
- Optimizer: Adam
- Loss: Categorical crossentropy
- Total Parameters: 1,341,226 (5.12 MB)

### Mixup Implementation
- For each batch: randomly pair images with lambda ~ Beta(α, α)
- Mixed image: λ × image₁ + (1-λ) × image₂
- Mixed label: λ × label₁ + (1-λ) × label₂
- Applied on-the-fly during training
- Alpha parameter: 0.2 (as recommended in original paper)

### CutMix Implementation
- For each batch: randomly pair images with lambda ~ Beta(α, α)
- Extract random patch from second image based on lambda
- Replace corresponding region in first image with patch
- Adjust lambda based on actual patch-to-image ratio
- Mixed label: λ × label₁ + (1-λ) × label₂
- Alpha parameter: 1.0 (as recommended in original paper)

## Experimental Results

Based on the comparison results in `results/comparison_results.json`:

| Model | Test Accuracy | Test Loss | Training Time (s) | Final Train Acc |
|-------|--------------|-----------|-------------------|-----------------|
| Baseline CNN | 79.24% | 0.6993 | 1,015.36 | 86.70% |
| Mixup CNN (α=0.2) | 81.48% | 0.5780 | 962.97 | 81.17% |
| CutMix CNN (α=1.0) | 80.30% | 0.5952 | 1,031.16 | 73.16% |

Note: The CutMix results differ from what was reported in PROJECT_REPORT.md, which showed 58.77% accuracy. The current saved metrics show CutMix performing slightly better than the baseline but not as well as Mixup.

The Mixup and CutMix training loops also recorded diagnostic evaluations on the final 10% of the training array. Those samples were included in the custom training loop, so the previously reported Mixup and CutMix values of 92.78% and 89.20% must not be interpreted as independent validation accuracy. The held-out test-set results above are the primary comparison.

### Per-Class Performance Highlights

**Baseline CNN:**
- Best performing classes: automobile (90.7% F1), ship (87.5% F1), frog (83.4% F1)
- Weakest performing classes: cat (61.8% F1), dog (70.5% F1), bird (71.5% F1)

**Mixup CNN:**
- Best performing classes: automobile (90.7% F1), truck (88.4% F1), ship (89.0% F1)
- Most improved over baseline: frog (+1.0% F1), horse (+2.0% F1), airplane (+2.7% F1)

**CutMix CNN:**
- Best performing classes: automobile (90.7% F1), truck (88.7% F1), ship (88.4% F1)
- Performance similar to baseline but with different strengths/weaknesses

## Analysis and Discussion

### Key Findings
1. **Mixup provides improvement:** The Mixup-augmented model achieved the highest test accuracy (81.48%), outperforming the baseline by 2.24 percentage points.
2. **Baseline performs strongly:** The baseline CNN achieved respectable performance (79.24%) without complex augmentations.
3. **CutMix shows moderate improvement:** CutMix improved accuracy to 80.30% (1.06 percentage points over baseline), though less than Mixup.
4. **Training efficiency:** All three approaches had similar training times (~16-17 minutes), contrary to expectations that CutMix would be significantly faster.
5. **Evidence scope:** The comparison is based on one run per configuration. The accuracy differences are empirical observations, not statistical significance claims.

### Why Mixup Helped:
- Encourages linear behavior between classes
- Reduces sensitivity to adversarial examples
- Acts as a form of label smoothing
- Particularly effective with our relatively simple CNN architecture

### Performance Comparison with Literature:
- **Original Mixup paper:** Reported improvements on CIFAR-10 (typically 1-2% over baselines) - our results align well
- **Original CutMix paper:** Reported significant improvements (often 2-4% over baselines) - our results show more modest improvement
- Possible explanations for CutMix's relatively lower performance:
  - Hyperparameter sensitivity (alpha=1.0 may not be optimal for our architecture)
  - Implementation details in patch extraction/mixing
  - Dataset characteristics (CIFAR-10's small 32x32 images may limit CutMix effectiveness)

### Threats to Validity

- The experiment uses one random seed and one alpha value for each augmentation method, so results may vary across repetitions or hyperparameter settings.
- The baseline uses Keras' built-in validation split, while Mixup and CutMix use custom training loops; their validation histories are therefore not directly comparable.
- The custom Mixup and CutMix loops reuse the reported validation subset during training. A future study should create an explicit train/validation split before augmentation.
- The conclusions apply to this CNN, CIFAR-10 preprocessing pipeline, and software environment; they should not be generalized to all datasets or architectures.

## Instructions for Reproducing Results

### Environment Setup
```bash
# Navigate to project directory
cd C:\Users\sanjay\Documents\ds-project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### Running the Models
```bash
# Train and compare all models (50 epochs each)
python src/train.py --epochs 50

# Train individual models
python src/train.py --baseline-only --epochs 50
python src/train.py --mixup-only --epochs 50 --mixup-alpha 0.2
python src/train.py --cutmix-only --epochs 50 --cutmix-alpha 1.0

# Quick test (2 epochs for verification)
python src/train.py --epochs 2

# Evaluate the saved Mixup model on the complete test set
python evaluate_mixup.py

# Generate dataset and model-comparison visualizations
python create_dataset_visualizations.py

# Launch the interactive prediction explorer
python web_app.py
```

### Visualizing Augmentations
```bash
python src/demo_augmentations.py
```

## Files and Directories

- **Trained Models:** `models/` directory (`*.h5` files)
- **Evaluation Metrics:** `results/` directory (`*_metrics.json` files)
- **Training History:** `results/` directory (`*_history.json` files)
- **Visualizations:** `results/` directory (`*.png` files)
- **Comparison Results:** `results/comparison_results.json`
- **Source Code:** `src/` directory
- **Notebooks:** `notebooks/` directory

## Limitations

1. Single hyperparameter setting for each augmentation technique (no extensive grid search)
2. Fixed training duration (50 epochs) without early stopping or learning rate scheduling
3. Simple CNN architecture without modern improvements (batch norm, residual connections)
4. GPU acceleration not fully utilized due to Windows TensorFlow limitations
5. One primary run per model, with no confidence intervals or statistical significance testing
6. The custom Mixup and CutMix loops reuse the reported validation subset during training

## Future Work

1. **Correct validation protocol:** Create explicit 45,000/5,000 train-validation splits before augmentation and training
2. **Architecture enhancements:** Implement ResNet, DenseNet, or WideResNet backbones
3. **Advanced training techniques:** Learning rate schedules, batch normalization, ensemble methods
4. **Combined augmentations:** Experiment with Mixup + CutMix or other augmentation combinations
5. **Robustness testing:** Evaluate on corrupted CIFAR-10-C or other distribution shifts
6. **Transfer learning:** Pre-train on larger datasets before CIFAR-10 fine-tuning
7. **Extended analysis:** Investigate why CutMix underperformed relative to expectations
8. **Visualization improvements:** Feature visualization, saliency maps, etc.
9. **Statistical replication:** Repeat each configuration with multiple seeds and report mean, standard deviation, and confidence intervals
10. **Hyperparameter optimization:** Search systematically over Mixup/CutMix alpha values and patch settings

## References

1. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, A. (2017). mixup: Beyond Empirical Risk Minimization. *arXiv preprint arXiv:1710.09412*.
2. Yun, S., Han, D., Oh, S. J., Chun, S., Jye, Y., & Poh, H. S. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *In Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 6023-6032).
3. Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. *Technical Report, University of Toronto*.
4. Abadi, M., et al. (2015). TensorFlow: Large-scale machine learning on heterogeneous distributed systems. *arXiv preprint arXiv:1603.04467*.

---

*Report generated: October 2026*
*For academic use in Data Science course project*