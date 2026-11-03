# Source Code Directory

This directory contains all the source code for the CNN CIFAR-10 image classification project.

## Files

- `baseline_cnn.py`: Implementation of the baseline CNN model
- `mixup_cnn.py`: Implementation of CNN with Mixup augmentation
- `cutmix_cnn.py`: Implementation of CNN with CutMix augmentation
- `utils.py`: Utility functions used across all implementations
- `train.py`: Main script to train and compare all three approaches
- `demo_augmentations.py`: Demo script to visualize Mixup and CutMix augmentations
- `exploration.ipynb`: Jupyter notebook for data exploration and result visualization

## Usage

To train and compare all three models:
```bash
python train.py --epochs 30 --batch-size 64
```

To train only the baseline CNN:
```bash
python train.py --baseline-only --epochs 30
```

To train only the CNN with Mixup:
```bash
python train.py --mixup-only --epochs 30
```

To train only the CNN with CutMix:
```bash
python train.py --cutmix-only --epochs 30
```

To see augmentation demonstrations:
```bash
python demo_augmentations.py
```

To test the installation:
```bash
python ..\test_installation.py
```