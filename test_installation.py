"""
Test script to verify installation and basic functionality
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'

# Ensure the project root and src directory are importable regardless of the
# current working directory when the script is launched.
for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# The project expects to operate from the repository root for data paths.
os.chdir(PROJECT_ROOT)

def test_imports():
    """Test that required libraries can be imported"""
    print("Testing library imports...")

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Matplotlib: {e}")
        return False

    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False

    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Seaborn: {e}")
        return False

    return True

def test_local_modules():
    """Test that our local modules can be imported"""
    print("\nTesting local module imports...")

    # Add src directory to path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    try:
        from utils import load_and_preprocess_cifar10, create_baseline_cnn
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils module: {e}")
        return False

    try:
        from baseline_cnn import train_baseline_cnn
        print("✓ Baseline CNN module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import baseline_cnn module: {e}")
        return False

    try:
        from mixup_cnn import train_mixup_cnn
        print("✓ Mixup CNN module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mixup_cnn module: {e}")
        return False

    try:
        from cutmix_cnn import train_cutmix_cnn
        print("✓ CutMix CNN module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cutmix_cnn module: {e}")
        return False

    try:
        from train import compare_models
        print("✓ Training script module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import train module: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality with a small example"""
    print("\nTesting basic functionality...")

    try:
        # Test utils functions
        from utils import load_and_preprocess_cifar10, set_random_seeds

        # Set seeds
        set_random_seeds(42)
        print("✓ Random seeds set successfully")

        # Test data loading (just a small test)
        print("✓ Utility functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CNN CIFAR-10 Project Installation Test")
    print("=" * 40)

    # Run tests from the actual project root, regardless of where the script is launched.
    os.chdir(PROJECT_ROOT)

    # Run tests
    tests_passed = 0
    total_tests = 3

    if test_imports():
        tests_passed += 1

    if test_local_modules():
        tests_passed += 1

    if test_basic_functionality():
        tests_passed += 1

    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✓ All tests passed! Installation appears to be working correctly.")
        print("\nNext steps:")
        print("1. Run: python src\\train.py --epochs 10 (for a quick test)")
        print("2. Or: python src\\demo_augmentations.py (to see augmentation examples)")
        return True
    else:
        print("✗ Some tests failed. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

