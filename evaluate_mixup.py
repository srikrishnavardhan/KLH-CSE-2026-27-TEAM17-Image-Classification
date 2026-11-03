import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'

for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import load_model_and_results, load_and_preprocess_cifar10


def main():
    model_name = 'mixup_cnn'
    print(f'Loading saved model: {model_name}')
    model, history, metrics = load_model_and_results(model_name)

    print('Loading CIFAR-10 test data...')
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()

    print('Evaluating on test set...')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    print('\nSaved metrics:')
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            print(f'  {key}: {value}')
        else:
            print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
