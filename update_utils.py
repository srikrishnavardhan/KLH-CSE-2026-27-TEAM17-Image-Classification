import re

# Read the file
with open(r'C:\Users\sanjay\Documents\ds-project\src\utils.py', 'r') as f:
    content = f.read()

# Define the new function
new_function = '''def load_and_preprocess_cifar10():
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
    
    return (x_train, y_train), (x_test, y_test)'''

# Replace the function using regex
pattern = r'def load_and_preprocess_cifar10\(\):.*?return \(x_train, y_train\), \(x_test, y_test\)'
content = re.sub(pattern, new_function, content, flags=re.DOTALL)

# Write back to file
with open(r'C:\Users\sanjay\Documents\ds-project\src\utils.py', 'w') as f:
    f.write(content)

print("Updated load_and_preprocess_cifar10 function in utils.py")
