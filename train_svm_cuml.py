#!/usr/bin/env python3
"""
GPU-Accelerated SVM Training using cuML (RAPIDS)
CSC14120 - Parallel Programming

This script trains an SVM classifier on features extracted by the C++ autoencoder.
Uses cuML for 100-1000x faster training compared to CPU LIBSVM.

Usage:
    python train_svm_cuml.py --train models/train_features_gpu.bin \
                             --test models/test_features_gpu.bin \
                             --data /path/to/cifar-10-batches-bin \
                             --model-name gpu_opt_v1
"""

import argparse
import numpy as np
import time
import os
import pickle

# CIFAR-10 constants (must match C++ code)
CIFAR_TRAIN_SIZE = 50000
CIFAR_TEST_SIZE = 10000
FEATURE_DIM = 8192  # 8x8x128 from autoencoder latent space
CIFAR_CLASSES = 10

CIFAR_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_cifar_labels(data_path):
    """Load CIFAR-10 labels from binary files."""
    train_labels = []
    test_labels = []
    
    # Load training labels from 5 batches
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f'data_batch_{i}.bin')
        if not os.path.exists(batch_file):
            raise FileNotFoundError(f"CIFAR-10 batch file not found: {batch_file}")
        
        with open(batch_file, 'rb') as f:
            # Each record: 1 byte label + 3072 bytes image
            for _ in range(10000):
                label = int.from_bytes(f.read(1), byteorder='little')
                train_labels.append(label)
                f.read(3072)  # Skip image data
    
    # Load test labels
    test_file = os.path.join(data_path, 'test_batch.bin')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"CIFAR-10 test file not found: {test_file}")
    
    with open(test_file, 'rb') as f:
        for _ in range(10000):
            label = int.from_bytes(f.read(1), byteorder='little')
            test_labels.append(label)
            f.read(3072)
    
    return np.array(train_labels, dtype=np.int32), np.array(test_labels, dtype=np.int32)


def load_features(filepath, num_samples, feature_dim):
    """Load binary feature file created by C++ autoencoder."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")
    
    features = np.fromfile(filepath, dtype=np.float32)
    expected_size = num_samples * feature_dim
    
    if features.size != expected_size:
        raise ValueError(f"Feature file size mismatch. Expected {expected_size}, got {features.size}")
    
    return features.reshape(num_samples, feature_dim)


def print_evaluation(predictions, true_labels, class_names):
    """Print evaluation metrics similar to C++ version."""
    num_samples = len(predictions)
    correct = np.sum(predictions == true_labels)
    accuracy = correct / num_samples * 100
    
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40 + "\n")
    
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    
    # Confusion matrix
    confusion = np.zeros((CIFAR_CLASSES, CIFAR_CLASSES), dtype=np.int32)
    for pred, true in zip(predictions, true_labels):
        confusion[true][pred] += 1
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 40)
    for c in range(CIFAR_CLASSES):
        class_total = np.sum(confusion[c])
        class_correct = confusion[c][c]
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        print(f"{class_names[c]:>12}: {class_acc:6.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    print("           ", end="")
    for j in range(CIFAR_CLASSES):
        print(f"{j:5}", end="")
    print()
    
    for i in range(CIFAR_CLASSES):
        print(f"{class_names[i]:>10} ", end="")
        for j in range(CIFAR_CLASSES):
            print(f"{confusion[i][j]:5}", end="")
        print()
    
    print("\n" + "=" * 40)
    
    return accuracy, confusion


def train_svm_cuml(train_features, train_labels, test_features, test_labels,
                   C=10.0, kernel='rbf', gamma='scale', model_name='default'):
    """Train SVM using cuML GPU acceleration."""
    USE_GPU = False
    
    try:
        import cuml
        from cuml.svm import SVC
        import cupy as cp
        USE_GPU = True
        print("Using cuML GPU-accelerated SVM")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"cuML not available ({type(e).__name__}), falling back to scikit-learn CPU SVM")
        print("To fix cuML on Colab, run:")
        print("  !pip uninstall -y cupy-cuda11x cupy-cuda12x")
        print("  !pip install cupy-cuda12x cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
        print()
        from sklearn.svm import SVC
        USE_GPU = False
    
    print("\n" + "=" * 40)
    print("Training SVM Classifier (GPU)" if USE_GPU else "Training SVM Classifier (CPU)")
    print("=" * 40)
    print(f"C: {C}")
    print(f"Kernel: {kernel}")
    print(f"Gamma: {gamma}")
    print(f"Samples: {len(train_labels)}")
    print(f"Features: {train_features.shape[1]}")
    print("=" * 40 + "\n")
    
    # Convert to GPU arrays if using cuML
    if USE_GPU:
        train_features_gpu = cp.asarray(train_features)
        train_labels_gpu = cp.asarray(train_labels)
        test_features_gpu = cp.asarray(test_features)
    
    # Create and train SVM
    print("Training SVM...")
    start_time = time.time()
    
    if USE_GPU:
        svm = SVC(C=C, kernel=kernel, gamma=gamma, verbose=True)
        svm.fit(train_features_gpu, train_labels_gpu)
    else:
        svm = SVC(C=C, kernel=kernel, gamma=gamma, verbose=True)
        svm.fit(train_features, train_labels)
    
    train_time = time.time() - start_time
    print(f"\nSVM training completed in {train_time:.2f}s")
    
    # Predict
    print("\nPredicting on test set...")
    start_time = time.time()
    
    if USE_GPU:
        predictions = svm.predict(test_features_gpu)
        predictions = cp.asnumpy(predictions).astype(np.int32)
    else:
        predictions = svm.predict(test_features).astype(np.int32)
    
    predict_time = time.time() - start_time
    print(f"SVM prediction completed in {predict_time:.2f}s")
    
    # Evaluate
    accuracy, confusion = print_evaluation(predictions, test_labels, CIFAR_CLASS_NAMES)
    
    # Save model
    model_path = f"models/svm_cuml_model_{model_name}.pkl"
    os.makedirs("models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f"\nModel saved to: {model_path}")
    
    # Print summary
    print("\n" + "=" * 40)
    print("SVM Pipeline Summary (cuML GPU)" if USE_GPU else "SVM Pipeline Summary (sklearn CPU)")
    print("=" * 40)
    print(f"SVM training time:   {train_time:.2f}s")
    print(f"SVM prediction time: {predict_time:.2f}s")
    print(f"Total time:          {train_time + predict_time:.2f}s")
    print("-" * 40)
    print(f"Test Accuracy:       {accuracy:.2f}%")
    print("=" * 40 + "\n")
    
    return accuracy, train_time, predict_time


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated SVM Training using cuML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default paths
    python train_svm_cuml.py --data /content/data/cifar-10-batches-bin

    # Train with custom feature files
    python train_svm_cuml.py --train models/train_features_gpu_opt_v1.bin \\
                             --test models/test_features_gpu_opt_v1.bin \\
                             --data /content/data/cifar-10-batches-bin \\
                             --model-name gpu_opt_v1

    # Use Linear kernel (faster, slightly lower accuracy)
    python train_svm_cuml.py --data /path/to/cifar --kernel linear
        """
    )
    
    parser.add_argument('--train', type=str, default='models/train_features_gpu.bin',
                        help='Path to training features binary file')
    parser.add_argument('--test', type=str, default='models/test_features_gpu.bin',
                        help='Path to test features binary file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CIFAR-10 binary data directory')
    parser.add_argument('--model-name', type=str, default='gpu',
                        help='Model variant name for saving (e.g., gpu, gpu_opt_v1)')
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM regularization parameter')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly'],
                        help='SVM kernel type')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='Kernel coefficient (scale, auto, or float)')
    
    args = parser.parse_args()
    
    print("=" * 40)
    print("cuML GPU SVM Training")
    print("CSC14120 - Parallel Programming")
    print("=" * 40)
    print(f"Train features: {args.train}")
    print(f"Test features:  {args.test}")
    print(f"CIFAR-10 data:  {args.data}")
    print(f"Model name:     {args.model_name}")
    print("=" * 40 + "\n")
    
    # Load labels
    print("Loading CIFAR-10 labels...")
    train_labels, test_labels = load_cifar_labels(args.data)
    print(f"  Train labels: {len(train_labels)}")
    print(f"  Test labels:  {len(test_labels)}")
    
    # Load features
    print("\nLoading features from binary files...")
    train_features = load_features(args.train, CIFAR_TRAIN_SIZE, FEATURE_DIM)
    test_features = load_features(args.test, CIFAR_TEST_SIZE, FEATURE_DIM)
    print(f"  Train features shape: {train_features.shape}")
    print(f"  Test features shape:  {test_features.shape}")
    
    # Parse gamma
    try:
        gamma = float(args.gamma)
    except ValueError:
        gamma = args.gamma  # 'scale' or 'auto'
    
    # Train and evaluate
    train_svm_cuml(
        train_features, train_labels,
        test_features, test_labels,
        C=args.C,
        kernel=args.kernel,
        gamma=gamma,
        model_name=args.model_name
    )


if __name__ == '__main__':
    main()
