"""
Train Arrhythmia Detection Model

This script trains a multi-class arrhythmia classifier using either:
1. Synthetic data (for demonstration)
2. Real ECG datasets (MIT-BIH, PTB-XL, etc.)

Usage:
    python train_model.py --mode synthetic --samples 10000
    python train_model.py --mode real --data-path /path/to/ecg/data
"""

import argparse
import numpy as np
import pandas as pd
from arrhythmia_classifier import ArrhythmiaClassifier, create_synthetic_training_data
from ecg_processor import ECGProcessor, extract_features_from_signal
import os
import glob
from tqdm import tqdm

def train_with_synthetic_data(n_samples: int = 10000, 
                              model_type: str = 'random_forest',
                              output_path: str = 'models/arrhythmia_model.pkl'):
    """
    Train model using synthetic data
    
    Args:
        n_samples: Number of training samples to generate
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        output_path: Path to save trained model
    """
    print("=" * 60)
    print("🎓 Training Arrhythmia Classifier with Synthetic Data")
    print("=" * 60)
    print(f"Generating {n_samples} synthetic training samples...")
    
    # Generate synthetic data
    X, y = create_synthetic_training_data(n_samples)
    
    print(f"Training set size: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({ArrhythmiaClassifier.ARRHYTHMIA_CODES[cls]}): {count} samples")
    
    # Initialize classifier
    print(f"\nInitializing {model_type} classifier...")
    classifier = ArrhythmiaClassifier(model_type=model_type)
    
    # Train
    print("\nTraining model...")
    metrics = classifier.train(X, y, validation_split=0.2)
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = classifier.get_feature_importance()
    for feature, importance in sorted(feature_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifier.save_model(output_path)
    
    print("\n✓ Training complete!")
    print(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")
    
    return classifier, metrics


def train_with_real_data(data_path: str,
                         model_type: str = 'random_forest',
                         output_path: str = 'models/arrhythmia_model.pkl'):
    """
    Train model using real ECG data
    
    Expected data structure:
    data_path/
        ├── normal/
        │   ├── ecg1.csv
        │   ├── ecg2.csv
        │   └── ...
        ├── afib/
        │   ├── ecg1.csv
        │   └── ...
        └── ...
    
    Or a CSV file with columns: [file_path, label]
    
    Args:
        data_path: Path to ECG data directory or CSV file
        model_type: Type of model
        output_path: Path to save trained model
    """
    print("=" * 60)
    print("🎓 Training Arrhythmia Classifier with Real Data")
    print("=" * 60)
    
    processor = ECGProcessor()
    
    # Load data
    X_list = []
    y_list = []
    
    # Label mapping
    label_map = {
        'normal': 0, 'nsr': 0,
        'afib': 1, 'af': 1,
        'afl': 2, 'flutter': 2,
        'pvc': 3,
        'vt': 4,
        'svt': 5,
        'brady': 6, 'bradycardia': 6,
        'tachy': 7, 'tachycardia': 7,
        'other': 8
    }
    
    if os.path.isfile(data_path):
        # Load from CSV file
        print(f"Loading data from CSV: {data_path}")
        df = pd.read_csv(data_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG files"):
            try:
                file_path = row['file_path']
                label = row['label'].lower()
                
                # Load and process ECG
                ecg_signal, _ = processor.process_ecg_for_classification(file_path)
                
                # Extract features
                features = extract_features_from_signal(ecg_signal)
                
                X_list.append(features)
                y_list.append(label_map.get(label, 8))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    elif os.path.isdir(data_path):
        # Load from directory structure
        print(f"Loading data from directory: {data_path}")
        
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            
            if not os.path.isdir(label_path):
                continue
            
            label = label_dir.lower()
            label_code = label_map.get(label, 8)
            
            print(f"\nProcessing {label} files...")
            
            # Find all ECG files
            ecg_files = []
            for ext in ['*.csv', '*.txt', '*.dat']:
                ecg_files.extend(glob.glob(os.path.join(label_path, ext)))
            
            for file_path in tqdm(ecg_files, desc=f"Processing {label}"):
                try:
                    # Load and process ECG
                    ecg_signal, _ = processor.process_ecg_for_classification(file_path)
                    
                    # Extract features
                    features = extract_features_from_signal(ecg_signal)
                    
                    X_list.append(features)
                    y_list.append(label_code)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nTotal samples loaded: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({ArrhythmiaClassifier.ARRHYTHMIA_CODES[cls]}): {count} samples")
    
    # Initialize and train classifier
    print(f"\nInitializing {model_type} classifier...")
    classifier = ArrhythmiaClassifier(model_type=model_type)
    
    print("\nTraining model...")
    metrics = classifier.train(X, y, validation_split=0.2)
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = classifier.get_feature_importance()
    for feature, importance in sorted(feature_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifier.save_model(output_path)
    
    print("\n✓ Training complete!")
    print(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")
    
    return classifier, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Arrhythmia Detection Model')
    
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'real'],
                       help='Training mode: synthetic or real data')
    
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of synthetic samples to generate (synthetic mode)')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to real ECG data directory or CSV file (real mode)')
    
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting'],
                       help='Type of model to train')
    
    parser.add_argument('--output', type=str, default='models/arrhythmia_model.pkl',
                       help='Output path for trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        train_with_synthetic_data(
            n_samples=args.samples,
            model_type=args.model_type,
            output_path=args.output
        )
    
    elif args.mode == 'real':
        if args.data_path is None:
            print("Error: --data-path is required for real data mode")
            return
        
        train_with_real_data(
            data_path=args.data_path,
            model_type=args.model_type,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
