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
    X, y = create_synthetic_training_data(n_samples)
    classifier = ArrhythmiaClassifier(model_type=model_type)
    metrics = classifier.train(X, y, validation_split=0.2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifier.save_model(output_path)
    return classifier, metrics


def train_with_real_data(data_path: str,
                         model_type: str = 'random_forest',
                         output_path: str = 'models/arrhythmia_model.pkl'):
    
    processor = ECGProcessor()
    
    X_list = []
    y_list = []
    
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
        df = pd.read_csv(data_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG files"):
            try:
                file_path = row['file_path']
                label = row['label'].lower()
                ecg_signal, _ = processor.process_ecg_for_classification(file_path)
                features = extract_features_from_signal(ecg_signal)
                X_list.append(features)
                y_list.append(label_map.get(label, 8))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    elif os.path.isdir(data_path):
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            
            if not os.path.isdir(label_path):
                continue
            
            label = label_dir.lower()
            label_code = label_map.get(label, 8)
            
            ecg_files = []
            for ext in ['*.csv', '*.txt', '*.dat']:
                ecg_files.extend(glob.glob(os.path.join(label_path, ext)))
            
            for file_path in tqdm(ecg_files, desc=f"Processing {label}"):
                try:
                    ecg_signal, _ = processor.process_ecg_for_classification(file_path)
                    features = extract_features_from_signal(ecg_signal)
                    X_list.append(features)
                    y_list.append(label_code)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    classifier = ArrhythmiaClassifier(model_type=model_type)
    metrics = classifier.train(X, y, validation_split=0.2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifier.save_model(output_path)
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
