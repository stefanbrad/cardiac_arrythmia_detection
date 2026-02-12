import os
import wfdb
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import argparse
from collections import Counter

from ecg_processor import ECGProcessor, extract_features_from_signal
from arrhythmia_classifier import ArrhythmiaClassifier


MITBIH_ANNOTATION_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 5, "a": 5, "J": 5, "S": 5,
    "V": 3, "E": 3, "F": 3,
    "/": 8, "f": 8, "Q": 8, "?": 8,
    "[": 4, "!": 4, "]": 0,
}


class MITBIHTrainer:
    def __init__(self, mitbih_path: str):
        self.mitbih_path = mitbih_path
        self.processor = ECGProcessor(sampling_rate=360)

        self.record_numbers = [
            100,101,102,103,104,105,106,107,108,109,
            111,112,113,114,115,116,117,118,119,121,
            122,123,124,200,201,202,203,205,207,208,
            209,210,212,213,214,215,217,219,220,221,
            222,223,228,230,231,232,233,234
        ]

    def _label_segment_presence(self, segment_annotations: List[int]) -> int:
        if not segment_annotations:
            return 0

        total = len(segment_annotations)
        counts = Counter(segment_annotations)

        if counts.get(4, 0) >= 1:
            return 4
        if counts.get(3, 0) >= 1:
            return 3
        if counts.get(5, 0) >= 1:
            return 5
        if counts.get(8, 0) >= 1:
            return 8

        return 0

    def load_record(self, record_number: int, segment_length: int = 3600):
        record_name = str(record_number)
        record_path = os.path.join(self.mitbih_path, record_name)

        print(f"\n Loading record {record_number}...")

        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        ecg_signal = record.p_signal[:, 0]

        ecg_processed = self.processor.preprocess_signal(ecg_signal, 360)

        segments = []
        labels = []

        num_segments = len(ecg_processed) // segment_length

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = ecg_processed[start:end]

            segment_annotations = []
            for j, ann_sample in enumerate(annotation.sample):
                if start <= ann_sample < end:
                    sym = annotation.symbol[j]
                    if sym in MITBIH_ANNOTATION_MAP:
                        segment_annotations.append(MITBIH_ANNOTATION_MAP[sym])

            label = self._label_segment_presence(segment_annotations)

            segments.append(segment)
            labels.append(label)

        print(f"   → {len(segments)} segments extracted")

        return segments, labels

    def extract_features_from_segments(self, segments):
        print("\n Extracting features...")
        feats = []
        for seg in tqdm(segments):
            f = extract_features_from_signal(seg, sampling_rate=360)
            feats.append(f)
        return np.array(feats)

    def load_all_records(self, segment_length=3600, max_records=None):
        print("\n====================================================")
        print("Loading MIT-BIH Dataset")
        print("====================================================")

        all_segments = []
        all_labels = []

        records = self.record_numbers[:max_records] if max_records else self.record_numbers

        for rn in records:
            segs, labs = self.load_record(rn, segment_length)
            all_segments.extend(segs)
            all_labels.extend(labs)

        X = self.extract_features_from_segments(all_segments)
        y = np.array(all_labels)

        print("\n Dataset Summary:")
        print(f"   Total samples: {len(y)}")
        print("   Class distribution:", Counter(y))

        return X, y

    def train(
        self,
        model_type="random_forest",
        segment_length=3600,
        max_records=None,
        balance=True,
        output_path="models/mitbih_arrhythmia_model.pkl",
    ):

        X, y = self.load_all_records(segment_length, max_records)

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        print("\n Splitting train / validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Train size: {len(y_train)}")
        print(f"   Validation size: {len(y_val)}")

        if balance:
            print("\n Balancing training data...")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            print("   Balanced distribution:", Counter(y_train))

        print("\n Initializing model...")
        classifier = ArrhythmiaClassifier(model_type=model_type)

        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_val_scaled = classifier.scaler.transform(X_val)

        print("\n Training model...")
        classifier.model.fit(X_train_scaled, y_train)
        classifier.is_trained = True

        train_acc = classifier.model.score(X_train_scaled, y_train)
        val_acc = classifier.model.score(X_val_scaled, y_val)

        print("\n====================================================")
        print(" Training Results")
        print("====================================================")
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, classifier.model.predict(X_val_scaled)))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        classifier.save_model(output_path)

        print(f"\n Model saved to: {output_path}")

        return classifier


def main():
    parser = argparse.ArgumentParser(description="Train MIT-BIH Arrhythmia Model")
    parser.add_argument("--mitbih-path", required=True)
    parser.add_argument("--model-type", default="random_forest")
    parser.add_argument("--segment-length", type=int, default=3600)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", default="models/mitbih_arrhythmia_model.pkl")
    args = parser.parse_args()

    trainer = MITBIHTrainer(args.mitbih_path)
    trainer.train(
        model_type=args.model_type,
        segment_length=args.segment_length,
        max_records=args.max_records,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
