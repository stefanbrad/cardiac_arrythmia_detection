import os
import wfdb
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import argparse

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
        counts = {c: segment_annotations.count(c) for c in set(segment_annotations)}

        vt = counts.get(4, 0)
        pvc = counts.get(3, 0)
        svt = counts.get(5, 0)
        oth = counts.get(8, 0)

        if vt >= 1:
            return 4
        if pvc / total >= 0.05 or pvc >= 1:
            return 3
        if svt / total >= 0.05 or svt >= 1:
            return 5
        if oth / total >= 0.05 or oth >= 1:
            return 8
        return 0

    def load_record(self, record_number: int, segment_length: int = 3600) -> Tuple[List[np.ndarray], List[int]]:
        record_name = str(record_number)
        record_path = os.path.join(self.mitbih_path, record_name)

        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, "atr")
            ecg_signal = record.p_signal[:, 0]

            ecg_processed = self.processor.preprocess_signal(ecg_signal, original_sampling_rate=360)

            segments = []
            labels = []

            num_segments = len(ecg_processed) // segment_length
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segment = ecg_processed[start_idx:end_idx]

                segment_annotations = []
                for j, ann_sample in enumerate(annotation.sample):
                    if start_idx <= ann_sample < end_idx:
                        sym = annotation.symbol[j]
                        if sym in MITBIH_ANNOTATION_MAP:
                            segment_annotations.append(MITBIH_ANNOTATION_MAP[sym])

                label = self._label_segment_presence(segment_annotations)

                segments.append(segment)
                labels.append(label)

            return segments, labels

        except Exception as e:
            print(f"Error loading record {record_number}: {e}")
            return [], []

    def extract_features_from_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        feats = []
        for seg in tqdm(segments, desc="Extracting features"):
            try:
                f = extract_features_from_signal(seg, sampling_rate=360)
                feats.append(f)
            except Exception:
                continue
        return np.array(feats)

    def load_all_records(self, segment_length: int = 3600, max_records: int = None) -> Tuple[np.ndarray, np.ndarray]:
        all_segments = []
        all_labels = []

        records_to_load = self.record_numbers[:max_records] if max_records else self.record_numbers

        for rn in tqdm(records_to_load, desc="Loading records"):
            segs, labs = self.load_record(rn, segment_length)
            if segs:
                all_segments.extend(segs)
                all_labels.extend(labs)

        X = self.extract_features_from_segments(all_segments)
        y = np.array(all_labels)

        return X, y

    def balance_train_only(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample TRAIN only to reduce imbalance WITHOUT leaking into validation.
        """
        from collections import Counter
        rng = np.random.default_rng(42)
        class_counts = Counter(y_train)
        max_count = max(class_counts.values())

        Xb = []
        yb = []
        for cls in class_counts.keys():
            idx = np.where(y_train == cls)[0]
            cls_X = X_train[idx]
            if len(cls_X) == 0:
                continue
            pick = rng.choice(len(cls_X), size=max_count, replace=True)
            Xb.append(cls_X[pick])
            yb.extend([cls] * max_count)

        Xb = np.vstack(Xb)
        yb = np.array(yb)
        p = rng.permutation(len(yb))
        return Xb[p], yb[p]

    def train(
        self,
        model_type: str = "random_forest",
        segment_length: int = 3600,
        max_records: int = None,
        balance: bool = True,
        output_path: str = "models/mitbih_arrhythmia_model.pkl",
    ) -> ArrhythmiaClassifier:
        X, y = self.load_all_records(segment_length, max_records)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if balance:
            X_train, y_train = self.balance_train_only(X_train, y_train)

        classifier = ArrhythmiaClassifier(model_type=model_type)

        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_val_scaled = classifier.scaler.transform(X_val)

        classifier.model.fit(X_train_scaled, y_train)
        classifier.is_trained = True

        train_acc = classifier.model.score(X_train_scaled, y_train)
        val_acc = classifier.model.score(X_val_scaled, y_val)
        y_pred = classifier.model.predict(X_val_scaled)

        labels = np.unique(y_val)
        from sklearn.metrics import classification_report

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        classifier.save_model(output_path)
        return classifier


def main():
    parser = argparse.ArgumentParser(description="Train arrhythmia classifier on MIT-BIH Database (Improved)")
    parser.add_argument("--mitbih-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "gradient_boosting"])
    parser.add_argument("--segment-length", type=int, default=3600)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--output", type=str, default="models/mitbih_arrhythmia_model.pkl")
    args = parser.parse_args()

    if not os.path.exists(args.mitbih_path):
        print(f"Error: MIT-BIH path not found: {args.mitbih_path}")
        return

    trainer = MITBIHTrainer(args.mitbih_path)
    trainer.train(
        model_type=args.model_type,
        segment_length=args.segment_length,
        max_records=args.max_records,
        balance=(not args.no_balance),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
