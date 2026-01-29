import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class ArrhythmiaClassifier:
    ARRHYTHMIA_TYPES = {
        0: "Normal Sinus Rhythm",
        1: "Atrial Fibrillation",
        2: "Atrial Flutter",
        3: "Premature Ventricular Contraction",
        4: "Ventricular Tachycardia",
        5: "Supraventricular Tachycardia",
        6: "Bradycardia",
        7: "Tachycardia",
        8: "Other Arrhythmia",
    }

    ARRHYTHMIA_CODES = {
        0: "NSR",
        1: "AFIB",
        2: "AFL",
        3: "PVC",
        4: "VT",
        5: "SVT",
        6: "BRADY",
        7: "TACHY",
        8: "OTHER",
    }

    ARRHYTHMIA_RISK = {
        0: "Low",
        1: "High",
        2: "Moderate",
        3: "Moderate",
        4: "Very High",
        5: "Moderate",
        6: "Moderate",
        7: "Moderate",
        8: "Unknown",
    }

    FEATURE_NAMES = [
        "heart_rate",
        "mean_rr",
        "sdnn",
        "rmssd",
        "pnn50",
        "qrs_duration",
        "pr_interval",
        "qt_interval",
        "p_wave_amplitude",
        "r_wave_amplitude",
        "t_wave_amplitude",
        "signal_mean",
        "signal_std",
        "signal_max",
        "signal_min",
    ]

    MITBIH_TRAINED_CLASSES = {0, 3, 4, 5, 8}

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _get_code_index_static(code: str) -> int:
        for idx, c in ArrhythmiaClassifier.ARRHYTHMIA_CODES.items():
            if c == code:
                return idx
        return 8

    def classify_by_features(self, features: Dict[str, float]) -> int:
        hr = float(features.get("heart_rate", 75))
        sdnn = float(features.get("sdnn", 50))
        rmssd = float(features.get("rmssd", 40))
        pnn50 = float(features.get("pnn50", 10))

        if hr < 55:
            return 6

        if hr > 110:
            if sdnn > 100 or rmssd > 80:
                return 1 if pnn50 > 20 else 7
            return 4 if hr > 160 else 5

        if sdnn > 90 or rmssd > 70:
            return 1 if pnn50 > 15 else 3

        if 60 <= hr <= 100 and sdnn < 50:
            return 0

        return 8

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        y_pred = self.model.predict(X_val)
        labels = np.unique(y_val)

        return {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "confusion_matrix": confusion_matrix(y_val, y_pred, labels=labels).tolist(),
        }

    def predict_single(self, features: np.ndarray) -> Dict:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        feature_dict = dict(zip(self.FEATURE_NAMES, features.flatten()))

        if not self.is_trained:
            rule_pred = self.classify_by_features(feature_dict)
            return self._format_result(rule_pred, 0.5, source="rule_based")

        features_scaled = self.scaler.transform(features)

        probs = self.model.predict_proba(features_scaled)[0]
        classes = list(self.model.classes_)

        prob_dict = {self.ARRHYTHMIA_CODES[c]: float(p) for c, p in zip(classes, probs)}

        best_idx = int(np.argmax(probs))
        ml_pred_class = int(classes[best_idx])
        ml_conf = float(probs[best_idx])

        if ml_conf < 0.55:
            rule_pred = self.classify_by_features(feature_dict)
            return self._format_result(rule_pred, ml_conf, prob_dict, source="hybrid_rule_fallback")

        if ml_pred_class not in self.MITBIH_TRAINED_CLASSES:
            ml_pred_class = 8

        return self._format_result(ml_pred_class, ml_conf, prob_dict, source="ml")

    def _format_result(
        self,
        prediction: int,
        confidence: float,
        probabilities: Dict[str, float] | None = None,
        source: str = "ml",
    ) -> Dict:
        prediction = int(prediction)
        return {
            "arrhythmia_detected": prediction != 0,
            "arrhythmia_code": self.ARRHYTHMIA_CODES.get(prediction, "OTHER"),
            "arrhythmia_type": self.ARRHYTHMIA_TYPES.get(prediction, "Other Arrhythmia"),
            "risk_level": self.ARRHYTHMIA_RISK.get(prediction, "Unknown"),
            "confidence": float(confidence),
            "source": source,
            "all_probabilities": probabilities or {},
        }

    def save_model(self, path: str):
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "model_type": self.model_type},
            path,
        )

    def load_model(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]
        self.is_trained = True

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))


def create_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, 15))
    y = np.random.choice([0, 3, 4, 5, 8], size=n_samples)
    return X, y
