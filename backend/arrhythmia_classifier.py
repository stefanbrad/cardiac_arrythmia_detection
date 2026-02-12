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

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.scaler     = StandardScaler()
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
        """
        Rule-based classifier for all 9 arrhythmia types.

        This runs on a 10-second window so HRV features are meaningful:
          - sdnn  (ms): std of RR intervals  - high = irregular rhythm
          - rmssd (ms): short-term HRV       - high = beat-to-beat variability
          - pnn50 (%):  % of consecutive RR pairs differing > 50 ms
          - hr    (bpm): beats per minute

        NOTE: features come from a window of ~10 s containing multiple beats.
        When called on a single beat these metrics are 0 and this function
        should not be relied upon — that was the original bug.
        """
        hr     = float(features.get("heart_rate", 75))
        sdnn   = float(features.get("sdnn",   50))
        rmssd  = float(features.get("rmssd",  40))
        pnn50  = float(features.get("pnn50",  10))

        # ── Ventricular Tachycardia: very fast + very regular (low HRV) ──────
        if hr > 150 and sdnn < 25:
            return 4  # VT

        # ── Bradycardia: slow heart rate ──────────────────────────────────────
        if hr < 55:
            return 6  # BRADY

        # ── Atrial Fibrillation: chaotic rhythm → high HRV at any rate ───────
        # AFIB has the highest HRV of all arrhythmias
        if sdnn > 80 and rmssd > 55 and pnn50 > 12:
            return 1  # AFIB

        # ── Atrial Flutter: fast + moderately regular ─────────────────────────
        if 100 < hr <= 175 and 20 < sdnn <= 70:
            return 2  # AFL

        # ── Supraventricular Tachycardia: fast + regular ──────────────────────
        if hr > 130 and sdnn < 40:
            return 5  # SVT

        # ── General Tachycardia ───────────────────────────────────────────────
        if hr > 100:
            return 7  # TACHY

        # ── PVC: normal rate but mildly elevated HRV ─────────────────────────
        if 55 <= hr <= 105 and (35 < sdnn <= 80 or 25 < rmssd <= 55):
            return 3  # PVC

        # ── Normal Sinus Rhythm ───────────────────────────────────────────────
        if 55 <= hr <= 100 and sdnn <= 50 and rmssd <= 40:
            return 0  # NSR

        return 8  # OTHER

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        val_acc   = self.model.score(X_val, y_val)
        y_pred    = self.model.predict(X_val)
        labels    = np.unique(y_val)

        return {
            "train_accuracy":   float(train_acc),
            "val_accuracy":     float(val_acc),
            "confusion_matrix": confusion_matrix(y_val, y_pred, labels=labels).tolist(),
        }

    def predict_single(self, features: np.ndarray) -> Dict:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        feature_dict = dict(zip(self.FEATURE_NAMES, features.flatten()))

        # No trained model → rule-based only
        if not self.is_trained:
            pred = self.classify_by_features(feature_dict)
            return self._format_result(pred, 0.6, source="rule_based")

        # ML prediction
        features_scaled = self.scaler.transform(features)
        probs    = self.model.predict_proba(features_scaled)[0]
        classes  = list(self.model.classes_)
        prob_dict = {self.ARRHYTHMIA_CODES.get(c, "OTHER"): float(p)
                     for c, p in zip(classes, probs)}

        best_idx      = int(np.argmax(probs))
        ml_pred_class = int(classes[best_idx])
        ml_conf       = float(probs[best_idx])

        # Low confidence → trust rule-based result, keep ML probs for reference
        if ml_conf < 0.40:
            pred = self.classify_by_features(feature_dict)
            return self._format_result(pred, ml_conf, prob_dict,
                                       source="rule_based_fallback")

        # Accept whatever the ML model predicted — no class whitelist
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
            "arrhythmia_code":     self.ARRHYTHMIA_CODES.get(prediction, "OTHER"),
            "arrhythmia_type":     self.ARRHYTHMIA_TYPES.get(prediction, "Other Arrhythmia"),
            "risk_level":          self.ARRHYTHMIA_RISK.get(prediction, "Unknown"),
            "confidence":          float(confidence),
            "source":              source,
            "all_probabilities":   probabilities or {},
        }

    def save_model(self, path: str):
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "model_type": self.model_type},
            path,
        )

    def load_model(self, path: str):
        data            = joblib.load(path)
        self.model      = data["model"]
        self.scaler     = data["scaler"]
        self.model_type = data["model_type"]
        self.is_trained = True

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))


def create_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, 15))
    y = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], size=n_samples)
    return X, y