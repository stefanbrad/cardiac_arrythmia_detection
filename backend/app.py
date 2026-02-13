import os
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import wfdb

from ecg_processor import ECGProcessor, extract_features_from_signal
from arrhythmia_classifier import ArrhythmiaClassifier


MODEL_PATH  = os.environ.get("MODEL_PATH", "models/mitbih_arrhythmia_model.pkl")
MAX_FILE_MB = 50
FS_DEFAULT  = 360  # MIT-BIH standard

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

processor  = ECGProcessor(sampling_rate=FS_DEFAULT)
classifier = ArrhythmiaClassifier(model_type="random_forest")

if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
else:
    print(f"[WARN] Model not found at {MODEL_PATH}. Using rule-based classification.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def json_error(message: str, status: int = 400, **extra):
    return jsonify({"error": message, **extra}), status

def get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def allowed_ext(filename: str) -> bool:
    return get_ext(filename) in {"csv", "txt", "json", "zip", "dat", "hea"}


# ── File readers ──────────────────────────────────────────────────────────────

def read_signal_from_csv(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Empty CSV file.")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows  = []
    for ln in lines:
        ln    = ln.replace(";", ",")
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        if not parts:
            continue
        if any(ch.isalpha() for ch in parts[0]):
            continue
        rows.append(parts)
    if not rows:
        raise ValueError("CSV has no numeric rows.")
    amps = []
    for r in rows:
        try:
            amps.append(float(r[-1]))
        except Exception:
            continue
    if len(amps) < 200:
        raise ValueError("CSV does not contain enough samples (need >= 200).")
    return np.array(amps, dtype=float)


def read_signal_from_txt(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Empty TXT file.")
    text = text.replace(",", ".")
    vals = []
    for p in text.split():
        try:
            vals.append(float(p))
        except Exception:
            pass
    if len(vals) < 200:
        raise ValueError("TXT does not contain enough samples (need >= 200).")
    return np.array(vals, dtype=float)


def read_signal_from_json(file_bytes: bytes) -> np.ndarray:
    obj = json.loads(file_bytes.decode("utf-8", errors="ignore"))
    if isinstance(obj, list):
        arr = obj
    elif isinstance(obj, dict):
        arr = (obj.get("signal") or obj.get("ecg")
               or obj.get("data") or obj.get("samples"))
    else:
        arr = None
    if not isinstance(arr, list) or len(arr) < 200:
        raise ValueError(
            "JSON must contain a list of >= 200 samples under key "
            "'signal', 'ecg', 'data', or 'samples'."
        )
    vals = []
    for x in arr:
        try:
            vals.append(float(x))
        except Exception:
            pass
    if len(vals) < 200:
        raise ValueError("JSON does not contain enough numeric samples.")
    return np.array(vals, dtype=float)


def read_signal_from_mitbih_zip(zip_bytes: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "upload.zip")
        with open(zpath, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(tmpdir)
        hea_files = list(Path(tmpdir).rglob("*.hea"))
        if not hea_files:
            raise ValueError("ZIP must contain a .hea file (MIT-BIH header).")
        for hea_path in hea_files:
            stem     = hea_path.with_suffix("")
            dat_path = stem.with_suffix(".dat")
            if dat_path.exists():
                record = wfdb.rdrecord(str(stem))
                return record.p_signal[:, 0].astype(float)
        raise ValueError("ZIP must contain matching .dat for at least one .hea file.")


# ── R-peak detection ──────────────────────────────────────────────────────────

def detect_r_peaks(sig: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    from scipy.signal import find_peaks
    x = np.asarray(sig, dtype=float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)
    dx     = np.diff(x, prepend=x[0])
    win    = max(int(0.08 * fs), 3)
    energy = np.convolve(dx ** 2, np.ones(win) / win, mode="same")
    thr    = np.percentile(energy, 75)
    peaks, _ = find_peaks(energy, height=thr, distance=int(0.25 * fs))
    # refine to true signal peak within +-40 ms
    half = int(0.04 * fs)
    refined = []
    for p in peaks:
        lo = max(0, p - half)
        hi = min(len(sig), p + half + 1)
        refined.append(int(lo + np.argmax(np.abs(sig[lo:hi]))))
    return np.array(refined, dtype=int)


def heart_rate_from_peaks(peaks: np.ndarray, fs: int = FS_DEFAULT) -> float:
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    return float(60.0 / np.mean(rr)) if len(rr) > 0 else 0.0



def classify_windows(signal_processed: np.ndarray, fs: int = FS_DEFAULT) -> dict:
    win_len = int(10 * fs)   # 10 s = 3600 samples at 360 Hz
    step    = int(5  * fs)   # 50 % overlap
    n       = len(signal_processed)

    if n < win_len:
        windows = [signal_processed]
    else:
        starts  = range(0, n - win_len + 1, step)
        windows = [signal_processed[s: s + win_len] for s in starts]

    code_counts: dict[str, int] = {}
    for win in windows:
        feats = extract_features_from_signal(win, sampling_rate=fs)
        pred  = classifier.predict_single(np.array(feats, dtype=float))
        code  = pred.get("arrhythmia_code", "OTHER")
        code_counts[code] = code_counts.get(code, 0) + 1

    total_w   = max(sum(code_counts.values()), 1)
    normal_w  = code_counts.get("NSR", 0)
    abnorm_w  = total_w - normal_w

    events = []
    for code, count in code_counts.items():
        if code == "NSR":
            continue
        events.append({
            "code":    code,
            "label":   code,
            "count":   int(count),
            "percent": round(count / total_w, 4),
        })
    events.sort(key=lambda e: e["count"], reverse=True)

    # Heart rate + beat counts from the full signal
    peaks         = detect_r_peaks(signal_processed, fs=fs)
    hr            = heart_rate_from_peaks(peaks, fs=fs)
    total_beats   = len(peaks)
    abnorm_frac   = abnorm_w / total_w
    abnorm_beats  = int(round(total_beats * abnorm_frac))

    summary = {
        "heart_rate_bpm":   int(round(hr)) if hr > 0 else 0,
        "total_beats":      total_beats,
        "normal_beats":     total_beats - abnorm_beats,
        "abnormal_beats":   abnorm_beats,
        "normal_percent":   round(1 - abnorm_frac, 4),
        "abnormal_percent": round(abnorm_frac, 4),
    }

    return {
        "events":      events,
        "summary":     summary,
        "code_counts": code_counts,
        "peaks":       peaks,
    }


# ── Final decision ────────────────────────────────────────────────────────────

def final_decision(window_info: dict, seg_pred: dict) -> tuple[str, bool]:
    # 1. Trust window-level events first
    if window_info["events"]:
        return window_info["events"][0]["code"], True
    # 2. Fall back to whole-segment prediction if confident enough
    seg_code = seg_pred.get("arrhythmia_code", "NSR")
    seg_conf = float(seg_pred.get("confidence", 0.0))
    if seg_code != "NSR" and seg_conf >= 0.40:
        return seg_code, True
    return "NSR", False


def dominant_code(code_counts: dict) -> str:
    if not code_counts:
        return "OTHER"
    return max(code_counts.items(), key=lambda x: x[1])[0]


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": bool(classifier.is_trained),
    })


@app.post("/api/predict")
def predict():
    try:
        if "file" not in request.files:
            return json_error("No file provided. Use form-data field 'file'.", 400)

        f = request.files["file"]
        if not f or f.filename == "":
            return json_error("Empty filename.", 400)

        if not allowed_ext(f.filename):
            return json_error("Unsupported file type.", 400, filename=f.filename)

        ext        = get_ext(f.filename)
        file_bytes = f.read()

        if not file_bytes:
            return json_error("Uploaded file is empty.", 400)

        if ext == "dat":
            return json_error(
                "MIT-BIH .dat requires a matching .hea. "
                "Upload a .zip containing both files.", 400)
        if ext == "hea":
            return json_error(
                ".hea alone is not enough. "
                "Upload a .zip containing both .hea and .dat.", 400)

        # Parse
        if ext == "csv":
            raw_signal = read_signal_from_csv(file_bytes)
        elif ext == "txt":
            raw_signal = read_signal_from_txt(file_bytes)
        elif ext == "json":
            raw_signal = read_signal_from_json(file_bytes)
        elif ext == "zip":
            raw_signal = read_signal_from_mitbih_zip(file_bytes)
        else:
            return json_error("Unsupported file type.", 400)

        # Pre-process
        signal_processed = processor.preprocess_signal(
            raw_signal, original_sampling_rate=FS_DEFAULT
        )

        # Window-based classification (primary)
        window_info = classify_windows(signal_processed, fs=FS_DEFAULT)

        # Whole-segment classification (secondary / tiebreaker)
        seg_pred = classifier.predict_single(
            np.array(
                extract_features_from_signal(signal_processed, sampling_rate=FS_DEFAULT),
                dtype=float
            )
        )

        # Combine
        final_code, arrhythmia_detected = final_decision(window_info, seg_pred)
        final_idx  = ArrhythmiaClassifier._get_code_index_static(final_code)
        final_type = ArrhythmiaClassifier.ARRHYTHMIA_TYPES.get(final_idx, "Unknown")
        final_risk = ArrhythmiaClassifier.ARRHYTHMIA_RISK.get(final_idx, "Unknown")
        confidence = float(seg_pred.get("confidence", 0.5))

        s = window_info["summary"]
        response = {
            "arrhythmia_detected":  arrhythmia_detected,
            "arrhythmia_code":      final_code,
            "arrhythmia_type":      final_type,
            "risk_level":           final_risk,
            "confidence":           confidence,
            "source":               seg_pred.get("source", "rule_based"),
            "base_rhythm_code":     dominant_code(window_info["code_counts"]),
            "main_arrhythmia_code": final_code,
            "all_probabilities":    seg_pred.get("all_probabilities", {}),
            "summary":              s,
            "events":               window_info["events"],
            "message": (
                f"Analysis identified {s['abnormal_beats']} abnormal beat(s) "
                f"out of {s['total_beats']} total."
                if s["total_beats"] > 0
                else "Not enough beats detected. "
                     "Check that the file contains a valid ECG signal."
            ),
        }

        return jsonify(response)

    except zipfile.BadZipFile:
        return json_error("Invalid ZIP file.", 400)
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        return json_error(
            "Internal server error during prediction.", 500, detail=str(e)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)