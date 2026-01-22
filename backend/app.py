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


MODEL_PATH = os.environ.get("MODEL_PATH", "models/mitbih_arrhythmia_model.pkl")
MAX_FILE_MB = 50
FS_DEFAULT = 360  # MIT-BIH uses 360Hz
MAX_BEATS_TO_ANALYZE = 500  # limit beats for performance

EVENT_MIN_COUNT = 2          
EVENT_MIN_PERCENT = 0.05     

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

processor = ECGProcessor(sampling_rate=FS_DEFAULT)

classifier = ArrhythmiaClassifier(model_type="random_forest")
if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
else:
    print(f"[WARN] Model not found at {MODEL_PATH}. "
          f"Backend will still run, but predictions may be rule-based.")


def json_error(message: str, status: int = 400, **extra):
    return jsonify({"error": message, **extra}), status


def get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def allowed_ext(filename: str) -> bool:
    return get_ext(filename) in {"csv", "txt", "json", "zip", "dat", "hea"}


def read_signal_from_csv(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Empty CSV")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        ln = ln.replace(";", ",")
        parts = [p.strip() for p in ln.split(",") if p.strip() != ""]
        if not parts:
            continue
        if any(ch.isalpha() for ch in parts[0]):
            continue
        rows.append(parts)

    if not rows:
        raise ValueError("CSV has no numeric rows")

    amps = []
    for r in rows:
        try:
            amps.append(float(r[-1]))
        except:
            continue

    if len(amps) < 200:
        raise ValueError("CSV does not contain enough samples (need >= 200).")

    return np.array(amps, dtype=float)


def read_signal_from_txt(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Empty TXT")

    text = text.replace(",", ".")
    parts = text.split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass

    if len(vals) < 200:
        raise ValueError("TXT does not contain enough samples (need >= 200).")

    return np.array(vals, dtype=float)


def read_signal_from_json(file_bytes: bytes) -> np.ndarray:
    obj = json.loads(file_bytes.decode("utf-8", errors="ignore"))
    if isinstance(obj, list):
        arr = obj
    elif isinstance(obj, dict):
        arr = obj.get("signal") or obj.get("ecg") or obj.get("data")
    else:
        arr = None

    if not isinstance(arr, list) or len(arr) < 200:
        raise ValueError("JSON must contain a list of samples (signal/ecg/data) with >= 200 points.")

    vals = []
    for x in arr:
        try:
            vals.append(float(x))
        except:
            pass

    if len(vals) < 200:
        raise ValueError("JSON does not contain enough numeric samples.")

    return np.array(vals, dtype=float)


def read_signal_from_mitbih_zip(zip_bytes: bytes) -> np.ndarray:
    """
    Robust ZIP reader for MIT-BIH:
    - supports subfolders in zip
    - finds a .hea that has matching .dat with same stem name
    """
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
            record_stem = hea_path.with_suffix("")  # path without extension
            dat_path = record_stem.with_suffix(".dat")
            if dat_path.exists():
                record = wfdb.rdrecord(str(record_stem))
                signal = record.p_signal[:, 0].astype(float)  # channel 0
                return signal

        raise ValueError("ZIP must contain matching .dat for at least one .hea file (same filename).")


def detect_r_peaks_simple(signal: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    std = np.std(x) + 1e-9
    x = x / std

    win = int(0.12 * fs)
    win = max(win, 5)
    energy = np.convolve(x * x, np.ones(win) / win, mode="same")

    thr = np.percentile(energy, 95)
    candidates = np.where(energy > thr)[0]
    if candidates.size == 0:
        return np.array([], dtype=int)

    refractory = int(0.25 * fs)
    peaks = []
    last = -10**9
    for idx in candidates:
        if idx - last >= refractory:
            peaks.append(idx)
            last = idx
    return np.array(peaks, dtype=int)


def heart_rate_from_peaks(peaks: np.ndarray, fs: int = FS_DEFAULT) -> float:
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    rr = rr[rr > 0]
    if len(rr) == 0:
        return 0.0
    return float(60.0 / np.mean(rr))


def classify_beats(signal_processed: np.ndarray, fs: int = FS_DEFAULT) -> dict:
    peaks = detect_r_peaks_simple(signal_processed, fs=fs)

    if len(peaks) > MAX_BEATS_TO_ANALYZE:
        idx = np.linspace(0, len(peaks) - 1, num=MAX_BEATS_TO_ANALYZE, dtype=int)
        peaks = peaks[idx]

    hr = heart_rate_from_peaks(peaks, fs=fs)

    pre = int(0.30 * fs)
    post = int(0.50 * fs)

    code_counts: dict[str, int] = {}

    for p in peaks:
        a = max(0, p - pre)
        b = min(len(signal_processed), p + post)
        beat = signal_processed[a:b]

        if len(beat) < int(0.5 * fs):
            continue

        feats = extract_features_from_signal(beat, sampling_rate=fs)
        feats = np.array(feats, dtype=float)

        pred = classifier.predict_single(feats)
        code = pred.get("arrhythmia_code", "OTHER")

        code_counts[code] = code_counts.get(code, 0) + 1

    total_beats = int(sum(code_counts.values()))
    normal_beats = int(code_counts.get("NSR", 0))
    abnormal_beats = int(total_beats - normal_beats)

    events = []
    if total_beats > 0:
        for code, count in code_counts.items():
            if code == "NSR":
                continue
            pct = count / total_beats
            if count >= EVENT_MIN_COUNT or pct >= EVENT_MIN_PERCENT:
                events.append({
                    "code": code,
                    "label": code,
                    "count": int(count),
                    "percent": float(pct),
                })
        events.sort(key=lambda e: e["count"], reverse=True)

    summary = {
        "heart_rate_bpm": int(round(hr)) if hr > 0 else 0,
        "total_beats": total_beats,
        "abnormal_beats": abnormal_beats,
        "normal_beats": normal_beats,
        "abnormal_percent": float(abnormal_beats / total_beats) if total_beats else 0.0,
        "normal_percent": float(normal_beats / total_beats) if total_beats else 0.0,
    }

    return {
        "peaks": peaks,
        "summary": summary,
        "events": events,
        "code_counts": code_counts,
    }


def segment_probabilities(signal_processed: np.ndarray, fs: int = FS_DEFAULT) -> dict:
    feats = extract_features_from_signal(signal_processed, sampling_rate=fs)
    feats = np.array(feats, dtype=float)
    return classifier.predict_single(feats)


def dominant_code(code_counts: dict) -> str:
    if not code_counts:
        return "OTHER"
    return max(code_counts.items(), key=lambda x: x[1])[0]

def sample_windows(
    signal: np.ndarray,
    fs: int,
    window_seconds: int = 10,
    n_windows: int = 6,
) -> np.ndarray:
    """
    Adaptive window sampling:
    - if signal is shorter than one window -> return full signal
    - if signal is shorter than total requested duration -> return full signal
    - otherwise sample n_windows windows uniformly across signal
    """

    win_len = int(window_seconds * fs)
    total_requested = win_len * n_windows
    signal_len = len(signal)

    # case 1: very short signal
    if signal_len <= win_len:
        return signal

    # case 2: signal shorter than requested total duration
    if signal_len <= total_requested:
        return signal

    # case 3: long signal -> sample windows
    max_start = signal_len - win_len
    starts = np.linspace(0, max_start, num=n_windows, dtype=int)

    chunks = [signal[s:s + win_len] for s in starts]
    return np.concatenate(chunks)



@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": bool(classifier.is_trained)})


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

        ext = get_ext(f.filename)
        file_bytes = f.read()
        if not file_bytes:
            return json_error("Uploaded file is empty.", 400)

        if ext == "dat":
            return json_error(
                "MIT-BIH .dat requires matching .hea. Upload a .zip containing both .dat and .hea.",
                400,
            )
        if ext == "hea":
            return json_error(
                ".hea alone is not enough. Upload a .zip containing both .hea and .dat.",
                400,
            )

        if ext == "csv":
            signal = read_signal_from_csv(file_bytes)
        elif ext == "txt":
            signal = read_signal_from_txt(file_bytes)
        elif ext == "json":
            signal = read_signal_from_json(file_bytes)
        elif ext == "zip":
            signal = read_signal_from_mitbih_zip(file_bytes)
        else:
            return json_error("Unsupported file type.", 400)

        print(
            f"[UPLOAD] {f.filename} ext={ext} len={len(signal)} "
            f"min={float(np.min(signal)):.4f} max={float(np.max(signal)):.4f} "
            f"mean={float(np.mean(signal)):.4f} std={float(np.std(signal)):.4f}"
        )

        signal = sample_windows(
            signal,
            fs=FS_DEFAULT,
            window_seconds=10,
            n_windows=6
        )


        signal_processed = processor.preprocess_signal(signal, original_sampling_rate=FS_DEFAULT)

        # beat-level stats + events
        beat_info = classify_beats(signal_processed, fs=FS_DEFAULT)
        base_rhythm_code = dominant_code(beat_info["code_counts"])  # often NSR
        main_arrhythmia_code = beat_info["events"][0]["code"] if beat_info["events"] else "NSR"

        # segment-level probabilities 
        seg_pred = segment_probabilities(signal_processed, fs=FS_DEFAULT)

        s = beat_info["summary"]
        response = {
            "base_rhythm_code": base_rhythm_code,
            "arrhythmia_detected": len(beat_info["events"]) > 0,
            "main_arrhythmia_code": main_arrhythmia_code,

            "arrhythmia_code": main_arrhythmia_code,
            "arrhythmia_type": ArrhythmiaClassifier.ARRHYTHMIA_TYPES.get(
                ArrhythmiaClassifier._get_code_index_static(main_arrhythmia_code), "Unknown"
            ),
            "risk_level": ArrhythmiaClassifier.ARRHYTHMIA_RISK.get(
                ArrhythmiaClassifier._get_code_index_static(main_arrhythmia_code), "Unknown"
            ),

            "confidence": float(seg_pred.get("confidence", 0.5)),
            "source": seg_pred.get("source", "ml"),
            "all_probabilities": seg_pred.get("all_probabilities", {}),

            "summary": beat_info["summary"],
            "events": beat_info["events"],
        }

        response["message"] = (
            f"Analysis identified {s['abnormal_beats']} abnormal beats out of {s['total_beats']} total."
            if s["total_beats"] > 0
            else "Not enough beats detected to provide beat statistics."
        )

        return jsonify(response)

    except zipfile.BadZipFile:
        return json_error("Invalid ZIP file.", 400)
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        # always JSON
        print("[ERROR]", repr(e))
        return json_error("Internal server error during prediction.", 500, detail=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
