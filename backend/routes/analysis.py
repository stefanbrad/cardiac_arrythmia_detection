from flask import Blueprint, request, jsonify
import wfdb
import numpy as np
import joblib
import os
from datetime import datetime
import scipy.signal

bp = Blueprint('analysis', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'ecg_model.pkl')

# MAP FOR PREDICTIONS
PREDICTION_MAP = {
    0: {'name': 'Normal Beat', 'type': 'green', 'code': 'N'},
    1: {'name': 'Left Bundle Branch Block', 'type': 'orange', 'code': 'LBBB'},
    2: {'name': 'Right Bundle Branch Block', 'type': 'orange', 'code': 'RBBB'},
    3: {'name': 'Atrial Premature Beat', 'type': 'red', 'code': 'APB'},
    4: {'name': 'Premature Ventricular Contraction', 'type': 'red', 'code': 'PVC'}
}

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Model was not found at path")
except Exception as e:
    print(f"Error loading model")

@bp.route('/analyze', methods=['POST'])
def analyze():
    if not model:
        return jsonify({'error': 'AI Model not loaded.'}), 500

    try:
        data = request.get_json()
        record_id = data.get('record_id')
        
        if not record_id:
            return jsonify({'error': 'Record ID is required'}), 400

        try:
            record = wfdb.rdrecord(record_id, pn_dir='mitdb', channels=[0])
        except:
             return jsonify({'error': 'Record not found in MIT-BIH database'}), 404
        
        signal_data = record.p_signal[:2000, 0]
        peaks, _ = scipy.signal.find_peaks(signal_data, distance=150, height=0.5)
        
        window_size = 30
        timeline = []
        beat_confidences = []
        
        # contoare pentru fiecare diagnostic
        diagnosis_counts = {k: 0 for k in PREDICTION_MAP.keys()}
        
        for peak in peaks:
            # limitele segmentului analizat de AI
            start_index = peak - window_size
            end_index = peak + window_size

            if start_index < 0 or end_index >= len(signal_data):
                continue
                
            segment = signal_data[start_index : end_index]
            
            # predictie
            pred_class = int(model.predict([segment])[0])
            diagnosis_counts[pred_class] += 1
            
            # probabilitate (Siguranta modelului)
            probabilities = model.predict_proba([segment])[0]
            confidence_score = np.max(probabilities) * 100
            beat_confidences.append(confidence_score)
            
            timestamp = peak / record.fs
            
            # adaugam in timeline doar daca NU este normal (0)
            if pred_class != 0:
                disease_info = PREDICTION_MAP[pred_class]
                timeline.append({
                    'time': f"{timestamp:.2f}s",
                    'start_index': int(start_index),
                    'end_index': int(end_index),
                    'type': disease_info['code'],
                    'description': disease_info['name'],
                    'color': 'linear-gradient(135deg, #ef4444, #ec4899)' if disease_info['type'] == 'red' else 'linear-gradient(135deg, #f97316, #fbbf24)'
                })

        # final stats
        total_beats = sum(diagnosis_counts.values())
        normal_beats = diagnosis_counts[0]
        abnormal_beats = total_beats - normal_beats
        
        events_list = []
        main_diagnosis = "Normal Sinus Rhythm"
        
        detected_issues = []
        for code, count in diagnosis_counts.items():
            if code != 0 and count > 0:
                info = PREDICTION_MAP[code]
                events_list.append({
                    'name': info['name'],
                    'count': count,
                    'type': info['type']
                })
                detected_issues.append(info['code'])

        if detected_issues:
            main_diagnosis = f"Arrhythmia Detected: {', '.join(set(detected_issues))}"

        final_confidence = int(np.mean(beat_confidences)) if beat_confidences else 0
        
        response_data = {
            'analysis_id': f"{record_id}-{int(datetime.now().timestamp())}",
            'diagnosis': main_diagnosis,
            'description': f"Analysis identified {abnormal_beats} abnormal beats out of {total_beats} total.",
            'confidence': final_confidence,
            'normal_beats': round(normal_beats/total_beats * 100) if total_beats else 0,
            'abnormal_beats': round(abnormal_beats/total_beats * 100) if total_beats else 0,
            'heart_rate': int(total_beats * (60 / (len(signal_data)/record.fs))) if total_beats else 0,
            'events': events_list,
            'timeline': timeline,
            'ecg_data': signal_data.tolist()
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500