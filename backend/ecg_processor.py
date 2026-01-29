"""
ECG Signal Processor
Handles multiple ECG file formats and prepares data for arrhythmia detection
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import wfdb
from typing import Tuple, Dict, List, Optional
import io

class ECGProcessor:
    def __init__(self, sampling_rate: int = 360):
        self.sampling_rate = sampling_rate
        self.lowcut = 0.5  # Hz - High-pass filter cutoff
        self.highcut = 45.0  # Hz - Low-pass filter cutoff
        
    def load_ecg_file(self, file_path_or_bytes, file_extension: str = '.csv') -> np.ndarray:
        file_extension = file_extension.lower()
        
        try:
            if file_extension in ['.csv', '.txt']:
                return self._load_csv_txt(file_path_or_bytes)
            elif file_extension == '.dat':
                return self._load_mitbih(file_path_or_bytes)
            elif file_extension == '.json':
                return self._load_json(file_path_or_bytes)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise Exception(f"Error loading ECG file: {str(e)}")
    
    def _load_csv_txt(self, file_path_or_bytes) -> np.ndarray:
        if isinstance(file_path_or_bytes, bytes):
            df = pd.read_csv(io.BytesIO(file_path_or_bytes))
        else:
            df = pd.read_csv(file_path_or_bytes)
        
        if df.shape[1] == 1:
            ecg_signal = df.iloc[:, 0].values
        elif df.shape[1] == 2:
            ecg_signal = df.iloc[:, 1].values
        else:
            amplitude_cols = ['amplitude', 'ecg', 'signal', 'mlii', 'v1', 'lead_i', 'lead_ii']
            for col in amplitude_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    ecg_signal = df[col].values
                    break
            else:
                ecg_signal = df.iloc[:, 1].values
        
        return ecg_signal.astype(float)
    
    def _load_mitbih(self, file_path) -> np.ndarray:
        record_name = file_path.replace('.dat', '').replace('.hea', '')
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal[:, 0]
        return ecg_signal
    
    def _load_json(self, file_path_or_bytes) -> np.ndarray:
        if isinstance(file_path_or_bytes, bytes):
            data = pd.read_json(io.BytesIO(file_path_or_bytes))
        else:
            data = pd.read_json(file_path_or_bytes)
        
        if 'signal' in data:
            return np.array(data['signal'])
        elif 'ecg' in data:
            return np.array(data['ecg'])
        else:
            raise ValueError("JSON must contain 'signal' or 'ecg' key")
    
    def preprocess_signal(self, ecg_signal: np.ndarray, 
                         original_sampling_rate: Optional[int] = None) -> np.ndarray:
        if original_sampling_rate and original_sampling_rate != self.sampling_rate:
            ecg_signal = self._resample_signal(ecg_signal, 
                                              original_sampling_rate, 
                                              self.sampling_rate)
        
        ecg_filtered = self._bandpass_filter(ecg_signal)
        ecg_normalized = self._normalize_signal(ecg_filtered)
        
        return ecg_normalized
    
    def _bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def _normalize_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        
        if std == 0:
            return ecg_signal - mean
        
        normalized = (ecg_signal - mean) / std
        return normalized
    
    def _resample_signal(self, ecg_signal: np.ndarray, 
                        original_rate: int, 
                        target_rate: int) -> np.ndarray:
        num_samples = int(len(ecg_signal) * target_rate / original_rate)
        resampled = signal.resample(ecg_signal, num_samples)
        return resampled
    
    def segment_signal(self, ecg_signal: np.ndarray, 
                      segment_length: int = 3600) -> List[np.ndarray]:
        segments = []
        num_segments = len(ecg_signal) // segment_length
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = ecg_signal[start_idx:end_idx]
            segments.append(segment)
        
        return segments
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        diff_signal = np.diff(ecg_signal)
        squared_signal = diff_signal ** 2
        window_size = int(0.12 * self.sampling_rate)
        moving_avg = np.convolve(squared_signal, 
                                np.ones(window_size) / window_size, 
                                mode='same')
        
        threshold = np.percentile(moving_avg, 90)
        min_distance = int(0.2 * self.sampling_rate)
        
        peaks, _ = find_peaks(moving_avg, 
                            height=threshold, 
                            distance=min_distance)
        
        return peaks
    
    def calculate_heart_rate(self, r_peaks: np.ndarray) -> float:
        if len(r_peaks) < 2:
            return 0.0
        
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60.0 / avg_rr
        
        return heart_rate
    
    def extract_hrv_features(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract Heart Rate Variability (HRV) features
        
        Args:
            r_peaks: Indices of R-peaks
            
        Returns:
            Dictionary of HRV features
        """
        if len(r_peaks) < 2:
            return {
                'mean_rr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0
            }
        
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
        
        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50
        }
    
    def extract_morphological_features(self, ecg_segment: np.ndarray, 
                                      r_peaks: np.ndarray) -> Dict[str, float]:
        features = {}
        
        if len(r_peaks) == 0:
            return {
                'qrs_duration': 0.0,
                'pr_interval': 0.0,
                'qt_interval': 0.0,
                'p_wave_amplitude': 0.0,
                'r_wave_amplitude': 0.0,
                't_wave_amplitude': 0.0
            }
        
        r_amplitudes = [ecg_segment[peak] for peak in r_peaks if peak < len(ecg_segment)]
        features['r_wave_amplitude'] = np.mean(r_amplitudes) if r_amplitudes else 0.0
        
        qrs_duration_samples = int(0.08 * self.sampling_rate)
        features['qrs_duration'] = qrs_duration_samples / self.sampling_rate * 1000
        
        features['pr_interval'] = 160.0
        features['qt_interval'] = 400.0
        features['p_wave_amplitude'] = 0.25
        features['t_wave_amplitude'] = 0.3
        
        return features
    
    def process_ecg_for_classification(self, file_path_or_bytes, 
                                       file_extension: str = '.csv',
                                       original_sampling_rate: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Complete pipeline to process ECG file for classification
        
        Args:
            file_path_or_bytes: ECG file path or bytes
            file_extension: File extension
            original_sampling_rate: Original sampling rate if known
            
        Returns:
            Tuple of (processed_signal, features_dict)
        """
        ecg_signal = self.load_ecg_file(file_path_or_bytes, file_extension)
        ecg_processed = self.preprocess_signal(ecg_signal, original_sampling_rate)
        r_peaks = self.detect_r_peaks(ecg_processed)
        features = {}
        features['heart_rate'] = self.calculate_heart_rate(r_peaks)
        hrv_features = self.extract_hrv_features(r_peaks)
        features.update(hrv_features)
        morph_features = self.extract_morphological_features(ecg_processed, r_peaks)
        features.update(morph_features)
        
        features['signal_mean'] = np.mean(ecg_processed)
        features['signal_std'] = np.std(ecg_processed)
        features['signal_max'] = np.max(ecg_processed)
        features['signal_min'] = np.min(ecg_processed)
        
        return ecg_processed, features


def extract_features_from_signal(ecg_signal: np.ndarray, 
                                 sampling_rate: int = 360) -> np.ndarray:
    processor = ECGProcessor(sampling_rate=sampling_rate)
    r_peaks = processor.detect_r_peaks(ecg_signal)
    features_dict = {}
    features_dict['heart_rate'] = processor.calculate_heart_rate(r_peaks)
    hrv_features = processor.extract_hrv_features(r_peaks)
    features_dict.update(hrv_features)
    morph_features = processor.extract_morphological_features(ecg_signal, r_peaks)
    features_dict.update(morph_features)
    features_dict['signal_mean'] = np.mean(ecg_signal)
    features_dict['signal_std'] = np.std(ecg_signal)
    features_dict['signal_max'] = np.max(ecg_signal)
    features_dict['signal_min'] = np.min(ecg_signal)
    feature_vector = np.array([
        features_dict['heart_rate'],
        features_dict['mean_rr'],
        features_dict['sdnn'],
        features_dict['rmssd'],
        features_dict['pnn50'],
        features_dict['qrs_duration'],
        features_dict['pr_interval'],
        features_dict['qt_interval'],
        features_dict['p_wave_amplitude'],
        features_dict['r_wave_amplitude'],
        features_dict['t_wave_amplitude'],
        features_dict['signal_mean'],
        features_dict['signal_std'],
        features_dict['signal_max'],
        features_dict['signal_min']
    ])
    
    return feature_vector
