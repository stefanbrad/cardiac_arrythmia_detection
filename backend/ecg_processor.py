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
    """
    Processes ECG signals from various file formats and prepares them for analysis
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize ECG processor
        
        Args:
            sampling_rate: Target sampling rate in Hz (default: 360 Hz - MIT-BIH standard)
        """
        self.sampling_rate = sampling_rate
        self.lowcut = 0.5  # Hz - High-pass filter cutoff
        self.highcut = 45.0  # Hz - Low-pass filter cutoff
        
    def load_ecg_file(self, file_path_or_bytes, file_extension: str = '.csv') -> np.ndarray:
        """
        Load ECG data from various file formats
        
        Supported formats:
        - CSV (.csv)
        - TXT (.txt)
        - MIT-BIH format (.dat, .hea)
        - JSON (.json)
        
        Args:
            file_path_or_bytes: File path or bytes object
            file_extension: File extension to determine format
            
        Returns:
            ECG signal as numpy array
        """
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
        """
        Load ECG from CSV or TXT file
        
        Expected formats:
        1. Single column with ECG values
        2. Two columns: time, amplitude
        3. Multiple columns: uses first signal column
        """
        if isinstance(file_path_or_bytes, bytes):
            df = pd.read_csv(io.BytesIO(file_path_or_bytes))
        else:
            df = pd.read_csv(file_path_or_bytes)
        
        # Handle different CSV structures
        if df.shape[1] == 1:
            # Single column - assume it's the signal
            ecg_signal = df.iloc[:, 0].values
        elif df.shape[1] == 2:
            # Two columns - assume second is amplitude
            ecg_signal = df.iloc[:, 1].values
        else:
            # Multiple columns - look for common column names
            amplitude_cols = ['amplitude', 'ecg', 'signal', 'mlii', 'v1', 'lead_i', 'lead_ii']
            for col in amplitude_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    ecg_signal = df[col].values
                    break
            else:
                # Default to second column if no match found
                ecg_signal = df.iloc[:, 1].values
        
        return ecg_signal.astype(float)
    
    def _load_mitbih(self, file_path) -> np.ndarray:
        """
        Load ECG from MIT-BIH format using wfdb
        """
        # Remove extension for wfdb
        record_name = file_path.replace('.dat', '').replace('.hea', '')
        record = wfdb.rdrecord(record_name)
        
        # Use first channel if multiple channels available
        ecg_signal = record.p_signal[:, 0]
        return ecg_signal
    
    def _load_json(self, file_path_or_bytes) -> np.ndarray:
        """
        Load ECG from JSON file
        
        Expected format: {"signal": [...]} or {"ecg": [...]}
        """
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
        """
        Preprocess ECG signal:
        1. Remove baseline wander
        2. Remove high-frequency noise
        3. Normalize
        4. Resample if needed
        
        Args:
            ecg_signal: Raw ECG signal
            original_sampling_rate: Original sampling rate (if different from target)
            
        Returns:
            Preprocessed ECG signal
        """
        # Resample if needed
        if original_sampling_rate and original_sampling_rate != self.sampling_rate:
            ecg_signal = self._resample_signal(ecg_signal, 
                                              original_sampling_rate, 
                                              self.sampling_rate)
        
        # Apply bandpass filter (0.5-45 Hz)
        ecg_filtered = self._bandpass_filter(ecg_signal)
        
        # Normalize signal
        ecg_normalized = self._normalize_signal(ecg_filtered)
        
        return ecg_normalized
    
    def _bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to remove baseline wander and noise
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # 4th order Butterworth filter
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def _normalize_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Normalize ECG signal to zero mean and unit variance
        """
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        
        if std == 0:
            return ecg_signal - mean
        
        normalized = (ecg_signal - mean) / std
        return normalized
    
    def _resample_signal(self, ecg_signal: np.ndarray, 
                        original_rate: int, 
                        target_rate: int) -> np.ndarray:
        """
        Resample ECG signal to target sampling rate
        """
        num_samples = int(len(ecg_signal) * target_rate / original_rate)
        resampled = signal.resample(ecg_signal, num_samples)
        return resampled
    
    def segment_signal(self, ecg_signal: np.ndarray, 
                      segment_length: int = 3600) -> List[np.ndarray]:
        """
        Segment ECG signal into fixed-length windows
        
        Args:
            ecg_signal: Preprocessed ECG signal
            segment_length: Length of each segment in samples (default: 3600 = 10 seconds at 360 Hz)
            
        Returns:
            List of ECG segments
        """
        segments = []
        num_segments = len(ecg_signal) // segment_length
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = ecg_signal[start_idx:end_idx]
            segments.append(segment)
        
        return segments
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in ECG signal using Pan-Tompkins-like algorithm
        
        Args:
            ecg_signal: Preprocessed ECG signal
            
        Returns:
            Indices of R-peaks
        """
        # Differentiate signal
        diff_signal = np.diff(ecg_signal)
        
        # Square the signal
        squared_signal = diff_signal ** 2
        
        # Moving average filter
        window_size = int(0.12 * self.sampling_rate)  # 120ms window
        moving_avg = np.convolve(squared_signal, 
                                np.ones(window_size) / window_size, 
                                mode='same')
        
        # Find peaks
        # Adaptive threshold: 60% of max value
        threshold = np.percentile(moving_avg, 90)
        
        # Minimum distance between peaks (200ms = 0.2 * sampling_rate)
        min_distance = int(0.2 * self.sampling_rate)
        
        peaks, _ = find_peaks(moving_avg, 
                            height=threshold, 
                            distance=min_distance)
        
        return peaks
    
    def calculate_heart_rate(self, r_peaks: np.ndarray) -> float:
        """
        Calculate average heart rate from R-peaks
        
        Args:
            r_peaks: Indices of R-peaks
            
        Returns:
            Heart rate in BPM
        """
        if len(r_peaks) < 2:
            return 0.0
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate  # in seconds
        
        # Calculate heart rate
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60.0 / avg_rr  # BPM
        
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
        
        # RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        # Mean RR interval
        mean_rr = np.mean(rr_intervals)
        
        # SDNN: Standard deviation of RR intervals
        sdnn = np.std(rr_intervals)
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # pNN50: Percentage of successive RR intervals that differ by more than 50ms
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
        """
        Extract morphological features from ECG segment
        
        Args:
            ecg_segment: ECG signal segment
            r_peaks: R-peak locations in segment
            
        Returns:
            Dictionary of morphological features
        """
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
        
        # Average R-wave amplitude
        r_amplitudes = [ecg_segment[peak] for peak in r_peaks if peak < len(ecg_segment)]
        features['r_wave_amplitude'] = np.mean(r_amplitudes) if r_amplitudes else 0.0
        
        # Estimate QRS duration (typically 0.06-0.10 seconds)
        qrs_duration_samples = int(0.08 * self.sampling_rate)  # 80ms
        features['qrs_duration'] = qrs_duration_samples / self.sampling_rate * 1000  # in ms
        
        # Additional features (simplified estimates)
        features['pr_interval'] = 160.0  # typical value in ms
        features['qt_interval'] = 400.0  # typical value in ms
        features['p_wave_amplitude'] = 0.25  # typical value
        features['t_wave_amplitude'] = 0.3  # typical value
        
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
        # Load signal
        ecg_signal = self.load_ecg_file(file_path_or_bytes, file_extension)
        
        # Preprocess
        ecg_processed = self.preprocess_signal(ecg_signal, original_sampling_rate)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(ecg_processed)
        
        # Extract features
        features = {}
        
        # Heart rate
        features['heart_rate'] = self.calculate_heart_rate(r_peaks)
        
        # HRV features
        hrv_features = self.extract_hrv_features(r_peaks)
        features.update(hrv_features)
        
        # Morphological features
        morph_features = self.extract_morphological_features(ecg_processed, r_peaks)
        features.update(morph_features)
        
        # Signal statistics
        features['signal_mean'] = np.mean(ecg_processed)
        features['signal_std'] = np.std(ecg_processed)
        features['signal_max'] = np.max(ecg_processed)
        features['signal_min'] = np.min(ecg_processed)
        
        return ecg_processed, features


def extract_features_from_signal(ecg_signal: np.ndarray, 
                                 sampling_rate: int = 360) -> np.ndarray:
    """
    Helper function to extract feature vector from ECG signal
    
    Args:
        ecg_signal: Preprocessed ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Feature vector as numpy array
    """
    processor = ECGProcessor(sampling_rate=sampling_rate)
    
    # Detect R-peaks
    r_peaks = processor.detect_r_peaks(ecg_signal)
    
    # Extract all features
    features_dict = {}
    
    # Heart rate
    features_dict['heart_rate'] = processor.calculate_heart_rate(r_peaks)
    
    # HRV features
    hrv_features = processor.extract_hrv_features(r_peaks)
    features_dict.update(hrv_features)
    
    # Morphological features
    morph_features = processor.extract_morphological_features(ecg_signal, r_peaks)
    features_dict.update(morph_features)
    
    # Signal statistics
    features_dict['signal_mean'] = np.mean(ecg_signal)
    features_dict['signal_std'] = np.std(ecg_signal)
    features_dict['signal_max'] = np.max(ecg_signal)
    features_dict['signal_min'] = np.min(ecg_signal)
    
    # Convert to feature vector
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
