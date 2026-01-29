import numpy as np
import pandas as pd
import json
import os

def generate_ecg_signal(duration=10, sampling_rate=360, arrhythmia_type='normal'):
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    if arrhythmia_type == 'normal':
        heart_rate = 75 / 60
        ecg = np.zeros_like(t)
        beat_interval = 1.0 / heart_rate
        num_beats = int(duration * heart_rate)
        
        for i in range(num_beats):
            beat_time = i * beat_interval
            p_wave = 0.25 * np.exp(-((t - (beat_time + 0.12)) ** 2) / 0.0005)
            qrs = 1.5 * np.exp(-((t - (beat_time + 0.18)) ** 2) / 0.0001)
            t_wave = 0.35 * np.exp(-((t - (beat_time + 0.36)) ** 2) / 0.001)
            ecg += p_wave + qrs + t_wave
        
        ecg += 0.05 * np.random.randn(len(t))
    
    elif arrhythmia_type == 'afib':
        current_time = 0
        ecg = np.zeros_like(t)
        
        while current_time < duration:
            rr_interval = np.random.uniform(0.3, 0.9)
            qrs = 1.2 * np.exp(-((t - current_time) ** 2) / 0.0001)
            t_wave = 0.25 * np.exp(-((t - (current_time + 0.25)) ** 2) / 0.001)
            ecg += qrs + t_wave
            current_time += rr_interval
        
        ecg += 0.15 * np.random.randn(len(t))
    
    elif arrhythmia_type == 'brady':
        heart_rate = 50 / 60
        ecg = np.zeros_like(t)
        beat_interval = 1.0 / heart_rate
        num_beats = int(duration * heart_rate)
        
        for i in range(num_beats):
            beat_time = i * beat_interval
            p_wave = 0.3 * np.exp(-((t - (beat_time + 0.15)) ** 2) / 0.0005)
            qrs = 1.6 * np.exp(-((t - (beat_time + 0.22)) ** 2) / 0.0001)
            t_wave = 0.4 * np.exp(-((t - (beat_time + 0.45)) ** 2) / 0.001)
            ecg += p_wave + qrs + t_wave
        
        ecg += 0.04 * np.random.randn(len(t))
    
    elif arrhythmia_type == 'tachy':
        heart_rate = 120 / 60
        ecg = np.zeros_like(t)
        beat_interval = 1.0 / heart_rate
        num_beats = int(duration * heart_rate)
        
        for i in range(num_beats):
            beat_time = i * beat_interval
            p_wave = 0.2 * np.exp(-((t - (beat_time + 0.08)) ** 2) / 0.0004)
            qrs = 1.4 * np.exp(-((t - (beat_time + 0.13)) ** 2) / 0.0001)
            t_wave = 0.3 * np.exp(-((t - (beat_time + 0.28)) ** 2) / 0.0008)
            ecg += p_wave + qrs + t_wave
        
        ecg += 0.05 * np.random.randn(len(t))
    
    elif arrhythmia_type == 'pvc':
        heart_rate = 75 / 60
        ecg = np.zeros_like(t)
        beat_interval = 1.0 / heart_rate
        num_beats = int(duration * heart_rate)
        
        for i in range(num_beats):
            beat_time = i * beat_interval
            if i % 4 == 3:
                qrs = 2.0 * np.exp(-((t - beat_time) ** 2) / 0.0003)
                t_wave = -0.5 * np.exp(-((t - (beat_time + 0.35)) ** 2) / 0.0015)
                ecg += qrs + t_wave
            else:
                p_wave = 0.25 * np.exp(-((t - (beat_time + 0.12)) ** 2) / 0.0005)
                qrs = 1.5 * np.exp(-((t - (beat_time + 0.18)) ** 2) / 0.0001)
                t_wave = 0.35 * np.exp(-((t - (beat_time + 0.36)) ** 2) / 0.001)
                ecg += p_wave + qrs + t_wave
        
        ecg += 0.05 * np.random.randn(len(t))
    
    else:
        return generate_ecg_signal(duration, sampling_rate, 'normal')
    
    return t, ecg


def save_examples():
    os.makedirs('examples', exist_ok=True)
    
    arrhythmia_types = {
        'normal': 'Normal Sinus Rhythm',
        'afib': 'Atrial Fibrillation',
        'brady': 'Bradycardia',
        'tachy': 'Tachycardia',
        'pvc': 'Premature Ventricular Contractions'
    }
    
    print("Generating example ECG files...")
    print("=" * 60)
    
    for arr_type, arr_name in arrhythmia_types.items():
        print(f"\nGenerating {arr_name} ({arr_type})...")
        
        # Generate signal
        t, ecg = generate_ecg_signal(duration=10, arrhythmia_type=arr_type)
        
        # Save as CSV (two columns)
        csv_file = f'examples/ecg_{arr_type}_2col.csv'
        df = pd.DataFrame({'time': t, 'amplitude': ecg})
        df.to_csv(csv_file, index=False)
        print(f"  ✓ Created {csv_file}")
        
        # Save as CSV (single column)
        csv_file_single = f'examples/ecg_{arr_type}_1col.csv'
        pd.DataFrame({'amplitude': ecg}).to_csv(csv_file_single, index=False)
        print(f"  ✓ Created {csv_file_single}")
        
        # Save as TXT
        txt_file = f'examples/ecg_{arr_type}.txt'
        np.savetxt(txt_file, ecg, fmt='%.6f')
        print(f"  ✓ Created {txt_file}")
        
        # Save as JSON
        json_file = f'examples/ecg_{arr_type}.json'
        data = {
            'signal': ecg.tolist(),
            'sampling_rate': 360,
            'duration': 10,
            'arrhythmia_type': arr_name
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Created {json_file}")
    
    print("\n" + "=" * 60)
    print("✓ All example files generated successfully!")
    print("\nExample files are in the 'examples/' directory:")
    print("  - CSV files (2 columns): time, amplitude")
    print("  - CSV files (1 column): amplitude only")
    print("  - TXT files: amplitude values, one per line")
    print("  - JSON files: structured format with metadata")
    print("\nYou can use these files to test the API:")
    print("  curl -X POST -F 'file=@examples/ecg_afib_2col.csv' http://localhost:5000/predict")
    print("=" * 60)


def create_readme():
    """Create README for examples"""
    readme_content = """# Example ECG Files

This directory contains synthetic ECG files for testing the arrhythmia detection API.

## Available Examples

1. **Normal Sinus Rhythm** (`ecg_normal_*`)
   - Regular rhythm, ~75 bpm
   - Normal P-QRS-T morphology

2. **Atrial Fibrillation** (`ecg_afib_*`)
   - Irregular rhythm, ~110 bpm average
   - No clear P waves
   - Irregular RR intervals

3. **Bradycardia** (`ecg_brady_*`)
   - Slow rhythm, ~50 bpm
   - Normal morphology

4. **Tachycardia** (`ecg_tachy_*`)
   - Fast rhythm, ~120 bpm
   - Normal morphology

5. **Premature Ventricular Contractions** (`ecg_pvc_*`)
   - Normal rhythm with premature beats
   - Wide QRS complexes for PVCs
   - Occurs every 4th beat

## File Formats

Each arrhythmia type is available in multiple formats:

- `*_2col.csv`: Two-column CSV (time, amplitude)
- `*_1col.csv`: Single-column CSV (amplitude only)
- `*.txt`: Plain text, one amplitude value per line
- `*.json`: JSON format with metadata

## Usage

### Test with cURL

```bash
curl -X POST -F "file=@examples/ecg_afib_2col.csv" http://localhost:5000/predict
```

### Test with Python

```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('examples/ecg_normal_2col.csv', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### Test with Frontend

Simply drag and drop any of these files into the upload area in the web interface.

## Specifications

- **Duration**: 10 seconds
- **Sampling Rate**: 360 Hz
- **Total Samples**: 3,600 per file
- **Format**: Normalized amplitude values

## Notes

These are **synthetic** ECG signals generated for testing purposes. They simulate typical characteristics of each arrhythmia type but are not real patient data.

For production use, you should test with real ECG recordings from validated medical databases like MIT-BIH or PTB-XL.
"""
    
    with open('examples/README.md', 'w') as f:
        f.write(readme_content)
    
    print("\n✓ Created examples/README.md")


if __name__ == '__main__':
    save_examples()
    create_readme()
