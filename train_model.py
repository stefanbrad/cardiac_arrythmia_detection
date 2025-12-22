import wfdb
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os


# class mapping
CLASS_MAP = {
    'N': 0, 
    'L': 1, 
    'R': 2, 
    'A': 3, 
    'V': 4  
}

# using all records from MIT-BIH database to check for every heart event
try:
    record_list = wfdb.get_record_list('mitdb')
except:
    # backup list 
    record_list = [str(x) for x in range(100, 235) if x not in [110, 120]]

print(f"Scanăm {len(record_list)} înregistrări...")

data_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}

window_size = 30

for record_id in record_list:
    try:
        record = wfdb.rdrecord(record_id, pn_dir='mitdb')
        annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        
        signal = record.p_signal[:, 0]
        peaks = annotation.sample
        symbols = annotation.symbol
        
        for i, peak in enumerate(peaks):
            if peak < window_size or peak > len(signal) - window_size:
                continue
            
            symbol = symbols[i]
            if symbol in CLASS_MAP:
                label = CLASS_MAP[symbol]
                segment = signal[peak - window_size : peak + window_size]

                # append to specific class list
                data_by_class[label].append(segment)
                
    except Exception as e:
        continue

print("\noriginal data distribution:")
for k, v in data_by_class.items():
    print(f"   class {k}: {len(v)} examples")

# ECHILIBRAREA (Undersampling)
# taiem toate clasele la un maxim fix, 5000 in cazul nostru
# gasim minimul, dar nu mai putin de 500 (ca sa nu eliminam clasele rare de tot)
counts = [len(v) for v in data_by_class.values()]
target_count = 5000 

X_balanced = []
y_balanced = []

print(f"\nEchilibram datele (Maxim {target_count} exemple per clasa)")

for label, samples in data_by_class.items():
    if not samples:
        continue

    samples_array = np.array(samples)
    
    # daca avem mai mult decat target-ul luam random doar target_count
    if len(samples) > target_count:
        samples_subset = resample(samples_array, 
                                n_samples=target_count, 
                                replace=False, 
                                random_state=42)
    else:
        # daca avem putine le luam pe toate
        samples_subset = samples_array
    
    # append la datasetul final
    for s in samples_subset:
        X_balanced.append(s)
        y_balanced.append(label)

X = np.array(X_balanced)
y = np.array(y_balanced)

print(f"Dataset FINAL: {len(X)} batai totale")

# antrenare
print("Antrenare Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# verificare acuratete
print("\nAcuratete pe test:")
print(model.score(X_test, y_test))

joblib.dump(model, 'ecg_model.pkl')
print("Modelul ECHILIBRAT a fost salvat")