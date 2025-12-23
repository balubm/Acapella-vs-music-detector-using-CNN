import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
import time
import os

#Step 1: Data Loading and Balanced Sampling
meta_data = pd.read_csv("/Users/balamuralibalu/PythonProjects/CLAP_AI_music_detection/Out/audio_files_metadata.csv")
human_files = meta_data[meta_data['source'] == 'Human'].sample(n=1000, random_state=42)
selected_meta = human_files.sample(frac=1, random_state=42).reset_index(drop=True)
selected_meta.to_csv("/Users/balamuralibalu/PythonProjects/voice_controlla/datasets/human_files_metadata.csv", index=False)
print(f"Selected {len(selected_meta)} Human files for processing")

BATCH_SIZE = 1000
data_for_ml = []
batch_num = 1
total_count = 0

TARGET_TIME = 1723

# Step 3: Audio Processing and Feature Extraction
for index, row in selected_meta.iterrows():
    y, sr = librosa.load(row["filepath"], sr=44100)
    # Peak normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    samples_needed = sr * 20
    
    if len(y) > samples_needed:
        max_start = len(y) - samples_needed
        sample_start = random.randint(0, max_start)
        y = y[sample_start:sample_start+samples_needed]
    elif len(y) < samples_needed:
        print(f"File {row['filepath']},{row['source']}  is shorter than 20 seconds. Skipping.")
        continue
    # Step 4: Feature Extraction (STFT Spectrogram)
    s_pow = np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) ** 2
    s_db = librosa.power_to_db(s_pow, ref=np.max)
    label = 1 # Since all selected files are Human



    if s_db.shape[1] > TARGET_TIME:
        s_db = s_db[:, :TARGET_TIME]
    elif s_db.shape[1] < TARGET_TIME:
        pad_width = TARGET_TIME - s_db.shape[1]
        s_db = np.pad(s_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=s_db.min())



    data_for_ml.append((s_db, label))
    total_count += 1
    # Step 5: Batch Saving
    if total_count % BATCH_SIZE == 0:
        spectrograms = np.array([d[0] for d in data_for_ml])
        labels = np.array([d[1] for d in data_for_ml])
        save_path = f'/Users/balamuralibalu/PythonProjects/voice_controlla/npz_folder/fma_data_batch_{batch_num}.npz'
        np.savez(save_path, spectrograms=spectrograms, labels=labels)
        print(f"Saved batch {batch_num} with {len(spectrograms)} items.")
        print(f"Processed {total_count} files so far.")
        batch_num += 1
        data_for_ml = []

# Step 6: Save any remainder
if data_for_ml:
    spectrograms = np.array([d[0] for d in data_for_ml])
    labels = np.array([d[1] for d in data_for_ml])
    save_path = f'/Users/balamuralibalu/PythonProjects/voice_controlla/npz_folder/fma_data_batch_{batch_num}.npz'
    np.savez(save_path, spectrograms=spectrograms, labels=labels)
    print(f"Saved final batch {batch_num} with {len(spectrograms)} items.")

