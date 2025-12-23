import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
import time
import os


AUDIO_DURATION = 20  # seconds
BATCH_SIZE = 1000
OUT_PATH = "/Users/balamuralibalu/PythonProjects/voice_controlla/npz_folder"

data_for_ml = []
batch_num = 1
total_count = 0
short =0

wav_folder = '/Volumes/QL SSD1/ai_human_music_data/Acapella'

acap_full_metadata = pd.read_csv("/Users/balamuralibalu/PythonProjects/voice_controlla/datasets/full_dataset.csv")


# Step 3: Audio Processing and Feature Extraction
for root, dirs, files in os.walk(wav_folder):
    for file in files:
        if file.endswith('.wav'):
            filepath = os.path.join(root, file)
            try:
                y, sr = librosa.load(filepath, sr=44100)
                # Peak normalize
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y / peak

            except Exception as e:
                print(f"ERROR loading {os.path.basename(filepath)}: {e}")
                continue

            samples_needed = sr * AUDIO_DURATION
        
            if len(y) > samples_needed:
                max_start = len(y) - samples_needed
                sample_start = random.randint(0, max_start)
                y = y[sample_start:sample_start+samples_needed]
            elif len(y) < samples_needed:
                print(f"File {filepath} is shorter than {AUDIO_DURATION} seconds. Skipping.")
                continue
            # Step 4: Feature Extraction (STFT Spectrogram)
            s_pow = np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) ** 2
            s_db = librosa.power_to_db(s_pow, ref=np.max)

            # Get label from metadata
            if file[0] == "_":     # Skip "_abc.wav"
                continue

            video_id = file.rsplit("_", 2)[0]  # "afqfkjtb-KA"
            acap_metadata_row = acap_full_metadata[acap_full_metadata['ID'].str.contains(video_id, na=False)]

            if acap_metadata_row.empty:
                print(f"No metadata for {file} (ID: {video_id}). Skipping.")
                continue

            #label = 1 if acap_metadata_row["Gender"].iloc[0] == "Male" else 0
            label = 0  # Since all selected files are Acapella
            
            TARGET_TIME = 1723  
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
                save_path = f'{OUT_PATH}/acap_data_batch_{batch_num}.npz'
                np.savez(save_path, spectrograms=spectrograms, labels=labels)
                print(f"Saved batch {batch_num} with {len(spectrograms)} items.")
                print(f"Processed {total_count} files so far.")
                batch_num += 1
                data_for_ml = []

# Step 6: Save any remainder
if data_for_ml:
    spectrograms = np.array([d[0] for d in data_for_ml])
    labels = np.array([d[1] for d in data_for_ml])
    save_path = f'{OUT_PATH}/acap_data_batch_{batch_num}.npz'
    np.savez(save_path, spectrograms=spectrograms, labels=labels)
    print(f"Saved final batch {batch_num} with {len(spectrograms)} items.")

    
