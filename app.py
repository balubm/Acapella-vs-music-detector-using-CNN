import streamlit as st
from tensorflow import keras
from keras.models import load_model
import librosa
import numpy as np


# ----------------- CONSTANTS (from training) -----------------
TRAIN_MEAN = -69.777504
TRAIN_STD  = 14.743397


TARGET_TIME = 1723      # time frames
SR = 44100              # sample rate
AUDIO_DURATION = 20     # seconds
SAMPLES_NEEDED = SR * AUDIO_DURATION

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def get_music_model():
    # Place this file next to app.py (or adjust path if needed)
    return load_model("acapella_vs_music_detector.keras")

model = get_music_model()

# ----------------- APP UI -----------------
st.title("🎵 Music vs Acapella Detector")
st.info("Upload 20+ second audio files (WAV/MP3). Model was trained to distinguish Acapella (0) vs Music (1).")

uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # ---------- AUDIO LOADING ----------
    y, sr = librosa.load(uploaded_file, sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak  # peak-normalize

    duration_sec = len(y) / sr

    if len(y) >= SAMPLES_NEEDED:
        # Deterministic first 20s (simpler for debugging)
        y = y[:SAMPLES_NEEDED]
        st.write(f"✅ Using first 20 s of the uploaded audio)")
    else:
        st.error(f"❌ Too short: {duration_sec:.1f} s (need ≥ 20 s)")
        st.stop()

    # ---------- FEATURE EXTRACTION (MATCH TRAINING) ----------
    # STFT → power → dB with ref=np.max
    s_pow = np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) ** 2
    s_db = librosa.power_to_db(s_pow, ref=np.max)

    # Force time dimension to 1723 (model expects 2049 × 1723 × 1)
    if s_db.shape[1] < TARGET_TIME:
        s_db = np.pad(s_db, ((0, 0), (0, TARGET_TIME - s_db.shape[1])))
    elif s_db.shape[1] > TARGET_TIME:
        s_db = s_db[:, :TARGET_TIME]


    # SAME GLOBAL NORMALIZATION AS TRAINING
    s_db = (s_db - TRAIN_MEAN) / TRAIN_STD

    # Add channel + batch dims
    s_db = s_db[..., np.newaxis]         # (2049, 1723, 1)
    s_db = np.expand_dims(s_db, axis=0)  # (1, 2049, 1723, 1)

    # ---------- PREDICTION ----------
    # Model output: prob(class = 1) = prob(Music)
    raw = float(model.predict(s_db, verbose=0)[0][0])
    music_prob = raw
    acapella_prob = 1.0 - music_prob

    #st.write(f"🔍 Raw model output (prob Music=1): **{raw:.4f}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎤 Acapella probability", f"{acapella_prob:.1%}")
    with col2:
        st.metric("🎸 Music probability", f"{music_prob:.1%}")

    # Threshold (0.5) – you can tune later if needed
    if acapella_prob >= 0.5:
        st.balloons()
        st.success("🎤 **Predicted: A CAPPELLA (voice-only)**")
    else:
        st.info("🎸 **Predicted: MUSIC (with instruments)**")
