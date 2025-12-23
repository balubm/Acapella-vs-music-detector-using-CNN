# Music vs Acapella Audio Classifier (CNN)

This project is a **Streamlit web app** and **TensorFlow/Keras model** that classifies 20‑second audio clips as either:

- **Acapella (0)** – voice‑only, no instruments  
- **Music (1)** – full‑mix with instruments and/or backing tracks  

The model is trained on spectrograms of 20‑second clips and achieves **~94% test accuracy**, with **1.0 precision** and **0.87 recall** on the Music class (balanced test set of 400 clips: 200 Acapella, 200 Music).

---

## How it works

- Audio is loaded at **44.1 kHz**, converted to mono, and cropped to **20 seconds**.  
- The 20‑second segment is **peak‑normalized**.  
- A **STFT spectrogram** is computed (`n_fft=4096`, `hop_length=512`), converted to **power** and then **dB**.  
- The spectrogram is cropped/padded to a fixed shape of **2049 × 1723** and standardized using global mean/std from training.  
- A compact **2D CNN with GlobalAveragePooling2D** predicts the probability that the clip is Music (1) vs Acapella (0).  
- The Streamlit app exposes this pipeline with a simple file uploader UI.

---

## Running the app locally

1. Clone the repo:

    git clone https://github.com/<your-username>/acapella-vs-music-detector-using-cnn.git
    cd acapella-vs-music-detector-using-cnn

2. (Optional) Create and activate a virtual environment.

3. Install dependencies:

    pip install -r requirements.txt

4. Run the Streamlit app:

    streamlit run app.py

5. Open the local URL shown in the terminal (e.g. `http://localhost:8501`) and upload a WAV/MP3 file of at least 20 seconds.

---

## Model details

  - **Input:** spectrograms of shape `(2049, 1723, 1)`  
  - **Architecture (high level):**
  - Conv2D → MaxPooling → Dropout  
  - Conv2D → MaxPooling → Dropout  
  - Conv2D → GlobalAveragePooling2D  
  - Dense(64) + Dropout  
  - Dense(1, sigmoid)
  
  - **Training setup:**
  - Loss: binary cross‑entropy  
  - Optimizer: Adam (lr = 1e‑3)  
  - Metrics: accuracy, precision, recall  
  - Data split (balanced): 56% train, 24% validation, 20% test  

**Test performance:**

  - Accuracy: **0.935**  
  - Music class (1): precision **1.00**, recall **0.87**  
  - Acapella class (0): precision **0.88**, recall **1.00**

---

## Data sources

  - **Acapella:**  
  - Acappella dataset – https://ipcv.github.io/Acappella/acappella/  
  
  - **Music (Human/FMA):**  
  - FMA Free Music Archive (small/medium) –  
   https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium  
  
  Both sources were downsampled to 20‑second clips and balanced to **1,000 Acapella / 1,000 Music** examples.

---

## Future work

  - Scale to **tens of thousands** of clips with more genres, languages, and recording setups.  
  - Add **lightweight baselines** (MFCC + SVM/logistic regression) for low‑compute scenarios.  
  - Add **monitoring and stress tests** (noise, loudness, codecs) for robustness.  
  - Extend to related tasks using the same pipeline, e.g. **gender classification** of vocals and **language ID** for voice‑only clips.

---

## Live demo

  Try the app here: https://acapella-vs-music-detector-using-cnn-dtydruynorappy6gakjkp6y.streamlit.app/
