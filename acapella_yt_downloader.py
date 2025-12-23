import os
import subprocess
import pandas as pd
from urllib.parse import urlparse, parse_qs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CSV_PATH = "/Users/balamuralibalu/PythonProjects/voice_controlla/full_dataset.csv"
OUTPUT_DIR = "/Volumes/QL SSD1/ai_human_music_data/Acapella"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

def extract_video_id_from_link(link):
    """Extract YouTube video ID from URL"""
    parsed = urlparse(link)
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    return parsed.path.strip("/")

def min_dot_sec_to_seconds(x):
    """Convert min.sec format to seconds (0.31=31s, 1.31=91s)"""
    s = str(x)
    mins_str, secs_str = s.split(".")
    mins = int(mins_str)
    secs = int(secs_str)
    return mins * 60 + secs

print(f"Processing {len(df)} samples from {CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

for idx, row in df.iterrows():
    print(f"\n{'='*50}")
    print(f"Processing sample {idx+1}/{len(df)}")
    print(f"{'='*50}")
    
    sample_id = str(row["ID"])
    link = str(row["Link"])
    
    # Parse timestamps (min.sec format)
    start = min_dot_sec_to_seconds(row["Init"])
    end = min_dot_sec_to_seconds(row["Fin"])
    duration = end - start
    
    print(f"Sample ID: {sample_id}")
    print(f"Link: {link}")
    print(f"Start: {start}s ({row['Init']}), End: {end}s ({row['Fin']}), Duration: {duration}s")
    
    video_id = extract_video_id_from_link(link)
    temp_audio = os.path.join(OUTPUT_DIR, f"{video_id}_full.m4a")
    out_wav = os.path.join(OUTPUT_DIR, f"{sample_id}_{int(start)}_{int(end)}.wav")

    # Skip if output exists
    if os.path.exists(out_wav):
        print(f"✅ Skipping existing: {out_wav}")
        continue

    # Download full audio if needed
    if not os.path.exists(temp_audio):
        print(f"📥 Downloading: {video_id}")
        cmd_dl = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=default",
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "-o", temp_audio,
            link,
        ]
        try:
            result = subprocess.run(cmd_dl, check=True, capture_output=True, text=True)
            print(f"✅ Downloaded: {temp_audio}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed for {video_id}: {e.stderr}")
            continue
    else:
        print(f"🔄 Using existing: {temp_audio}")

    # Extract segment with ffmpeg
    print(f"✂️  Extracting: {out_wav}")
    cmd_ffmpeg = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", temp_audio,
        "-t", str(duration),
        "-ac", "1",
        "-ar", "16000",
        out_wav,
    ]
    try:
        result = subprocess.run(cmd_ffmpeg, check=True, capture_output=True, text=True)
        print(f"✅ Extracted: {out_wav}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed for {out_wav}: {e.stderr}")
        continue

    # Cleanup temp audio
    try:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            print(f"🗑️  Removed temp: {temp_audio}")
    except OSError as e:
        print(f"⚠️  Could not remove {temp_audio}: {e}")

print(f"\n🎉 Processing complete! Check {OUTPUT_DIR} for WAV files.")
