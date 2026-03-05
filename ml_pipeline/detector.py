# ml_pipeline/detector.py
import os
import torch
import torch.nn as nn
import torchaudio.transforms as T
from dataset import VocalGateDataset

DATA_DIR = "./data" 

print("Loading VocalGate dataset...")

# 1. Define the exact same Log Mel Transform used in train.py
log_mel_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=40,
        center=False,
        window_fn=torch.hann_window
    ),
    T.AmplitudeToDB(stype='power', top_db=80.0)
)

# 2. Initialize the dataset WITHOUT the transform argument
dataset = VocalGateDataset(split_dir=os.path.join(DATA_DIR, "train"), augment=False)

# 3. Grab the first item (which is now raw audio!)
raw_waveform, label = dataset[0]

# 4. Apply the transform manually to inspect the features
log_mel_features = log_mel_transform(raw_waveform)

print(f"Label: {label.item()} (1.0 = Artifact, 0.0 = Vocal)")
print(f"Log Mel Shape: {log_mel_features.shape} (channels x Mel bins x time frames)")

