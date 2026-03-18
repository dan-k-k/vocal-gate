# ml_pipeline/detector.py
import os
import torch
import torch.nn as nn
import torchaudio.transforms as T
from dataset import VocalGateDataset

DATA_DIR = "./data" 

print("Loading VocalGate dataset...")

# Same Log Mel Transform used in train.py
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

dataset = VocalGateDataset(split_dir=os.path.join(DATA_DIR, "train"), augment=False)
raw_waveform, label = dataset[0]
log_mel_features = log_mel_transform(raw_waveform)

print(f"Label: {label.item()} (1.0 = artifact, 0.0 = vocal)")
print(f"Log mel shape: {log_mel_features.shape} (channels x Mel bins x time frames)")

