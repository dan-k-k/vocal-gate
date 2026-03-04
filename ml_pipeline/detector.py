# ml_pipeline/detector.py
import os
import torch
import torchaudio
import torchaudio.transforms as T
from dataset import VocalGateDataset

DATA_DIR = "./data" # Make sure your ESC-50 data is extracted here!

print("Loading ESC-50 dataset...")

# Define the C++ friendly MFCC Transform
n_mfcc = 40
mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": 512,        # POWER OF 2: Crucial for JUCE C++ FFT
        "hop_length": 256,   # POWER OF 2: Step size
        "n_mels": 40,        
        "center": False,     # Keep False to avoid hidden PyTorch padding magic
        "window_fn": torch.hann_window # Explicitly define the window
    }
)

dataset = VocalGateDataset(split_dir=os.path.join(DATA_DIR, "train"), transform=mfcc_transform)

# Grab the first item
mfcc_features, label = dataset[0]

print(f"Label: {label.item()} (1.0 = Artifact, 0.0 = Vocal)")
print(f"MFCC Shape: {mfcc_features.shape} (channels x MFCC bins x time frames)")

