# ml_pipeline/dataset.py
# find . -maxdepth 2 -not -path '*/.*'
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf

class VocalGateDataset(Dataset):
    def __init__(self, split_dir, chunk_duration=1.0, sample_rate=16000, augment=False):
        self.split_dir = split_dir
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.augment = augment
        
        self.class_map = {
            'speech': 0,
            'coughing': 1,
            'sneezing': 1,
            'breathing': 1,
            'fsd50k_noise': 1 # FSD50K bug fixed!
        }
        
        self.samples = []
        self.speech_files = [] 
        
        print(f"⏳ Scanning {split_dir} and caching file metadata...")
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    parent_folder = os.path.basename(root) 
                    
                    if parent_folder in self.class_map:
                        label = self.class_map[parent_folder]
                        
                        # ISSUE 2 FIX: Read the metadata ONCE during init
                        try:
                            info = sf.info(file_path)
                            total_frames = info.frames
                            self.samples.append((file_path, label, total_frames))
                            
                            if label == 0:
                                self.speech_files.append((file_path, total_frames))
                        except Exception as e:
                            print(f"⚠️ Skipping {file_path} due to read error: {e}")
                            
        print(f"✅ Cached {len(self.samples)} valid files from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def _get_active_chunk(self, file_path, gate_threshold_db):
        """Loads audio into RAM once, then slices an active chunk."""
        peak_threshold = 10 ** (gate_threshold_db / 20.0) 
        
        # ISSUE 2 FIX: Load the entire file into RAM (Fast because they are short files)
        waveform, sr = torchaudio.load(file_path)
        
        # Force 16kHz & Mono
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        frames_in_ram = waveform.shape[1]
        
        # If it's shorter than 1 second, just pad and return
        if frames_in_ram <= self.chunk_samples:
            pad_amount = self.chunk_samples - frames_in_ram
            return torch.nn.functional.pad(waveform, (0, pad_amount))

        # If it's longer, find a loud enough chunk entirely in RAM
        best_chunk = None
        for _ in range(10):
            start_frame = random.randint(0, frames_in_ram - self.chunk_samples)
            chunk = waveform[:, start_frame:start_frame + self.chunk_samples]
            
            if torch.max(torch.abs(chunk)) > peak_threshold:
                return chunk
            
            best_chunk = chunk # Keep as fallback
            
        return best_chunk

    def __getitem__(self, idx):
        file_path, label, _ = self.samples[idx]
        threshold_db = -18.0 if 'breathing' in file_path else -10.0
        
        # 1. Get the base audio chunk
        waveform = self._get_active_chunk(file_path, threshold_db)

        # 2. Apply all augmentations ONLY if we are in training mode
        if self.augment: 
            
            # --- AUGMENTATION A: 60% Chance of Background Bleed ---
            if random.random() < 0.6: 
                if label == 1:
                    bg_file, _ = random.choice(self.speech_files)
                    bg_waveform = self._get_active_chunk(bg_file, -24.0)
                    bleed = random.uniform(0.05, 0.2) 
                    waveform = waveform + (bg_waveform * bleed)
                    
                elif label == 0:
                    artifact_files = [s for s in self.samples if s[1] == 1]
                    bg_file, _, _ = random.choice(artifact_files)
                    
                    if random.random() < 0.5:
                        bg_waveform = self._get_active_chunk(bg_file, -24.0)
                        bleed = random.uniform(0.05, 0.2) 
                        waveform = waveform + (bg_waveform * bleed)
                    else:
                        bg_waveform = self._get_active_chunk(bg_file, -10.0)
                        mix = random.uniform(0.5, 0.9) 
                        waveform = waveform + (bg_waveform * mix)
            
            # --- AUGMENTATION B: 100% Chance of Random Gain (Training Only) ---
            gain_db = random.uniform(-6.0, 6.0) 
            linear_gain = 10 ** (gain_db / 20.0)
            waveform = waveform * linear_gain
            
            # 3. Final safety clamp to prevent digital clipping after adding/boosting
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Return RAW waveform
        return waveform, torch.tensor([label], dtype=torch.float32)
    
