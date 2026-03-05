# ml_pipeline/plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
import csv 
import torch
import torch.nn as nn # <--- Added nn here
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Make sure your imports at the top also include your model
from model import VocalGateModel

# Setup Images Directory
IMAGES_DIR = "../images"
MODELS_DIR = "./models"
os.makedirs(IMAGES_DIR, exist_ok=True)

def plot_training_curve(csv_file="training_log.csv", output_img="loss_curve.png"):
    """Reads the CSV generated during training and plots the loss curves."""
    csv_path = os.path.join(MODELS_DIR, csv_file)
    out_path = os.path.join(IMAGES_DIR, output_img)

    if not os.path.exists(csv_path):
        print(f"⚠️ Could not find {csv_path}. Make sure training has started!")
        return

    epochs, train_losses, val_losses = [], [], []

    # Read the CSV
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))

    # Plot the Loss Curve 
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label='Train Loss', color='dodgerblue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='darkorange', linewidth=2)
    
    plt.title('Training & Validation Loss', fontsize=14, pad=10)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300)
    print(f"📈 Loss curve saved to: '{out_path}'")
    plt.close()

def save_audio_and_features(waveform, log_mel, sample_rate, label):
    waveform_np = waveform.squeeze().numpy()
    log_mel_np = log_mel.squeeze().numpy()
    
    # ---------------------------------------------------------
    # Plot 1: Raw Waveform
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    time_axis = np.linspace(0, len(waveform_np) / sample_rate, num=len(waveform_np))
    
    ax1.plot(time_axis, waveform_np, color='dodgerblue')
    ax1.set_title(f"Raw Audio Waveform: '{label}' (16,000 samples)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_xlim([0, time_axis[-1]])
    ax1.grid(True, alpha=0.3)
    
    save_path1 = os.path.join(IMAGES_DIR, f"waveform_{label}.png")
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Waveform plot saved to: {save_path1}")
    plt.close(fig1)

    # ---------------------------------------------------------
    # Plot 2: Log Mel Spectrogram
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    cax = ax2.imshow(log_mel_np, origin='lower', aspect='auto', cmap='magma')
    
    ax2.set_title(f"Log Mel Features: '{label}' ({log_mel_np.shape[0]} bins x {log_mel_np.shape[1]} frames)")
    ax2.set_ylabel("Mel Bins")
    ax2.set_xlabel("Time Frames")
    fig2.colorbar(cax, ax=ax2, format="%+2.0f")
        
    save_path2 = os.path.join(IMAGES_DIR, f"log_mel_{label}.png")
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Log Mel plot saved to: {save_path2}")
    plt.close(fig2)


# ---------------------------------------------------------
# 1. The Confusion Matrix
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name="Model", filename="confusion_matrix.png"):
    """Generates a clean, readable confusion matrix for the README."""
    cm = confusion_matrix(y_true, y_pred)
    
    # We use scikit-learn's built in display tool
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Clean Vocal (0)', 'Artifact (1)']
    )
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, text_kw={'fontsize': 14})
    
    plt.title(f'Vocal Gate Confusion Matrix\n({model_name})', fontsize=14, pad=15)
    
    # Customizing axes for better readability
    ax.set_xlabel('Predicted Label (What the VST did)', fontsize=12)
    ax.set_ylabel('True Label (What the audio actually was)', fontsize=12)
    
    out_path = os.path.join(IMAGES_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"🔲 Confusion matrix saved to: '{out_path}'")
    plt.close()

# ---------------------------------------------------------
# 2. The "In-Action" Gating Simulation
# ---------------------------------------------------------
def plot_in_action_gate(model_path, speech_path, artifact_path, threshold=0.5):
    """
    Simulates a streaming audio buffer, runs the model, and plots the ducking.
    Stitches a speech file and an artifact file together to create a test scenario.
    """
    print("🎬 Running In-Action Gate Simulation...")
    device = torch.device("cpu") # Run simulation on CPU for simplicity
    
    # 1. Load Model
    model = VocalGateModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load and Stitch Audio (Speech -> Artifact -> Speech)
    speech, sr_s = torchaudio.load(speech_path)
    artifact, sr_a = torchaudio.load(artifact_path)
    
    # Force 16kHz Mono for both
    if sr_s != 16000: speech = torchaudio.functional.resample(speech, sr_s, 16000)
    if sr_a != 16000: artifact = torchaudio.functional.resample(artifact, sr_a, 16000)
    speech = torch.mean(speech, dim=0, keepdim=True)
    artifact = torch.mean(artifact, dim=0, keepdim=True)

    # Trim to 2 seconds of speech, 1 second of artifact, 1 second of speech
    combined_audio = torch.cat([
        speech[:, :32000], 
        artifact[:, :16000], 
        speech[:, 32000:48000]
    ], dim=1)
    
    audio_np = combined_audio.squeeze().numpy()
    total_samples = len(audio_np)
    
    # 3. Simulate a DAW Buffer (Sliding Window)
    window_size = 16000 # 1 second window
    hop_size = 4000     # 250ms jumps for a smoother probability curve
    
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

    probs = []
    time_points = []

    with torch.no_grad():
        for start in range(0, total_samples - window_size, hop_size):
            chunk = combined_audio[:, start:start + window_size]
            features = log_mel_transform(chunk).unsqueeze(0) # Add batch dim
            
            logit = model(features).squeeze(-1)
            prob = torch.sigmoid(logit).item()
            
            probs.append(prob)
            # Record the time point at the *center* of the window
            time_points.append((start + window_size / 2) / 16000)

    # 4. Create the Gating Array (Gain Reduction)
    # Interpolate our jumpy probability points up to the full audio sample rate
    probs_interp = np.interp(np.linspace(0, total_samples/16000, total_samples), time_points, probs)
    
    # If prob > threshold, gain goes to 0 (muted). Otherwise 1 (unmuted).
    # We invert the probability to act as a gain multiplier (1.0 - prob)
    # and apply a harsh threshold so it snaps shut.
    raw_gain = np.where(probs_interp > threshold, 0.0, 1.0)
    
    # Smooth the gain reduction so it doesn't click (simulating Attack/Release)
    smoothed_gain = np.convolve(raw_gain, np.ones(2000)/2000, mode='same') 
    
    gated_audio = audio_np * smoothed_gain

    # 5. Plot the Results
    time_axis = np.linspace(0, total_samples / 16000, total_samples)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Top: Original Audio
    ax1.plot(time_axis, audio_np, color='dodgerblue', alpha=0.8)
    ax1.set_title('1. Original Input (Clean Speech + Interrupted by Cough)', fontsize=12)
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # Middle: Model Probability
    ax2.plot(time_axis, probs_interp, color='darkorange', lw=2)
    ax2.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Mute Threshold ({threshold})')
    ax2.fill_between(time_axis, probs_interp, threshold, where=(probs_interp > threshold), color='red', alpha=0.3)
    ax2.set_title('2. Neural Network Artifact Probability', fontsize=12)
    ax2.set_ylabel('Confidence (0 to 1)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom: Gated Output
    ax3.plot(time_axis, gated_audio, color='forestgreen', alpha=0.8)
    ax3.set_title('3. Final Gated Output', fontsize=12)
    ax3.set_xlabel('Time (Seconds)', fontsize=12)
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "in_action_gate.png")
    plt.savefig(out_path, dpi=300)
    print(f"📉 In-Action plot saved to: '{out_path}'")
    plt.close()

if __name__ == '__main__':
    plot_training_curve()
    # You can manually set these paths to specific files you know sound good!
    best_model = "./models/vocalgate_best.pt"

    speech_files = [f for f in os.listdir("./data/test/vocals/speech") if f.endswith('.wav')]
    artifact_files = [f for f in os.listdir("./data/test/artifacts/coughing") if f.endswith('.wav')]
    test_speech = os.path.join("./data/test/vocals/speech", speech_files[0])
    test_artifact = os.path.join("./data/test/artifacts/coughing", artifact_files[0])
    
    plot_in_action_gate(best_model, test_speech, test_artifact)

