# ml_pipeline/train.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler # <--- ADDED HERE
import torchaudio.transforms as T
from tqdm import tqdm
import shutil 

from dataset import VocalGateDataset
from model import VocalGateModel

def process_batch(features, labels, model, criterion, device, mfcc_transform):
    # 1. Move raw audio to the GPU
    features = features.to(device)
    labels = labels.float().to(device)
    
    # ISSUE 1 FIX: Apply the heavy MFCC math to the whole batch at once on the GPU
    features = mfcc_transform(features)

    outputs = model(features)
    loss = criterion(outputs, labels)
    
    # Calculate True Positives, False Positives, False Negatives
    predictions = (outputs > 0.0).float()
    tp = ((predictions == 1.0) & (labels == 1.0)).float().sum()
    fp = ((predictions == 1.0) & (labels == 0.0)).float().sum()
    fn = ((predictions == 0.0) & (labels == 1.0)).float().sum()
    
    return loss, tp, fp, fn

def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🚀 Training on device: {device}.")

    batch_size = 128
    epochs = 60 
    learning_rate = 1e-3
    data_dir = "./data"

    # ISSUE 1 FIX: Initialize the transform and send it to the GPU
    mfcc_transform = T.MFCC(
        sample_rate=16000, 
        n_mfcc=40,
        melkwargs={
            "n_fft": 512,
            "hop_length": 256,
            "n_mels": 40, 
            "center": False, 
            "window_fn": torch.hann_window
        }
    ).to(device)

    # Initialize datasets WITHOUT the transform argument
    train_dataset = VocalGateDataset(split_dir=os.path.join(data_dir, "train"), augment=True)
    val_dataset = VocalGateDataset(split_dir=os.path.join(data_dir, "val"), augment=False)
    
    print("⚖️ Calculating class weights for the sampler...")
    class_counts = [0, 0] # [Clean Vocals (0), Artifacts (1)]
    for _, label, _ in train_dataset.samples:
        class_counts[label] += 1
    print(f"📊 Training Class Counts - Vocals: {class_counts[0]} | Artifacts: {class_counts[1]}")

    # 2. Calculate the weight for each class (inverse frequency)
    class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    
    # 3. Assign a weight to every single sample in the dataset
    sample_weights = [class_weights[label] for _, label, _ in train_dataset.samples]
    
    # 4. Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # 5. Pass the sampler to the DataLoader (NOTE: shuffle must be False when using a sampler!)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VocalGateModel().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Checkpoint logic remains exactly the same...
    patience = 10
    start_epoch = 0
    epochs_no_improve = 0
    best_val_loss = float('inf')
    os.makedirs("./models", exist_ok=True)
    best_checkpoint_path = "./models/vocalgate_best.pt"
    latest_checkpoint_path = "./models/vocalgate_latest.pt"

    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resuming from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, epochs):
            # --- TRAINING ---
            model.train()
            train_loss, train_tp, train_fp, train_fn = 0.0, 0.0, 0.0, 0.0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for features, labels in train_bar:
                optimizer.zero_grad()
                # Pass mfcc_transform into the batch processor
                loss, tp, fp, fn = process_batch(features, labels, model, criterion, device, mfcc_transform)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_tp += tp.item()
                train_fp += fp.item()
                train_fn += fn.item()
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            train_precision = train_tp / (train_tp + train_fp + 1e-7)
            train_recall = train_tp / (train_tp + train_fn + 1e-7)

            # --- VALIDATION ---
            model.eval()
            val_loss, val_tp, val_fp, val_fn = 0.0, 0.0, 0.0, 0.0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

            with torch.no_grad():
                for features, labels in val_bar:
                    loss, tp, fp, fn = process_batch(features, labels, model, criterion, device, mfcc_transform)
                    val_loss += loss.item()
                    val_tp += tp.item()
                    val_fp += fp.item()
                    val_fn += fn.item()
                    val_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            avg_val_loss = val_loss / len(val_loader)
            val_precision = val_tp / (val_tp + val_fp + 1e-7)
            val_recall = val_tp / (val_tp + val_fn + 1e-7)
            
            print(f"Epoch {epoch+1} Summary | Val: | Loss: {avg_val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

            # Logging & Checkpointing logic remains the same...
            log_file = "./models/training_log.csv"
            file_exists = os.path.exists(log_file)
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_precision', 'val_recall']) 
                writer.writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}"])

            is_best = False
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0 
                is_best = True
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")


            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            torch.save(checkpoint, latest_checkpoint_path)
            if is_best:
                shutil.copyfile(latest_checkpoint_path, best_checkpoint_path)
                print(f"💾 Saved new best model to {best_checkpoint_path}")
            if epochs_no_improve >= patience:
                print(f"\n✋ Early stopping triggered.")
                break 

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted safely.")

if __name__ == "__main__":
    train()

