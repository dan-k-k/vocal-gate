# ml_pipeline/prune.py
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from tqdm import tqdm

from dataset import VocalGateDataset
from model import VocalGateModel
from train import process_batch # Reusing your existing logic!

def apply_pruning(model, amount=0.2):
    print(f"Pruning {amount*100}% of the smallest weights...")
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return parameters_to_prune

def make_pruning_permanent(parameters_to_prune):
    for module, name in parameters_to_prune:
        prune.remove(module, name)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting... on: {device}")

    weights_path = "./models/vocalgate_best.pt"
    pruned_save_path = "./models/vocalgate_pruned.pt"
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Run train.py first.")
        return

    model = VocalGateModel().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    pruned_params = apply_pruning(model, amount=0.2)

    print("Starting recovery...")
    data_dir = "./data"
    
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
    ).to(device)
    
    train_dataset = VocalGateDataset(split_dir=os.path.join(data_dir, "train"), augment=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Much smaller lr

    epochs = 5 
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Recovery Epoch {epoch+1}/{epochs}")

        for features, labels in train_bar:
            optimizer.zero_grad()
            loss, _, _, _ = process_batch(features, labels, model, criterion, device, log_mel_transform)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    make_pruning_permanent(pruned_params)
    
    torch.save({'model_state_dict': model.state_dict()}, pruned_save_path)
    print(f"Saved to: {pruned_save_path}")

if __name__ == "__main__":
    main()

