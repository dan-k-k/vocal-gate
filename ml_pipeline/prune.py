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
    """Applies L1 Unstructured pruning to Conv2d and Linear layers."""
    print(f"🔪 Pruning {amount*100}% of the smallest weights...")
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    # Apply the pruning mask
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return parameters_to_prune

def make_pruning_permanent(parameters_to_prune):
    """Removes the pruning masks so the weights are permanently zeroed."""
    print("🔒 Making pruning permanent for ONNX export...")
    for module, name in parameters_to_prune:
        prune.remove(module, name)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Pruning & Fine-tuning on: {device}")

    weights_path = "./models/vocalgate_best.pt"
    pruned_save_path = "./models/vocalgate_pruned.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found. Run train.py first.")
        return

    # 1. Load the best dense model
    model = VocalGateModel().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. Apply the Pruning Mask (Knock out 20% of the network)
    pruned_params = apply_pruning(model, amount=0.2)

    # 3. Setup Fine-Tuning (Recovery Phase)
    print("🩹 Starting recovery fine-tuning...")
    data_dir = "./data"
    mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=40,
        melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 40, "center": False, "window_fn": torch.hann_window}).to(device)
    
    # We only need the train loader for a quick recovery
    train_dataset = VocalGateDataset(split_dir=os.path.join(data_dir, "train"), augment=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    # Use a MUCH smaller learning rate (1e-4) so we don't wreck the existing good weights
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 

    epochs = 5 # Just enough to let the network adapt
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Recovery Epoch {epoch+1}/{epochs}")

        for features, labels in train_bar:
            optimizer.zero_grad()
            # FIX: Catch the extra return values and pass the transform!
            loss, _, _, _ = process_batch(features, labels, model, criterion, device, mfcc_transform)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    # 4. Make it permanent and save
    make_pruning_permanent(pruned_params)
    
    # Save a clean state dict without the pruning masks overhead
    torch.save({'model_state_dict': model.state_dict()}, pruned_save_path)
    print(f"✅ Pruned and healed model saved to: {pruned_save_path}")

if __name__ == "__main__":
    main()

