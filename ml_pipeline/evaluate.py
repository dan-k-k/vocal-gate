# ml_pipeline/evaluate.py
import os
import torch
import torch.nn as nn # <--- Added nn for the Sequential block
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from dataset import VocalGateDataset
from model import VocalGateModel
from plots import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

DATA_DIR = "./data"
IMAGES_DIR = "../images"
# We now evaluate a list of models to compare them
MODELS_TO_TEST = {
    "Original": "./models/vocalgate_best.pt",
    "Pruned": "./models/vocalgate_pruned.pt",
    "Quantized": "../plugin/vocalgate_int8.onnx" # <--- Update this path!
}
threshold=0.5; print(f"Threshold={threshold}")

def evaluate_single_model(model_name, model_path, test_loader, device, log_mel_transform):
    """Evaluates a single model and returns its scores and metrics."""
    print(f"\nEvaluating: {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name}: Model not found at {model_path}")
        return None

    model = VocalGateModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 

    y_true, y_scores = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            features = log_mel_transform(features)
            logits = model(features).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_scores.extend(probs.flatten())
            y_true.extend(labels.cpu().numpy().flatten())

    y_pred = [1 if score >= threshold else 0 for score in y_scores] 

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = (fp / (fp + tn)) * 100

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"🚨 False Positive Rate (Hallucinated Mutes): {fpr:.2f}%")
    
    plot_confusion_matrix(y_true, y_pred, model_name, filename=f"cm_{model_name.replace(' ', '_')}.png")

    return y_true, y_scores

def evaluate_onnx_model(model_name, model_path, test_loader, log_mel_transform, device):
    """Evaluates an ONNX model using the ONNX Runtime engine."""
    print(f"\nEvaluating: {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name}: Model not found at {model_path}")
        return None

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    y_true, y_scores = [], []

    for features, labels in test_loader:
        features = features.to(device)      # <-- ADDED THIS LINE
        features = log_mel_transform(features) # <-- ADDED THIS LINE
        
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy().flatten()
        
        # ONNX is locked to batch_size=1, so we process the batch 1 by 1
        for i in range(features_np.shape[0]):
            # Slice to keep the batch dimension: shape [1, 1, 40, 61]
            single_feature = features_np[i:i+1] 
            
            outputs = session.run(None, {input_name: single_feature})
            logit = outputs[0].squeeze(-1) 
            
            logit_clipped = np.clip(logit, -500, 500) 
            prob = 1 / (1 + np.exp(-logit_clipped))
            
            y_scores.append(prob.item())
            
        y_true.extend(labels_np)

    y_pred = [1 if score >= threshold else 0 for score in y_scores] 

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = (fp / (fp + tn)) * 100

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"🚨 False Positive Rate (Hallucinated Mutes): {fpr:.2f}%")
    
    plot_confusion_matrix(y_true, y_pred, model_name, filename=f"cm_{model_name.replace(' ', '_')}.png")
    
    return y_true, y_scores

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on: {device}")

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

    # REMOVED transform=mfcc_transform
    test_dataset = VocalGateDataset(split_dir=os.path.join(DATA_DIR, "test"), augment=False) 
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Setup the plot
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    colors = ['darkorange', 'dodgerblue', 'forestgreen'] # Added a 3rd color

    # Loop through our models and overlay them on the same graph
    for i, (model_name, model_path) in enumerate(MODELS_TO_TEST.items()):
        
        # Route to the correct evaluator based on file type
        if model_path.endswith('.onnx'):
            results = evaluate_onnx_model(model_name, model_path, test_loader, log_mel_transform, device) 
        else:
            results = evaluate_single_model(model_name, model_path, test_loader, device, log_mel_transform)
        
        if results:
            y_true, y_scores = results
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                     label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Muting clean vocals)')
    plt.ylabel('True Positive Rate (Muting actual coughs)')
    plt.title('ROC Curve: Original vs Pruned Gate Performance')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    save_path = os.path.join(IMAGES_DIR, "roc_curve_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison ROC Curve saved to: {save_path}")

if __name__ == "__main__":
    main()

