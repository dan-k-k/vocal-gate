# ml_pipeline/export_onnx.py
import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference

from model import VocalGateModel

def print_size_of_model(model_path):
    return os.path.getsize(model_path) / 1024

def export_to_onnx():
    print("Initialising PyTorch model...")
    device = torch.device("cpu")
    model = VocalGateModel().to(device)
    
    weights_path = "./models/vocalgate_pruned.pt" 
    plugin_dir = "../plugin" 
    os.makedirs(plugin_dir, exist_ok=True)

    fp32_onnx_path = os.path.join(plugin_dir, "vocalgate_fp32.onnx")
    preprocessed_onnx_path = os.path.join(plugin_dir, "vocalgate_fp32_prep.onnx") # For preprocessing
    int8_onnx_path = os.path.join(plugin_dir, "vocalgate_int8.onnx")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: trained weights '{weights_path}' not found. Run train.py and prune.py first.")

    print(f"Loading trained weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    dummy_input = torch.randn(1, 1, 40, 61)

    print(f"Exporting FP32 model to {fp32_onnx_path}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        fp32_onnx_path,
        export_params=True,
        opset_version=18, 
        do_constant_folding=True,
        input_names=['input_log_mel'],
        output_names=['gate_logit'], 
    )
    
    onnx_model = onnx.load(fp32_onnx_path)
    onnx.save_model(onnx_model, fp32_onnx_path)

    print("FP32 export complete.")

    # Infer and lock shapes
    shape_inference.quant_pre_process(
        input_model_path=fp32_onnx_path,
        output_model_path=preprocessed_onnx_path,
        skip_optimization=False,
        save_as_external_data=False
    )

    print("Pre-processing complete. Starting ONNX INT8 quantisation...")

    # Use pre-processed model for final quantisation
    quantize_dynamic(
        model_input=preprocessed_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=False 
    )

    size_pt = print_size_of_model(weights_path)
    size_fp32 = print_size_of_model(fp32_onnx_path)
    size_int8 = print_size_of_model(int8_onnx_path)
    compression = (1 - (size_int8 / size_fp32)) * 100

    print(f"Raw PyTorch size {size_pt:.2f} KB")
    print(f"ONNX FP32 size: {size_fp32:.2f} KB")
    print(f"ONNX INT8 size: {size_int8:.2f} KB")
    print(f"{compression:.1f}% smaller.")
    print(f"\nSaved to: '{int8_onnx_path}'")

    if os.path.exists(preprocessed_onnx_path):
        os.remove(preprocessed_onnx_path) 

if __name__ == "__main__":
    export_to_onnx()

