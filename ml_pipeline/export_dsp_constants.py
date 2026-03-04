# ml_pipeline/export_dsp_constants.py
import os
import torch
import torchaudio.transforms as T

def tensor_to_cpp_array(tensor, name, is_2d=False):
    """Converts a PyTorch tensor into a C++ const float array string."""
    tensor = tensor.squeeze().numpy()
    
    if not is_2d:
        # 1D Array (e.g., Window function)
        arr_str = ", ".join([f"{val:.6f}f" for val in tensor])
        return f"const float {name}[{len(tensor)}] = {{\n    {arr_str}\n}};\n"
    else:
        # 2D Matrix (e.g., Mel Filterbank, DCT)
        rows, cols = tensor.shape
        cpp_str = f"const float {name}[{rows}][{cols}] = {{\n"
        for i in range(rows):
            row_str = ", ".join([f"{val:.8f}f" for val in tensor[i]])
            cpp_str += f"    {{{row_str}}},\n"
        cpp_str += "};\n"
        return cpp_str

def main():
    print("🧪 Initializing exact PyTorch MFCC Transform...")
    
    # Use the exact same parameters from your training script!
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
    )

    # 1. Steal the Mel Filterbank Matrix
    # Shape: [257, 40] (Freq bins x Mel bins)
    mel_fb = mfcc_transform.MelSpectrogram.mel_scale.fb
    
    # 2. Steal the DCT Matrix
    # Shape: [40, 40] (Mel bins x MFCC bins)
    dct_mat = mfcc_transform.dct_mat
    
    # 3. Steal the Window Function
    # Shape: [512]
    window = torch.hann_window(512)

    print("📝 Formatting matrices into C++ header...")
    
    cpp_code = "#pragma once\n\n"
    cpp_code += "namespace DSPConstants {\n\n"
    
    cpp_code += tensor_to_cpp_array(window, "hannWindow512", is_2d=False) + "\n"
    cpp_code += tensor_to_cpp_array(mel_fb, "melFilterBank", is_2d=True) + "\n"
    cpp_code += tensor_to_cpp_array(dct_mat, "dctMatrix", is_2d=True) + "\n"
    
    cpp_code += "} // namespace DSPConstants\n"

    # Save directly into your JUCE plugin source folder
    output_path = "../plugin/Source/DSPConstants.h"
    
    # Ensure the directory exists (just in case)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(cpp_code)

    print(f"✅ Success! C++ constants saved to: {output_path}")
    print(f"   -> Mel Filterbank shape: {mel_fb.shape}")
    print(f"   -> DCT Matrix shape: {dct_mat.shape}")

if __name__ == "__main__":
    main()

