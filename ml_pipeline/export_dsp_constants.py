# ml_pipeline/export_dsp_constants.py
import os
import torch
import torchaudio.transforms as T

def tensor_to_cpp_array(tensor, name, is_2d=False):
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
    print("Initialising PyTorch MFCC transform...")
    
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=40,
        center=False,
        window_fn=torch.hann_window
    )

    mel_fb = mel_transform.mel_scale.fb # [257, 40]
    
    window = torch.hann_window(512) # [512]

    print("Formatting matrices into C++ header...")
    
    cpp_code = "#pragma once\n\n"
    cpp_code += "namespace DSPConstants {\n\n"
    
    cpp_code += tensor_to_cpp_array(window, "hannWindow512", is_2d=False) + "\n"
    cpp_code += tensor_to_cpp_array(mel_fb, "melFilterBank", is_2d=True) + "\n"
    
    cpp_code += "} // namespace DSPConstants\n"

    output_path = "../plugin/Source/DSPConstants.h"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(cpp_code)

    print(f"C++ constants saved to: {output_path}")
    print(f"Mel filterbank shape: {mel_fb.shape}")

if __name__ == "__main__":
    main()

