## Vocal Gate VST3/AU/Standalone

An AI-powered noise gate plugin specifically trained to separate clean speech from artifacts and background noise. 
For M-series Macs, macOS 11+ | Windows 10+ 

<p align="center">
  <img src="images/LogMel_Plugin_Demo1.gif" alt="LogMel_Plugin_Demo1" width="500">
</p>

<p align="center">
  <a href="https://github.com/dan-k-k/vocal-gate/releases/download/v1.0.1/VocalGate_Mac_Installer.pkg">
    <img src="https://img.shields.io/badge/Download_for_macOS-v1.0.1-black?style=for-the-badge&logo=apple" alt="Download for macOS">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/dan-k-k/vocal-gate/releases/download/v1.0.1/VocalGate_Windows_Installer.exe">
    <img src="https://img.shields.io/badge/Download_for_Windows-v1.0.1-blue?style=for-the-badge&logo=windows" alt="Download for Windows">
  </a>
</p>

*Note: This installer is unsigned. You must right-click open the installer on macOS | press 'More info' and 'Run anyway' on Windows.*

### Real-World Use Examples

<p align="center">
  <a href="https://youtu.be/z2ef61ITh04">
    <img src="images/VocalGateThumbnail.png" alt="Vocal Gate Demo" width="500">
  </a>
</p>
<p align="center"><i>Click the image above to watch the full demo on YouTube</i></p>

---

## Model Performance

The plugin relies on a pruned and quantised ONNX model to achieve real-time inference with incredibly low latency (~0.2 - 0.3 ms per buffer).

<p align="center">
  <img src="images/AI_Inference.png" alt="AI_Inference" width="500">
</p>

### Dataset Energies
<p align="center">
  <img src="images/dataset_energy_comparison.png" alt="dataset_energy_comparison" width="500">
</p>

### Training Loss 
<p align="center">
  <img src="images/loss_curve.png" alt="loss_curve" width="500">
</p>

### ROC 
Pruning and quantising the model led to better performance in both ability (generalisation) and inference time.
<p align="center">
  <img src="images/roc_curve_comparison.png" alt="roc_curve_comparison" width="500">
</p>

