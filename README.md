## Vocal Gate VST3/AU/Standalone

An AI-powered noise gate plugin specifically trained to separate clean speech from artifacts and background noise. 

This plugin is intended for Creators and Editors (Podcast, Youtube, etc.) who need a quick and accurate filter. 

<p align="center">
  <img src="images/LogMel_Plugin_Demo1.gif" alt="LogMel_Plugin_Demo1" width="500">
</p>

<p align="center">
  <a href="https://github.com/dan-k-k/vocal-gate/releases/download/v1.0.3/VocalGate_Mac_Installer.pkg">
    <img src="https://img.shields.io/badge/Download_for_macOS-v1.0.3-black?style=for-the-badge" alt="Download for macOS">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/dan-k-k/vocal-gate/releases/download/v1.0.3/VocalGate_Windows_Installer.exe">
    <img src="https://img.shields.io/badge/Download_for_Windows-v1.0.3-blue?style=for-the-badge" alt="Download for Windows">
  </a>
</p>

<p align="center">
  <small><b>Requires:</b> macOS 11+ (M-series) &nbsp;|&nbsp; Windows 10+ (64-bit)</small><br>
  <i><a href="https://github.com/dan-k-k/vocal-gate/releases/">Release notes</a></i>
</p>

*Note: This installer is unsigned*. On macOS, right-click open the installer in your downloads. On Windows, press 'More info' and 'Run anyway'.

### Real-World Use Examples

<p align="center">
  <a href="https://youtu.be/z2ef61ITh04">
    <img src="images/VocalGateThumbnail.png" alt="Vocal Gate Demo" width="500">
  </a>
</p>
<p align="center"><i>Watch the full demo on YouTube</i></p>

---

## Model Performance

The plugin relies on a pruned and quantised int8 ONNX model to achieve real-time inference with incredibly low latency (~0.2 - 0.3 ms per buffer). The plugin reports 750ms of latency to build the spectrogram for inference. 

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
The pruned and quantised model has better performance in both inference time and ability on the test set (generalisation).
<p align="center">
  <img src="images/roc_curve_comparison1.png" alt="roc_curve_comparison" width="500">
</p>

