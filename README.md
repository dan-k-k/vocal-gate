# Vocal Gate macOS VST3/AU/Standalone

An AI-powered noise gate plugin specifically trained to separate clean speech from artifacts and background noise. Built for macOS (Apple Silicon M-Series).

### See it in Action

<p align="center">
  <img src="images/LogMel_Plugin_Demo1.gif" alt="LogMel_Plugin_Demo1" width="500">
</p>

<p align="center">
  <a href="https://youtu.be/z2ef61ITh04">
    <img src="images/VocalGateThumbnail.png" alt="Vocal Gate Demo" width="500">
  </a>
</p>
<p align="center"><i>Click the image above to watch the full demo on YouTube</i></p>

---

## Installation (macOS: M-Series)

You can download the compiled installer from the [Releases](../../releases) tab.

**Note on Installation:** This installer is unsigned. To install it, you must right-click open the `.pkg` file.
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

