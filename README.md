## Vocal Gate VST3 & AU (Free)

Vocal Gate is a free AI noise gate VST3/AU plugin trained to separate clean speech from unwanted microphone artifacts. This plugin is intended for Creators and Editors. Whether you need an automatic cough remover for your podcast/Youtube video, a sneeze ducker for your Twitch live stream, or a smart filter to block keyboard clicks and heavy breathing, this plugin processes your audio in real-time. 

<p align="center">
  <img src="images/LogMel_Plugin_Demo1.gif" alt="Free AI Noise Gate VST3 removing coughs and sneezes in real time" width="500">
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
  <small><b>Requires:</b> macOS 11+ (M-series) &nbsp;|&nbsp; Windows 10+ (64-bit) | <i><a href="https://github.com/dan-k-k/vocal-gate/releases/">Release notes</a></i>
</small><br>
  <small><i>Note: This installer is unsigned. On macOS, right-click open the installer in your downloads. On Windows, press 'More info' and 'Run anyway'.</small></i>
</p>

### Real-World Use Examples

<p align="center">
  <a href="https://youtu.be/z2ef61ITh04">
    <img src="images/VocalGateThumbnail.png" alt="Vocal Gate Demo" width="500">
  </a>
</p>
<p align="center"><i>Watch the full demo on YouTube</i></p>

#### Can be used for:
* **Live Streaming:** Twitch, YouTube Live, Kick, Facebook Gaming
* **Podcasting:** Spotify, Apple Podcasts, Patreon
* **Video Content:** YouTube VODs, TikTok, Instagram Reels
*NOTE TO LIVE STREAMERS: This AI model looks 750ms ahead to cleanly cut noise, which induces 750ms latency/delay to ONLY your Mic/Aux channel. You MUST delay your webcam/game or app audio/game or app video/anything else by 750ms in OBS to keep everything synced.

#### Can be used in:
* **Broadcasting:** OBS Studio, Streamlabs, vMix
* **Video Editing:** DaVinci Resolve, Adobe Premiere Pro, Final Cut Pro
* **DAWs & Audio:** Reaper, Logic Pro, FL Studio, Ableton Live, Audacity

---

## Model Performance

The plugin relies on a pruned and quantised int8 ONNX model to achieve real-time inference in just ~0.3 ms per buffer. 

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
The pruned and quantised model has better performance in both inference time and ability on the test set (it is better generalised).
<p align="center">
  <img src="images/roc_curve_comparison1.png" alt="roc_curve_comparison" width="500">
</p>

