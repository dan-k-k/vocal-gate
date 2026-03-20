// plugin/Source/FeatureExtractor.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "DSPConstants.h" 

class FeatureExtractor
{
public:
    FeatureExtractor();
    ~FeatureExtractor() = default;

    void prepare(double dawSampleRate, int dawSamplesPerHop);
    const std::vector<float>& process(const float* hopData);

private:
    // Configuration
    static constexpr int targetSampleRate = 16000;
    static constexpr int targetHopSamples = 800; // 50ms at 16kHz

    // Mel Spectrogram
    static constexpr size_t n_fft = 512;
    static constexpr size_t hop_length = 256;
    static constexpr size_t num_frames = 61;     
    static constexpr size_t num_freq_bins = 257; 
    static constexpr size_t num_mels = 40;

    double sourceSampleRate = 44100.0;
    int sourceSamplesPerHop = 0;

    // Resampling
    juce::LagrangeInterpolator resampler;

    // Buffers
    std::vector<float> resampledHopBuffer;
    
    std::vector<float> timeDomain;
    std::vector<float> powerSpec;
    std::vector<float> melEnergies;
    
    // Dynamic Overlap-Save Tail
    std::vector<float> audioTailBuffer; 
    size_t numSamplesInTail = 0; // Tracks the fluctuating leftover samples
    
    // The final output buffer fed to the ONNX model
    std::vector<float> logMelFeatures;

    // JUCE FFT (Order 9 = 512 bins)
    juce::dsp::FFT forwardFFT { 9 }; 

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FeatureExtractor)
};

