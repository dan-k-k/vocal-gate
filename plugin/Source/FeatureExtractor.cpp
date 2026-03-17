// plugin/Source/FeatureExtractor.cpp
#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
    // Pre-allocate all memory to avoid allocations on the audio/ML threads
    resampledHopBuffer.assign(targetHopSamples, 0.0f);
    rolling16kBuffer.assign(rollingBufferSize, 0.0f);
    
    timeDomain.assign(n_fft * 2, 0.0f); // *2 for complex FFT output
    powerSpec.assign(num_freq_bins, 0.0f);
    melEnergies.assign(num_mels, 0.0f);
    
    logMelFeatures.assign(num_mels * num_frames, 0.0f);
}

void FeatureExtractor::prepare(double dawSampleRate, int dawSamplesPerHop)
{
    sourceSampleRate = dawSampleRate;
    sourceSamplesPerHop = dawSamplesPerHop;
    
    resampler.reset();
    
    // Clear rolling buffers on playback start
    std::fill(rolling16kBuffer.begin(), rolling16kBuffer.end(), 0.0f);
    std::fill(logMelFeatures.begin(), logMelFeatures.end(), 0.0f);
}

const std::vector<float>& FeatureExtractor::process(const float* hopData)
{
    // 1. Resample the incoming DAW hop to 16kHz
    double ratio = sourceSampleRate / static_cast<double>(targetSampleRate);
    resampler.process(ratio, hopData, resampledHopBuffer.data(), targetHopSamples);

    // 2. Shift the rolling 1-second buffer to the left
    int shiftAmount = targetHopSamples;
    int keepAmount = rollingBufferSize - shiftAmount;
    
    std::memmove(rolling16kBuffer.data(), rolling16kBuffer.data() + shiftAmount, keepAmount * sizeof(float));
    
    // 3. Append the new resampled audio to the end of the rolling buffer
    std::memcpy(rolling16kBuffer.data() + keepAmount, resampledHopBuffer.data(), shiftAmount * sizeof(float));

    // 4. Compute the actual Mel Spectrogram
    computeLogMels();

    return logMelFeatures;
}

void FeatureExtractor::computeLogMels()
{
    for (size_t frame = 0; frame < num_frames; ++frame)
    {
        size_t start_sample = frame * hop_length;

        // Clear time domain buffer
        std::fill(timeDomain.begin(), timeDomain.end(), 0.0f);

        // Apply Hann Window
        for (size_t i = 0; i < n_fft; ++i) {
            timeDomain[i] = rolling16kBuffer[start_sample + i] * DSPConstants::hannWindow512[i];
        }

        // Perform in-place FFT
        forwardFFT.performFrequencyOnlyForwardTransform(timeDomain.data());
        
        // Compute Power Spectrum
        for (size_t i = 0; i < num_freq_bins; ++i) {
            powerSpec[i] = timeDomain[i] * timeDomain[i];
        }

        // Apply Mel Filterbank
        for (size_t m = 0; m < num_mels; ++m) 
        {
            float sum = 0.0f;
            for (size_t f = 0; f < num_freq_bins; ++f) {
                sum += powerSpec[f] * DSPConstants::melFilterBank[f][m];
            }
            
            // Convert to Log (dB) and avoid log(0) with a tiny noise floor
            melEnergies[m] = 10.0f * std::log10(std::max(sum, 1e-10f));
            
            // Store in the flat output tensor array
            logMelFeatures[m * num_frames + frame] = melEnergies[m];
        }
    }
}

