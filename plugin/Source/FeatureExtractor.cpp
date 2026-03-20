// plugin/Source/FeatureExtractor.cpp
#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
    resampledHopBuffer.assign(targetHopSamples, 0.0f);
    
    // Allocate a generous buffer. The buildup reaches ~1280 samples max before clearing.
    // 2048 gives a completely safe margin to prevent vector reallocations.
    audioTailBuffer.assign(2048, 0.0f); 
    
    timeDomain.assign(n_fft * 2, 0.0f); 
    powerSpec.assign(num_freq_bins, 0.0f);
    melEnergies.assign(num_mels, 0.0f);
    
    logMelFeatures.assign(num_mels * num_frames, 0.0f);
}

void FeatureExtractor::prepare(double dawSampleRate, int dawSamplesPerHop)
{
    sourceSampleRate = dawSampleRate;
    sourceSamplesPerHop = dawSamplesPerHop;
    resampler.reset();
    
    std::fill(audioTailBuffer.begin(), audioTailBuffer.end(), 0.0f);
    std::fill(logMelFeatures.begin(), logMelFeatures.end(), 0.0f);
    
    // Reset tail tracker on playback start
    numSamplesInTail = 0; 
}

const std::vector<float>& FeatureExtractor::process(const float* hopData)
{
    double ratio = sourceSampleRate / static_cast<double>(targetSampleRate);
    resampler.process(ratio, hopData, resampledHopBuffer.data(), targetHopSamples);

    // Append new audio strictly after whatever tail 
    std::memcpy(audioTailBuffer.data() + numSamplesInTail, resampledHopBuffer.data(), targetHopSamples * sizeof(float));

    size_t totalAvailable = numSamplesInTail + targetHopSamples;
    size_t newFramesCount = 0;
    
    if (totalAvailable >= n_fft) {
        newFramesCount = ((totalAvailable - n_fft) / hop_length) + 1;
    }

    if (newFramesCount > 0) 
    {
        size_t framesToKeep = num_frames - newFramesCount;
        for (size_t m = 0; m < num_mels; ++m) 
        {
            float* melRow = logMelFeatures.data() + (m * num_frames);
            // Shift old frames to the left
            std::memmove(melRow, melRow + newFramesCount, framesToKeep * sizeof(float));
        }

        // Compute only the new frames
        for (size_t f = 0; f < newFramesCount; ++f)
        {
            size_t startSample = f * hop_length;
            
            // Apply Hann Window
            std::fill(timeDomain.begin(), timeDomain.end(), 0.0f);
            for (size_t i = 0; i < n_fft; ++i) {
                timeDomain[i] = audioTailBuffer[startSample + i] * DSPConstants::hannWindow512[i];
            }

            forwardFFT.performFrequencyOnlyForwardTransform(timeDomain.data());

            // Power Spectrum
            for (size_t i = 0; i < num_freq_bins; ++i) {
                powerSpec[i] = timeDomain[i] * timeDomain[i];
            }

            // Mel Filterbank
            for (size_t m = 0; m < num_mels; ++m) {
                float sum = 0.0f;
                for (size_t bin = 0; bin < num_freq_bins; ++bin) {
                    sum += powerSpec[bin] * DSPConstants::melFilterBank[bin][m];
                }
                
                float logMel = 10.0f * std::log10(std::max(sum, 1e-10f));
                size_t featureIdx = (m * num_frames) + (num_frames - newFramesCount + f);
                logMelFeatures[featureIdx] = logMel;
            }
        }
    }

    // Update tail for NEXT `process` call
    size_t consumedSamples = newFramesCount * hop_length;
    numSamplesInTail = totalAvailable - consumedSamples; // Accurately retain all uncalculated samples
    
    std::memmove(audioTailBuffer.data(), audioTailBuffer.data() + consumedSamples, numSamplesInTail * sizeof(float));

    return logMelFeatures;
}

