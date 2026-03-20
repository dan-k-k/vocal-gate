// plugin/Source/EmaRmsTracker.h
#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>

class EmaRmsTracker
{
public:
    EmaRmsTracker() : smoothedPower(0.0f), currentSampleRate(44100.0) {}

    void prepare(double sampleRate, int /*samplesPerBlock*/)
    {
        if (sampleRate > 0.0) 
        {
            currentSampleRate = sampleRate;
        }
        
        reset();
    }

    void reset()
    {
        // 0.0 linear = -inf dB
        smoothedPower.store(0.0f, std::memory_order_relaxed);
    }

    void processBlock(const juce::AudioBuffer<float>& buffer)
    {
        int numChannels = buffer.getNumChannels();
        int numSamples = buffer.getNumSamples(); // Actual block size

        if (numSamples == 0 || numChannels == 0) return;

        float sumSquares = 0.0f;
        for (int channel = 0; channel < numChannels; ++channel)
        {
            const float* readPtr = buffer.getReadPointer(channel);
            for (int i = 0; i < numSamples; ++i)
            {
                sumSquares += readPtr[i] * readPtr[i];
            }
        }

        // Calc linear power
        float meanSquare = sumSquares / static_cast<float>(numSamples * numChannels);
        float currentRms = std::sqrt(meanSquare);
        float currentDb = juce::Decibels::gainToDecibels(currentRms, -100.0f);

        if (currentDb >= -50.0f)
        {
            // Dynamic alpha based on the number of samples in the block
            const float timeConstantSeconds = 5.0f;
            float dynamicAlpha = 1.0f - std::exp(-static_cast<float>(numSamples) / static_cast<float>(currentSampleRate * timeConstantSeconds));

            float prevPower = smoothedPower.load(std::memory_order_relaxed);
            
            // First run snap (if power is basically zero)
            if (prevPower <= 0.0000000001f) { 
                smoothedPower.store(meanSquare, std::memory_order_relaxed);
            } else {
                float newPower = prevPower + dynamicAlpha * (meanSquare - prevPower);
                smoothedPower.store(newPower, std::memory_order_relaxed);
            }
        }
    }

    float getDbDifferenceFromTarget(float targetDb = -21.5f) const
    {
        float currentPower = smoothedPower.load(std::memory_order_relaxed);
        float smoothedRms = std::sqrt(currentPower);
        float smoothedDb = juce::Decibels::gainToDecibels(smoothedRms, -100.0f);
        
        return smoothedDb - targetDb;
    }

private:
    std::atomic<float> smoothedPower;
    double currentSampleRate;
};

