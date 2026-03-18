// plugin/Source/EmaRmsTracker.h
#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>

class EmaRmsTracker
{
public:
    EmaRmsTracker() : smoothedPower(0.0f), alpha(1.0f) {}

    void prepare(double sampleRate, int samplesPerBlock)
    {
        const float timeConstantSeconds = 5.0f;
        
        if (sampleRate > 0.0 && timeConstantSeconds > 0.0f) 
        {
            alpha = 1.0f - std::exp(-static_cast<float>(samplesPerBlock) / static_cast<float>(sampleRate * timeConstantSeconds));
        }
        
        reset();
    }

    void reset()
    {
        // 0.0 linear power equals -inf dB
        smoothedPower.store(0.0f, std::memory_order_relaxed);
    }

    void processBlock(const juce::AudioBuffer<float>& buffer)
    {
        int numChannels = buffer.getNumChannels();
        int numSamples = buffer.getNumSamples();

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

        // 1. Calculate the linear power of this current block
        float meanSquare = sumSquares / static_cast<float>(numSamples * numChannels);
        
        // 2. We only calculate currentDb here to check our -50dB Noise Floor gate
        float currentRms = std::sqrt(meanSquare);
        float currentDb = juce::Decibels::gainToDecibels(currentRms, -100.0f);

        // 3. Apply the EMA to the LINEAR POWER (not the decibels)
        if (currentDb >= -50.0f)
        {
            float prevPower = smoothedPower.load(std::memory_order_relaxed);
            
            // First run snap (if power is basically zero)
            if (prevPower <= 0.0000000001f) { 
                smoothedPower.store(meanSquare, std::memory_order_relaxed);
            } else {
                float newPower = prevPower + alpha * (meanSquare - prevPower);
                smoothedPower.store(newPower, std::memory_order_relaxed);
            }
        }
    }

    float getDbDifferenceFromTarget(float targetDb = -21.5f) const
    {
        // 4. Convert the smoothed power back to decibels just for the UI
        float currentPower = smoothedPower.load(std::memory_order_relaxed);
        float smoothedRms = std::sqrt(currentPower);
        float smoothedDb = juce::Decibels::gainToDecibels(smoothedRms, -100.0f);
        
        return smoothedDb - targetDb;
    }

private:
    std::atomic<float> smoothedPower;
    float alpha;
};

