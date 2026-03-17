// plugin/Source/GateDSP.cpp
#include "GateDSP.h"

GateDSP::GateDSP()
{
}

void GateDSP::prepare(const juce::dsp::ProcessSpec& spec, int samplesPerHop)
{
    dawSampleRate = spec.sampleRate;
    dawSamplesPerHop = samplesPerHop;
    
    // Default lookahead latency (0.750 seconds)
    lookaheadSamples = static_cast<int>(dawSampleRate * 0.750); 

    delayLine.setMaximumDelayInSamples(static_cast<int>(dawSampleRate * 2.0));
    delayLine.prepare(spec);
    delayLine.setDelay(lookaheadSamples);

    currentGainEnvelope = 1.0f;
    audioReadIndex.store(0);
    inputLevel.store(0.0f);
    outputLevel.store(0.0f);
    gateProbability.store(0.0f);
}

void GateDSP::process(juce::AudioBuffer<float>& buffer, 
                      const ParameterManager& params,
                      const std::atomic<float>* probRingBuffer, 
                      int probBufferSize)
{
    int numSamples = buffer.getNumSamples();
    int numChannels = buffer.getNumChannels();
    
    const float* leftChannelIn = buffer.getReadPointer(0);
    float* outL = buffer.getWritePointer(0);
    float* outR = (numChannels > 1) ? buffer.getWritePointer(1) : nullptr;

    // Grab current parameters from our clean manager
    float currentThreshold = params.getThreshold();
    float currentFloorDB   = params.getFloor();
    float currentAttackMs  = params.getAttack();
    float currentReleaseMs = params.getRelease();
    float shiftMs          = params.getShift();

    // Calculate coefficients
    float attackCoef = std::exp(-1.0f / (currentAttackMs * 0.001f * dawSampleRate));
    float releaseCoef = std::exp(-1.0f / (currentReleaseMs * 0.001f * dawSampleRate));
    float duckingGain = (currentFloorDB <= -99.9f) ? 0.0f : juce::Decibels::decibelsToGain(currentFloorDB);

    // Shift in samples
    int halfWindowSamples = static_cast<int>(dawSampleRate * 0.55);
    int shiftSamples = static_cast<int>(shiftMs * (dawSampleRate / 1000.0f));

    float currentInPeak = 0.0f;
    float currentOutPeak = 0.0f;
    uint64_t localReadIndex = audioReadIndex.load(std::memory_order_relaxed);

    for (int i = 0; i < numSamples; ++i)
    {
        // 1. Process fixed audio delay
        float inL = buffer.getSample(0, i);
        float delayedL = delayLine.popSample(0);
        delayLine.pushSample(0, inL);
        outL[i] = delayedL;

        if (outR != nullptr) {
            float inR = buffer.getSample(1, i);
            float delayedR = delayLine.popSample(1);
            delayLine.pushSample(1, inR);
            outR[i] = delayedR;
        }

        // 2. Read the synchronized ML probability
        int64_t syncedIndex = localReadIndex - lookaheadSamples + halfWindowSamples - dawSamplesPerHop - shiftSamples;
        int readPos = static_cast<int>(syncedIndex % probBufferSize); 
        if (readPos < 0) { readPos += probBufferSize; }

        float currentProb = 0.0f;
        if (probRingBuffer != nullptr) {
            currentProb = probRingBuffer[readPos].load(std::memory_order_relaxed);
        }
        gateProbability.store(currentProb, std::memory_order_relaxed);

        // 3. Apply gate envelope
        float targetGain = (currentProb < currentThreshold) ? 1.0f : duckingGain;

        if (targetGain < currentGainEnvelope) {
            currentGainEnvelope = attackCoef * currentGainEnvelope + (1.0f - attackCoef) * targetGain;
        } else {
            currentGainEnvelope = releaseCoef * currentGainEnvelope + (1.0f - releaseCoef) * targetGain;
        }

        // 4. Output math
        outL[i] *= currentGainEnvelope;
        if (outR != nullptr) outR[i] *= currentGainEnvelope;

        // 5. Update peaks
        currentInPeak = std::max(currentInPeak, std::abs(delayedL));
        currentOutPeak = std::max(currentOutPeak, std::abs(outL[i]));

        localReadIndex++; 
    }

    // Save state for the next block
    audioReadIndex.store(localReadIndex, std::memory_order_relaxed);
    inputLevel.store(currentInPeak, std::memory_order_relaxed);  
    outputLevel.store(currentOutPeak, std::memory_order_relaxed);
}

