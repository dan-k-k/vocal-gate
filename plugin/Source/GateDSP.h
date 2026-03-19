// plugin/Source/GateDSP.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "ParameterManager.h"

class GateDSP
{
public:
    GateDSP();
    ~GateDSP() = default;

    // For Processor::prepareToPlay
    void prepare(const juce::dsp::ProcessSpec& spec, int dawSamplesPerHop);

    // For Processor::processBlock
    void process(juce::AudioBuffer<float>& buffer, const ParameterManager& params, 
        const std::atomic<float>* probRingBuffer, int probBufferSize);

    // Getters for GUI
    float getInputLevel() const  { return inputLevel.load(std::memory_order_relaxed); }
    float getOutputLevel() const { return outputLevel.load(std::memory_order_relaxed); }
    float getGateProbability() const { return gateProbability.load(std::memory_order_relaxed); }

    int getLookaheadSamples() const { return lookaheadSamples; }

private:
    double dawSampleRate = 44100.0;
    int lookaheadSamples = 0;
    int dawSamplesPerHop = 0;

    // Audio state
    float currentGainEnvelope = 1.0f;
    std::atomic<uint64_t> audioReadIndex { 0 };

    // Delay line for lookahead
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLine { 96000 };

    // Metering
    std::atomic<float> inputLevel { 0.0f };
    std::atomic<float> outputLevel { 0.0f };
    std::atomic<float> gateProbability { 0.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GateDSP)
};

