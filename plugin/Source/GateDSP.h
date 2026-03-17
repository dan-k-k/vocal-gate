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

    // Called in Processor::prepareToPlay
    void prepare(const juce::dsp::ProcessSpec& spec, int dawSamplesPerHop);

    // Called in Processor::processBlock
    // We pass the shared probability buffer that the ML thread writes to
    void process(juce::AudioBuffer<float>& buffer, 
                 const ParameterManager& params,
                 const std::atomic<float>* probRingBuffer, 
                 int probBufferSize);

    // Real-time safe getters for your GUI meters
    float getInputLevel() const  { return inputLevel.load(std::memory_order_relaxed); }
    float getOutputLevel() const { return outputLevel.load(std::memory_order_relaxed); }
    float getGateProbability() const { return gateProbability.load(std::memory_order_relaxed); }

    int getLookaheadSamples() const { return lookaheadSamples; }

private:
    double dawSampleRate = 44100.0;
    int lookaheadSamples = 0;
    int dawSamplesPerHop = 0;

    // Audio State
    float currentGainEnvelope = 1.0f;
    std::atomic<uint64_t> audioReadIndex { 0 };

    // Delay line for lookahead
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLine { 96000 };

    // Metering state
    std::atomic<float> inputLevel { 0.0f };
    std::atomic<float> outputLevel { 0.0f };
    std::atomic<float> gateProbability { 0.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GateDSP)
};

