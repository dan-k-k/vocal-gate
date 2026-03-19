// plugin/Source/ParameterManager.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

class ParameterManager
{
public:
    explicit ParameterManager(juce::AudioProcessor& processor);
    juce::AudioProcessorValueTreeState apvts;

    float getThreshold() const     { return thresholdParam->load(std::memory_order_relaxed); }
    float getFloor() const         { return floorParam->load(std::memory_order_relaxed); }
    float getAttack() const        { return attackParam->load(std::memory_order_relaxed); }
    float getRelease() const       { return releaseParam->load(std::memory_order_relaxed); }
    float getShift() const         { return shiftParam->load(std::memory_order_relaxed); }
    float getProbSmoothing() const { return probSmoothingParam->load(std::memory_order_relaxed); }
    float getInputGain() const     { return inputGainParam->load(std::memory_order_relaxed); } // <-- Changed here

    void saveState(juce::MemoryBlock& destData);
    void loadState(const void* data, int sizeInBytes);

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    std::atomic<float>* thresholdParam = nullptr;
    std::atomic<float>* floorParam     = nullptr;
    std::atomic<float>* attackParam    = nullptr;
    std::atomic<float>* releaseParam   = nullptr;
    std::atomic<float>* shiftParam     = nullptr;
    std::atomic<float>* probSmoothingParam = nullptr;
    std::atomic<float>* inputGainParam = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ParameterManager)
};

