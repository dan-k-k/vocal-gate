// plugin/Source/ParameterManager.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

class ParameterManager
{
public:
    // We pass the main processor in so APVTS can hook into it
    explicit ParameterManager(juce::AudioProcessor& processor);

    // The APVTS is public so your Editor and Processor can easily access it
    juce::AudioProcessorValueTreeState apvts;

    // Clean getters for the DSP and ML threads (Real-time safe)
    float getThreshold() const     { return thresholdParam->load(std::memory_order_relaxed); }
    float getFloor() const         { return floorParam->load(std::memory_order_relaxed); }
    float getAttack() const        { return attackParam->load(std::memory_order_relaxed); }
    float getRelease() const       { return releaseParam->load(std::memory_order_relaxed); }
    float getShift() const         { return shiftParam->load(std::memory_order_relaxed); }
    float getProbSmoothing() const { return probSmoothingParam->load(std::memory_order_relaxed); }

    // DAW Save/Load State Handling
    void saveState(juce::MemoryBlock& destData);
    void loadState(const void* data, int sizeInBytes);

private:
    // The layout definition lives exclusively here now
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Cached atomic pointers
    std::atomic<float>* thresholdParam = nullptr;
    std::atomic<float>* floorParam     = nullptr;
    std::atomic<float>* attackParam    = nullptr;
    std::atomic<float>* releaseParam   = nullptr;
    std::atomic<float>* shiftParam     = nullptr;
    std::atomic<float>* probSmoothingParam = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ParameterManager)
};