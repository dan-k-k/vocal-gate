// plugin/Source/PluginProcessor.h
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "ParameterManager.h"
#include "GateDSP.h"
#include "BackgroundMLThread.h"
#include <vector>

class VocalGateProcessor : public juce::AudioProcessor
{
public:
    VocalGateProcessor();
    ~VocalGateProcessor() override;

    // Standard JUCE audio lifecycle
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // Editor integration
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // Standard JUCE metadata
    const juce::String getName() const override { return "Vocal Gate"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    // Programs
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    // State handling (Delegated to ParameterManager)
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Public module for the Editor to connect to
    ParameterManager parameterManager;

    // Public getters for the Editor to read meters
    float getInputLevel() const  { return dspCore.getInputLevel(); }
    float getOutputLevel() const { return dspCore.getOutputLevel(); }
    float getGateProbability() const { return dspCore.getGateProbability(); }

private:
    // Core Modules
    GateDSP dspCore;
    BackgroundMLThread mlThread;

    int dawSamplesPerHop = 0;
    std::vector<float> offlineHopBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateProcessor)
};

