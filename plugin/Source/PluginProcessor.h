// plugin/Source/PluginProcessor.h
#pragma once

#include "ParameterManager.h"
#include "GateDSP.h"
#include "BackgroundMLThread.h"
#include "EmaRmsTracker.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>
#include <juce_dsp/juce_dsp.h>

class VocalGateProcessor : public juce::AudioProcessor
{
public:
    VocalGateProcessor();
    ~VocalGateProcessor() override;

    // JUCE audio lifecycle
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // Editor
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // JUCE metadata
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

    // State handling (delegated to ParameterManager)
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Public module for the Editor to connect to
    ParameterManager parameterManager;

    float getInputLevel() const  { return dspCore.getInputLevel(); }
    float getOutputLevel() const { return dspCore.getOutputLevel(); }
    float getGateProbability() const { return dspCore.getGateProbability(); }
    float getDbDifferenceFromTarget() const { return inputVolumeTracker.getDbDifferenceFromTarget(); }

private:
    GateDSP dspCore;
    BackgroundMLThread mlThread;
    EmaRmsTracker inputVolumeTracker;

    int dawSamplesPerHop = 0;
    std::vector<float> offlineHopBuffer;
    juce::dsp::Gain<float> inputGainModule;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateProcessor)
};

