// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h" // Assuming you have an Editor class

VocalGateProcessor::VocalGateProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
       parameterManager(*this) // Initialize APVTS
#endif
{
}

VocalGateProcessor::~VocalGateProcessor()
{
    mlThread.stopProcessing();
}

void VocalGateProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // 1. Calculate our hop size (50ms of audio)
    dawSamplesPerHop = static_cast<int>(sampleRate * 0.05);
    offlineHopBuffer.assign(dawSamplesPerHop, 0.0f);

    // 2. Prepare our sub-modules
    juce::dsp::ProcessSpec spec { sampleRate, static_cast<uint32_t>(samplesPerBlock), static_cast<uint32_t>(getTotalNumOutputChannels()) };

    inputGainModule.prepare(spec);
    inputGainModule.setRampDurationSeconds(0.02);
    
    dspCore.prepare(spec, dawSamplesPerHop);
    mlThread.prepare(sampleRate, dawSamplesPerHop, parameterManager);
    
    // <--- 1. Prepare the tracker with sample rate and block size
    inputVolumeTracker.prepare(sampleRate, samplesPerBlock);

    // 3. Report our lookahead latency to the DAW (Ableton, Logic, etc.)
    setLatencySamples(dspCore.getLookaheadSamples());

    // 4. Fire up the background inference thread
    mlThread.startProcessing();
}

void VocalGateProcessor::releaseResources()
{
    mlThread.stopProcessing();
}

void VocalGateProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    int numSamples = buffer.getNumSamples();

    // 1. Apply Smoothed Input Gain FIRST
    inputGainModule.setGainDecibels(parameterManager.getInputGain());
    
    juce::dsp::AudioBlock<float> audioBlock (buffer);
    juce::dsp::ProcessContextReplacing<float> context (audioBlock);
    inputGainModule.process(context);

    // <--- 2. Process tracker immediately after input gain, before DSP/ML
    inputVolumeTracker.processBlock(buffer);

    const float* leftChannelIn = buffer.getReadPointer(0);

    // 2. Route Audio to the Machine Learning Thread
    mlThread.setOfflineMode(isNonRealtime());
    mlThread.pushAudio(leftChannelIn, numSamples);

    if (isNonRealtime()) 
    {
        // Offline rendering: Synchronously pop and process hops on the main thread
        while (mlThread.getNumReadySamples() >= dawSamplesPerHop) 
        {
            mlThread.processNextOfflineHop(parameterManager);
        }
    }
    else 
    {
        // Realtime rendering: Wake the background thread ONLY if we have enough data
        if (mlThread.getNumReadySamples() >= dawSamplesPerHop) 
        {
            mlThread.notifyDataReady(); 
        }
    }

    // -----------------------------------------------------------------------
    // DSP Processing
    // -----------------------------------------------------------------------
    dspCore.process(buffer, parameterManager, mlThread.getProbRingBuffer(), mlThread.getProbBufferSize());
}

// -----------------------------------------------------------------------
// State Management (Delegated to ParameterManager)
// -----------------------------------------------------------------------

void VocalGateProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    parameterManager.saveState(destData);
}

void VocalGateProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    parameterManager.loadState(data, sizeInBytes);
}

// -----------------------------------------------------------------------
// Editor Creation
// -----------------------------------------------------------------------

juce::AudioProcessorEditor* VocalGateProcessor::createEditor()
{
    return new VocalGateEditor (*this); 
}

// -----------------------------------------------------------------------
// JUCE Plugin Entry Point
// -----------------------------------------------------------------------
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

