// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h" 

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
       parameterManager(*this) // Initialise APVTS
#endif
{
}

VocalGateProcessor::~VocalGateProcessor()
{
    mlThread.stopProcessing();
}

void VocalGateProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Hop size (50ms)
    dawSamplesPerHop = static_cast<int>(sampleRate * 0.05);
    offlineHopBuffer.assign(dawSamplesPerHop, 0.0f);

    juce::dsp::ProcessSpec spec { sampleRate, static_cast<uint32_t>(samplesPerBlock), static_cast<uint32_t>(getTotalNumOutputChannels()) };

    inputGainModule.prepare(spec);
    inputGainModule.setRampDurationSeconds(0.02);
    
    dspCore.prepare(spec, dawSamplesPerHop);
    mlThread.prepare(sampleRate, dawSamplesPerHop, parameterManager);
    inputVolumeTracker.prepare(sampleRate, samplesPerBlock);

    // Report lookahead latency 
    setLatencySamples(dspCore.getLookaheadSamples());
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

    inputGainModule.setGainDecibels(parameterManager.getInputGain());
    
    juce::dsp::AudioBlock<float> audioBlock (buffer);
    juce::dsp::ProcessContextReplacing<float> context (audioBlock);
    inputGainModule.process(context);

    inputVolumeTracker.processBlock(buffer);

    const float* leftChannelIn = buffer.getReadPointer(0);

    // Route Audio to ML thread
    mlThread.setOfflineMode(isNonRealtime());
    mlThread.pushAudio(leftChannelIn, numSamples);

    if (isNonRealtime()) 
    {
        while (mlThread.getNumReadySamples() >= dawSamplesPerHop) 
        {
            mlThread.processNextOfflineHop(parameterManager);
        }
    }
    else 
    {
        if (mlThread.getNumReadySamples() >= dawSamplesPerHop) 
        {
            mlThread.notifyDataReady(); 
        }
    }
    // Process DSP 
    dspCore.process(buffer, parameterManager, mlThread.getProbRingBuffer(), mlThread.getProbBufferSize());
}

// State management
void VocalGateProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    parameterManager.saveState(destData);
}

void VocalGateProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    parameterManager.loadState(data, sizeInBytes);
}

// Create editor
juce::AudioProcessorEditor* VocalGateProcessor::createEditor()
{
    return new VocalGateEditor (*this); 
}

// JUCE plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

