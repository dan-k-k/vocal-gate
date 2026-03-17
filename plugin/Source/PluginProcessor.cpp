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
    
    dspCore.prepare(spec, dawSamplesPerHop);
    mlThread.prepare(sampleRate, dawSamplesPerHop);

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
    const float* leftChannelIn = buffer.getReadPointer(0);

    // -----------------------------------------------------------------------
    // 1. Route Audio to the Machine Learning Thread
    // -----------------------------------------------------------------------
    if (isNonRealtime()) 
    {
        // Offline rendering: Block the thread and process synchronously
        // Note: You would adapt mlThread to expose a synchronous push/pop for offline processing
        // For brevity, assuming mlThread has a blocking processOffline block here:
        int processed = 0;
        while (processed < numSamples) {
            int chunk = std::min(dawSamplesPerHop, numSamples - processed);
            mlThread.processOfflineBlock(leftChannelIn + processed, parameterManager);
            processed += chunk;
        }
    }
    else 
    {
        // Realtime rendering: Push to the lock-free FIFO and wake the ML thread
        mlThread.pushAudio(leftChannelIn, numSamples);
        
        // If we've pushed enough for a new hop, notify the thread
        if (numSamples >= dawSamplesPerHop) {
            mlThread.notifyDataReady(); 
        }
    }

    // -----------------------------------------------------------------------
    // 2. Apply DSP (Delay Line & Gate Envelope)
    // -----------------------------------------------------------------------
    // We pass the parameter manager so DSP can read the latest threshold/attack/etc.
    // We pass the ML thread's ring buffer so DSP knows when to open/close the gate.
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
    // Return your UI here. It will connect to `this->parameterManager.apvts`
    // return new VocalGateEditor (*this); 
    return nullptr; // Temporary until you link your editor
}

// -----------------------------------------------------------------------
// JUCE Plugin Entry Point
// -----------------------------------------------------------------------
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

