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

    // -----------------------------------------------------------------------
    // NEW: Apply Smoothed Input Gain FIRST
    // -----------------------------------------------------------------------
    inputGainModule.setGainDecibels(parameterManager.getInputGain());
    
    juce::dsp::AudioBlock<float> audioBlock (buffer);
    juce::dsp::ProcessContextReplacing<float> context (audioBlock);
    inputGainModule.process(context);

    // Now grab the read pointer AFTER the gain has been applied
    const float* leftChannelIn = buffer.getReadPointer(0);

    // -----------------------------------------------------------------------
    // 1. Route Audio to the Machine Learning Thread
    // -----------------------------------------------------------------------
    if (isNonRealtime()) 
    {
        int processed = 0;
        while (processed < numSamples) 
        {
            int chunk = std::min(dawSamplesPerHop, numSamples - processed);

            if (chunk == dawSamplesPerHop) 
            {
                // Fast path: We have exactly enough samples for a full hop
                mlThread.processOfflineBlock(leftChannelIn + processed, parameterManager);
            } 
            else 
            {
                // Edge case: The final chunk of the render is smaller than our hop size.
                // Copy what we have, zero-pad the rest, and process safely.
                std::fill(offlineHopBuffer.begin(), offlineHopBuffer.end(), 0.0f);
                std::copy(leftChannelIn + processed, leftChannelIn + processed + chunk, offlineHopBuffer.begin());
                
                mlThread.processOfflineBlock(offlineHopBuffer.data(), parameterManager);
            }
            processed += chunk;
        }
    }
    else 
    {
        // Realtime rendering: Push to the lock-free FIFO
        mlThread.pushAudio(leftChannelIn, numSamples);
        
        // Wake the ML thread ONLY if the FIFO has accumulated a full hop's worth of data
        if (mlThread.getNumReadySamples() >= dawSamplesPerHop) 
        {
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
    return new VocalGateEditor (*this); 
}

// -----------------------------------------------------------------------
// JUCE Plugin Entry Point
// -----------------------------------------------------------------------
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

