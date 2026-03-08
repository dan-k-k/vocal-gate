// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "DSPConstants.h" // <--- ADD THIS
#include <cstring>   // For std::memset, std::memcpy, std::memmove
#include <cmath>     // For std::log10, std::exp
#include <algorithm> // For std::max
#include <BinaryData.h>

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
       juce::Thread ("ONNX_ML_Thread") 
#endif
{
    // --- 1. Floor (Already good!) ---
    juce::NormalisableRange<float> floorRange (-100.0f, 0.0f, 0.1f);
    floorRange.setSkewForCentre (-18.0f);
    auto floorAttributes = juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction([] (float value, int maximumStringLength) 
        {
            if (value <= -99.9f) return juce::String ("-inf");
            return juce::String (value, 1); 
        });

    // --- 2. Attack (Skewed for fast transients) ---
    juce::NormalisableRange<float> attackRange (1.0f, 500.0f, 0.1f);
    attackRange.setSkewForCentre (20.0f);

    // --- 3. Release (Skewed for typical vocal tails) ---
    juce::NormalisableRange<float> releaseRange (10.0f, 2000.0f, 0.1f);
    releaseRange.setSkewForCentre (150.0f);

    // --- 4. Shift (Skewed so 0ms is exactly dead-center on the knob) ---
    juce::NormalisableRange<float> shiftRange (-100.0f, 200.0f, 0.1f);
    shiftRange.setSkewForCentre (0.0f);

    // --- 5. P Smooth (Skewed so 400ms is perfectly at 12 o'clock) ---
    juce::NormalisableRange<float> probSmoothRange (100.0f, 1200.0f, 1.0f);
    probSmoothRange.setSkewForCentre (400.0f);

    // --- Create and Add the Parameters ---
    thresholdParam = new juce::AudioParameterFloat("threshold", "Threshold", 0.001f, 0.999f, 0.50f);
    floorParam = new juce::AudioParameterFloat("floor", "Floor", floorRange, -25.0f, floorAttributes);
    attackParam  = new juce::AudioParameterFloat("attack", "Attack", attackRange, 10.0f);
    releaseParam = new juce::AudioParameterFloat("release", "Release", releaseRange, 150.0f);
    shiftParam   = new juce::AudioParameterFloat("shift", "Shift", shiftRange, 0.0f);
    probSmoothingParam = new juce::AudioParameterFloat("probsmoothing", "P Smooth", probSmoothRange, 400.0f);

    addParameter(thresholdParam);
    addParameter(floorParam);
    addParameter(attackParam);
    addParameter(releaseParam);
    addParameter(shiftParam);
    addParameter(probSmoothingParam);

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        // Load directly from the baked-in BinaryData memory
        onnxSession = std::make_unique<Ort::Session>(
            onnxEnv, 
            BinaryData::vocalgate_int8_onnx, 
            BinaryData::vocalgate_int8_onnxSize, 
            sessionOptions
        );
            
        Ort::AllocatorWithDefaultOptions allocator;
        auto expectedInput = onnxSession->GetInputNameAllocated(0, allocator);
        auto expectedOutput = onnxSession->GetOutputNameAllocated(0, allocator);
        
        juce::String msg = "✅ ONNX Loaded! Expected Input Name: " + juce::String(expectedInput.get()) + 
                           ", Expected Output Name: " + juce::String(expectedOutput.get());
        
        juce::Logger::writeToLog(msg); 
    } 
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog("🚨 ONNX LOAD CRASH: " + juce::String(e.what()));
    }
} // <--- The constructor now ends cleanly right after the catch block

VocalGateProcessor::~VocalGateProcessor()
{
    // Crucial: Stop the thread safely before the plugin is destroyed
    stopThread (4000); 
}

void VocalGateProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // Create an outer XML element
    juce::XmlElement xml ("VOCAL_GATE_SETTINGS");

    // Store the current values
    xml.setAttribute ("threshold", thresholdParam->get());
    xml.setAttribute ("floor", floorParam->get());
    xml.setAttribute ("attack", attackParam->get());
    xml.setAttribute ("release", releaseParam->get());
    xml.setAttribute ("shift", shiftParam->get());
    xml.setAttribute ("probsmoothing", probSmoothingParam->get());

    // Save it to the memory block
    copyXmlToBinary (xml, destData);
}

void VocalGateProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // Retrieve the XML from the DAW's saved memory block
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState != nullptr)
    {
        // Make sure it's our plugin's state
        if (xmlState->hasTagName ("VOCAL_GATE_SETTINGS"))
        {
            // Restore the values (with fallbacks to current values just in case)
            *thresholdParam = (float) xmlState->getDoubleAttribute ("threshold", thresholdParam->get());
            *floorParam = (float) xmlState->getDoubleAttribute ("floor", floorParam->get());
            *attackParam = (float) xmlState->getDoubleAttribute ("attack", attackParam->get());
            *releaseParam = (float) xmlState->getDoubleAttribute ("release", releaseParam->get());
            *shiftParam = (float) xmlState->getDoubleAttribute ("shift", shiftParam->get());
            *probSmoothingParam = (float) xmlState->getDoubleAttribute ("probsmoothing", probSmoothingParam->get());
        }
    }
}

void VocalGateProcessor::pushAndAveragePrediction(float rawProb)
{
    // 1. Write the new prediction to the circular buffer
    predictionHistory[predictionWriteIndex] = rawProb;
    predictionWriteIndex = (predictionWriteIndex + 1) % maxPredictionFrames;

    // 2. Determine how many frames to average
    float smoothMs = probSmoothingParam->get();
    int framesToKeep = std::max(1, static_cast<int>(smoothMs / 50.0f));
    framesToKeep = std::min(framesToKeep, maxPredictionFrames); // Safety clamp

    // 3. Sum backwards from the most recently written index
    float sum = 0.0f;
    for (int i = 0; i < framesToKeep; ++i) 
    {
        // (WriteIndex - 1) is the newest item. We subtract 'i' to go back in time, 
        // and add maxPredictionFrames to ensure the modulo doesn't fail on negative numbers.
        int readIndex = (predictionWriteIndex - 1 - i + maxPredictionFrames) % maxPredictionFrames;
        sum += predictionHistory[readIndex];
    }
            
    // 4. Update the atomic float for the audio thread and UI
    gateProbability.store(sum / static_cast<float>(framesToKeep));
}

void VocalGateProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Stop the thread if it's already running from a previous prepareToPlay call
    if (isThreadRunning())
        stopThread(2000); 

    dawSampleRate = sampleRate;
    audioFifo = std::make_unique<AudioFIFO> (static_cast<int>(dawSampleRate * 2.0));
    currentGainEnvelope = 1.0f;

    // --- Setup Lookahead Latency ---
    double lookaheadSeconds = 0.550;
    
    lookaheadSamples = static_cast<int>(sampleRate * lookaheadSeconds); 
    
    // Tell Ableton to delay all other tracks by 550ms
    setLatencySamples(lookaheadSamples); 

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = samplesPerBlock;
    spec.numChannels = getTotalNumOutputChannels();
    
    delayLine.prepare(spec);
    smoothedDelay.reset(sampleRate, 0.05); 
    
    // Calculate the initial delay time so it starts perfectly in place
    float initialShift = shiftParam->get();
    float initialDelay = lookaheadSamples - (initialShift * (sampleRate / 1000.0f));
    smoothedDelay.setCurrentAndTargetValue(initialDelay);

    delayLine.setDelay(initialDelay);

    // Set the glide time to 20ms
    double rampTimeSeconds = 0.02; 
    smoothedThreshold.reset(sampleRate, rampTimeSeconds);
    smoothedFloorDB.reset(sampleRate, rampTimeSeconds);
    smoothedAttack.reset(sampleRate, rampTimeSeconds);
    smoothedRelease.reset(sampleRate, rampTimeSeconds);

    // Snap them immediately to the current knob positions so they don't fade in from zero
    smoothedThreshold.setCurrentAndTargetValue(thresholdParam->get());
    smoothedFloorDB.setCurrentAndTargetValue(floorParam->get());
    smoothedAttack.setCurrentAndTargetValue(attackParam->get());
    smoothedRelease.setCurrentAndTargetValue(releaseParam->get());

    // --- Pre-allocate ML Thread Memory ---
    int dawSamplesPerHop = static_cast<int>(sampleRate * 0.05);
    
    dawHopBuffer.assign(dawSamplesPerHop, 0.0f);
    resampledHopBuffer.assign(800, 0.0f);
    rolling16kBuffer.assign(16000, 0.0f);

    predictionHistory.fill(0.0f);
    predictionWriteIndex = 0;

    smoothedProbability.reset(sampleRate, 0.05); // 50ms glide to match the hop
    smoothedProbability.setCurrentAndTargetValue(0.0f);
    
    logMelFeatures.assign(40 * 61, 0.0f);
    timeDomain.assign(512 * 2, 0.0f); 
    powerSpec.assign(257, 0.0f);
    melEnergies.assign(40, 0.0f);
    
    startThread(); 
}

void VocalGateProcessor::releaseResources()
{
    // Safely stop the background ML thread
    stopThread(4000);
    
    // Free up memory when Ableton stops the transport
    audioFifo.reset();
}

void VocalGateProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    int numSamples = buffer.getNumSamples();

    // --- 1. PUSH TO ML FIFO (Undelayed Audio) ---
    // We MUST send the real-time audio to the neural network before we delay it!
    const float* leftChannelIn = buffer.getReadPointer(0);
    if (audioFifo != nullptr && audioFifo->getFreeSpace() >= numSamples)
    {
        audioFifo->push (leftChannelIn, numSamples);
    }

    // --- 2. PUSH TO DELAY LINE (Smoothed to prevent clicks!) ---
    float shiftMs = shiftParam->get();
    float targetDelaySamples = lookaheadSamples - (shiftMs * (dawSampleRate / 1000.0f));
    targetDelaySamples = juce::jlimit(0.0f, (float)delayLine.getMaximumDelayInSamples() - 1.0f, targetDelaySamples);
    
    // Tell the smoother where we want to go
    smoothedDelay.setTargetValue(targetDelaySamples);

    smoothedThreshold.setTargetValue(thresholdParam->get());
    smoothedFloorDB.setTargetValue(floorParam->get());
    smoothedAttack.setTargetValue(attackParam->get());
    smoothedRelease.setTargetValue(releaseParam->get());
    
    int numChannels = getTotalNumInputChannels();
    
    // Process the delay line sample-by-sample to allow the smooth glide
    for (int i = 0; i < numSamples; ++i)
    {
        // Get the interpolated delay time for this exact sample
        delayLine.setDelay(smoothedDelay.getNextValue());

        for (int ch = 0; ch < numChannels; ++ch)
        {
            float in = buffer.getSample(ch, i);
            float out = delayLine.popSample(ch); // Read delayed audio
            delayLine.pushSample(ch, in);        // Write new audio
            buffer.setSample(ch, i, out);        // Overwrite buffer with delayed audio
        }
    }

    // --- 3. MEASURE DELAYED INPUT PEAK ---
    float currentInPeak = 0.0f;
    const float* delayedInL = buffer.getReadPointer(0);
    for (int i = 0; i < numSamples; ++i) {
        currentInPeak = std::max(currentInPeak, std::abs(delayedInL[i]));
    }
    inputLevel.store(currentInPeak);

    // Read the ML brain once per block
    float currentProb = gateProbability.load(); 

    float* outL = buffer.getWritePointer(0);
    float* outR = (getTotalNumInputChannels() > 1) ? buffer.getWritePointer(1) : nullptr;
    float currentOutPeak = 0.0f;

    // --- NEW: Calculate Exp Coefficients ONCE per block ---
    // We grab the current value of the smoother at the start of the block
    float currentAttackMs = smoothedAttack.getCurrentValue();
    float currentReleaseMs = smoothedRelease.getCurrentValue();

    float attackCoef = std::exp(-1.0f / (currentAttackMs * 0.001f * dawSampleRate));
    float releaseCoef = std::exp(-1.0f / (currentReleaseMs * 0.001f * dawSampleRate));

    // --- 4. APPLY GATE ENVELOPES (Sample-by-Sample) ---
    for (int i = 0; i < numSamples; ++i)
    {
        // Advance smoothers for threshold and floor to prevent zipper noise
        float currentThreshold = smoothedThreshold.getNextValue();
        float currentFloorDB = smoothedFloorDB.getNextValue();
        
        // We still need to call getNextValue() on Attack/Release so the smoothers 
        // advance their internal state, even if we aren't using the per-sample output here.
        smoothedAttack.getNextValue();
        smoothedRelease.getNextValue();

        // Calculate target gain
        float duckingGain = (currentFloorDB <= -99.9f) ? 0.0f : juce::Decibels::decibelsToGain(currentFloorDB);
        float targetGain = (currentProb >= currentThreshold) ? duckingGain : 1.0f;

        // Apply envelopes using our pre-calculated block-rate coefficients
        if (targetGain < currentGainEnvelope) {
            currentGainEnvelope = attackCoef * currentGainEnvelope + (1.0f - attackCoef) * targetGain;
        } 
        else {
            currentGainEnvelope = releaseCoef * currentGainEnvelope + (1.0f - releaseCoef) * targetGain;
        }

        outL[i] *= currentGainEnvelope;
        currentOutPeak = std::max(currentOutPeak, std::abs(outL[i]));

        if (outR != nullptr)
            outR[i] *= currentGainEnvelope;
    }

    // --- 6. STORE OUTPUT PEAK ---
    outputLevel.store(currentOutPeak);
}

void VocalGateProcessor::computeLogMels(const std::vector<float>& audio16k)
{
    size_t n_fft = 512;
    size_t hop_length = 256;
    size_t num_frames = 61;     
    size_t num_freq_bins = 257; 
    size_t num_mels = 40;
    
    for (size_t frame = 0; frame < num_frames; ++frame)
    {
        // Safely exit if the user deletes the plugin!
        if (threadShouldExit()) return; 

        size_t start_sample = frame * hop_length;

        // Zero out the pre-allocated timeDomain buffer
        std::fill(timeDomain.begin(), timeDomain.end(), 0.0f);

        for (size_t i = 0; i < n_fft; ++i) {
            timeDomain[i] = audio16k[start_sample + i] * DSPConstants::hannWindow512[i];
        }

        forwardFFT.performFrequencyOnlyForwardTransform(timeDomain.data());
        
        // Overwrite the pre-allocated powerSpec buffer
        for (size_t i = 0; i < num_freq_bins; ++i) {
            powerSpec[i] = timeDomain[i] * timeDomain[i];
        }

        // Overwrite the pre-allocated melEnergies buffer
        for (size_t m = 0; m < num_mels; ++m) 
        {
            float sum = 0.0f;
            for (size_t f = 0; f < num_freq_bins; ++f) {
                sum += powerSpec[f] * DSPConstants::melFilterBank[f][m];
            }
            
            melEnergies[m] = 10.0f * std::log10(std::max(sum, 1e-10f));
            logMelFeatures[m * num_frames + frame] = melEnergies[m];
        }
    }
}

void VocalGateProcessor::runONNXModel()
{
    // 1. Define ONNX Tensor Shapes (memoryInfo is now pulled from the class member)
    std::vector<int64_t> inputShape = {1, 1, 40, 61};
    
    // Create the input tensor pointing to our MFCC array using the pre-allocated memoryInfo
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, 
        const_cast<float*>(logMelFeatures.data()), 
        logMelFeatures.size(), 
        inputShape.data(), 
        inputShape.size()
    );

    const char* inputNames[] = {"input_log_mel"};
    const char* outputNames[] = {"gate_logit"};

    // 2. Run the Model
    auto outputTensors = onnxSession->Run(
        Ort::RunOptions{nullptr}, 
        inputNames, 
        &inputTensor, 
        1, 
        outputNames, 
        1
    );

    // 3. Extract the Logit and Apply Sigmoid
    float* outLogit = outputTensors.front().GetTensorMutableData<float>();
    float rawProb = 1.0f / (1.0f + std::exp(-outLogit[0]));

    // 4. APPLY MOVING AVERAGE (Allocation-Free)
    pushAndAveragePrediction(rawProb);
}

// --- The Background ML Thread ---
void VocalGateProcessor::run()
{
    int dawSamplesPerHop = static_cast<int>(dawSampleRate * 0.05);
    
    // Calculate our "silence" threshold (-50 dB converted to linear gain)
    // -50 dB is roughly 0.00316
    const float silenceThreshold = juce::Decibels::decibelsToGain(-50.0f);

    while (! threadShouldExit())
    {
        if (audioFifo != nullptr && audioFifo->getNumReady() >= dawSamplesPerHop)
        {
            audioFifo->pop(dawHopBuffer.data(), dawSamplesPerHop);

            // --- 1. QUICK SILENCE CHECK ---
            float peakLevel = 0.0f;
            for (int i = 0; i < dawSamplesPerHop; ++i) {
                peakLevel = std::max(peakLevel, std::abs(dawHopBuffer[i]));
            }

            // --- 2. BRANCH BASED ON AUDIO LEVEL ---
            if (peakLevel < silenceThreshold) 
            {
                // The audio is essentially silent. Skip ONNX and FFTs!
                // Push a 0.0f probability to gracefully glide down the envelope
                pushAndAveragePrediction(0.0f);
            }
            else 
            {
                // The audio is loud enough to matter. Run the heavy ML math.
                const float* inData = dawHopBuffer.data();
                float* outData = resampledHopBuffer.data();
                
                resampler.process(dawSampleRate / 16000.0, inData, outData, 800);

                std::memmove(rolling16kBuffer.data(), 
                             rolling16kBuffer.data() + 800, 
                             15200 * sizeof(float));
                             
                std::memcpy(rolling16kBuffer.data() + 15200, 
                            resampledHopBuffer.data(), 
                            800 * sizeof(float));

                computeLogMels(rolling16kBuffer);

                if (onnxSession != nullptr && !threadShouldExit())
                {
                    try {
                        runONNXModel(); // This updates predictionHistory & gateProbability internally
                    } 
                    catch (const Ort::Exception& e) {
                        juce::Logger::writeToLog("ONNX RUN ERROR: " + juce::String(e.what()));
                        wait(1000); 
                    }
                }
            }
        }
        else
        {
            wait(10);
        }
    }
}

juce::AudioProcessorEditor* VocalGateProcessor::createEditor()
{
    // Assuming your custom editor class is named VocalGateEditor
    return new VocalGateEditor (*this); 
}

// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

