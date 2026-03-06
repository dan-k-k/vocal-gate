// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "DSPConstants.h" // <--- ADD THIS
#include <cstring>   // For std::memset, std::memcpy, std::memmove
#include <cmath>     // For std::log10, std::exp
#include <algorithm> // For std::max

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
    // Register the parameter with the DAW (ID, Name, Min, Max, Default)
    thresholdParam = new juce::AudioParameterFloat("threshold", "Threshold", 0.01f, 0.99f, 0.50f);
    addParameter(thresholdParam);

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // --- FIX: CROSS-PLATFORM PATH HANDLING ---
    juce::File systemLibraryDir = juce::File::getSpecialLocation(juce::File::commonApplicationDataDirectory);
    juce::File manufacturerFolder;

#if JUCE_MAC
    // Mac: /Library/Application Support/DanK
    manufacturerFolder = systemLibraryDir.getChildFile("Application Support").getChildFile("DanK");
#else
    // Windows: C:\ProgramData\DanK
    manufacturerFolder = systemLibraryDir.getChildFile("DanK");
#endif

    juce::File pluginFolder = manufacturerFolder.getChildFile("VocalGate");
    juce::File modelFile = pluginFolder.getChildFile("vocalgate_int8.onnx");

    // --- FIX: STANDARDIZED LOGGING ---
    // We REMOVED the custom VocalGate_Debug.txt file logic. 
    // Now we use juce::Logger so it doesn't crash on standard Windows user accounts!

    if (modelFile.existsAsFile())
    {
        try {
            onnxSession = std::make_unique<Ort::Session>(
                onnxEnv, 
#if JUCE_WINDOWS
                modelFile.getFullPathName().toWideCharPointer(), // Windows ONNX requires wide chars
#else
                modelFile.getFullPathName().toStdString().c_str(), // Mac uses standard C-strings
#endif
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
    }
    else
    {
        juce::Logger::writeToLog("🚨 ERROR: Could not find ONNX at: " + modelFile.getFullPathName());
    }
}

VocalGateProcessor::~VocalGateProcessor()
{
    // Crucial: Stop the thread safely before the plugin is destroyed
    stopThread (4000); 
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
    delayLine.setDelay(static_cast<float>(lookaheadSamples));

    // --- Pre-allocate ML Thread Memory ---
    int dawSamplesPerHop = static_cast<int>(sampleRate * 0.25);
    
    dawHopBuffer.assign(dawSamplesPerHop, 0.0f);
    resampledHopBuffer.assign(4000, 0.0f);
    rolling16kBuffer.assign(16000, 0.0f);
    
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
    auto totalNumInputChannels = getTotalNumInputChannels();

    // 1. PUSH TO ML FIFO (Undelayed Audio)
    const float* leftChannelIn = buffer.getReadPointer(0);
    if (audioFifo != nullptr && audioFifo->getFreeSpace() >= numSamples)
    {
        audioFifo->push (leftChannelIn, numSamples);
    }

    // 2. PUSH TO DELAY LINE (The Waiting Room)
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    delayLine.process(context); 
    // ^ At this point, 'buffer' now contains the audio from 300ms ago!

    // 3. READ THE ML PREDICTION 
    float currentProb = gateProbability.load();
    float currentThreshold = thresholdParam->get();

    // Match Python logic: If it's a cough, drop to 0.1f (-20dB), otherwise 1.0f
    float duckingGain = 0.1f; 
    float targetGain = (currentProb >= currentThreshold) ? duckingGain : 1.0f;

    // 4. APPLY SMOOTHING TO THE DELAYED AUDIO
    // To match the ~125ms moving average window from Python at 44.1kHz/48kHz,
    // we need much slower coefficients. 
    float envelopeCoef = 0.9998f; 

    float* outL = buffer.getWritePointer(0);
    float* outR = (totalNumInputChannels > 1) ? buffer.getWritePointer(1) : nullptr;

    for (int i = 0; i < numSamples; ++i)
    {
        // Smooth the envelope
        currentGainEnvelope = envelopeCoef * currentGainEnvelope + (1.0f - envelopeCoef) * targetGain; 

        outL[i] *= currentGainEnvelope;
        if (outR != nullptr)
            outR[i] *= currentGainEnvelope;
    }

    // Inside processBlock, after applying the envelope:
    float currentPeak = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        currentPeak = std::max(currentPeak, std::abs(outL[i]));
    }
    // Smooth it slightly for the UI, or just store the block's peak
    latestAudioLevel.store(currentPeak);
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
    // 1. Define ONNX Tensor Shapes
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {1, 1, 40, 61};
    
    // Create the input tensor pointing to our MFCC array
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
    
    // Manual sigmoid since we used BCEWithLogitsLoss
    float prob = 1.0f / (1.0f + std::exp(-outLogit[0]));

    // 4. Update the Atomic Float for the Audio Thread!
    gateProbability.store(prob);
}

// --- The Background ML Thread ---
void VocalGateProcessor::run()
{
    int dawSamplesPerHop = static_cast<int>(dawSampleRate * 0.25);

    while (! threadShouldExit())
    {
        if (audioFifo != nullptr && audioFifo->getNumReady() >= dawSamplesPerHop)
        {
            // Pop directly into our pre-allocated buffer
            audioFifo->pop(dawHopBuffer.data(), dawSamplesPerHop);

            const float* inData = dawHopBuffer.data();
            float* outData = resampledHopBuffer.data();
            resampler.process(dawSampleRate / 16000.0, inData, outData, 4000);

            std::memmove(rolling16kBuffer.data(), 
                         rolling16kBuffer.data() + 4000, 
                         12000 * sizeof(float));
                         
            std::memcpy(rolling16kBuffer.data() + 12000, 
                        resampledHopBuffer.data(), 
                        4000 * sizeof(float));

            // computeLogMels now writes directly to our class member 'logMelFeatures'
            computeLogMels(rolling16kBuffer);

            if (onnxSession != nullptr)
            {
                if (threadShouldExit()) return; // Another safety check before inference

                try {
                    auto startTime = juce::Time::getMillisecondCounterHiRes(); 
                    
                    // runONNXModel can just use the 'logMelFeatures' class member
                    runONNXModel();
                    
                    auto endTime = juce::Time::getMillisecondCounterHiRes();  
                    juce::Logger::writeToLog("VocalGate ONNX Inference Took: " + juce::String(endTime - startTime) + " ms");
                } 
                catch (const Ort::Exception& e) {
                    juce::Logger::writeToLog("ONNX RUN ERROR: " + juce::String(e.what()));
                    wait(1000); 
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

