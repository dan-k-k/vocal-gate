// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "DSPConstants.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <BinaryData.h>
#if JUCE_WINDOWS
#include <windows.h>
#endif

juce::AudioProcessorValueTreeState::ParameterLayout VocalGateProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // 1. Threshold
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"threshold", 1}, "P Threshold", 0.001f, 0.999f, 0.60f));

    // 2. Floor
    juce::NormalisableRange<float> floorRange (-100.0f, 0.0f, 0.1f);
    floorRange.setSkewForCentre (-18.0f);
    auto floorAttributes = juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction([] (float value, int) {
            if (value <= -99.9f) return juce::String ("-inf");
            return juce::String (value, 1); 
        });
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"floor", 1}, "Floor", floorRange, -25.0f, floorAttributes));

    // 3. Attack
    juce::NormalisableRange<float> attackRange (1.0f, 500.0f, 0.1f);
    attackRange.setSkewForCentre (20.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"attack", 1}, "Attack", attackRange, 10.0f));

    // 4. Release
    juce::NormalisableRange<float> releaseRange (10.0f, 2000.0f, 0.1f);
    releaseRange.setSkewForCentre (150.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"release", 1}, "Release", releaseRange, 150.0f));

    // 5. Shift
    juce::NormalisableRange<float> shiftRange (-200.0f, 200.0f, 0.1f);
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"shift", 1}, "Shift", shiftRange, 0.0f));

    // 6. Smooth
    juce::NormalisableRange<float> probSmoothRange (100.0f, 1200.0f, 1.0f);
    probSmoothRange.setSkewForCentre (400.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"probsmoothing", 1}, "Smooth", probSmoothRange, 400.0f));

    return { params.begin(), params.end() };
}

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
       juce::Thread ("ONNX_ML_Thread"),
       apvts(*this, nullptr, "Parameters", createParameterLayout()) 
#endif
{
    // --- 1. SET WINDOWS DLL DIRECTORY FIRST ---
    #if JUCE_WINDOWS
    // Get the handle to THIS specific VST3 module in memory, NOT the host DAW
    HMODULE hModule = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)&createPluginFilter, &hModule);

    if (hModule != nullptr)
    {
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(hModule, path, MAX_PATH);
        juce::File pluginDllFile(juce::String(path));
        
        // This targets the VST3's x86_64-win folder where your CMake puts the dll
        juce::String pluginDirectory = pluginDllFile.getParentDirectory().getFullPathName();
        SetDllDirectoryW(pluginDirectory.toWideCharPointer());
    }
    #endif
    
    // Atomic pointers
    thresholdParam     = apvts.getRawParameterValue("threshold");
    floorParam         = apvts.getRawParameterValue("floor");
    attackParam        = apvts.getRawParameterValue("attack");
    releaseParam       = apvts.getRawParameterValue("release");
    shiftParam         = apvts.getRawParameterValue("shift");
    probSmoothingParam = apvts.getRawParameterValue("probsmoothing");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VocalGate");
        memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        onnxSession = std::make_unique<Ort::Session>(
            *onnxEnv, // Dereference the pointer
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
}

VocalGateProcessor::~VocalGateProcessor()
{
    stopThread (4000); 
}

void VocalGateProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void VocalGateProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState != nullptr)
        if (xmlState->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}

float VocalGateProcessor::pushAndAveragePrediction(float rawProb)
{
    predictionHistory[predictionWriteIndex] = rawProb;
    predictionWriteIndex = (predictionWriteIndex + 1) % maxPredictionFrames;

    float smoothMs = probSmoothingParam->load();
    int framesToKeep = std::max(1, static_cast<int>(smoothMs / 50.0f));
    framesToKeep = std::min(framesToKeep, maxPredictionFrames);

    float sum = 0.0f;
    for (int i = 0; i < framesToKeep; ++i) 
    {
        int readIndex = (predictionWriteIndex - 1 - i + maxPredictionFrames) % maxPredictionFrames;
        sum += predictionHistory[readIndex];
    }
            
    return sum / static_cast<float>(framesToKeep); 
}

void VocalGateProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    if (isThreadRunning())
        stopThread(2000); 

    dawSampleRate = sampleRate;
    audioFifo = std::make_unique<AudioFIFO> (static_cast<int>(dawSampleRate * 2.0));
    currentGainEnvelope = 1.0f;

    // Default lookahead latency
    double lookaheadSeconds = 0.750; 
    
    lookaheadSamples = static_cast<int>(sampleRate * lookaheadSeconds); 
    
    // Tell Ableton to delay
    setLatencySamples(lookaheadSamples);

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = samplesPerBlock;
    spec.numChannels = getTotalNumOutputChannels();
    
    // Prob ring buffer
    probBufferSize = static_cast<int>(sampleRate * 2.0);
    probRingBuffer = std::make_unique<std::atomic<float>[]>(probBufferSize);
    for (int i = 0; i < probBufferSize; ++i) {
        probRingBuffer[i].store(0.0f, std::memory_order_relaxed);
    }

    mlWriteIndex = 0;
    audioReadIndex.store(0);

    delayLine.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));
    delayLine.prepare(spec);
    delayLine.setDelay(lookaheadSamples); // This never changes
    
    resampler.reset();

    // ML thread memory 
    dawSamplesPerHop = static_cast<int>(sampleRate * 0.05);
    
    dawHopBuffer.assign(dawSamplesPerHop, 0.0f);
    offlineHopBuffer.assign(dawSamplesPerHop, 0.0f);
    resampledHopBuffer.assign(800, 0.0f);
    rolling16kBuffer.assign(16000, 0.0f);

    predictionHistory.fill(0.0f);
    predictionWriteIndex = 0;
    
    logMelFeatures.assign(40 * 61, 0.0f);
    timeDomain.assign(512 * 2, 0.0f); 
    powerSpec.assign(257, 0.0f);
    melEnergies.assign(40, 0.0f);
    
    startThread(); 
}

void VocalGateProcessor::releaseResources()
{
    stopThread(4000);
    audioFifo.reset();
}

void VocalGateProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    int numSamples = buffer.getNumSamples();
    const float* leftChannelIn = buffer.getReadPointer(0);

    if (isNonRealtime()) 
    {
        // Offline rendering
        if (audioFifo != nullptr) 
        {
            audioFifo->push(leftChannelIn, numSamples);
            while (audioFifo->getNumReady() >= dawSamplesPerHop) 
            {
                audioFifo->pop(offlineHopBuffer.data(), dawSamplesPerHop);
                processMLHop(offlineHopBuffer.data()); // Blocks audio thread until ML finishes
            }
        }
    }
    else 
    {
        // Realtime: background thread
        if (audioFifo != nullptr && audioFifo->getFreeSpace() >= numSamples)
        {
            audioFifo->push (leftChannelIn, numSamples);
            if (audioFifo->getNumReady() >= dawSamplesPerHop)
            {
                mlDataReady.store(true, std::memory_order_release); 
            }
        }
    }

    float currentThreshold = thresholdParam->load();
    float currentFloorDB = floorParam->load();
    float currentAttackMs = attackParam->load();
    float currentReleaseMs = releaseParam->load();

    float attackCoef = std::exp(-1.0f / (currentAttackMs * 0.001f * dawSampleRate));
    float releaseCoef = std::exp(-1.0f / (currentReleaseMs * 0.001f * dawSampleRate));

    float duckingGain = (currentFloorDB <= -99.9f) ? 0.0f : juce::Decibels::decibelsToGain(currentFloorDB);

    // Shift in samples
    int halfWindowSamples = static_cast<int>(dawSampleRate * 0.55); // Allows time to look ahead
    float shiftMs = shiftParam->load();
    int shiftSamples = static_cast<int>(shiftMs * (dawSampleRate / 1000.0f));

    int numChannels = getTotalNumInputChannels();
    float* outL = buffer.getWritePointer(0);
    float* outR = (numChannels > 1) ? buffer.getWritePointer(1) : nullptr;

    float currentInPeak = 0.0f;
    float currentOutPeak = 0.0f;
    uint64_t localReadIndex = audioReadIndex.load(std::memory_order_relaxed);

    for (int i = 0; i < numSamples; ++i)
    {
        // Process fixed audio delay
        float inL = buffer.getSample(0, i);
        float delayedL = delayLine.popSample(0);
        delayLine.pushSample(0, inL);
        outL[i] = delayedL;

        if (outR != nullptr) {
            float inR = buffer.getSample(1, i);
            float delayedR = delayLine.popSample(1);
            delayLine.pushSample(1, inR);
            outR[i] = delayedR;
        }

        // Read the synchronised prob
        int64_t syncedIndex = localReadIndex - lookaheadSamples + halfWindowSamples - dawSamplesPerHop - shiftSamples;
        int readPos = static_cast<int>(syncedIndex % probBufferSize); 
        if (readPos < 0) { readPos += probBufferSize; }

        float currentProb = probRingBuffer[readPos].load(std::memory_order_relaxed);
        gateProbability.store(currentProb, std::memory_order_relaxed);

        // Apply gate env
        float targetGain = (currentProb < currentThreshold) ? 1.0f : duckingGain;

        if (targetGain < currentGainEnvelope) {
            currentGainEnvelope = attackCoef * currentGainEnvelope + (1.0f - attackCoef) * targetGain;
        } else {
            currentGainEnvelope = releaseCoef * currentGainEnvelope + (1.0f - releaseCoef) * targetGain;
        }

        outL[i] *= currentGainEnvelope;
        if (outR != nullptr) outR[i] *= currentGainEnvelope;

        currentInPeak = std::max(currentInPeak, std::abs(delayedL));
        currentOutPeak = std::max(currentOutPeak, std::abs(outL[i]));

        localReadIndex++; 
    }

    audioReadIndex.store(localReadIndex, std::memory_order_relaxed);

    inputLevel.store(currentInPeak);  
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
        if (threadShouldExit()) return; 

        size_t start_sample = frame * hop_length;

        std::fill(timeDomain.begin(), timeDomain.end(), 0.0f);

        for (size_t i = 0; i < n_fft; ++i) {
            timeDomain[i] = audio16k[start_sample + i] * DSPConstants::hannWindow512[i];
        }

        forwardFFT.performFrequencyOnlyForwardTransform(timeDomain.data());
        
        for (size_t i = 0; i < num_freq_bins; ++i) {
            powerSpec[i] = timeDomain[i] * timeDomain[i];
        }

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

void VocalGateProcessor::processMLHop(const float* hopData)
{
    const float silenceThreshold = juce::Decibels::decibelsToGain(-50.0f);
    
    float peakLevel = 0.0f;
    for (int i = 0; i < dawSamplesPerHop; ++i) {
        peakLevel = std::max(peakLevel, std::abs(hopData[i]));
    }

    if (peakLevel < silenceThreshold) 
    {
        float smoothedProb = pushAndAveragePrediction(0.0f);
        
        for (int i = 0; i < dawSamplesPerHop; ++i) 
        {
            int writePos = (mlWriteIndex + i) % probBufferSize;
            probRingBuffer[writePos].store(smoothedProb, std::memory_order_relaxed);
        }
        mlWriteIndex += dawSamplesPerHop;
    }
    else 
    {
        resampler.process(dawSampleRate / 16000.0, hopData, resampledHopBuffer.data(), 800);

        std::memmove(rolling16kBuffer.data(), rolling16kBuffer.data() + 800, 15200 * sizeof(float));
        std::memcpy(rolling16kBuffer.data() + 15200, resampledHopBuffer.data(), 800 * sizeof(float));

        computeLogMels(rolling16kBuffer);

        if (onnxSession != nullptr)
        {
            try {
                runONNXModel(); 
            } 
            catch (const Ort::Exception& e) {
                juce::Logger::writeToLog("ONNX RUN ERROR: " + juce::String(e.what()));
            }
        }
    }
}

void VocalGateProcessor::runONNXModel()
{
    // Input tensor wrapper (add * to memoryInfo)
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        *memoryInfo, 
        logMelFeatures.data(), 
        logMelFeatures.size(), 
        inputShape, 
        4
    );

    // Output tensor wrapper (add * to memoryInfo)
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        *memoryInfo,
        outputLogitData.data(),
        outputLogitData.size(),
        outputShape,
        2
    );

    const char* inputNames[] = {"input_log_mel"};
    const char* outputNames[] = {"gate_logit"};

    // Run model 
    onnxSession->Run(
        Ort::RunOptions{nullptr}, 
        inputNames, 
        &inputTensor, 
        1, 
        outputNames, 
        &outputTensor,
        1
    );

    float rawProb = 1.0f / (1.0f + std::exp(-outputLogitData[0]));
    float finalProb = pushAndAveragePrediction(rawProb); 

    for (int i = 0; i < dawSamplesPerHop; ++i) 
    {
        int writePos = (mlWriteIndex + i) % probBufferSize;
        probRingBuffer[writePos].store(finalProb, std::memory_order_relaxed);
    }
    mlWriteIndex += dawSamplesPerHop;
}

// Background ML thread
void VocalGateProcessor::run()
{    
    while (! threadShouldExit())
    {
        // Check if the audio thread told us data is ready OR if the FIFO just has enough data
        if (mlDataReady.load(std::memory_order_acquire) || 
           (!isNonRealtime() && audioFifo != nullptr && audioFifo->getNumReady() >= dawSamplesPerHop))
        {
            while (audioFifo != nullptr && audioFifo->getNumReady() >= dawSamplesPerHop && !threadShouldExit())
            {
                audioFifo->pop(dawHopBuffer.data(), dawSamplesPerHop);
                processMLHop(dawHopBuffer.data());
            }
            mlDataReady.store(false, std::memory_order_release);
        }
        else
        {
            juce::Thread::sleep(5); 
        }
    }
}

juce::AudioProcessorEditor* VocalGateProcessor::createEditor()
{
    return new VocalGateEditor (*this); 
}

// Create plugin instance
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VocalGateProcessor();
}

