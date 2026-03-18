// plugin/Source/BackgroundMLThread.cpp
#include "BackgroundMLThread.h"
#include "FeatureExtractor.h"  // (To be implemented)
#include "InferenceEngine.h"   // (To be implemented)

BackgroundMLThread::BackgroundMLThread()
    : juce::Thread("ONNX_ML_Thread")
{
    featureExtractor = std::make_unique<FeatureExtractor>();
    inferenceEngine = std::make_unique<InferenceEngine>();
}

BackgroundMLThread::~BackgroundMLThread()
{
    stopProcessing();
}

void BackgroundMLThread::prepare(double sampleRate, int samplesPerHop, const ParameterManager& params)
{
    stopProcessing();
    currentParams = &params;

    // 1. Setup Audio FIFO (2 seconds worth of buffering)
    audioFifo = std::make_unique<AudioFIFO>(static_cast<int>(sampleRate * 2.0));
    dawHopBuffer.assign(samplesPerHop, 0.0f);

    // 2. Setup Probability Ring Buffer
    probBufferSize = static_cast<int>(sampleRate * 2.0);
    probRingBuffer = std::make_unique<std::atomic<float>[]>(probBufferSize);
    for (int i = 0; i < probBufferSize; ++i) {
        probRingBuffer[i].store(0.0f, std::memory_order_relaxed);
    }

    // 3. Reset state
    mlWriteIndex = 0;
    predictionHistory.fill(0.0f);
    predictionWriteIndex = 0;
    mlDataReady.store(false);

    double padDurationSeconds = 1.0; // To completely fill the ends of the spectrogram
    double armingSeconds = 0.3;      // How much silence is needed to re-arm
    int safeSamplesPerHop = std::max(1, samplesPerHop); 
    padDurationHops = static_cast<int>(std::ceil((padDurationSeconds * sampleRate) / safeSamplesPerHop));
    armingHops = static_cast<int>(std::ceil((armingSeconds * sampleRate) / safeSamplesPerHop));
    consecutiveSilentHops = padDurationHops; 
    padActiveHopsRemaining = 0;

    featureExtractor->prepare(sampleRate, samplesPerHop);
}

void BackgroundMLThread::startProcessing()
{
    isOffline = false;
    startThread(juce::Thread::Priority::normal);
}

void BackgroundMLThread::stopProcessing()
{
    if (isThreadRunning())
        stopThread(4000); // 4 seconds timeout
}

// -----------------------------------------------------------------------------
// REAL-TIME COMMUNICATION
// -----------------------------------------------------------------------------

void BackgroundMLThread::pushAudio(const float* data, int numSamples)
{
    if (audioFifo != nullptr && audioFifo->getFreeSpace() >= numSamples)
        audioFifo->push(data, numSamples);
}

void BackgroundMLThread::notifyDataReady()
{
    mlDataReady.store(true, std::memory_order_release);
}

void BackgroundMLThread::setOfflineMode(bool offline)
{
    isOffline.store(offline, std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// THREAD LOOP
// -----------------------------------------------------------------------------

void BackgroundMLThread::run()
{
    int hopSize = static_cast<int>(dawHopBuffer.size());

    while (!threadShouldExit())
    {
        // Prevent the background thread from draining the FIFO during an offline bounce
        bool offline = isOffline.load(std::memory_order_relaxed);

        if (!offline && (mlDataReady.load(std::memory_order_acquire) || 
           (audioFifo != nullptr && audioFifo->getNumReady() >= hopSize)))
        {
            // Drain the FIFO in hop-sized chunks
            while (audioFifo != nullptr && audioFifo->getNumReady() >= hopSize && !threadShouldExit())
            {
                audioFifo->pop(dawHopBuffer.data(), hopSize);
                processMLHop(dawHopBuffer.data(), *currentParams);
            }
            mlDataReady.store(false, std::memory_order_release);
        }
        else
        {
            juce::Thread::sleep(5); // Rest the CPU
        }
    }
}

// -----------------------------------------------------------------------------
// CORE ML PIPELINE
// -----------------------------------------------------------------------------

void BackgroundMLThread::processNextOfflineHop(const ParameterManager& params)
{
    int hopSize = static_cast<int>(dawHopBuffer.size());
    
    if (audioFifo != nullptr && audioFifo->getNumReady() >= hopSize)
    {
        // Safe to use dawHopBuffer here because the background run() loop 
        // ignores the FIFO when isOffline is true.
        audioFifo->pop(dawHopBuffer.data(), hopSize);
        processMLHop(dawHopBuffer.data(), params);
    }
}

void BackgroundMLThread::processMLHop(const float* hopData, const ParameterManager& params)
{
    int hopSize = static_cast<int>(dawHopBuffer.size());
    const float silenceThreshold = juce::Decibels::decibelsToGain(-50.0f);
    
    // 1. Find Peak Level
    float peakLevel = 0.0f;
    for (int i = 0; i < hopSize; ++i) {
        peakLevel = std::max(peakLevel, std::abs(hopData[i]));
    }

    bool isSilent = (peakLevel < silenceThreshold);
    float rawProb = 0.0f;

    // ------------------------------------------------------------------
    // THE FIX: ALWAYS EXTRACT FEATURES!
    // This ensures the rolling spectrogram buffer is always filling up 
    // with real audio, even while we are bypassing the ML inference.
    // ------------------------------------------------------------------
    auto features = featureExtractor->process(hopData);

    // 2. Transient Pad Logic & Inference
    if (isSilent) 
    {
        consecutiveSilentHops++;
        
        if (padActiveHopsRemaining > 0) {
            padActiveHopsRemaining--;
        }
        // rawProb remains 0.0f
    } 
    else 
    {
        // Check against the 0.3s arming requirement, NOT the 1.0s pad duration!
        if (consecutiveSilentHops >= armingHops) {
            padActiveHopsRemaining = padDurationHops;
        }
        
        consecutiveSilentHops = 0;

        if (padActiveHopsRemaining > 0) 
        {
            // BYPASS INFERENCE ONLY: Force rawProb to 0.0f
            rawProb = 0.0f;
            padActiveHopsRemaining--;
        } 
        else 
        {
            // RUN INFERENCE: The model now receives a fully populated, 
            // natural-looking spectrogram!
            rawProb = inferenceEngine->run(features);
        }
    }

    // 3. Smooth the prediction
    float smoothedProb = pushAndAveragePrediction(rawProb, params.getProbSmoothing());
    
    // 4. Write back to the Probability Ring Buffer
    for (int i = 0; i < hopSize; ++i) 
    {
        int writePos = (mlWriteIndex + i) % probBufferSize;
        probRingBuffer[writePos].store(smoothedProb, std::memory_order_relaxed);
    }
    
    mlWriteIndex += hopSize;
}

float BackgroundMLThread::pushAndAveragePrediction(float rawProb, float smoothMs)
{
    predictionHistory[predictionWriteIndex] = rawProb;
    predictionWriteIndex = (predictionWriteIndex + 1) % maxPredictionFrames;

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

