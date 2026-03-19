// plugin/Source/BackgroundMLThread.cpp
#include "BackgroundMLThread.h"
#include "FeatureExtractor.h"
#include "InferenceEngine.h"

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

    // Setup audio FIFO buffer (2sec)
    audioFifo = std::make_unique<AudioFIFO>(static_cast<int>(sampleRate * 2.0));
    dawHopBuffer.assign(samplesPerHop, 0.0f);

    // Setup prob ring buffer
    probBufferSize = static_cast<int>(sampleRate * 2.0);
    probRingBuffer = std::make_unique<std::atomic<float>[]>(probBufferSize);
    for (int i = 0; i < probBufferSize; ++i) {
        probRingBuffer[i].store(0.0f, std::memory_order_relaxed);
    }

    // Reset state
    mlWriteIndex = 0;
    predictionHistory.fill(0.0f);
    predictionWriteIndex = 0;
    mlDataReady.store(false);

    double padDurationSeconds = 1.0; // Fills the ends of the spectrogram
    double armingSeconds = 0.3;      // Patience needed to re-arm
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

// Real-time communication

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

// Thread loop

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
            juce::Thread::sleep(5); // Rest 
        }
    }
}

// Core ML pipeline

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
    
    // Find peak
    float peakLevel = 0.0f;
    for (int i = 0; i < hopSize; ++i) {
        peakLevel = std::max(peakLevel, std::abs(hopData[i]));
    }

    bool isSilent = (peakLevel < silenceThreshold);
    float rawProb = 0.0f;

    // Always extract features to fill the spectrofram buffer completely when bypassing ML inference
    auto features = featureExtractor->process(hopData);

    // Transient/silence padding logic and inference
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
        // If silence lasts longer than number of armingHops (0.3sec)
        if (consecutiveSilentHops >= armingHops) {
            padActiveHopsRemaining = padDurationHops;
        }
        
        consecutiveSilentHops = 0;

        if (padActiveHopsRemaining > 0) 
        {
            // Bypass inference and set rawProb to 0.0f
            rawProb = 0.0f;
            padActiveHopsRemaining--;
        } 
        else 
        {
            // Run inference
            rawProb = inferenceEngine->run(features);
        }
    }

    float smoothedProb = pushAndAveragePrediction(rawProb, params.getProbSmoothing());
    
    // Write back to the prob ring buffer
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

