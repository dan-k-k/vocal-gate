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

void BackgroundMLThread::prepare(double sampleRate, int samplesPerHop, const ParameterManager& params) // <-- Add the 3rd argument
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

// -----------------------------------------------------------------------------
// THREAD LOOP
// -----------------------------------------------------------------------------

void BackgroundMLThread::run()
{
    int hopSize = static_cast<int>(dawHopBuffer.size());

    while (!threadShouldExit())
    {
        // Wait until there is enough data in the FIFO, or we are signaled
        if (mlDataReady.load(std::memory_order_acquire) || 
           (audioFifo != nullptr && audioFifo->getNumReady() >= hopSize))
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

void BackgroundMLThread::processOfflineBlock(const float* data, const ParameterManager& params)
{
    isOffline = true;
    processMLHop(data, params);
}

void BackgroundMLThread::processMLHop(const float* hopData, const ParameterManager& params)
{
    int hopSize = static_cast<int>(dawHopBuffer.size());
    const float silenceThreshold = juce::Decibels::decibelsToGain(-50.0f);
    
    // 1. Simple Silence Detection (Bypass ML if it's completely quiet)
    float peakLevel = 0.0f;
    for (int i = 0; i < hopSize; ++i) {
        peakLevel = std::max(peakLevel, std::abs(hopData[i]));
    }

    float rawProb = 0.0f;

    if (peakLevel >= silenceThreshold) 
    {
        // 2. Extract Features (Audio -> Mel Spectrogram)
        auto features = featureExtractor->process(hopData);
        
        // 3. Run Inference (Mel Spectrogram -> Probability)
        rawProb = inferenceEngine->run(features);
    }

    // 4. Smooth the prediction
    float smoothedProb = pushAndAveragePrediction(rawProb, params.getProbSmoothing());
    
    // 5. Write back to the Probability Ring Buffer
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

