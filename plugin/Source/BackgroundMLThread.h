// plugin/Source/BackgroundMLThread.h
#pragma once
#include <juce_core/juce_core.h>
#include "AudioFIFO.h"
#include "ParameterManager.h"

// Forward declarations for the modules we will build next
class FeatureExtractor;
class InferenceEngine;

class BackgroundMLThread : public juce::Thread
{
public:
    BackgroundMLThread();
    ~BackgroundMLThread() override;

    // Lifecycle
    void prepare(double sampleRate, int samplesPerHop);
    void startProcessing();
    void stopProcessing();

    // The thread loop
    void run() override;

    // Real-time safe method for GateDSP to read the probability
    const std::atomic<float>* getProbRingBuffer() const { return probRingBuffer.get(); }
    int getProbBufferSize() const { return probBufferSize; }

    // Real-time safe methods for Processor to push audio
    void pushAudio(const float* data, int numSamples);
    void notifyDataReady();
    
    // Offline rendering bypass
    void processOfflineBlock(const float* data, const ParameterManager& params);

private:
    void processMLHop(const float* hopData, const ParameterManager& params);
    float pushAndAveragePrediction(float rawProb, float smoothMs);

    // Thread synchronization
    std::atomic<bool> mlDataReady { false };
    bool isOffline = false;

    // Sub-modules (Owned by this thread)
    std::unique_ptr<FeatureExtractor> featureExtractor;
    std::unique_ptr<InferenceEngine> inferenceEngine;

    // Buffers
    std::unique_ptr<AudioFIFO> audioFifo;
    std::vector<float> dawHopBuffer;
    
    // Communication back to the audio thread
    int probBufferSize = 0;
    std::unique_ptr<std::atomic<float>[]> probRingBuffer;
    uint64_t mlWriteIndex = 0;

    // Smoothing history
    static constexpr int maxPredictionFrames = 32;
    std::array<float, maxPredictionFrames> predictionHistory { 0.0f };
    int predictionWriteIndex = 0;

    // We store a reference to params safely via a pointer that gets passed in
    const ParameterManager* currentParams = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BackgroundMLThread)
};

