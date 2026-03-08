// plugin/Source/PluginProcessor.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <onnxruntime_cxx_api.h>
#include "AudioFIFO.h"
#include <mutex> // Add this at the top of your header

class VocalGateProcessor : public juce::AudioProcessor, public juce::Thread
{
public:
    VocalGateProcessor();
    ~VocalGateProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // --- Threading ---
    void run() override; // This is the Background ML Thread

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Vocal Gate"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    bool isModelLoaded() const { return onnxSession != nullptr; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    // The Processor updates this, the Editor reads it for the visualizer
    std::atomic<float> inputLevel { 0.0f };
    std::atomic<float> outputLevel { 0.0f };
    std::atomic<float> gateProbability { 0.0f }; 

private:
    // --- Neural Network Constants ---
    static constexpr int targetSampleRate = 16000;
    static constexpr int modelInputSamples = 16000; // 1 second of audio
    static constexpr int hopSizeSamples = 4000;     // 250ms evaluation hop

    // --- Buffering & Threading ---
    std::unique_ptr<AudioFIFO> audioFifo;
    void pushAndAveragePrediction(float rawProb);
    static constexpr int maxPredictionFrames = 32; // > 24 frames needed for 1200ms
    std::array<float, maxPredictionFrames> predictionHistory { 0.0f };
    int predictionWriteIndex = 0;
        
    // Smooths the gate movement in the audio thread (Attack/Release)
    float currentGainEnvelope = 1.0f;       

    // --- Resampling ---
    // DAW might run at 44.1k or 48k, but ONNX *needs* 16k.
    double dawSampleRate = 44100.0;

    // --- DSP & ML Helpers ---
    void computeLogMels(const std::vector<float>& audio16k);
    void runONNXModel();

    // --- ONNX Runtime ---
    Ort::Env onnxEnv{ORT_LOGGING_LEVEL_WARNING, "VocalGate"};
    std::unique_ptr<Ort::Session> onnxSession;
    
    // --- Plugin Memory ---
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    std::atomic<float>* thresholdParam = nullptr;
    std::atomic<float>* floorParam     = nullptr;
    std::atomic<float>* attackParam    = nullptr;
    std::atomic<float>* releaseParam   = nullptr;
    std::atomic<float>* shiftParam     = nullptr;
    std::atomic<float>* probSmoothingParam = nullptr;

    // --- Delay / Lookahead ---
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLine { 96000 };
    int lookaheadSamples = 0;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedDelay;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedThreshold;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedFloorDB;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedAttack;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedRelease;

    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> smoothedProbability;

    //// --- Pre-allocated Memory (Avoids Heap Thrashing) ---
    std::vector<float> dawHopBuffer;
    std::vector<float> resampledHopBuffer;
    std::vector<float> rolling16kBuffer;
    std::vector<float> logMelFeatures;

    std::vector<float> timeDomain;
    std::vector<float> powerSpec;
    std::vector<float> melEnergies;

    // ADD THESE: Pre-allocated output buffer and static tensor shapes
    std::array<float, 1> outputLogitData { 0.0f }; 
    static constexpr int64_t inputShape[] = {1, 1, 40, 61};
    static constexpr int64_t outputShape[] = {1, 1}; // Note: Change to {1} if your model expects a 1D tensor

    // Move FFT and Resampler here so they only initialize once
    juce::dsp::FFT forwardFFT { 9 }; 
    juce::LagrangeInterpolator resampler;

    // --- Thread Synchronization ---
    juce::WaitableEvent mlTriggerEvent;
    int dawSamplesPerHop = 0;

    // --- Offline Buffer ---
    void processMLHop(const float* hopData);
    std::mutex mlMutex; 
    std::vector<float> offlineHopBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateProcessor)
};

