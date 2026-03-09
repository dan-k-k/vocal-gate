// plugin/Source/PluginProcessor.h
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <onnxruntime_cxx_api.h>
#include "AudioFIFO.h"
#include <mutex> 

class VocalGateProcessor : public juce::AudioProcessor, public juce::Thread
{
public:
    VocalGateProcessor();
    ~VocalGateProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void run() override; // Background ML thread

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

    std::atomic<float> inputLevel { 0.0f }; // Processor updates, editor reads
    std::atomic<float> outputLevel { 0.0f };
    std::atomic<float> gateProbability { 0.0f }; 

private:
    // NN constants
    static constexpr int targetSampleRate = 16000;
    static constexpr int modelInputSamples = 16000; // 1 second of audio
    static constexpr int hopSizeSamples = 4000;     // 250ms evaluation hop

    // Buffering, threading
    std::unique_ptr<AudioFIFO> audioFifo;
    float pushAndAveragePrediction(float rawProb);
    static constexpr int maxPredictionFrames = 32; // > 24 frames needed for 1200ms
    std::array<float, maxPredictionFrames> predictionHistory { 0.0f };
    int predictionWriteIndex = 0;
        
    float currentGainEnvelope = 1.0f;       

    // Resampling
    double dawSampleRate = 44100.0;

    // DSP, ML helpers
    void computeLogMels(const std::vector<float>& audio16k);
    void runONNXModel();

    // ONNX runtime
    Ort::Env onnxEnv{ORT_LOGGING_LEVEL_WARNING, "VocalGate"};
    std::unique_ptr<Ort::Session> onnxSession;
    
    // Plugin memory
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    std::atomic<float>* thresholdParam = nullptr;
    std::atomic<float>* floorParam     = nullptr;
    std::atomic<float>* attackParam    = nullptr;
    std::atomic<float>* releaseParam   = nullptr;
    std::atomic<float>* shiftParam     = nullptr;
    std::atomic<float>* probSmoothingParam = nullptr;

    // Delay / lookahead
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLine { 96000 };
    int lookaheadSamples = 0;
    int probBufferSize = 0;
    std::unique_ptr<std::atomic<float>[]> probRingBuffer;

    uint64_t mlWriteIndex = 0; // Exact sample being fed to ML

    std::atomic<uint64_t> audioReadIndex { 0 }; // Exact sample being processed in audio thread

    // Pre-allocated memory
    std::vector<float> dawHopBuffer;
    std::vector<float> resampledHopBuffer;
    std::vector<float> rolling16kBuffer;
    std::vector<float> logMelFeatures;

    std::vector<float> timeDomain;
    std::vector<float> powerSpec;
    std::vector<float> melEnergies;

    // Pre-allocated output and tensor shapes
    std::array<float, 1> outputLogitData { 0.0f }; 
    static constexpr int64_t inputShape[] = {1, 1, 40, 61};
    static constexpr int64_t outputShape[] = {1, 1}; 

    // FFT and resampler
    juce::dsp::FFT forwardFFT { 9 }; 
    juce::LagrangeInterpolator resampler;

    // Thread sync
    std::atomic<bool> mlDataReady { false };
    int dawSamplesPerHop = 0;

    // Offline buffer
    void processMLHop(const float* hopData);
    std::mutex mlMutex; 
    std::vector<float> offlineHopBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateProcessor)
};

