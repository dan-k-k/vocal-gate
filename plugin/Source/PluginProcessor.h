// plugin/Source/PluginProcessor.h
#pragma once
#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include "AudioFIFO.h"
#include <juce_dsp/juce_dsp.h> // Ensure this is included at the top

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

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

    // The Editor writes to this, the Processor reads from it
    juce::AudioParameterFloat* thresholdParam;
    
    // The Processor updates this, the Editor reads it for the visualizer
    std::atomic<float> latestAudioLevel { 0.0f };

        // We use atomic so the background thread can update it and the audio thread 
    // can read it simultaneously without data races.
    std::atomic<float> gateProbability { 0.0f }; 

private:
    // --- Neural Network Constants ---
    static constexpr int targetSampleRate = 16000;
    static constexpr int modelInputSamples = 16000; // 1 second of audio
    static constexpr int hopSizeSamples = 4000;     // 250ms evaluation hop

    // --- Buffering & Threading ---
    std::unique_ptr<AudioFIFO> audioFifo;
        
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

    // --- Delay / Lookahead ---
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::None> delayLine { 48000 * 2 }; // Max 2 seconds of delay
    int lookaheadSamples = 0;

    // --- Pre-allocated Memory (Avoids Heap Thrashing) ---
    std::vector<float> dawHopBuffer;
    std::vector<float> resampledHopBuffer;
    std::vector<float> rolling16kBuffer;
    std::vector<float> logMelFeatures;

    std::vector<float> timeDomain;
    std::vector<float> powerSpec;
    std::vector<float> melEnergies;

    // Move FFT and Resampler here so they only initialize once
    juce::dsp::FFT forwardFFT { 9 }; 
    juce::LagrangeInterpolator resampler;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateProcessor)
};

