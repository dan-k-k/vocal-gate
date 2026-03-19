// plugin/Source/InferenceEngine.h
#pragma once
#include <juce_core/juce_core.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>

class InferenceEngine
{
public:
    InferenceEngine();
    ~InferenceEngine() = default;

    // Returns a gate probability [0.0, 1.0] from the spectrogram
    float run(std::vector<float>& logMelFeatures);

    bool isModelLoaded() const { return onnxSession != nullptr; }

private:
    void loadWindowsDLL();

    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> onnxEnv;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;
    std::unique_ptr<Ort::Session> onnxSession;

    // Output and tensor shapes
    std::array<float, 1> outputLogitData { 0.0f }; 
    
    // 1 Batch, 1 Channel, 40 Mels, 61 Frames
    static constexpr int64_t inputShape[] = {1, 1, 40, 61};
    static constexpr int64_t outputShape[] = {1, 1}; 

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (InferenceEngine)
};

