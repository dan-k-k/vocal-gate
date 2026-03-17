// plugin/Source/InferenceEngine.cpp
#include "InferenceEngine.h"
#include <BinaryData.h>
#include <cmath>

#if JUCE_WINDOWS
#define NOMINMAX
#include <windows.h>
// We need a dummy address to find the current DLL module in memory
static void dummyFunctionForModuleHandle() {}
#endif

InferenceEngine::InferenceEngine()
{
    loadWindowsDLL();

    try {
        onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VocalGate");
        memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1); // Keep it strictly single-threaded to avoid audio dropouts
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model directly from JUCE BinaryData
        onnxSession = std::make_unique<Ort::Session>(
            *onnxEnv, 
            BinaryData::vocalgate_int8_onnx, 
            BinaryData::vocalgate_int8_onnxSize, 
            sessionOptions
        );
            
        Ort::AllocatorWithDefaultOptions allocator;
        auto expectedInput = onnxSession->GetInputNameAllocated(0, allocator);
        auto expectedOutput = onnxSession->GetOutputNameAllocated(0, allocator);
        
        juce::Logger::writeToLog("ONNX loaded successfully."); 
    } 
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog("ONNX load crash: " + juce::String(e.what()));
    }
}

float InferenceEngine::run(std::vector<float>& logMelFeatures)
{
    if (onnxSession == nullptr) return 0.0f;

    try {
        // 1. Wrap the input memory block into an ONNX tensor
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo, 
            logMelFeatures.data(), 
            logMelFeatures.size(), 
            inputShape, 
            4
        );

        // 2. Wrap the output memory block into an ONNX tensor
        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            outputLogitData.data(),
            outputLogitData.size(),
            outputShape,
            2
        );

        const char* inputNames[] = {"input_log_mel"};
        const char* outputNames[] = {"gate_logit"};

        // 3. Run Inference
        onnxSession->Run(
            Ort::RunOptions{nullptr}, 
            inputNames, 
            &inputTensor, 
            1, 
            outputNames, 
            &outputTensor,
            1
        );

        // 4. Apply Sigmoid activation to the raw logit to get a 0.0 - 1.0 probability
        return 1.0f / (1.0f + std::exp(-outputLogitData[0]));

    } catch (const Ort::Exception& e) {
        juce::Logger::writeToLog("ONNX RUN ERROR: " + juce::String(e.what()));
        return 0.0f; // Fail gracefully to silence/open gate
    }
}

void InferenceEngine::loadWindowsDLL()
{
#if JUCE_WINDOWS
    HMODULE hModule = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)&dummyFunctionForModuleHandle, &hModule);

    if (hModule != nullptr)
    {
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(hModule, path, MAX_PATH);
        auto pluginDllFile = juce::File(juce::String(path));
        auto onnxDllPath = pluginDllFile.getParentDirectory().getChildFile("onnxruntime.dll").getFullPathName();
        
        LoadLibraryW(onnxDllPath.toWideCharPointer());
    }
#endif
}

