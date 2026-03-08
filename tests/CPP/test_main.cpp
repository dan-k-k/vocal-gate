// tests/CPP/test_main.cpp
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

// --- ADD THIS: Include your processor header ---
#include "../../plugin/Source/PluginProcessor.h" 

// ---------------------------------------------------------
// 1. Define your FIFO Test
// ---------------------------------------------------------
class FifoTest : public juce::UnitTest {
public:
    FifoTest() : juce::UnitTest ("FIFO Wraparound Test") {}
    void runTest() override {
        beginTest ("Circular buffer wraparound logic");
        juce::AudioBuffer<float> testFifo (2, 512);
        int writeIdx = 510;
        for (int i = 0; i < 4; ++i) {
            testFifo.setSample(0, writeIdx, 1.0f);
            writeIdx = (writeIdx + 1) % testFifo.getNumSamples();
        }
        expectEquals (writeIdx, 2);
    }
};

// ---------------------------------------------------------
// 2. Define your Audio Buffer Test
// ---------------------------------------------------------
class AudioFifoTest : public juce::UnitTest {
public:
    AudioFifoTest() : juce::UnitTest ("Audio FIFO Wraparound and Hop Logic Test") {}
    void runTest() override {
        beginTest ("Circular buffer wraparound logic for 256 hop size");
        const int bufferSize = 16384; 
        juce::AudioBuffer<float> ringBuffer (1, bufferSize);
        ringBuffer.clear();

        int writeIdx = bufferSize - 100; 
        const int hopSize = 256;         
        
        int samplesWritten = 0;
        for (int i = 0; i < hopSize; ++i) {
            ringBuffer.setSample(0, writeIdx, 0.5f); 
            writeIdx = (writeIdx + 1) % bufferSize;
            samplesWritten++;
        }

        expectEquals (writeIdx, 156);
        expectEquals (samplesWritten, 256);
        expectEquals (ringBuffer.getSample(0, 0), 0.5f);
        expectEquals (ringBuffer.getSample(0, 155), 0.5f);
    }
};

// ---------------------------------------------------------
// 3. Define the Model Test (UPDATED)
// ---------------------------------------------------------
class ModelLoadingTest : public juce::UnitTest {
public:
    ModelLoadingTest() : juce::UnitTest ("ONNX Model State Test") {}
    void runTest() override {
        beginTest ("Processor successfully loads baked-in ONNX model");
        VocalGateProcessor processor;
        
        // This must be TRUE now, since it loads from memory!
        expect (processor.isModelLoaded()); 
    }
};

// ---------------------------------------------------------
// 4. Instantiate them statically so JUCE finds them
// ---------------------------------------------------------
static FifoTest fifoTest; 
static AudioFifoTest audioFifoTest; 
static ModelLoadingTest modelLoadingTest; // <-- Added this

// ---------------------------------------------------------
// 5. ONE Main function to rule them all
// ---------------------------------------------------------
int main (int argc, char* argv[]) {
    juce::UnitTestRunner runner;
    runner.runAllTests();

    for (int i = 0; i < runner.getNumResults(); ++i) {
        if (runner.getResult(i)->failures > 0)
            return 1; // Return non-zero to fail CI pipeline
    }
    
    return 0; 
}

