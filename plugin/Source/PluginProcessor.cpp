// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "DSPConstants.h" // <--- ADD THIS

VocalGateProcessor::VocalGateProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
       juce::Thread ("ONNX_ML_Thread") 
#endif
{
    // Initialize ONNX Runtime options
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Look for the model on the Mac Desktop
    juce::File modelFile = juce::File::getSpecialLocation(juce::File::userDesktopDirectory)
                             .getChildFile("vocalgate_int8.onnx");

    if (modelFile.existsAsFile())
    {
        // Load the model into memory
        onnxSession = std::make_unique<Ort::Session>(
            onnxEnv, 
            modelFile.getFullPathName().toStdString().c_str(), 
            sessionOptions
        );
    }
    else
    {
        // If it can't find the file, we will print a warning to the console
        juce::Logger::writeToLog("🚨 ERROR: Could not find vocalgate_int8.onnx on the Desktop!");
    }

    startThread (juce::Thread::Priority::lowest);
}

VocalGateProcessor::~VocalGateProcessor()
{
    // Crucial: Stop the thread safely before the plugin is destroyed
    stopThread (4000); 
}

void VocalGateProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    dawSampleRate = sampleRate;

    // Make the FIFO large enough to hold 2 seconds of audio at the DAW's sample rate
    // This gives our background thread plenty of breathing room.
    audioFifo = std::make_unique<AudioFIFO> (static_cast<int>(dawSampleRate * 2.0));
    
    // Reset our smoothing envelope when playback starts
    currentGainEnvelope = 1.0f;
}

void VocalGateProcessor::releaseResources()
{
    // Free up memory when Ableton stops the transport
    audioFifo.reset();
}

void VocalGateProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    int numSamples = buffer.getNumSamples();
    
    // 1. PUSH TO FIFO
    // We grab just the left channel (Channel 0) to feed the neural network. 
    // It's a vocal gate, so mono is standard.
    const float* leftChannelIn = buffer.getReadPointer(0);
    if (audioFifo != nullptr && audioFifo->getFreeSpace() >= numSamples)
    {
        audioFifo->push (leftChannelIn, numSamples);
    }

    // 2. READ THE ML PREDICTION
    // This read is atomic, meaning it will never block the audio thread, 
    // even if the ML thread is writing to it at the exact same microsecond.
    float currentProb = gateProbability.load();

    // Determine target gain (ducking)
    // If the model is > 50% sure it's a cough/breath, target gain is 0.0 (mute). Otherwise 1.0.
    float targetGain = (currentProb > 0.5f) ? 0.0f : 1.0f;

    // 3. APPLY SMOOTHING (Attack / Release)
    // We use a simple one-pole lowpass filter to smooth the volume changes.
    // (In a production plugin, you'd expose these to Ableton as user-tweakable knobs).
    float attackCoef  = 0.99f;   // Fast close (snaps shut on coughs)
    float releaseCoef = 0.999f;  // Slower open (fades back in smoothly)

    float* outL = buffer.getWritePointer(0);
    float* outR = (totalNumInputChannels > 1) ? buffer.getWritePointer(1) : nullptr;

    for (int i = 0; i < numSamples; ++i)
    {
        // Smooth the gain trajectory
        if (targetGain < currentGainEnvelope)
            currentGainEnvelope = attackCoef * currentGainEnvelope + (1.0f - attackCoef) * targetGain; // Attack
        else
            currentGainEnvelope = releaseCoef * currentGainEnvelope + (1.0f - releaseCoef) * targetGain; // Release

        // Apply the smoothed gain to the actual audio samples
        outL[i] *= currentGainEnvelope;
        if (outR != nullptr)
            outR[i] *= currentGainEnvelope;
    }
}

void VocalGateProcessor::computeMFCCs(const std::vector<float>& audio16k, std::vector<float>& mfccOut)
{
    int n_fft = 512;
    int hop_length = 256;
    int num_frames = 61;     // (16000 - 512) / 256 + 1
    int num_freq_bins = 257; // n_fft / 2 + 1
    int num_mels = 40;

    juce::dsp::FFT forwardFFT(9); // 2^9 = 512
    std::vector<float> timeDomain(n_fft * 2, 0.0f); 

    for (int frame = 0; frame < num_frames; ++frame)
    {
        int start_sample = frame * hop_length;

        // 1. Copy audio chunk & apply Hann window
        std::memset(timeDomain.data(), 0, timeDomain.size() * sizeof(float));
        for (int i = 0; i < n_fft; ++i) {
            timeDomain[i] = audio16k[start_sample + i] * DSPConstants::hannWindow512[i];
        }

        // 2. Perform FFT (JUCE replaces the input array with the output magnitudes)
        forwardFFT.performFrequencyOnlyForwardTransform(timeDomain.data());
        
        // PyTorch MelSpectrogram uses Power Spectrogram (magnitude squared) by default!
        std::vector<float> powerSpec(num_freq_bins, 0.0f);
        for (int i = 0; i < num_freq_bins; ++i) {
            powerSpec[i] = timeDomain[i] * timeDomain[i];
        }

        // 3. Mel Filterbank Multiplication (PowerSpec @ MelFB)
        std::vector<float> melEnergies(num_mels, 0.0f);
        for (int m = 0; m < num_mels; ++m) 
        {
            float sum = 0.0f;
            for (int f = 0; f < num_freq_bins; ++f) {
                sum += powerSpec[f] * DSPConstants::melFilterBank[f][m];
            }
            
            // 4. Amplitude to DB (PyTorch default: 10 * log10(max(x, 1e-10)))
            melEnergies[m] = 10.0f * std::log10(std::max(sum, 1e-10f));
        }

        // 5. Discrete Cosine Transform (MelEnergies @ DCT)
        for (int mfcc_bin = 0; mfcc_bin < 40; ++mfcc_bin) 
        {
            float mfcc_val = 0.0f;
            for (int m = 0; m < num_mels; ++m) {
                mfcc_val += melEnergies[m] * DSPConstants::dctMatrix[m][mfcc_bin];
            }
            
            // 6. Store in flat 1D array representing [40, 61]
            mfccOut[mfcc_bin * num_frames + frame] = mfcc_val;
        }
    }
}

void VocalGateProcessor::runONNXModel(const std::vector<float>& mfccFeatures)
{
    // 1. Define ONNX Tensor Shapes
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {1, 1, 40, 61};
    
    // Create the input tensor pointing to our MFCC array
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, 
        const_cast<float*>(mfccFeatures.data()), 
        mfccFeatures.size(), 
        inputShape.data(), 
        inputShape.size()
    );

    const char* inputNames[] = {"input_mfcc"};
    const char* outputNames[] = {"gate_logit"};

    // 2. Run the Model
    auto outputTensors = onnxSession->Run(
        Ort::RunOptions{nullptr}, 
        inputNames, 
        &inputTensor, 
        1, 
        outputNames, 
        1
    );

    // 3. Extract the Logit and Apply Sigmoid
    float* outLogit = outputTensors.front().GetTensorMutableData<float>();
    
    // Manual sigmoid since we used BCEWithLogitsLoss
    float prob = 1.0f / (1.0f + std::exp(-outLogit[0]));

    // 4. Update the Atomic Float for the Audio Thread!
    gateProbability.store(prob);
}

// --- The Background ML Thread ---
void VocalGateProcessor::run()
// (Note: In a production plugin, you usually export PyTorch's Mel filterbank matrix and DCT matrix as a C++ header file filled with constants so the math matches 1-to-1. Here is the structure of how JUCE handles the DSP steps).
{
    // --- 1. Background Thread Setup ---
    juce::LagrangeInterpolator resampler;
    std::vector<float> dawHopBuffer;
    std::vector<float> resampledHopBuffer;
    
    // We need 1 second of audio at 16kHz (16,000 samples)
    std::vector<float> rolling16kBuffer (16000, 0.0f); 

    // Calculate how many DAW samples equal our 250ms (4000 sample) target hop at 16kHz
    int dawSamplesPerHop = static_cast<int>(dawSampleRate * 0.25);
    dawHopBuffer.resize(dawSamplesPerHop, 0.0f);
    resampledHopBuffer.resize(4000, 0.0f);

    while (! threadShouldExit())
    {
        // 2. Wait until we have enough new audio from Ableton
        if (audioFifo != nullptr && audioFifo->getNumReady() >= dawSamplesPerHop)
        {
            // Pop the new audio block
            audioFifo->pop(dawHopBuffer.data(), dawSamplesPerHop);

            // 3. Resample down to 16kHz
            const float* inData = dawHopBuffer.data();
            float* outData = resampledHopBuffer.data();
            resampler.process(dawSampleRate / 16000.0, inData, outData, 4000);

            // 4. Shift our 1-second rolling buffer back by 250ms, and append the new audio
            // This is standard FIFO array shifting
            std::memmove(rolling16kBuffer.data(), 
                         rolling16kBuffer.data() + 4000, 
                         12000 * sizeof(float));
            std::memcpy(rolling16kBuffer.data() + 12000, 
                        resampledHopBuffer.data(), 
                        4000 * sizeof(float));

            // 5. Extract MFCCs (The heavy lifting)
            // Target output shape: [1, 1, 40, 61] (Batch, Channel, Mels, Frames)
            std::vector<float> mfccFeatures(40 * 61, 0.0f);
            computeMFCCs(rolling16kBuffer, mfccFeatures);

            // 6. Run ONNX Inference
            if (onnxSession != nullptr)
            {
                runONNXModel(mfccFeatures);
            }
        }
        else
        {
            // Sleep for 10ms so we don't fry the user's CPU while waiting for audio
            wait(10);
        }
    }
}

