// plugin/Source/AudioFIFO.h
#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

class AudioFIFO
{
public:
    // Init fifo with specific cap
    AudioFIFO(int capacity) : abstractFifo(capacity), buffer(1, capacity)
    {
        buffer.clear();
    }

    // Called by the audio thread
    void push(const float* data, int numSamples)
    {
        int start1, size1, start2, size2;
        abstractFifo.prepareToWrite(numSamples, start1, size1, start2, size2);
        
        if (size1 > 0) buffer.copyFrom(0, start1, data, size1);
        if (size2 > 0) buffer.copyFrom(0, start2, data + size1, size2);
        
        abstractFifo.finishedWrite(size1 + size2);
    }

    // Called by the background thread (ML)
    void pop(float* dest, int numSamples)
    {
        int start1, size1, start2, size2;
        abstractFifo.prepareToRead(numSamples, start1, size1, start2, size2);
        
        if (size1 > 0) juce::FloatVectorOperations::copy(dest, buffer.getReadPointer(0, start1), size1);
        if (size2 > 0) juce::FloatVectorOperations::copy(dest + size1, buffer.getReadPointer(0, start2), size2);
        
        abstractFifo.finishedRead(size1 + size2);
    }

    int getNumReady() const { return abstractFifo.getNumReady(); }
    int getFreeSpace() const { return abstractFifo.getFreeSpace(); }

private:
    juce::AbstractFifo abstractFifo;
    juce::AudioBuffer<float> buffer;
};

