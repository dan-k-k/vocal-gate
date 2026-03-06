// plugin/Source/PluginEditor.h
#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"
#include <array>

class VocalGateEditor : public juce::AudioProcessorEditor, public juce::Timer
{
public:
    VocalGateEditor (VocalGateProcessor&);
    ~VocalGateEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override; // The 60Hz UI loop

private:
    VocalGateProcessor& audioProcessor;

    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    
    // ADD THIS:
    std::unique_ptr<juce::SliderParameterAttachment> thresholdAttachment;

    // --- Visualizer Data ---
    static constexpr int historySize = 120; // 120 frames = 2 seconds at 60fps
    std::array<float, historySize> probHistory { 0.0f };
    std::array<float, historySize> audioHistory { 0.0f };
    size_t writeIndex = 0; // Use size_t instead // Keeps track of where we are in the circular buffer

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateEditor)
};

