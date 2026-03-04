// plugin/Source/PluginEditor.h
#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class VocalGateEditor : public juce::AudioProcessorEditor
{
public:
    VocalGateEditor (VocalGateProcessor&);
    ~VocalGateEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    VocalGateProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateEditor)
};

