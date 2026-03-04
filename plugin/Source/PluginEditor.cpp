// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalGateEditor::VocalGateEditor (VocalGateProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (400, 300);
}

VocalGateEditor::~VocalGateEditor() {}

void VocalGateEditor::paint (juce::Graphics& g)
{
    // Dark grey background
    g.fillAll (juce::Colour::fromRGB (30, 30, 30));

    g.setColour (juce::Colours::white);
    g.setFont (20.0f);
    g.drawFittedText ("Vocal Gate (ONNX Ready)", getLocalBounds(), juce::Justification::centred, 1);
}

void VocalGateEditor::resized() {}

