// plugin/Source/PluginEditor.h
#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"
#include <array>

class CustomKnobLookAndFeel : public juce::LookAndFeel_V4
{
public:
    void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                           float sliderPos, const float rotaryStartAngle, 
                           const float rotaryEndAngle, juce::Slider& slider) override
    {
        auto radius = (float) juce::jmin (width / 2, height / 2) - 4.0f;
        auto centreX = (float) x + (float) width  * 0.5f;
        auto centreY = (float) y + (float) height * 0.5f;
        auto rx = centreX - radius;
        auto ry = centreY - radius;
        auto rw = radius * 2.0f;
        auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

        // Knob
        g.setColour (juce::Colour::fromRGB (45, 45, 50));
        g.fillEllipse (rx, ry, rw, rw);
        g.setColour (juce::Colour::fromRGB (20, 20, 22));
        g.drawEllipse (rx, ry, rw, rw, 2.0f);

        // Pointer
        juce::Path p;
        auto pointerLength = radius * 0.75f;
        auto pointerThickness = 4.0f;
        p.addRoundedRectangle (-pointerThickness * 0.5f, -radius + 4.0f, pointerThickness, pointerLength, 2.0f);
        p.applyTransform (juce::AffineTransform::rotation (angle).translated (centreX, centreY));

        g.setColour (slider.findColour(juce::Slider::thumbColourId));
        g.fillPath (p);
    }
};

class VocalGateEditor : public juce::AudioProcessorEditor, public juce::Timer
{
public:
    VocalGateEditor (VocalGateProcessor&);
    ~VocalGateEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    VocalGateProcessor& audioProcessor;
    CustomKnobLookAndFeel customKnobLookAndFeel;
    
    juce::TooltipWindow tooltipWindow;
    juce::Font titleFont; 

    juce::Slider inputGainSlider;
    juce::Label inputGainLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> inputGainAttachment;

    juce::Label inputLevelMeterLabel;
    juce::TextButton matchButton;

    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    
    juce::Slider floorSlider;
    juce::Label floorLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> floorAttachment;

    juce::Slider attackSlider;
    juce::Label attackLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment;

    juce::Slider releaseSlider;
    juce::Label releaseLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> releaseAttachment;

    juce::Slider shiftSlider;
    juce::Label shiftLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> shiftAttachment;

    juce::Slider probSmoothSlider;
    juce::Label probSmoothLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> probSmoothAttachment;
    
    static constexpr int historySize = 120; 
    std::array<float, historySize> inputHistory { 0.0f };
    std::array<float, historySize> outputHistory { 0.0f };
    std::array<float, historySize> probHistory { 0.0f }; 
    size_t writeIndex = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalGateEditor)
};

