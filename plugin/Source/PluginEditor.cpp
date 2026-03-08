// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalGateEditor::VocalGateEditor (VocalGateProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (600, 300); 

    // Updated lambda: takes a const juce::String& paramID instead of an atomic pointer
    auto setupKnob = [this](juce::Slider& slider, juce::Label& label, const juce::String& text, 
                            const juce::String& suffix, const juce::String& paramID, 
                            std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>& attachment,
                            juce::Colour pointerColor) 
    {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 20);
        slider.setTextValueSuffix(suffix);
        
        // Apply our custom LookAndFeel and color
        slider.setLookAndFeel(&customKnobLookAndFeel);
        slider.setColour(juce::Slider::thumbColourId, pointerColor);
        slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);

        addAndMakeVisible(slider);
        
        // Use the APVTS attachment, passing the APVTS, the parameter ID string, and the slider
        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.apvts, paramID, slider);

        label.setText(text, juce::dontSendNotification);
        label.attachToComponent(&slider, false);
        label.setJustificationType(juce::Justification::centred);
        label.setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(label);
    };

    // --- Define your theme colors ---
    juce::Colour detectionColor = juce::Colours::darkorange;
    juce::Colour envelopeColor = juce::Colour::fromRGB(40, 210, 180);

    // --- Apply them to the knobs using the Parameter IDs string ---
    setupKnob(thresholdSlider, thresholdLabel, "Threshold", "", "threshold", thresholdAttachment, detectionColor);
    setupKnob(floorSlider, floorLabel, "Floor", " dB", "floor", floorAttachment, detectionColor);
    setupKnob(probSmoothSlider, probSmoothLabel, "P Smooth", " ms", "probsmoothing", probSmoothAttachment, detectionColor);

    setupKnob(attackSlider, attackLabel, "Attack", " ms", "attack", attackAttachment, envelopeColor);
    setupKnob(releaseSlider, releaseLabel, "Release", " ms", "release", releaseAttachment, envelopeColor);
    setupKnob(shiftSlider, shiftLabel, "Shift", " ms", "shift", shiftAttachment, envelopeColor);

    startTimerHz(60); 
}

VocalGateEditor::~VocalGateEditor() 
{
    // CRITICAL: You must remove the custom LookAndFeel before the editor is destroyed
    thresholdSlider.setLookAndFeel(nullptr);
    floorSlider.setLookAndFeel(nullptr);
    probSmoothSlider.setLookAndFeel(nullptr); // <--- ADD THIS
    attackSlider.setLookAndFeel(nullptr);
    releaseSlider.setLookAndFeel(nullptr);
    shiftSlider.setLookAndFeel(nullptr);
}

void VocalGateEditor::timerCallback()
{
    // Read all three values
    inputHistory[writeIndex]  = audioProcessor.inputLevel.load();
    outputHistory[writeIndex] = audioProcessor.outputLevel.load();
    probHistory[writeIndex]   = audioProcessor.gateProbability.load(); // Read the ML brain

    writeIndex = (writeIndex + 1) % historySize;
    repaint(); 
}

void VocalGateEditor::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    g.fillAll (juce::Colour::fromRGB (20, 20, 22));

    auto graphArea = bounds.withTrimmedBottom(110).withTrimmedTop(15).withTrimmedRight(20).withTrimmedLeft(20);
    auto audioArea = graphArea.removeFromTop(graphArea.getHeight() / 2);
    auto probArea = graphArea; 

    g.setColour(juce::Colour::fromRGB(30, 30, 34));
    g.fillRect(audioArea);
    g.setColour(juce::Colour::fromRGB(25, 25, 28)); 
    g.fillRect(probArea);

    juce::Path inputPath, outputPath, probPath;

    float width = audioArea.getWidth();
    float xStep = width / (historySize - 1);

    inputPath.startNewSubPath(audioArea.getX(), audioArea.getBottom());
    outputPath.startNewSubPath(audioArea.getX(), audioArea.getBottom());

    // Look how simple this loop is now! Just a 1:1 mapping.
    for (int i = 0; i < historySize; ++i)
    {
        // One single read index for everything
        size_t readIndex = (writeIndex + static_cast<size_t>(i)) % historySize;

        float x = audioArea.getX() + (i * xStep);
        
        float inY = audioArea.getBottom() - (inputHistory[readIndex] * audioArea.getHeight());
        float outY = audioArea.getBottom() - (outputHistory[readIndex] * audioArea.getHeight());
        float pY = probArea.getBottom() - (probHistory[readIndex] * probArea.getHeight());

        inputPath.lineTo(x, inY);
        outputPath.lineTo(x, outY);

        if (i == 0) probPath.startNewSubPath(x, pY);
        else        probPath.lineTo(x, pY);
    }

    inputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    inputPath.closeSubPath();
    outputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    outputPath.closeSubPath();

    g.setColour(juce::Colour::fromRGB(80, 85, 95).withAlpha(0.4f));
    g.fillPath(inputPath);

    g.setColour(juce::Colour::fromRGB(40, 210, 180).withAlpha(0.85f));
    g.fillPath(outputPath);

    g.setColour (juce::Colours::darkorange);
    g.strokePath (probPath, juce::PathStrokeType(2.0f));

    float threshVal = static_cast<float>(thresholdSlider.getValue());
    float threshY = probArea.getBottom() - (threshVal * probArea.getHeight());
    
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    float dashLengths[2] = { 4.0f, 4.0f };
    g.drawDashedLine(juce::Line<float>(probArea.getX(), threshY, probArea.getRight(), threshY), dashLengths, 2, 1.0f);
}

void VocalGateEditor::resized()
{
    auto bounds = getLocalBounds();
    auto bottomArea = bounds.removeFromBottom(110); // Area for knobs
    
    int squeezePixels = 40; 
    bottomArea = bottomArea.withTrimmedLeft(squeezePixels).withTrimmedRight(squeezePixels);
    
    // CHANGE THIS: Divide by 6 columns instead of 5
    int knobWidth = bottomArea.getWidth() / 6; 
    int shiftDown = 8;

    // Lay them out sequentially from left to right
    thresholdSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    floorSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    probSmoothSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    
    attackSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    releaseSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    shiftSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
}

