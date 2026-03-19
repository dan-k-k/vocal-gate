// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <BinaryData.h> 

VocalGateEditor::VocalGateEditor (VocalGateProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (600, 300); 

    auto typeface = juce::Typeface::createSystemTypefaceFor (BinaryData::FredokaOne_ttf, BinaryData::FredokaOne_ttfSize);
    titleFont = juce::Font (typeface);
    titleFont.setHeight (28.0f); 

    auto setupKnob = [this](juce::Slider& slider, juce::Label& label, const juce::String& text, 
                            const juce::String& suffix, const juce::String& paramID, 
                            std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>& attachment,
                            juce::Colour pointerColor) 
    {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 20);
        slider.setTextValueSuffix(suffix);
        
        slider.setLookAndFeel(&customKnobLookAndFeel);
        slider.setColour(juce::Slider::thumbColourId, pointerColor);
        slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);

        addAndMakeVisible(slider);
        
        // APVTS attachment
        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.parameterManager.apvts, paramID, slider);

        label.setText(text, juce::dontSendNotification);
        label.attachToComponent(&slider, false);
        label.setJustificationType(juce::Justification::centred);
        label.setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(label);
    };

    juce::Colour utilityColor   = juce::Colour::fromRGB(220, 80, 150); // Magenta for Global IO
    juce::Colour detectionColor = juce::Colours::darkorange;           // Orange for ML
    juce::Colour envelopeColor  = juce::Colour::fromRGB(40, 210, 180); // Blue for DSP

    // Volume meter and Match UI
    inputLevelMeterLabel.setFont(juce::Font(12.0f)); 
    inputLevelMeterLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.45f));
    inputLevelMeterLabel.setJustificationType(juce::Justification::centred);
    inputLevelMeterLabel.setTooltip("How your input volume level compares\nwith the AI training data\n(5s time constant EMA RMS).\nIdeally within +-3 dB.");
    addAndMakeVisible(inputLevelMeterLabel);
    
    matchButton.setButtonText("Match");
    matchButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    matchButton.setColour(juce::TextButton::textColourOffId, utilityColor.withAlpha(0.8f)); 
    addAndMakeVisible(matchButton);

    matchButton.onClick = [this]() 
    {
        float currentDelta = audioProcessor.getDbDifferenceFromTarget();
        
        if (currentDelta <= -68.0f) return; 
        float currentGain = static_cast<float>(inputGainSlider.getValue());
        float newGain = currentGain - currentDelta;
        inputGainSlider.setValue(newGain, juce::sendNotificationSync);
    };

    // Utilities
    setupKnob(inputGainSlider, inputGainLabel, "Input", " dB", "input_gain", inputGainAttachment, utilityColor);

    // Detection
    setupKnob(thresholdSlider, thresholdLabel, "Threshold", "", "threshold", thresholdAttachment, detectionColor);
    setupKnob(probSmoothSlider, probSmoothLabel, "Smooth", " ms", "probsmoothing", probSmoothAttachment, detectionColor);
    setupKnob(shiftSlider, shiftLabel, "Shift", " ms", "shift", shiftAttachment, detectionColor);

    // Envelope
    setupKnob(floorSlider, floorLabel, "Floor", " dB", "floor", floorAttachment, envelopeColor);
    setupKnob(attackSlider, attackLabel, "Attack", " ms", "attack", attackAttachment, envelopeColor);
    setupKnob(releaseSlider, releaseLabel, "Release", " ms", "release", releaseAttachment, envelopeColor);

    startTimerHz(60); 
}

VocalGateEditor::~VocalGateEditor() 
{
    inputGainSlider.setLookAndFeel(nullptr);
    thresholdSlider.setLookAndFeel(nullptr);
    floorSlider.setLookAndFeel(nullptr);
    probSmoothSlider.setLookAndFeel(nullptr); 
    attackSlider.setLookAndFeel(nullptr);
    releaseSlider.setLookAndFeel(nullptr);
    shiftSlider.setLookAndFeel(nullptr);
}

void VocalGateEditor::timerCallback()
{
    inputHistory[writeIndex]  = audioProcessor.getInputLevel();
    outputHistory[writeIndex] = audioProcessor.getOutputLevel();
    float rawProb             = audioProcessor.getGateProbability(); 
    
    float dbDiff = audioProcessor.getDbDifferenceFromTarget();
    if (dbDiff < -80.0f) 
    {
        inputLevelMeterLabel.setText("-inf dB", juce::dontSendNotification);
    }
    else
    {
        juce::String sign = (dbDiff > 0.0f) ? "+" : "";
        inputLevelMeterLabel.setText(sign + juce::String(dbDiff, 1) + " dB", juce::dontSendNotification);
    }

    size_t prevIndex = (writeIndex + historySize - 1) % historySize;
    float prevProb = probHistory[prevIndex];
    
    float smoothAmount = 0.6f; 
    probHistory[writeIndex] = (rawProb * (1.0f - smoothAmount)) + (prevProb * smoothAmount);
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

    float shiftMs = static_cast<float>(shiftSlider.getValue());
    int shiftFrames = static_cast<int>(std::round(shiftMs / (1000.0f / 60.0f)));

    for (int i = 0; i < historySize; ++i)
    {
        // Audio uses the current real-time index
        size_t audioReadIndex = (writeIndex + static_cast<size_t>(i)) % historySize;
        // Probability uses the visually shifted index
        int source_i = i - shiftFrames;
        source_i = juce::jlimit(0, historySize - 1, source_i); // Prevent out-of-bounds drawing
        size_t probReadIndex = (writeIndex + static_cast<size_t>(source_i)) % historySize;

        float x = audioArea.getX() + (i * xStep);
        float inY = audioArea.getBottom() - (inputHistory[audioReadIndex] * audioArea.getHeight());
        float outY = audioArea.getBottom() - (outputHistory[audioReadIndex] * audioArea.getHeight());
        
        float pY = probArea.getBottom() - (probHistory[probReadIndex] * probArea.getHeight());

        inputPath.lineTo(x, inY);
        outputPath.lineTo(x, outY);

        if (i == 0) probPath.startNewSubPath(x, pY);
        else        probPath.lineTo(x, pY);
    }

    inputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    inputPath.closeSubPath();
    outputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    outputPath.closeSubPath();

    // Input and output waveforms
    g.setColour(juce::Colour::fromRGB(80, 85, 95).withAlpha(0.4f));
    g.fillPath(inputPath);
    g.setColour(juce::Colour::fromRGB(40, 210, 180).withAlpha(0.85f));
    g.fillPath(outputPath);

    // Prob and threshold
    g.setColour (juce::Colours::darkorange);
    g.strokePath (probPath, juce::PathStrokeType(2.0f));
    float threshVal = static_cast<float>(thresholdSlider.getValue());
    float threshY = probArea.getBottom() - (threshVal * probArea.getHeight());
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    float dashLengths[2] = { 4.0f, 4.0f };
    g.drawDashedLine(juce::Line<float>(probArea.getX(), threshY, probArea.getRight(), threshY), dashLengths, 2, 1.0f);

    // Logo
    g.setColour (juce::Colour::fromRGB (120, 125, 135)); 
    g.setFont (titleFont); 
    g.drawText ("Vocal Gate", audioArea.getX() + 8, audioArea.getY() + 8, 150, 40, juce::Justification::topLeft, true);
}

void VocalGateEditor::resized()
{
    auto bounds = getLocalBounds();
    auto bottomArea = bounds.removeFromBottom(110); 
    
    int squeezePixels = 25; 
    bottomArea = bottomArea.withTrimmedLeft(squeezePixels).withTrimmedRight(squeezePixels);
    
    int shiftDown = 8;

    // Utility
    auto utilityArea = bottomArea.removeFromLeft(120);
    auto knobArea = utilityArea.removeFromLeft(70);
    auto helperArea = utilityArea; 

    inputGainSlider.setBounds(knobArea.withSizeKeepingCentre(70, 50).translated(0, shiftDown));
    inputLevelMeterLabel.setBounds(helperArea.withSizeKeepingCentre(60, 19).translated(0, shiftDown - 17.f));
    matchButton.setBounds(helperArea.withSizeKeepingCentre(60, 19).translated(0, shiftDown + 4.0f));

    bottomArea.removeFromLeft(15); 
    int knobWidth = bottomArea.getWidth() / 6; 

    thresholdSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    probSmoothSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    shiftSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    
    floorSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    attackSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    releaseSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
}

