// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalGateEditor::VocalGateEditor (VocalGateProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (600, 300); 

    // Updated lambda to accept a color for the pointer
    auto setupKnob = [this](juce::Slider& slider, juce::Label& label, const juce::String& text, 
                            const juce::String& suffix, auto* param, 
                            std::unique_ptr<juce::SliderParameterAttachment>& attachment,
                            juce::Colour pointerColor) 
    {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 20);
        slider.setTextValueSuffix(suffix);
        
        // Apply our custom LookAndFeel and color
        slider.setLookAndFeel(&customKnobLookAndFeel);
        slider.setColour(juce::Slider::thumbColourId, pointerColor);
        slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack); // Clean up the text box

        addAndMakeVisible(slider);
        attachment = std::make_unique<juce::SliderParameterAttachment>(*param, slider, nullptr);

        label.setText(text, juce::dontSendNotification);
        label.attachToComponent(&slider, false);
        label.setJustificationType(juce::Justification::centred);
        label.setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(label);
    };

    // --- Define your theme colors ---
    juce::Colour detectionColor = juce::Colours::darkorange;
    juce::Colour envelopeColor = juce::Colour::fromRGB(40, 210, 180); // Cyan matching the audio output

    // --- Apply them to the knobs ---
    setupKnob(thresholdSlider, thresholdLabel, "Threshold", "", audioProcessor.thresholdParam, thresholdAttachment, detectionColor);
    setupKnob(floorSlider, floorLabel, "Floor", " dB", audioProcessor.floorParam, floorAttachment, detectionColor);
    
    setupKnob(attackSlider, attackLabel, "Attack", " ms", audioProcessor.attackParam, attackAttachment, envelopeColor);
    setupKnob(releaseSlider, releaseLabel, "Release", " ms", audioProcessor.releaseParam, releaseAttachment, envelopeColor);
    setupKnob(shiftSlider, shiftLabel, "Shift", " ms", audioProcessor.shiftParam, shiftAttachment, envelopeColor);

    startTimerHz(60); 
}

VocalGateEditor::~VocalGateEditor() 
{
    // CRITICAL: You must remove the custom LookAndFeel before the editor is destroyed
    thresholdSlider.setLookAndFeel(nullptr);
    floorSlider.setLookAndFeel(nullptr);
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

    // --- NEW LAYOUT MATH ---
    auto graphArea = bounds.withTrimmedBottom(110).withTrimmedTop(15).withTrimmedRight(20).withTrimmedLeft(20);
    auto audioArea = graphArea.removeFromTop(graphArea.getHeight() / 2);
    auto probArea = graphArea; 

    // Draw subtle backgrounds
    g.setColour(juce::Colour::fromRGB(30, 30, 34));
    g.fillRect(audioArea);
    g.setColour(juce::Colour::fromRGB(25, 25, 28)); 
    g.fillRect(probArea);

    juce::Path inputPath, outputPath, probPath;

    float width = audioArea.getWidth();
    float xStep = width / (historySize - 1);

    inputPath.startNewSubPath(audioArea.getX(), audioArea.getBottom());
    outputPath.startNewSubPath(audioArea.getX(), audioArea.getBottom());

    // --- INSTANT VISUAL SHIFT MATH ---
    // 60 frames per 1000ms means 0.06 frames per ms
    float shiftMs = static_cast<float>(shiftSlider.getValue());
    int offsetFrames = juce::roundToInt(shiftMs * 0.06f); 

    for (int i = 0; i < historySize; ++i)
    {
        // 1. Audio Read Index (Normal circular buffer read)
        int audioReadIndex = (writeIndex + i) % historySize;
        
        // 2. Prob Read Index (Clamped to prevent the wrap-around glitch!)
        // Instead of letting it loop, we clamp 'i + offsetFrames' to stay within our history limits
        int logicalProbIndex = juce::jlimit(0, historySize - 1, i + offsetFrames);
        int probReadIndex = (writeIndex + logicalProbIndex) % historySize;

        float x = audioArea.getX() + (i * xStep);
        
        // Map audio to the top area
        float inY = audioArea.getBottom() - (inputHistory[audioReadIndex] * audioArea.getHeight());
        float outY = audioArea.getBottom() - (outputHistory[audioReadIndex] * audioArea.getHeight());

        // Map probability to the bottom area using the CLAMPED shifted index
        float pY = probArea.getBottom() - (probHistory[probReadIndex] * probArea.getHeight());

        inputPath.lineTo(x, inY);
        outputPath.lineTo(x, outY);

        if (i == 0) probPath.startNewSubPath(x, pY);
        else        probPath.lineTo(x, pY);
    }

    // Close the audio paths
    inputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    inputPath.closeSubPath();
    outputPath.lineTo(audioArea.getRight(), audioArea.getBottom());
    outputPath.closeSubPath();

    // --- 1. Draw Audio Silhouettes ---
    g.setColour(juce::Colour::fromRGB(80, 85, 95).withAlpha(0.4f));
    g.fillPath(inputPath);

    g.setColour(juce::Colour::fromRGB(40, 210, 180).withAlpha(0.85f));
    g.fillPath(outputPath);

    // --- 2. Draw Probability Curve ---
    g.setColour (juce::Colours::darkorange);
    g.strokePath (probPath, juce::PathStrokeType(2.0f));

    // --- 3. Draw Threshold Line ---
    float threshVal = static_cast<float>(thresholdSlider.getValue());
    float threshY = probArea.getBottom() - (threshVal * probArea.getHeight());
    
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    float dashLengths[2] = { 4.0f, 4.0f };
    g.drawDashedLine(juce::Line<float>(probArea.getX(), threshY, probArea.getRight(), threshY), dashLengths, 2, 1.0f);

    // --- 4. ML BRAIN ERROR OVERLAY (Add this at the very end of paint!) ---
    if (! audioProcessor.isModelLoaded())
    {
        // Dim the entire graph area slightly to make the text pop
        g.setColour (juce::Colours::black.withAlpha (0.6f));
        g.fillRect (graphArea);

        // Draw the main warning text
        g.setColour (juce::Colours::red.brighter());
        g.setFont (juce::Font (22.0f, juce::Font::bold));
        g.drawFittedText ("MODEL MISSING: ML BRAIN OFFLINE", graphArea, juce::Justification::centred, 1);
        
        // Draw a helpful sub-text slightly below the center
        g.setFont (juce::Font (14.0f));
        g.setColour (juce::Colours::lightgrey);
        g.drawFittedText ("Could not find vocalgate_int8.onnx on this system.", 
                          graphArea.translated(0, 25), juce::Justification::centred, 1);
    }
}

void VocalGateEditor::resized()
{
    auto bounds = getLocalBounds();
    auto bottomArea = bounds.removeFromBottom(110); // Area for knobs
    
    int squeezePixels = 40; 
    bottomArea = bottomArea.withTrimmedLeft(squeezePixels).withTrimmedRight(squeezePixels);
    
    int knobWidth = bottomArea.getWidth() / 5; // 5 narrower columns
    int shiftDown = 8;

    thresholdSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    floorSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    attackSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    releaseSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
    shiftSlider.setBounds(bottomArea.removeFromLeft(knobWidth).withSizeKeepingCentre(70, 70).translated(0, shiftDown));
}

