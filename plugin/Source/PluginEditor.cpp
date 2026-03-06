// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalGateEditor::VocalGateEditor (VocalGateProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (500, 300); // Made it wider for the scrolling timeline

    // --- Setup Threshold Slider ---
    thresholdSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    thresholdSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(thresholdSlider);

    // ADD THIS: Link the slider to the processor's parameter.
    // This single line replaces setRange, setValue, and the old onValueChange lambda!
    thresholdAttachment = std::make_unique<juce::SliderParameterAttachment>(*audioProcessor.thresholdParam, thresholdSlider, nullptr);

    thresholdLabel.setText("Threshold", juce::dontSendNotification);
    thresholdLabel.attachToComponent(&thresholdSlider, true);
    thresholdLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(thresholdLabel);

    // Wake up 60 times a second to redraw the screen
    startTimerHz(60); 
}

VocalGateEditor::~VocalGateEditor() {}

void VocalGateEditor::timerCallback()
{
    // 1. Read the latest values from the background ML & Audio threads
    float currentProb = audioProcessor.gateProbability.load();
    float currentAudio = audioProcessor.latestAudioLevel.load();

    // 2. Save them into our circular arrays
    probHistory[writeIndex] = currentProb;
    audioHistory[writeIndex] = currentAudio;

    // 3. Advance the index, wrapping around if we hit the end
    writeIndex = (writeIndex + 1) % historySize;

    // 4. Tell JUCE to call paint() again
    repaint(); 
}

void VocalGateEditor::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Dark grey background
    g.fillAll (juce::Colour::fromRGB (30, 30, 30));

    // Define the graphing area
    auto graphArea = bounds.withTrimmedBottom(60).withTrimmedTop(20).withTrimmedRight(20).withTrimmedLeft(20);
    g.setColour(juce::Colour::fromRGB(45, 45, 45));
    g.fillRect(graphArea);

    // --- Draw the Scrolling Paths ---
    juce::Path probPath;
    juce::Path audioPath;

    float width = graphArea.getWidth();
    float height = graphArea.getHeight();
    float bottom = graphArea.getBottom();
    float xStep = width / (historySize - 1);

    for (int i = 0; i < historySize; ++i)
    {
        // Read from oldest to newest using the circular index
        int readIndex = (writeIndex + i) % historySize;
        
        float x = graphArea.getX() + (i * xStep);
        
        // Map probability (0.0 to 1.0) to Y pixels
        float probY = bottom - (probHistory[readIndex] * height);
        
        // Map audio (0.0 to 1.0) to Y pixels (drawn from the center line)
        float audioYOffset = audioHistory[readIndex] * (height * 0.4f);

        if (i == 0) {
            probPath.startNewSubPath(x, probY);
            audioPath.startNewSubPath(x, graphArea.getCentreY() - audioYOffset);
        } else {
            probPath.lineTo(x, probY);
            audioPath.lineTo(x, graphArea.getCentreY() - audioYOffset);
        }
    }

    // Draw Audio Waveform (Dodger Blue)
    g.setColour (juce::Colours::dodgerblue.withAlpha(0.6f));
    g.strokePath (audioPath, juce::PathStrokeType(2.0f));

    // Draw ML Probability (Dark Orange)
    g.setColour (juce::Colours::darkorange);
    g.strokePath (probPath, juce::PathStrokeType(3.0f));

    // --- Draw Threshold Line (Red Dashed) ---
    float threshVal = static_cast<float>(thresholdSlider.getValue());
    float threshY = bottom - (threshVal * height);
    g.setColour(juce::Colours::red.withAlpha(0.8f));
    g.drawLine(graphArea.getX(), threshY, graphArea.getRight(), threshY, 2.0f);

    // --- Draw The 550ms Lookahead Playhead! ---
    // 550ms is ~27.5% of our 2.0 second window. 
    // We draw the line 27.5% from the RIGHT edge.
    float lookaheadOffsetRatio = 0.550f / 2.0f; 
    float playheadX = graphArea.getRight() - (width * lookaheadOffsetRatio);
    
    g.setColour(juce::Colours::white);
    g.drawLine(playheadX, graphArea.getY(), playheadX, graphArea.getBottom(), 2.0f);
    
    g.setFont(14.0f);
}

void VocalGateEditor::resized()
{
    auto bounds = getLocalBounds();
    auto bottomArea = bounds.removeFromBottom(50);
    
    thresholdSlider.setBounds(bottomArea.withTrimmedLeft(100).withTrimmedRight(20).withTrimmedBottom(10));
}

