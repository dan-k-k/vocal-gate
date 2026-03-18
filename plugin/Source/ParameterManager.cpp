// plugin/Source/ParameterManager.cpp
#include "ParameterManager.h"

ParameterManager::ParameterManager(juce::AudioProcessor& processor)
    : apvts(processor, nullptr, "Parameters", createParameterLayout())
{
    // Cache the atomic pointers once during instantiation
    thresholdParam     = apvts.getRawParameterValue("threshold");
    floorParam         = apvts.getRawParameterValue("floor");
    attackParam        = apvts.getRawParameterValue("attack");
    releaseParam       = apvts.getRawParameterValue("release");
    shiftParam         = apvts.getRawParameterValue("shift");
    probSmoothingParam = apvts.getRawParameterValue("probsmoothing");
    inputGainParam     = apvts.getRawParameterValue("input_gain");
}

juce::AudioProcessorValueTreeState::ParameterLayout ParameterManager::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"threshold", 1}, "P Threshold", 0.001f, 0.999f, 0.60f));

    juce::NormalisableRange<float> floorRange (-100.0f, 0.0f, 0.1f);
    floorRange.setSkewForCentre (-18.0f);
    auto floorAttributes = juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction([] (float value, int) {
            if (value <= -99.9f) return juce::String ("-inf");
            return juce::String (value, 1); 
        });
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"floor", 1}, "Floor", floorRange, -25.0f, floorAttributes));

    juce::NormalisableRange<float> attackRange (1.0f, 500.0f, 0.1f);
    attackRange.setSkewForCentre (20.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"attack", 1}, "Attack", attackRange, 10.0f));

    juce::NormalisableRange<float> releaseRange (10.0f, 2000.0f, 0.1f);
    releaseRange.setSkewForCentre (150.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"release", 1}, "Release", releaseRange, 150.0f));

    juce::NormalisableRange<float> shiftRange (-200.0f, 200.0f, 0.1f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"shift", 1}, "Shift", shiftRange, 0.0f));

    juce::NormalisableRange<float> probSmoothRange (100.0f, 1200.0f, 1.0f);
    probSmoothRange.setSkewForCentre (400.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"probsmoothing", 1}, "Smooth", probSmoothRange, 400.0f));

    juce::NormalisableRange<float> gainRange (-12.0f, 12.0f, 0.1f);
    gainRange.setSkewForCentre (0.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"input_gain", 1}, "Input Gain", gainRange, 0.0f));

    return { std::make_move_iterator(params.begin()), std::make_move_iterator(params.end()) };
}

void ParameterManager::saveState(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    if (xml != nullptr)
        juce::AudioProcessor::copyXmlToBinary (*xml, destData);
}

void ParameterManager::loadState(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (juce::AudioProcessor::getXmlFromBinary (data, sizeInBytes));

    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
        apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}

