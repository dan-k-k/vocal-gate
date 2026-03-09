# plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(VocalGate VERSION 1.0.0)

# Fetch JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.0 
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(VocalGate
    COMPANY_NAME "DanK"                 
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE  
    VST3_AUTO_MANIFEST FALSE          
    PLUGIN_MANUFACTURER_CODE "DanK"
    PLUGIN_CODE "Vgt1"                      
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Vocal Gate"
)

juce_generate_juce_header(VocalGate)

juce_add_binary_data(VocalGateData
    SOURCES
        vocalgate_int8.onnx 
        FredokaOne.ttf 
)

# Source files
target_sources(VocalGate
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/AudioFIFO.h
        Source/DSPConstants.h
)

# Standard JUCE definitions
target_compile_definitions(VocalGate
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

# Link with ONNX 
target_link_libraries(VocalGate
    PRIVATE
        VocalGateData 
        juce::juce_audio_utils
        juce::juce_dsp
        ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

target_include_directories(VocalGate SYSTEM PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

# VST to look for dylibs inside Frameworks folder
set_target_properties(VocalGate_VST3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/../Frameworks"
)

# Construction
add_custom_command(TARGET VocalGate_VST3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:VocalGate_VST3>/../Frameworks"
    
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib"
        "$<TARGET_FILE_DIR:VocalGate_VST3>/../Frameworks/libonnxruntime.1.24.2.dylib"
        
    COMMAND xattr -rc "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
    COMMAND codesign --force --deep --sign - "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Vocal Gate.vst3"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
        "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Vocal Gate.vst3"
)

# CTest
enable_testing()

add_executable(RunCPPTests 
    ../tests/CPP/test_main.cpp
)

target_link_libraries(RunCPPTests 
    PRIVATE 
        juce::juce_audio_basics 
        juce::juce_core
)

target_include_directories(RunCPPTests 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

add_test(NAME FIFOTest COMMAND RunCPPTests)

