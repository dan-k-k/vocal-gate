# plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(VocalGate VERSION 1.0.0)

# 1. Fetch JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.0 
)
FetchContent_MakeAvailable(JUCE)

# 2. Define the Plugin
juce_add_plugin(VocalGate
    COMPANY_NAME "DanK"                 
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    # We turn this OFF because we handle the copy manually in the POST_BUILD step
    COPY_PLUGIN_AFTER_BUILD FALSE  
    VST3_AUTO_MANIFEST FALSE          
    PLUGIN_MANUFACTURER_CODE "DanK"
    PLUGIN_CODE "Vgt1"                      
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Vocal Gate"
)

# Explicitly generate the JUCE header (from your StemExtractor)
juce_generate_juce_header(VocalGate)

# 3. Add Source Files
target_sources(VocalGate
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/AudioFIFO.h
        Source/DSPConstants.h
)

# 4. Standard JUCE Definitions
target_compile_definitions(VocalGate
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

# 5. Link JUCE Modules and ONNX Runtime
target_link_libraries(VocalGate
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# 6. Include ONNX Directories
target_include_directories(VocalGate SYSTEM PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

# ---------------------------------------------------------------------------
# 7. macOS DYLD Routing & RPATH Setup
# ---------------------------------------------------------------------------
# Tell the VST3 binary to look for dylibs inside its own Frameworks folder
set_target_properties(VocalGate_VST3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/../Frameworks"
)

# 8. Custom Post-Build Bundle Construction
add_custom_command(TARGET VocalGate_VST3 POST_BUILD
    # Create the Frameworks directory inside the built .vst3 bundle
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:VocalGate_VST3>/../Frameworks"
    
    # Copy ONNX into the bundle. 
    # NOTE: Ensure "1.24.2" matches the actual internal version of the dylib you downloaded!
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib"
        "$<TARGET_FILE_DIR:VocalGate_VST3>/../Frameworks/libonnxruntime.1.24.2.dylib"
        
    # Strip Apple quarantine flags
    COMMAND xattr -rc "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
    
    # Ad-hoc sign the bundle
    COMMAND codesign --force --deep --sign - "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."

    # Remove the old VST3 from the system folder
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Vocal Gate.vst3"

    # Copy the freshly packaged and signed bundle into the system folder
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
        "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Vocal Gate.vst3"
)

# ---------------------------------------------------------------------------
# 9. Testing
# ---------------------------------------------------------------------------
enable_testing()

# We uncommented this and updated the file name to test_main.cpp!
add_executable(RunCPPTests ../tests/CPP/test_main.cpp)

# Link the JUCE modules your tests need
target_link_libraries(RunCPPTests PRIVATE juce::juce_audio_utils juce::juce_dsp)

# Register the test with CTest
add_test(NAME FIFOTest COMMAND RunCPPTests)

