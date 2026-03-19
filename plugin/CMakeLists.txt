# plugin/CMakeLists.txt
# cmake -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release --parallel 4

# rm -rf ~/Library/Caches/AudioUnitCache/* 
# sudo rm -rf /Library/Caches/AudioUnitCache/*
# killall -9 AudioComponentRegistrar
# sudo pkgutil --forget com.dank.vocalgate

cmake_minimum_required(VERSION 3.15)

if(DEFINED ENV{CI_VERSION})
    project(VocalGate VERSION $ENV{CI_VERSION})
    message(STATUS "Building CI Version: ${PROJECT_VERSION}")
else()
    project(VocalGate VERSION 1.0.0) # FALLBACK
    message(STATUS "Building Local Version: ${PROJECT_VERSION}")
endif()

# Fetch JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.12 
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
        Source/ParameterManager.cpp
        Source/GateDSP.cpp
        Source/BackgroundMLThread.cpp
        Source/FeatureExtractor.cpp
        Source/InferenceEngine.cpp
        Source/AudioFIFO.h
        Source/DSPConstants.h
)

target_include_directories(VocalGate
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

# Standard JUCE definitions 
target_compile_definitions(VocalGate
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

# Base JUCE linking 
target_link_libraries(VocalGate
    PRIVATE
        VocalGateData 
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# OS-specific linking
if(APPLE)
    # Mac M-series.
    target_link_libraries(VocalGate PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.1.24.2.dylib
    )

    set(PLUGIN_FORMATS VST3 AU Standalone)

    foreach(FORMAT IN LISTS PLUGIN_FORMATS)
        set(TARGET_NAME "VocalGate_${FORMAT}")

        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path/../Frameworks"
        )

        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/../Frameworks"
            
            COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.1.24.2.dylib"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/../Frameworks/libonnxruntime.1.24.2.dylib"
            
            COMMAND install_name_tool -change 
                @rpath/libonnxruntime.1.24.2.dylib 
                @loader_path/../Frameworks/libonnxruntime.1.24.2.dylib 
                "$<TARGET_FILE:${TARGET_NAME}>"

            COMMAND codesign --force --sign - "$<TARGET_FILE_DIR:${TARGET_NAME}>/../Frameworks/libonnxruntime.1.24.2.dylib"
            COMMAND codesign --force --deep --sign - "$<TARGET_FILE_DIR:${TARGET_NAME}>/../.."
        )
    endforeach()

    add_custom_command(TARGET VocalGate_VST3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
            "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Vocal Gate.vst3"
    )

elseif(WIN32)
    # Windows x64
    target_link_libraries(VocalGate PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime.lib
    )

    target_link_libraries(VocalGate_VST3 PRIVATE Delayimp)
    target_link_options(VocalGate_VST3 PRIVATE "/DELAYLOAD:onnxruntime.dll")

    add_custom_command(TARGET VocalGate_VST3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:VocalGate_VST3>/"
            
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_FILE_DIR:VocalGate_VST3>/../.."
            "$ENV{COMMONPROGRAMFILES}/VST3/Vocal Gate.vst3"
    )
endif()

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

