#!/bin/bash
# build_mac_pkg.sh

echo "Building macOS .pkg Installer..."

# 1. Create a staging directory that mimics the root of a Mac hard drive
mkdir -p Payload/Library/Audio/Plug-Ins/VST3
mkdir -p Payload/Library/Audio/Plug-Ins/Components

# 2. Copy the compiled plugins from the CMake build folder into the staging area
# (Assuming your CMake outputs to build/VocalGate_artefacts/Release/)
cp -R "build/VocalGate_artefacts/Release/VST3/Vocal Gate.vst3" "Payload/Library/Audio/Plug-Ins/VST3/"
cp -R "build/VocalGate_artefacts/Release/AU/Vocal Gate.component" "Payload/Library/Audio/Plug-Ins/Components/"

# 3. Use Apple's built-in pkgbuild tool to create the installer
# We tell it to take everything in "Payload" and install it to "/" (the system drive)
pkgbuild --root Payload \
         --identifier com.dank.vocalgate \
         --version 1.0.0 \
         --install-location "/" \
         VocalGate_Mac_Installer.pkg

echo "Done! Created VocalGate_Mac_Installer.pkg"

