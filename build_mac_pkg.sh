#!/bin/bash
# build_mac_pkg.sh

VERSION=${1:-1.0.0} # FALLBACK
echo "Building macOS .pkg Installer for version $VERSION..."

# Mimicthe root of a Mac HD
mkdir -p Payload/Library/Audio/Plug-Ins/VST3
mkdir -p Payload/Library/Audio/Plug-Ins/Components

# Copy the compiled plugins into staging area
cp -R "build/VocalGate_artefacts/Release/VST3/Vocal Gate.vst3" "Payload/Library/Audio/Plug-Ins/VST3/"
cp -R "build/VocalGate_artefacts/Release/AU/Vocal Gate.component" "Payload/Library/Audio/Plug-Ins/Components/"

# pkgbuild tool
pkgbuild --root Payload \
         --identifier com.dank.vocalgate \
         --version $VERSION \
         --install-location "/" \
         VocalGate_Mac_Installer.pkg

echo "Created VocalGate_Mac_Installer.pkg"

