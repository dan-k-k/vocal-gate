; installer.iss
#define MyAppName "Vocal Gate"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "DanK"

[Setup]
AppId={{38B39248-A699-43EF-B35B-376D32A6950C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
ArchitecturesInstallIn64BitMode=x64
; Output settings
OutputDir=Output
OutputBaseFilename=VocalGate_Windows_Installer
Compression=lzma
SolidCompression=yes
; Require admin rights to install to Program Files
PrivilegesRequired=admin
; No need for a Start Menu folder for a VST plugin
DisableProgramGroupPage=yes

[Files]
; Grab the entire VST3 bundle folder (which includes your .vst3 and the copied onnxruntime.dll)
; Note: Check your exact CMake output path. Usually it's build/plugin/VocalGate_VST3.dir/Release/ or similar.
Source: "build\VocalGate_artefacts\Release\VST3\Vocal Gate.vst3\*"; DestDir: "{commoncf64}\VST3\Vocal Gate.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs

[Messages]
BeveledLabel=DanK

