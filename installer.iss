; installer.iss
#define MyAppName "Vocal Gate"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "DanK"

[Setup]
AppId={{YOUR-UNIQUE-GUID-HERE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
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
Source: "build\VocalGate_VST3.vst3\*"; DestDir: "{commoncf64}\VST3\Vocal Gate.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs

[Messages]
BeveledLabel=DanK

