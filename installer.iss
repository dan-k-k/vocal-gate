; installer.iss
#define MyAppName "Vocal Gate"
#ifndef MyAppVersion
#define MyAppVersion "1.0.0" ; FALLBACK 
#endif
#define MyAppPublisher "DanK"

[Setup]
AppId={{38B39248-A699-43EF-B35B-376D32A6950C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
ArchitecturesInstallIn64BitMode=x64 arm64
OutputDir=Output
OutputBaseFilename=VocalGate_Windows_Installer
RestartIfNeededByRun=no
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
DisableProgramGroupPage=yes

[Files]
; VST3
Source: "build\VocalGate_artefacts\Release\VST3\Vocal Gate.vst3\*"; DestDir: "{commoncf64}\VST3\Vocal Gate.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs

; Bundle VC++ Redistributable into a temporary folder
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /passive /norestart"; StatusMsg: "Installing Visual C++ Redistributable (Required for AI inference)..."; Flags: waituntilterminated

[Messages]
BeveledLabel=DanK

