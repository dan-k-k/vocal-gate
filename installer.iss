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
ArchitecturesInstallIn64BitMode=x64 arm64
OutputDir=Output
OutputBaseFilename=VocalGate_Windows_Installer
; ADD THIS LINE:
RestartIfNeededByRun=no
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
DisableProgramGroupPage=yes

[Files]
; 1. The VST3 Plugin
Source: "build\VocalGate_artefacts\Release\VST3\Vocal Gate.vst3\*"; DestDir: "{commoncf64}\VST3\Vocal Gate.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs

; 2. Bundle the VC++ Redistributable into a temporary folder
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
; 3. Execute the Redistributable silently during installation
; /install = Installs it
; /passive = Shows a basic progress bar but requires no user interaction
; /norestart = Prevents the redist installer from rebooting the user's PC automatically
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /passive /norestart"; StatusMsg: "Installing Visual C++ Redistributable (Required for AI features)..."; Flags: waituntilterminated

[Messages]
BeveledLabel=DanK

