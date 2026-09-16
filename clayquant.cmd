@echo off
rem  ClayQuant, run from the checkout without activating anything.
rem
rem      .\clayquant.cmd gui                 (PowerShell needs the .\ )
rem      clayquant.cmd gui                   (cmd.exe is happy without it)
rem      .\clayquant.cmd build-library -o library\clays.npz
rem      .\clayquant.cmd import-structures structures.xml -o structures\phases.json
rem
rem  The same wrapper as install.cmd and for the same reason: Windows turns
rem  PowerShell script execution off by default, so ".\clayquant.ps1" is
rem  refused before it runs.  This starts PowerShell with the policy
rem  bypassed for one invocation and passes everything through unchanged.
rem
rem  No pause here, unlike install.cmd: this one is for running commands and
rem  the GUI, where a keypress at the end would be in the way.
rem
rem  The desktop and Start-menu shortcuts are unaffected either way - they
rem  point at the environment's own clayquant-gui.exe, not at a script.

setlocal EnableDelayedExpansion

set "PSEXE=powershell"
where /q powershell || set "PSEXE=pwsh"
where /q !PSEXE! || (
    echo ClayQuant: no PowerShell found on this system.
    echo Looked for "powershell" and "pwsh" on PATH.
    exit /b 9009
)

"!PSEXE!" -NoProfile -ExecutionPolicy Bypass -File "%~dp0clayquant.ps1" %*
exit /b !ERRORLEVEL!
