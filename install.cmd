@echo off
rem  ClayQuant installer for Windows - start here.
rem
rem  Double-click this file, or run it from a command prompt:
rem
rem      install.cmd
rem      install.cmd -NoDev
rem      install.cmd -Python "C:\Python312\python.exe"
rem
rem  Why this file exists.  Windows ships with PowerShell script execution
rem  turned off, so ".\install.ps1" fails before it runs a line:
rem
rem      install.ps1 cannot be loaded because running scripts is disabled
rem      on this system
rem
rem  That is the default on every client installation of Windows and has
rem  nothing to do with this program.  A .cmd file is not covered by that
rem  policy, so this one starts PowerShell with the policy bypassed for
rem  that single invocation - it changes no setting, needs no
rem  administrator, and leaves the machine exactly as it was.
rem
rem  Everything after the file name is passed through to install.ps1
rem  unchanged.

setlocal EnableDelayedExpansion

rem  Windows PowerShell 5.1 is present on every Windows installation and is
rem  what these scripts were written for; fall back to PowerShell 7 only if
rem  the built-in one is missing.
set "PSEXE=powershell"
where /q powershell || set "PSEXE=pwsh"
where /q %PSEXE% || (
    echo ClayQuant: no PowerShell found on this system.
    echo Looked for "powershell" and "pwsh" on PATH.
    set "EXITCODE=9009"
    goto :finish
)

"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

:finish
rem  Double-clicking from Explorer runs this through "cmd /c", which closes
rem  the window the moment the script ends - including on the error the
rem  reader needs to see.  Started from a prompt, there is nothing to keep
rem  open and pausing would be an annoyance.
echo "%CMDCMDLINE%" | find /i " /c " >nul 2>&1 && pause

exit /b %EXITCODE%
