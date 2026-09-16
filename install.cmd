@echo off
rem  ClayQuant installer for Windows - start here.
rem
rem  Double-click this file, or from a prompt:
rem
rem      .\install.cmd                 (PowerShell needs the .\ )
rem      install.cmd                   (cmd.exe is happy without it)
rem      .\install.cmd -NoDev
rem      .\install.cmd -Python "C:\Python312\python.exe"
rem
rem  PowerShell does not run programs from the current directory unless you
rem  say so, hence the .\ - without it you get "the term 'install.cmd' is not
rem  recognized", which is about the path and not about the file.
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
rem  unchanged.  The window waits for a keypress at the end so that a
rem  message can be read when this was double-clicked; set
rem  CLAYQUANT_NO_PAUSE to anything to skip that.

setlocal EnableDelayedExpansion

rem  Windows PowerShell 5.1 is present on every Windows installation and is
rem  what these scripts were written for; fall back to PowerShell 7 only if
rem  the built-in one is missing.
set "PSEXE=powershell"
where /q powershell || set "PSEXE=pwsh"
where /q !PSEXE! || (
    echo ClayQuant: no PowerShell found on this system.
    echo Looked for "powershell" and "pwsh" on PATH.
    if not defined CLAYQUANT_NO_PAUSE pause
    exit /b 9009
)

"!PSEXE!" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "CQEXIT=!ERRORLEVEL!"

rem  Always wait, rather than trying to work out whether Explorer started
rem  this.  Guessing from CMDCMDLINE looked clever and cost a reader the one
rem  error message they needed; a keypress at a prompt costs nothing.
if not defined CLAYQUANT_NO_PAUSE pause

exit /b !CQEXIT!
