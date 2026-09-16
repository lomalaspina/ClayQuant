<#
    ClayQuant installer for Windows PowerShell.

    RUN "install.cmd" INSTEAD OF THIS FILE.  Windows ships with PowerShell
    script execution disabled, so ".\install.ps1" is refused before it runs a
    line - "cannot be loaded because running scripts is disabled on this
    system".  That is the default on every client installation of Windows and
    says nothing about this script.  install.cmd, beside this file, is a .cmd
    and so exempt from the policy: it starts PowerShell with the policy
    bypassed for one invocation, passes these same arguments through, and
    changes no setting on the computer.  Typed by hand it is

        powershell -ExecutionPolicy Bypass -File .\install.ps1

    To allow local scripts permanently instead, for your own account and
    without administrator rights:

        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

    A git clone then runs as it stands; files unpacked from a downloaded ZIP
    are marked as remote and need Unblock-File first.

        .\install.ps1                 set everything up
        .\install.ps1 -Python "C:\Python312\python.exe"
                                      use a particular interpreter
        .\install.ps1 -NoDev          skip the test and import extras

    It chooses an interpreter, builds the virtual environment, installs
    ClayQuant and its dependencies into it, and checks that the result works.
    Re-running it is safe.

    If Windows opens the Microsoft Store when you type "python", that is the
    App Execution Alias stub rather than a real interpreter. Install Python
    from python.org, or turn the alias off under
    Settings > Apps > Advanced app settings > App execution aliases.
#>

[CmdletBinding()]
param(
    [string] $Python,
    [switch] $NoDev,
    [switch] $NoShortcut,
    [switch] $Yes
)

$ErrorActionPreference = 'Stop'

# PowerShell 7.3 and later turn a native program's non-zero exit into a
# terminating error when $ErrorActionPreference is Stop.  This script checks
# $LASTEXITCODE itself and reports what failed; left on, that setting would
# abort with a raw exception instead, and every "if ($LASTEXITCODE -ne 0)"
# below would be dead code.  Windows PowerShell has no such variable, so this
# only matters where the .cmd wrapper has fallen back to PowerShell 7.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir '.venv'
$Extras     = if ($NoDev) { 'gui,import' } else { 'gui,import,dev' }
$MinVersion = [Version]'3.10'

function Write-Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  ok  " -ForegroundColor Green -NoNewline; Write-Host $text }
function Write-Warn($text) { Write-Host "  !!  " -ForegroundColor Yellow -NoNewline; Write-Host $text }
function Stop-WithError($text) { Write-Host ""; Write-Host "  error  $text" -ForegroundColor Red; exit 1 }

# Probing an interpreter without going through PowerShell's argument passing.
#
# Windows PowerShell hands arguments to a native program through a legacy
# quoting path that strips embedded double quotes.  Written the obvious way,
#
#     & $exe -c 'import sys; print("%d.%d" % sys.version_info[:2])'
#
# the interpreter receives print(%d.%d % sys.version_info[:2]) - quotes gone -
# and dies with a SyntaxError.  The probe then looks like an interpreter that
# cannot report its version, every candidate is rejected in turn, and the
# script announces that no Python 3.10 or newer exists on a machine that has
# 3.12 on PATH.  PowerShell 7 passes the same line through correctly, so the
# failure appears only where most people will run this.
#
# Rather than tiptoe around the quoting rules, the probes below start the
# process themselves and build the command line explicitly.  That also buys two
# things the & operator does not give: a timeout, so an interpreter that blocks
# - the Microsoft Store stub used to open the Store and wait - cannot hang the
# installation; and the child's own output, which is the diagnosis and which the
# previous version of this script discarded.
$VersionProbe = 'import sys; print(sys.version_info[0], sys.version_info[1])'

$script:LastProbe = ''

function Invoke-Probe($exe, $arguments, $timeoutSeconds) {
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName               = $exe
    $info.Arguments              = $arguments
    $info.UseShellExecute        = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError  = $true
    $info.CreateNoWindow         = $true

    try {
        $process = [System.Diagnostics.Process]::Start($info)
    } catch {
        return [pscustomobject]@{ ExitCode = -1; Output = $_.Exception.Message; TimedOut = $false }
    }

    # Start reading before waiting.  A child that fills the pipe buffer blocks
    # on the write while the parent blocks on the exit, and neither moves again.
    $out = $process.StandardOutput.ReadToEndAsync()
    $err = $process.StandardError.ReadToEndAsync()

    if (-not $process.WaitForExit([int]($timeoutSeconds * 1000))) {
        try { $process.Kill() } catch { }
        return [pscustomobject]@{
            ExitCode = -1
            Output   = "no answer after $timeoutSeconds seconds"
            TimedOut = $true
        }
    }
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output   = ($out.Result + $err.Result).Trim()
        TimedOut = $false
    }
}

function Format-Probe($result) {
    if ($result.Output) { return ($result.Output -replace '\s+', ' ') }
    return "exit code $($result.ExitCode)"
}

function Get-PythonVersion($exe) {
    # The probe holds no double quote of its own, so wrapping it in one here is
    # safe and the interpreter receives it intact.
    $result = Invoke-Probe $exe ('-c "' + $VersionProbe + '"') 30
    if ($result.ExitCode -ne 0) {
        $script:LastProbe = Format-Probe $result
        return $null
    }
    $parts = $result.Output -split '\s+'
    if ($parts.Count -ge 2) {
        $script:LastProbe = ''
        try { return [Version]('{0}.{1}' -f $parts[0], $parts[1]) } catch { }
    }
    $script:LastProbe = "unexpected answer '$($result.Output)'"
    return $null
}

function Test-VenvWorks($exe) {
    $probe = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
    try {
        $result = Invoke-Probe $exe ('-m venv "' + $probe + '"') 180
        # Windows puts the interpreter in Scripts, POSIX in bin; this script is
        # for Windows, but PowerShell runs everywhere and a wrong answer here
        # would reject a working interpreter.
        $made = (Test-Path (Join-Path $probe 'Scripts\python.exe')) -or
                (Test-Path (Join-Path $probe 'bin/python'))
        if ($result.ExitCode -eq 0 -and $made) { $script:LastProbe = ''; return $true }
        $script:LastProbe = if ($result.ExitCode -eq 0) {
            'it reported success but produced no interpreter'
        } else { Format-Probe $result }
        return $false
    } finally {
        if (Test-Path $probe) { Remove-Item -Recurse -Force $probe -ErrorAction SilentlyContinue }
    }
}

# --------------------------------------------------------------------------- #
# 1. Find an interpreter that can actually build a virtual environment.
#
# Each Python version carries its own venv module, so the one that "python"
# resolves to is what matters, not the newest one installed. The Windows py
# launcher is asked first because it knows about every registered install.
# --------------------------------------------------------------------------- #
Write-Step "Looking for a usable Python (need $MinVersion or newer)"

$candidates = New-Object System.Collections.Generic.List[string]
if ($Python) {
    $candidates.Add($Python)
} else {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($tag in '3.14', '3.13', '3.12', '3.11', '3.10') {
            try {
                $exe = & py "-$tag" -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe) { $candidates.Add($exe.Trim()) }
            } catch { }
        }
    }
    foreach ($name in 'python3', 'python') {
        Get-Command $name -All -ErrorAction SilentlyContinue | ForEach-Object {
            # Skip the Microsoft Store stub, which is not a real interpreter.
            if ($_.Source -notlike '*WindowsApps*') { $candidates.Add($_.Source) }
        }
    }
}

if ($candidates.Count -eq 0) {
    Stop-WithError @"
no Python interpreter found.
    Install Python 3.10 or newer from https://www.python.org/downloads/
    and tick "Add python.exe to PATH" during setup.
"@
}

$PythonExe = $null
$rejected = @()
$seen = @{}
foreach ($candidate in $candidates) {
    if ($seen.ContainsKey($candidate)) { continue }
    $seen[$candidate] = $true

    # Every rejection is reported with its reason.  An earlier version of this
    # script skipped a candidate silently whenever the probe failed, so a
    # machine with Python 3.12 on PATH was told it had no Python 3.10 or newer
    # - a message naming the one thing that was not wrong.  Whatever goes wrong
    # next, the reason should be on the screen.
    $version = Get-PythonVersion $candidate
    if (-not $version) {
        $reason = "did not report its version - $script:LastProbe"
        $rejected += "$candidate : $reason"
        Write-Host "  $candidate $reason" -ForegroundColor DarkGray
        continue
    }
    if ($version -lt $MinVersion) {
        $rejected += "$candidate : Python $version, older than $MinVersion"
        Write-Host "  $candidate is Python $version, older than $MinVersion" -ForegroundColor DarkGray
        continue
    }
    if (Test-VenvWorks $candidate) {
        $PythonExe = $candidate
        Write-Ok "using $candidate (Python $version)"
        break
    }
    $reason = "Python $version, but cannot build a virtual environment - $script:LastProbe"
    $rejected += "$candidate : $reason"
    Write-Host "  $candidate $reason" -ForegroundColor DarkGray
}

if (-not $PythonExe) {
    $detail = if ($rejected) {
        "`n    Tried, and why each was rejected:`n      " + ($rejected -join "`n      ")
    } else { '' }
    Stop-WithError @"
no Python $MinVersion or newer that can build a virtual environment.$detail
    Install Python from https://www.python.org/downloads/ , ticking
    "Add python.exe to PATH", then run this script again.
"@
}

# --------------------------------------------------------------------------- #
# 2. Build the environment and install.
# --------------------------------------------------------------------------- #
Write-Step "Creating the virtual environment in .venv"
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
if (Test-Path $VenvPy) {
    $existing = Get-PythonVersion $VenvPy
    $wanted   = Get-PythonVersion $PythonExe
    if ($existing -eq $wanted) {
        Write-Ok "reusing the existing environment (Python $existing)"
    } else {
        Write-Warn "the existing .venv is Python $existing, but $wanted was selected; rebuilding it"
        Remove-Item -Recurse -Force $VenvDir
        & $PythonExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Stop-WithError "could not create the virtual environment." }
        Write-Ok "created"
    }
} else {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Stop-WithError "could not create the virtual environment." }
    Write-Ok "created"
}

Write-Step "Installing ClayQuant and its dependencies"
Write-Host "    this downloads numpy, scipy, dash and plotly; it takes a few minutes" -ForegroundColor DarkGray
& $VenvPy -m pip install --quiet --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Stop-WithError "could not upgrade pip inside the environment." }
& $VenvPy -m pip install --quiet -e "$ProjectDir[$Extras]"
if ($LASTEXITCODE -ne 0) {
    Stop-WithError "installation failed. If you are behind a proxy, set HTTPS_PROXY and try again."
}
Write-Ok "installed with extras: $Extras"

# --------------------------------------------------------------------------- #
# 3. Check that it actually works.
# --------------------------------------------------------------------------- #
Write-Step "Checking the installation"
$check = @'
import clayquant
from clayquant.models import available_phases, structure_directory
print(f"  ClayQuant {clayquant.__version__}")
have = available_phases()
n = sum(have.values())
print(f"  structure directory: {structure_directory()}")
if n == len(have):
    print(f"  all {n} clay structures found")
else:
    missing = [k for k, v in have.items() if not v]
    print(f"  {n} of {len(have)} clay structures found; missing: {', '.join(missing)}")
'@
$check | & $VenvPy -
if ($LASTEXITCODE -ne 0) { Stop-WithError "ClayQuant is installed but does not import correctly." }

# Each name in [project.scripts] becomes an .exe in the environment's Scripts
# directory, written at install time and only then.  Pulling a version that adds
# a command therefore does not create it: the environment keeps the set of
# commands it was built with, and the new one is not recognised even though its
# code is sitting in the checkout.  Re-installing writes them all, which has just
# happened above, so what is left is to prove each declared command is there.
Write-Step "Checking the commands"
$listing = & $VenvPy (Join-Path $ProjectDir 'scripts\list_commands.py') (Join-Path $ProjectDir 'pyproject.toml')
if ($LASTEXITCODE -ne 0) { Stop-WithError "could not read the command list from pyproject.toml." }
$missing = @()
foreach ($name in ($listing -split '\s+' | Where-Object { $_ })) {
    $exe = Join-Path $VenvDir "Scripts\$name.exe"
    if (Test-Path $exe) {
        & $exe --help *> $null
        if ($LASTEXITCODE -eq 0) { Write-Host "  $name" } else { $missing += $name }
    } else {
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    Stop-WithError ("these commands were not created or do not run: " + ($missing -join ' ') + [Environment]::NewLine +
        "     The environment is out of step with the source, which happens when a new" + [Environment]::NewLine +
        "     version adds a command. Run this script again, or delete $VenvDir first.")
}

if ($Extras -like '*dev*') {
    & $VenvPy -m pytest -q (Join-Path $ProjectDir 'tests') 2>&1 | Select-Object -Last 3
}
Write-Ok "ready"

# --------------------------------------------------------------------------- #
# Offer to put it on the desktop.
# --------------------------------------------------------------------------- #
Write-Step "Desktop shortcut"
Write-Host "     ClayQuant can be started from an icon instead of a terminal. Double-"
Write-Host "     clicking it starts the program and opens it in your default browser."
$makeShortcut = $true
if ($NoShortcut) {
    $makeShortcut = $false
    Write-Host "     Skipped (-NoShortcut)."
} elseif (-not $Yes) {
    $reply = Read-Host "`n     Create a desktop icon and a Start menu entry? [Y/n]"
    if ($reply -and $reply -notmatch '^(y|yes)$') { $makeShortcut = $false }
}
if ($makeShortcut) {
    & $VenvPy -m clayquant.desktop
    if ($LASTEXITCODE -ne 0) { Write-Warn "could not create the shortcuts." }
} else {
    Write-Host "     Not created. .\clayquant.ps1 shortcut does it later;"
    Write-Host "     .\clayquant.ps1 shortcut --remove takes them away again."
}

# --------------------------------------------------------------------------- #
# 4. Say what to do next.
# --------------------------------------------------------------------------- #
Write-Host @"

Done. To use ClayQuant, activate the environment first:

    $VenvDir\Scripts\Activate.ps1

then

    clayquant-gui                     open the interface in a browser
    clayquant-build-library -o library\clays.npz
    clayquant-import-structures your_structures.xml -o structures\phases.json

Or, without activating anything, from the project directory:

    .\clayquant.ps1 gui
    .\clayquant.ps1 build-library -o library\clays.npz
    .\clayquant.ps1 import-structures your_structures.xml -o structures\phases.json

which finds this environment itself. The installed commands are also there as
$VenvDir\Scripts\clayquant-gui.exe and so on.

If PowerShell refuses to run the activation script, allow local scripts once:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

If any clay structures were reported missing above, put the four ICSD CIF files
into $ProjectDir\structures\ ; see structures\README.md for their names.
"@
