<#
    ClayQuant installer for Windows PowerShell.

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
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir '.venv'
$Extras     = if ($NoDev) { 'gui,import' } else { 'gui,import,dev' }
$MinVersion = [Version]'3.10'

function Write-Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  ok  " -ForegroundColor Green -NoNewline; Write-Host $text }
function Write-Warn($text) { Write-Host "  !!  " -ForegroundColor Yellow -NoNewline; Write-Host $text }
function Stop-WithError($text) { Write-Host ""; Write-Host "  error  $text" -ForegroundColor Red; exit 1 }

function Get-PythonVersion($exe) {
    try {
        $v = & $exe -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return [Version]$v.Trim() }
    } catch { }
    return $null
}

function Test-VenvWorks($exe) {
    $probe = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
    try {
        & $exe -m venv $probe 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $probe 'Scripts\python.exe')))
    } catch {
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
$tooOld = @()
$seen = @{}
foreach ($candidate in $candidates) {
    if ($seen.ContainsKey($candidate)) { continue }
    $seen[$candidate] = $true
    $version = Get-PythonVersion $candidate
    if (-not $version) { continue }
    if ($version -lt $MinVersion) { $tooOld += "$candidate (Python $version)"; continue }
    if (Test-VenvWorks $candidate) {
        $PythonExe = $candidate
        Write-Ok "using $candidate (Python $version)"
        break
    }
    Write-Host "  $candidate (Python $version) cannot build a virtual environment" -ForegroundColor DarkGray
}

if (-not $PythonExe) {
    $detail = if ($tooOld) { "`n    Found but too old: " + ($tooOld -join ', ') } else { '' }
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
