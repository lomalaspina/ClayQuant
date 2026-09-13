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
    [switch] $NoDev
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

if ($Extras -like '*dev*') {
    & $VenvPy -m pytest -q (Join-Path $ProjectDir 'tests') 2>&1 | Select-Object -Last 3
}
Write-Ok "ready"

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

Without activating, the same commands work as:

    $VenvDir\Scripts\clayquant-gui.exe

If PowerShell refuses to run the activation script, allow local scripts once:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

If any clay structures were reported missing above, put the four ICSD CIF files
into $ProjectDir\structures\ ; see structures\README.md for their names.
"@
