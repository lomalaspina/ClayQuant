<#
ClayQuant, run from the checkout without activating anything.

    .\clayquant.ps1 gui
    .\clayquant.ps1 import-structures structures.xml -o structures\phases.json
    .\clayquant.ps1 build-library -o library\clays.npz

The installed commands live inside the virtual environment and need it on PATH,
which means activating it first.  This script finds the environment's own
interpreter and runs the module directly, so nothing has to be on PATH and no
command has to have been written at install time.  See the comments in the
POSIX script beside it for why both of those go wrong.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)] [string[]] $Rest
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
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Show-Usage {
    Write-Host @'
usage: .\clayquant.ps1 <command> [options]

  gui                 open the quantification workflow in a browser
  import-structures   convert a TOPAS structure library into a phase database
  build-library       calculate the reference pattern library
  setup               the same two steps in a window, with file pickers
                      (setup import-structures, setup build-library)
  shortcut            put ClayQuant and the two setup windows on the desktop
                      and in the Start menu
  python              run the environment's interpreter

Options after the command are passed through unchanged; --help on any of them
describes it.
'@
}

$python = $null
foreach ($candidate in @(
        $(if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }),
        (Join-Path $here '.venv\Scripts\python.exe'),
        (Join-Path $here '.venv/bin/python'))) {
    if ($candidate -and (Test-Path $candidate)) { $python = $candidate; break }
}
if (-not $python) {
    Write-Error "no virtual environment found at $here\.venv - run $here\install.ps1 first."
    exit 1
}

if (-not $Command) { Show-Usage; exit 0 }

$module = switch ($Command) {
    'gui'               { 'clayquant.gui.app' }
    'import-structures' { 'clayquant.bern' }
    'build-library'     { 'clayquant.library' }
    'setup'             { 'clayquant.tools' }
    'shortcut'          { 'clayquant.desktop' }
    'python'            { & $python @Rest; exit $LASTEXITCODE }
    { $_ -in '-h', '--help', 'help' } { Show-Usage; exit 0 }
    default {
        Write-Host "clayquant: unknown command '$Command'`n"
        Show-Usage
        exit 2
    }
}

& $python -c 'import clayquant' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "ClayQuant is not installed in that environment - run $here\install.ps1."
    exit 1
}

& $python -m $module @Rest
exit $LASTEXITCODE
