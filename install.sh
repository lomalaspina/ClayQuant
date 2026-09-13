#!/usr/bin/env bash
#
# ClayQuant installer for Linux, macOS and WSL.
#
#   ./install.sh              set everything up, asking before it installs
#                             anything with the system package manager
#   ./install.sh -y           don't ask
#   ./install.sh --python /usr/bin/python3.12
#                             use a particular interpreter
#   ./install.sh --no-dev     skip the test and import extras
#   ./install.sh --no-shortcut
#                             do not offer to create desktop shortcuts
#
# It chooses an interpreter, builds the virtual environment, installs
# ClayQuant and its dependencies into it, and checks that the result works.
# Re-running it is safe.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
MIN_MAJOR=3
MIN_MINOR=10
EXTRAS="gui,import,dev"
ASSUME_YES=0
NO_SHORTCUT=0
CHOSEN_PYTHON=""

bold=$'\033[1m'; red=$'\033[31m'; green=$'\033[32m'; yellow=$'\033[33m'; dim=$'\033[2m'; off=$'\033[0m'
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$bold" "$*" "$off"; }
ok()   { printf '%s  ok%s  %s\n' "$green" "$off" "$*"; }
warn() { printf '%s  !!%s  %s\n' "$yellow" "$off" "$*"; }
die()  { printf '\n%s  error%s  %s\n' "$red" "$off" "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes)      ASSUME_YES=1; shift ;;
    --python)      CHOSEN_PYTHON="${2:-}"; shift 2 ;;
    --no-dev)      EXTRAS="gui,import"; shift ;;
    --no-shortcut) NO_SHORTCUT=1; shift ;;
    -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             die "unknown option $1 (try --help)" ;;
  esac
done

# --------------------------------------------------------------------------- #
# 1. Find an interpreter that can actually build a virtual environment.
#
# "python -m venv" runs whichever interpreter "python" happens to be, and each
# Python version carries its own venv and ensurepip with its own bundled
# wheels. On Ubuntu and WSL, python3 is often an older version than the one you
# installed, and its venv package is missing - which is what produces
#
#     The virtual environment was not created successfully because ensurepip is
#     not available. ... apt install python3.10-venv
#
# even when python3.12-venv is installed. So rather than trusting any one name,
# every candidate is tried newest first and actually tested.
# --------------------------------------------------------------------------- #

python_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true; }

version_ok() {
  local v major minor
  v="$(python_version "$1")"; [ -n "$v" ] || return 1
  major="${v%%.*}"; minor="${v##*.}"
  [ "$major" -gt "$MIN_MAJOR" ] && return 0
  [ "$major" -eq "$MIN_MAJOR" ] && [ "$minor" -ge "$MIN_MINOR" ]
}

# Ask a yes/no question, defaulting to yes. Prefers the terminal so the prompt
# still works when the script's stdin is a pipe; falls back to stdin, and
# declines if there is neither. /dev/tty can exist yet fail to open when there
# is no controlling terminal, so the test has to be an actual open.
confirm() {
  local prompt="$1" reply
  if [ "$ASSUME_YES" -eq 1 ]; then return 0; fi
  printf '\n     %s [Y/n] ' "$prompt"
  if { : </dev/tty; } 2>/dev/null; then
    read -r reply </dev/tty || reply=n
  else
    read -r reply || reply=n
  fi
  printf '\n'
  case "${reply:-y}" in [Yy]*|"") return 0 ;; *) return 1 ;; esac
}

# Can this interpreter build a venv with pip in it? Tested by building one.
venv_works() {
  local probe rc
  probe="$(mktemp -d)"
  "$1" -m venv "${probe}/v" >/dev/null 2>&1 && [ -x "${probe}/v/bin/python" ]
  rc=$?
  rm -rf "$probe"
  return $rc
}

step "Looking for a usable Python (need ${MIN_MAJOR}.${MIN_MINOR} or newer)"

candidates=()
if [ -n "$CHOSEN_PYTHON" ]; then
  candidates=("$CHOSEN_PYTHON")
else
  for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    # type -aP lists every match on PATH, newest-named first; command -v would
    # only give the first, which on Ubuntu is usually the old system python.
    while IFS= read -r path; do
      [ -n "$path" ] && candidates+=("$path")
    done < <(type -aP "$name" 2>/dev/null || true)
  done
fi
[ "${#candidates[@]}" -gt 0 ] || die "no Python interpreter found on PATH at all."

PYTHON=""
best_but_broken=""
seen=""
for candidate in "${candidates[@]}"; do
  real="$(readlink -f "$candidate" 2>/dev/null || echo "$candidate")"
  case " $seen " in *" $real "*) continue ;; esac
  seen="$seen $real"
  version_ok "$candidate" || continue
  if venv_works "$candidate"; then
    PYTHON="$candidate"
    ok "using $candidate (Python $(python_version "$candidate"))"
    break
  fi
  say "${dim}  $candidate (Python $(python_version "$candidate")) cannot build a virtual environment${off}"
  [ -z "$best_but_broken" ] && best_but_broken="$candidate"
done

# --------------------------------------------------------------------------- #
# 2. If nothing works, it is almost always a missing distro package. Name the
#    right one for the version we found, rather than a generic instruction.
# --------------------------------------------------------------------------- #
if [ -z "$PYTHON" ]; then
  if [ -z "$best_but_broken" ]; then
    die "no Python ${MIN_MAJOR}.${MIN_MINOR}+ found. Install one, e.g.
      sudo apt update && sudo apt install python3.12 python3.12-venv"
  fi

  version="$(python_version "$best_but_broken")"
  if command -v apt-get >/dev/null 2>&1; then
    pkg="python${version}-venv"
    warn "Python ${version} is installed but its virtual-environment support is not."
    say  "     The package that provides it is ${bold}${pkg}${off}."
    if confirm "Install it now with sudo?"; then reply=y; else reply=n; fi
    case "$reply" in
      [Yy]*|"")
        step "Installing ${pkg}"
        sudo apt-get update && sudo apt-get install -y "$pkg" python3-pip \
          || die "could not install ${pkg}. Install it by hand and re-run this script."
        venv_works "$best_but_broken" \
          || die "${pkg} is installed but ${best_but_broken} still cannot build a virtual environment."
        PYTHON="$best_but_broken"
        ok "using $PYTHON (Python ${version})"
        ;;
      *)
        die "cannot continue without it. Run:
      sudo apt install ${pkg}
    then run this script again."
        ;;
    esac
  else
    die "Python ${version} cannot build a virtual environment, and this system has no apt.
    On Fedora/RHEL:  sudo dnf install python3-virtualenv
    On Alpine:       sudo apk add python3-venv
    On macOS:        install Python from python.org or 'brew install python'"
  fi
fi

# WSL only: a virtual environment on a Windows drive is very slow and its
# scripts are often not executable, because /mnt/c is a DrvFs mount.
case "$PROJECT_DIR" in
  /mnt/[a-z]/*)
    warn "This project is on a Windows drive ($PROJECT_DIR)."
    say  "     Virtual environments there are slow and their launchers may not be"
    say  "     executable. Consider moving the project into the Linux filesystem,"
    say  "     for example ~/ClayQuant, and running this script again."
    ;;
esac

# --------------------------------------------------------------------------- #
# 3. Build the environment and install.
# --------------------------------------------------------------------------- #
step "Creating the virtual environment in .venv"
if [ -x "${VENV_DIR}/bin/python" ]; then
  existing="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
  wanted="$(python_version "$PYTHON")"
  if [ "$existing" = "$wanted" ]; then
    ok "reusing the existing environment (Python ${existing})"
  else
    warn "the existing .venv is Python ${existing}, but ${wanted} was selected; rebuilding it"
    rm -rf "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR" || die "could not create the virtual environment."
    ok "created"
  fi
else
  "$PYTHON" -m venv "$VENV_DIR" || die "could not create the virtual environment."
  ok "created"
fi

VENV_PY="${VENV_DIR}/bin/python"

step "Installing ClayQuant and its dependencies"
say "${dim}    this downloads numpy, scipy, dash and plotly; it takes a few minutes${off}"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel \
  || die "could not upgrade pip inside the environment."
"$VENV_PY" -m pip install --quiet -e "${PROJECT_DIR}[${EXTRAS}]" \
  || die "installation failed. If the network is behind a proxy, set HTTPS_PROXY and try again."
ok "installed with extras: ${EXTRAS}"

# --------------------------------------------------------------------------- #
# 4. Check that it actually works.
# --------------------------------------------------------------------------- #
step "Checking the installation"
"$VENV_PY" - <<'PY' || die "ClayQuant is installed but does not import correctly."
import clayquant
from clayquant.models import available_phases, structure_directory
from clayquant.mixed_layer import lognormal_csds
print(f"  ClayQuant {clayquant.__version__}")
have = available_phases()
n = sum(have.values())
print(f"  structure directory: {structure_directory()}")
if n == len(have):
    print(f"  all {n} clay structures found")
else:
    missing = [k for k, v in have.items() if not v]
    print(f"  {n} of {len(have)} clay structures found; missing: {', '.join(missing)}")
PY

# Each name in [project.scripts] becomes a file in the environment's bin
# directory, written at install time and only then.  Pulling a version that adds
# a command therefore does not create it: the environment keeps the set of
# commands it was built with, and the new one is "command not found" even though
# its code is sitting in the checkout.  Re-installing writes them all, which has
# just happened above, so what is left is to prove that each declared command is
# there and answers - and to say so plainly if one is not.
step "Checking the commands"
SCRIPTS=$("$VENV_PY" "${PROJECT_DIR}/scripts/list_commands.py" "${PROJECT_DIR}/pyproject.toml") \
  || die "could not read the command list from pyproject.toml."
MISSING=""
for name in $SCRIPTS; do
  if [ -x "${VENV_DIR}/bin/${name}" ] && "${VENV_DIR}/bin/${name}" --help >/dev/null 2>&1; then
    say "  ${name}"
  else
    MISSING="${MISSING} ${name}"
  fi
done
[ -n "$MISSING" ] && die "these commands were not created or do not run:${MISSING}
     The environment is out of step with the source, which happens when a new
     version adds a command. Run this script again, or delete ${VENV_DIR} first."

if [ "${EXTRAS#*dev}" != "$EXTRAS" ]; then
  "$VENV_PY" -m pytest -q "${PROJECT_DIR}/tests" 2>&1 | tail -3 || true
fi
ok "ready"

# --------------------------------------------------------------------------- #
# 5. Offer to put it on the desktop.
# --------------------------------------------------------------------------- #
step "Desktop shortcut"
say "     ClayQuant can be started from an icon instead of a terminal. Double-"
say "     clicking it starts the program and opens it in your default browser."
if [ "$NO_SHORTCUT" -eq 1 ]; then
  say "     Skipped (--no-shortcut)."
elif confirm "Create a desktop icon and a menu entry?"; then
  "$VENV_PY" -m clayquant.desktop || warn "could not create the shortcuts."
else
  say "     Not created. ${bold}./clayquant shortcut${off} does it later;"
  say "     ${bold}./clayquant shortcut --remove${off} takes them away again."
fi

# --------------------------------------------------------------------------- #
# 6. Say what to do next.
# --------------------------------------------------------------------------- #
cat <<EOF

${bold}Done.${off} To use ClayQuant, activate the environment first:

    ${bold}source ${VENV_DIR}/bin/activate${off}

then

    clayquant-gui                     open the interface in a browser
    clayquant-build-library -o library/clays.npz
    clayquant-import-structures your_structures.xml -o structures/phases.json

Or, without activating anything, from the project directory:

    ${bold}./clayquant gui${off}
    ./clayquant build-library -o library/clays.npz
    ./clayquant import-structures your_structures.xml -o structures/phases.json

which finds this environment itself. The installed commands are also there as
${VENV_DIR}/bin/clayquant-gui and so on.

If any clay structures were reported missing above, put the four ICSD CIF files
into ${PROJECT_DIR}/structures/ ; see structures/README.md for their names.
EOF
