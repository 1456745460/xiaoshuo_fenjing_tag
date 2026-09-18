#!/usr/bin/env bash
set -u
cd "$(cd "$(dirname "$0")" && pwd)"

fail() {
  echo "[ERROR] $*" >&2
  echo
  read -r -p "Press Enter to exit..." _ || true
  exit 1
}

echo
echo "Novel Snapshot GUI"
echo

if [[ ! -f snapshot_gui.py ]]; then
  fail "snapshot_gui.py not found. Put this script in the project root."
fi

PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info[0] >= 3 else 1)" >/dev/null 2>&1; then
      PYTHON="$cand"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "macOS: brew install python python-tk" >&2
  echo "Ubuntu/Debian: sudo apt install python3 python3-tk" >&2
  fail "Python 3 not found."
fi

if ! "$PYTHON" -c "import tkinter" >/dev/null 2>&1; then
  echo "macOS: brew install python-tk" >&2
  echo "Ubuntu/Debian: sudo apt install python3-tk" >&2
  fail "tkinter is missing, so the GUI cannot start."
fi

echo "Starting GUI..."
exec "$PYTHON" snapshot_gui.py
