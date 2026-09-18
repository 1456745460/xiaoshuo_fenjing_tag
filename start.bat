@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Novel Snapshot GUI
echo.

if not exist "snapshot_gui.py" (
  echo [ERROR] snapshot_gui.py not found. Put this script in the project root.
  goto :fail
)

set "PY_CMD="

py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=py -3"
  goto :have_python
)

python -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=python"
  goto :have_python
)

python3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=python3"
  goto :have_python
)

echo [ERROR] Python 3 not found.
echo Install Python 3 and check "Add python.exe to PATH".
echo Download: https://www.python.org/downloads/
goto :fail

:have_python
%PY_CMD% -c "import tkinter" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] tkinter is missing, so the GUI cannot start.
  echo Reinstall official Python and keep tcl/tk plus "Add python.exe to PATH".
  goto :fail
)

echo Starting GUI...
%PY_CMD% "snapshot_gui.py"
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
pause
exit /b 1
