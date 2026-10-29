@echo off
rem Installs Guppy Colour: its own Python environment and a desktop icon. Just double-click.
rem With an NVIDIA card it installs PyTorch for the card (CUDA 13.0, requirements-win-cuda.txt), so outlines and
rem landmarks are much faster; otherwise the processor version (requirements-win.txt). Versions are pinned with SHA-256.
setlocal
set "APP=%~dp0"
set "HOME_APP=%LOCALAPPDATA%\GuppyColour"
set "VENV=%HOME_APP%\venv"
echo.
echo  Installing Guppy Colour (takes a few minutes, needs internet)
echo.
set "PY="
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "PY=python"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY="%ProgramFiles%\Python312\python.exe""
if not defined PY (
  echo  Python 3.12 is needed ^(from python.org^). Install it and run this file again.
  goto error
)
echo  Using Python: %PY%
set "REQ=%APP%requirements-win.txt"
set "EXTRA="
set "SIZE=1.5"
nvidia-smi >nul 2>nul && (
  set "REQ=%APP%requirements-win-cuda.txt"
  set "EXTRA=--extra-index-url https://download.pytorch.org/whl/cu130"
  set "SIZE=4"
  echo  NVIDIA card found: installing the version that uses it.
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
) || (
  echo  No NVIDIA card found: installing the processor version.
)
if not exist "%HOME_APP%" mkdir "%HOME_APP%"
rem An installation made by this installer (marker file) with Python 3.12 is updated in place: packages already
rem there are not downloaded again. Anything else is removed and installed from scratch.
set "MARK=%VENV%\guppycolour-installed.txt"
set "FRESH=1"
if exist "%MARK%" if exist "%VENV%\Scripts\python.exe" "%VENV%\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "FRESH="
if defined FRESH (
  if exist "%VENV%" rmdir /s /q "%VENV%"
  %PY% -m venv "%VENV%" || goto error
) else (
  echo  Updating the existing installation.
)
rem No pip cache (saves about 1 GB of disk) and no "new pip version" notice.
set "PIPOPT=--disable-pip-version-check --no-cache-dir"
echo.
if defined FRESH (
  echo  Downloading and installing the Python packages: about %SIZE% GB, 10-30 minutes on a slow connection.
  echo  The window may stay on the same line for minutes ^(e.g. "Installing collected packages"^): it is working.
) else (
  echo  Checking the Python packages: only changed ones are downloaded.
)
echo.
"%VENV%\Scripts\python.exe" -m pip install %PIPOPT% --require-hashes --only-binary=:all: %EXTRA% -r "%REQ%" || goto error
"%VENV%\Scripts\python.exe" -m pip install %PIPOPT% --no-deps --no-build-isolation --force-reinstall "%APP%." || goto error
> "%MARK%" echo Installed by install_windows.bat from "%REQ%" with hash checks.
"%VENV%\Scripts\python.exe" -c "import torch; ok = torch.cuda.is_available(); print('  GPU OK: ' + torch.cuda.get_device_name(0) if ok else '  No GPU in use: the processor will be used.')"
powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Guppy Colour.lnk');" ^
  "$s.TargetPath = '%VENV%\Scripts\pythonw.exe'; $s.Arguments = '-m guppycolor'; $s.WorkingDirectory = '%HOME_APP%'; $s.Save()" || goto error
echo.
echo  Done. The "Guppy Colour" icon is on your desktop.
pause
exit /b 0
:error
echo.
echo  Something went wrong. Copy the text above and send it to Claude.
pause
exit /b 1
