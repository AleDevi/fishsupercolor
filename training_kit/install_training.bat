@echo off
rem Installs the Guppy Colour training kit in its own Python environment (folder "env" next to this file).
rem With an NVIDIA card it installs PyTorch for NVIDIA (CUDA 13.0), otherwise the processor version.
rem Package versions are pinned, with SHA-256 hashes, in the requirements files. Just double-click.
setlocal
set "KIT=%~dp0"
set "APP=%KIT%app\"
set "VENV=%KIT%env"
echo.
echo  Installing the Guppy Colour training kit (needs internet, about 3 GB with an NVIDIA card)
echo.
set "PY="
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "PY=python"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY="%ProgramFiles%\Python312\python.exe""
if not defined PY (
  echo  Python 3.12 is needed ^(from python.org, tick "Add python.exe to PATH"^). Install it and run this file again.
  goto error
)
set "REQ=%APP%requirements-win.txt"
set "EXTRA="
nvidia-smi >nul 2>nul && (
  set "REQ=%KIT%requirements-train-cuda-win.txt"
  set "EXTRA=--extra-index-url https://download.pytorch.org/whl/cu130"
  echo  NVIDIA card found:
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
) || (
  echo  No NVIDIA card found: installing the processor version ^(fine for the short test, too slow for the real training^).
)
echo.
rem An environment made by this installer (marker file) with Python 3.12 is updated in place; anything else is
rem removed and made again.
set "MARK=%VENV%\guppytraining-installed.txt"
set "FRESH=1"
if exist "%MARK%" if exist "%VENV%\Scripts\python.exe" "%VENV%\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "FRESH="
if defined FRESH (
  if exist "%VENV%" rmdir /s /q "%VENV%"
  %PY% -m venv "%VENV%" || goto error
) else (
  echo  Updating the existing environment.
)
set "PIPOPT=--disable-pip-version-check --no-cache-dir"
echo  Downloading and installing the packages: 10-30 minutes. The window may stay on the same line for minutes.
"%VENV%\Scripts\python.exe" -m pip install %PIPOPT% --require-hashes --only-binary=:all: %EXTRA% -r "%REQ%" || goto error
"%VENV%\Scripts\python.exe" -m pip install %PIPOPT% --no-deps --no-build-isolation --force-reinstall "%APP%." || goto error
> "%MARK%" echo Installed by install_training.bat from "%REQ%" with hash checks.
echo.
"%VENV%\Scripts\python.exe" -c "import torch; ok = torch.cuda.is_available(); print('  GPU OK: ' + torch.cuda.get_device_name(0) if ok else '  No GPU in use: the processor will be used.')" || goto error
echo.
echo  Done. Now double-click start_training.bat
pause
exit /b 0
:error
echo.
echo  Something went wrong. Copy the text above and send it to Claude.
pause
exit /b 1
