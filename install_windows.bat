@echo off
rem Installs Guppy Colour: its own Python environment and a desktop icon. Just double-click.
rem Package versions are pinned, with SHA-256 hashes, in requirements-win.txt.
setlocal
set "APP=%~dp0"
set "HOME_APP=%LOCALAPPDATA%\GuppyColour"
set "VENV=%HOME_APP%\venv"
echo.
echo  Installing Guppy Colour (takes a few minutes, needs internet)
echo.
set "PY="
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3.13 -c "" >nul 2>nul && set "PY=py -3.13"
if not defined PY python -c "import sys; sys.exit(sys.version_info[:2] not in ((3, 12), (3, 13)))" >nul 2>nul && set "PY=python"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY="%ProgramFiles%\Python312\python.exe""
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not defined PY if exist "%ProgramFiles%\Python313\python.exe" set "PY="%ProgramFiles%\Python313\python.exe""
if not defined PY (
  echo  Python 3.12 or 3.13 is needed ^(from python.org^). Install it and run this file again.
  goto error
)
echo  Using Python: %PY%
if not exist "%HOME_APP%" mkdir "%HOME_APP%"
%PY% -m venv "%VENV%" || goto error
"%VENV%\Scripts\python.exe" -m pip install --require-hashes --only-binary=:all: -r "%APP%requirements-win.txt" || goto error
"%VENV%\Scripts\python.exe" -m pip install --no-deps --no-build-isolation "%APP%." || goto error
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
