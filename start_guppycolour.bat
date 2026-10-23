@echo off
rem Starts Guppy Colour without the icon (useful if the desktop icon does not open: it shows the errors).
cd /d "%LOCALAPPDATA%\GuppyColour"
"%LOCALAPPDATA%\GuppyColour\venv\Scripts\python.exe" -m guppycolor
pause
