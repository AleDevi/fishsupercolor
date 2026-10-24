@echo off
rem Runs the whole training: prepare the photos, train the outline and landmark models, check them, write the
rem model files in the folder "results". If it stops, run it again: it continues where it was.
rem The data folder is the folder of this file, with the van der Bijl zips, photo_database.csv and foto_2000.
setlocal
set "KIT=%~dp0"
if not exist "%KIT%env\Scripts\python.exe" (
  echo  Run install_training.bat first.
  goto error
)
set "PYTHONPATH=%KIT%"
set "QUICK="
if /i "%~1"=="quick" set "QUICK=--quick"
echo.
echo  Guppy Colour training. It takes hours: leave the PC on ^(sleep off^). The log is in training_log.txt.
echo.
"%KIT%env\Scripts\python.exe" -m gctrain "%KIT%data" %QUICK% || goto error
echo.
pause
exit /b 0
:error
echo.
echo  Something went wrong. Copy the text above ^(or training_log.txt^) and send it to Claude.
pause
exit /b 1
