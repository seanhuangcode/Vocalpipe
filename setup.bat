@echo off
rem vocalpipe setup for Windows: Python packages, ffmpeg and the pitch model. Safe to run again.
setlocal
cd /d "%~dp0"
title vocalpipe setup
echo.
echo   vocalpipe setup
echo   ---------------
echo.

rem 1. Python 3.12 or newer
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.12 or newer is needed. Install it from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
  goto :fail
)

rem 2. ffmpeg, which reads and writes the audio
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg isn't installed yet. Installing it with winget...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  if errorlevel 1 (
    echo Couldn't install ffmpeg automatically. Download it from https://www.gyan.dev/ffmpeg/builds/
    echo and add its bin folder to PATH, then run setup.bat again.
    goto :fail
  )
  echo ffmpeg is installed. It will be found once you open a new window.
)

rem 3. A private Python environment with the app's packages
if not exist venv\Scripts\python.exe (
  echo Creating the Python environment in the venv folder...
  %PY% -m venv venv
  if errorlevel 1 goto :fail
)
set "VPY=venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA graphics card found: installing the GPU build of PyTorch, so songs process much faster...
  "%VPY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 echo The GPU build didn't install, so the standard build will be used instead.
)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem 4. The pitch model (181 MB, not in the repository)
if not exist models mkdir models
if not exist models\rmvpe.pt (
  echo Downloading the pitch model, 181 MB...
  curl.exe -L --fail -o models\rmvpe.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
  if errorlevel 1 (
    del models\rmvpe.pt 2>nul
    echo The download failed. Check your internet connection and run setup.bat again.
    goto :fail
  )
)

echo.
echo   All set. Double-click start.bat to open vocalpipe.
echo.
pause
exit /b 0

:fail
echo.
echo   Setup stopped. Fix the problem above, then run setup.bat again.
echo.
pause
exit /b 1
