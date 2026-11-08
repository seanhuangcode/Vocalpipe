@echo off
rem Starts vocalpipe and opens it in your browser. Close this window to stop it.
setlocal
cd /d "%~dp0"
title vocalpipe
if not exist venv\Scripts\python.exe (
  echo vocalpipe isn't set up yet, so running setup first...
  call setup.bat
  if errorlevel 1 exit /b 1
)
set PYTHONUTF8=1
rem open the browser once the server has had a moment to start
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://localhost:8765"
venv\Scripts\python.exe serve.py %*
pause
