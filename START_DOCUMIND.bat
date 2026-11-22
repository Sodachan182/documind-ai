@echo off
REM DocuMind AI - double-click to set up and launch.
setlocal
cd /d "%~dp0"
title DocuMind AI
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 (
  echo.
  echo DocuMind could not start. Read the messages above for the reason.
  pause
)
endlocal
