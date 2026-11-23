@echo off
rem Launch the interview copilot hub on Windows (the counterpart of scripts/launch_copilot.sh):
rem one process = the opening menu (Start a call / Train / Review / Generate context / Settings)
rem plus every mode view. It opens the browser itself. This is what the Desktop shortcut runs
rem (create it with scripts\install_desktop_shortcut.ps1).
rem
rem   launch_copilot.bat [session_id]   optional: preselect this context bundle on the Call screen
rem   set COPILOT_SESSION=<id>          same, via the environment
rem   set COPILOT_PYTHON=<path>         a different interpreter (default: .venv, else python on PATH)
rem
rem Keep this window open during a call; Ctrl-C here (or closing it) stops everything and releases
rem the mic. The hub MUST run in a console window: stopping transcription sends Ctrl-Break to the
rem recorder, which only works between processes that share a console. Started without one
rem (pythonw, a hidden window) the stop falls back to a hard kill and the transcript loses its
rem closing lines.
setlocal EnableExtensions
cd /d "%~dp0.."

if defined COPILOT_PYTHON (
  set "PY=%COPILOT_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
  set "PY=%CD%\.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

set "SESSION=%~1"
if not defined SESSION set "SESSION=%COPILOT_SESSION%"
set "OPEN_PATH=/"
if defined SESSION set "OPEN_PATH=/call?session=%SESSION%"

title Interview Copilot

rem Keep the local model resident for the whole call: Ollama unloads it after 5 idle minutes and a
rem cold reload costs ~10 s on the next question. /b keeps the pinger in this console; it exits by
rem itself when the hub stops (see scripts\ollama_keepalive.py).
start "" /b "%PY%" scripts\ollama_keepalive.py --follow-hub

"%PY%" scripts\app_hub.py --open-path "%OPEN_PATH%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo The copilot stopped with exit code %RC%. Read the messages above; logs are in the logs folder.
  pause
)
endlocal & exit /b %RC%
