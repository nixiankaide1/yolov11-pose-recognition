@echo off
chcp 65001 >nul

set PYTHON_PATH=%~dp0py311\python.exe
"%PYTHON_PATH%" "%~dp0app.py"
pause