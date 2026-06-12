@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
    echo Die virtuelle Umgebung fehlt. Bitte zuerst installer_starten.bat ausfuehren.
    pause
    exit /b 1
)
.venv\Scripts\python.exe script.py
pause
