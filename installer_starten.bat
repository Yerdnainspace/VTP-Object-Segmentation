@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0

set "PYTHON_CMD="

if not exist .venv\Scripts\python.exe (
    call :find_python
    if not defined PYTHON_CMD (
        echo Python wurde nicht gefunden.
        echo.
        echo Versuche Python 3.12 automatisch ueber winget zu installieren ...
        where winget >nul 2>nul
        if errorlevel 1 (
            echo winget wurde nicht gefunden.
            echo Bitte Python 3.10 oder neuer installieren:
            echo https://www.python.org/downloads/windows/
            echo Wichtig: Beim Installieren "Add python.exe to PATH" aktivieren.
            pause
            exit /b 1
        )
        winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
        call :find_python
    )
    if defined PYTHON_CMD (
        echo Erstelle virtuelle Umgebung mit: !PYTHON_CMD!
        !PYTHON_CMD! -m venv .venv
    )
)
if not exist .venv\Scripts\python.exe (
    echo Python wurde nicht gefunden. Bitte Python 3.10 oder neuer installieren.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip customtkinter
.venv\Scripts\python.exe install_windows.py
pause
exit /b 0

:find_python
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)

where python3 >nul 2>nul
if not errorlevel 1 (
    python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python3"
        exit /b 0
    )
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
) do (
    if exist "%%~P" (
        "%%~P" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_CMD="%%~P""
            exit /b 0
        )
    )
)

exit /b 1
