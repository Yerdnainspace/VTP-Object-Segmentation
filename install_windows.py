import os
import queue
import json
import ssl
import subprocess
import shutil
import sys
import threading
import urllib.request
import zipfile
import tempfile
import re
from pathlib import Path

try:
    import customtkinter as ctk
except Exception:
    ctk = None
    import tkinter as tk
    from tkinter import ttk


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
SCRIPT_PATH = PROJECT_DIR / "script.py"
MEDIAPIPE_MODEL = PROJECT_DIR / "selfie_multiclass.tflite"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
CORRIDORKEY_SOURCE_ZIP_URL = "https://github.com/nikopueringer/CorridorKey/archive/refs/heads/main.zip"

BASE_PACKAGES = [
    "customtkinter",
    "numpy",
    "pillow",
    "opencv-contrib-python",
    "imageio-ffmpeg",
    "mediapipe",
    "transformers",
    "timm",
    "safetensors",
    "einops",
    "kornia",
    "ultralytics",
    "huggingface_hub",
    "comtypes",
]

ULTRALYTICS_SUPPORT_PACKAGES = [
    "matplotlib",
    "pyyaml",
    "requests",
    "scipy",
    "psutil",
    "polars",
    "ultralytics-thop",
]

PACKAGE_IMPORTS = {
    "customtkinter": "customtkinter",
    "numpy": "numpy",
    "pillow": "PIL",
    "opencv-contrib-python": "cv2",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "mediapipe": "mediapipe",
    "transformers": "transformers",
    "timm": "timm",
    "safetensors": "safetensors",
    "einops": "einops",
    "kornia": "kornia",
    "ultralytics": "ultralytics",
    "huggingface_hub": "huggingface_hub",
    "comtypes": "comtypes",
    "matplotlib": "matplotlib",
    "pyyaml": "yaml",
    "requests": "requests",
    "scipy": "scipy",
    "psutil": "psutil",
    "polars": "polars",
    "ultralytics-thop": "ultralytics_thop",
}

CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def project_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe"


def project_pythonw() -> Path:
    return VENV_DIR / "Scripts" / "pythonw.exe"


def run_command(args, log, cwd=PROJECT_DIR):
    log("> " + " ".join(str(part) for part in args))
    process = subprocess.Popen(
        [str(part) for part in args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        safe_line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line.rstrip())
        safe_line = "".join(ch for ch in safe_line if ch == "\t" or ch >= " ")
        safe_line = safe_line.encode("cp1252", errors="replace").decode("cp1252")
        safe_line = re.sub(r"\?{6,}", "...", safe_line)
        log(safe_line)
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"Befehl fehlgeschlagen ({rc}): {' '.join(str(part) for part in args)}")


def _run_quiet(args, timeout=8):
    try:
        return subprocess.run(
            [str(part) for part in args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception:
        return None


def scan_windows_video_controllers():
    ps_script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility,DriverVersion,PNPDeviceID | "
        "ConvertTo-Json -Depth 3"
    )
    result = _run_quiet(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script], timeout=12)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    gpus = []
    for item in data:
        name = str(item.get("Name") or "").strip()
        vendor = str(item.get("AdapterCompatibility") or "").strip()
        driver = str(item.get("DriverVersion") or "").strip()
        pnp = str(item.get("PNPDeviceID") or "").strip()
        if not name:
            continue
        is_nvidia = "nvidia" in (name + " " + vendor + " " + pnp).lower() or "ven_10de" in pnp.lower()
        is_rtx = "rtx" in name.lower()
        gpus.append(
            {
                "name": name,
                "vendor": vendor or "unbekannt",
                "driver": driver or "unbekannt",
                "pnp": pnp,
                "is_nvidia": is_nvidia,
                "is_rtx": is_rtx,
            }
        )
    return gpus


def scan_nvidia_smi():
    result = _run_quiet(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version",
            "--format=csv,noheader",
        ],
        timeout=8,
    )
    if result is None:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "name": parts[0] if len(parts) > 0 else "NVIDIA GPU",
                "driver": parts[1] if len(parts) > 1 else "unbekannt",
                "cuda": parts[2] if len(parts) > 2 else "unbekannt",
            }
        )
    return gpus


def hardware_scan():
    controllers = scan_windows_video_controllers()
    nvidia_smi = scan_nvidia_smi()
    nvidia_controller = next((gpu for gpu in controllers if gpu["is_nvidia"]), None)
    rtx_controller = next((gpu for gpu in controllers if gpu["is_nvidia"] and gpu["is_rtx"]), None)
    nvidia_smi_gpu = nvidia_smi[0] if nvidia_smi else None
    selected_name = None
    selected_driver = None
    selected_cuda = None

    if nvidia_smi_gpu:
        selected_name = nvidia_smi_gpu["name"]
        selected_driver = nvidia_smi_gpu["driver"]
        selected_cuda = nvidia_smi_gpu["cuda"]
    elif rtx_controller:
        selected_name = rtx_controller["name"]
        selected_driver = rtx_controller["driver"]
    elif nvidia_controller:
        selected_name = nvidia_controller["name"]
        selected_driver = nvidia_controller["driver"]

    return {
        "controllers": controllers,
        "nvidia_smi": nvidia_smi,
        "nvidia_available": bool(nvidia_smi_gpu or nvidia_controller),
        "rtx_available": bool(rtx_controller or (selected_name and "rtx" in selected_name.lower())),
        "selected_name": selected_name,
        "selected_driver": selected_driver,
        "selected_cuda": selected_cuda,
        "nvidia_smi_available": bool(nvidia_smi),
    }


def detect_nvidia_gpu():
    scan = hardware_scan()
    if not scan["nvidia_available"]:
        return None
    return {
        "name": scan["selected_name"] or "NVIDIA GPU",
        "driver": scan["selected_driver"] or "unbekannt",
        "cuda": scan["selected_cuda"] or "ueber PyTorch-CUDA-Wheel",
    }


def format_hardware_report(scan):
    lines = []
    if scan["controllers"]:
        lines.append("Windows-Grafikkarten:")
        for index, gpu in enumerate(scan["controllers"], start=1):
            marker = " [NVIDIA/RTX]" if gpu["is_nvidia"] and gpu["is_rtx"] else ""
            if gpu["is_nvidia"] and not gpu["is_rtx"]:
                marker = " [NVIDIA]"
            lines.append(f"{index}. {gpu['name']}{marker}")
            lines.append(f"   Hersteller: {gpu['vendor']} | Treiber: {gpu['driver']}")
    else:
        lines.append("Windows-Grafikkarten: keine WMI-Daten gefunden")

    if scan["nvidia_smi"]:
        lines.append("nvidia-smi:")
        for gpu in scan["nvidia_smi"]:
            lines.append(f"- {gpu['name']} | Treiber {gpu['driver']} | CUDA {gpu['cuda']}")
    else:
        lines.append("nvidia-smi: nicht verfuegbar oder liefert keine GPU. Windows-WMI wird als Fallback genutzt.")

    if scan["nvidia_available"]:
        lines.append(f"Auswahl fuer Installation: {scan['selected_name']} -> PyTorch mit CUDA")
    else:
        lines.append("Auswahl fuer Installation: CPU-PyTorch")
    return "\n".join(lines)


def ensure_venv(log):
    if project_python().exists():
        log(f"Virtuelle Umgebung gefunden: {VENV_DIR}")
        return
    log(f"Erstelle virtuelle Umgebung: {VENV_DIR}")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)], log)


def pip_install(log, packages, extra_args=None):
    args = [project_python(), "-m", "pip", "install", "--upgrade"]
    if extra_args:
        args.extend(extra_args)
    args.extend(packages)
    run_command(args, log)


def pip_install_no_upgrade(log, packages, extra_args=None):
    args = [project_python(), "-m", "pip", "install"]
    if extra_args:
        args.extend(extra_args)
    args.extend(packages)
    run_command(args, log)


def pip_uninstall(log, packages):
    args = [project_python(), "-m", "pip", "uninstall", "-y"]
    args.extend(packages)
    run_command(args, log)


def installed_package_version(package_name):
    code = (
        "import importlib.metadata as md, json\n"
        f"name = {json.dumps(package_name)}\n"
        "try:\n"
        "    print(json.dumps(md.version(name)))\n"
        "except md.PackageNotFoundError:\n"
        "    print('null')\n"
    )
    result = _run_quiet([project_python(), "-c", code], timeout=20)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def opencv_ximgproc_available():
    code = "import cv2, json; print(json.dumps(bool(hasattr(cv2, 'ximgproc'))))"
    result = _run_quiet([project_python(), "-c", code], timeout=20)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return False
    try:
        return bool(json.loads(result.stdout))
    except json.JSONDecodeError:
        return False


def ensure_opencv_contrib_last(log):
    plain_opencv = installed_package_version("opencv-python")
    contrib_opencv = installed_package_version("opencv-contrib-python")

    if not plain_opencv:
        log("OpenCV: installiere opencv-python-Metadaten fuer ultralytics.")
        pip_install_no_upgrade(log, ["opencv-python"], ["--no-deps"])
        plain_opencv = installed_package_version("opencv-python")

    if not contrib_opencv:
        log("opencv-contrib-python fehlt und wird installiert.")
        pip_install_no_upgrade(log, ["opencv-contrib-python"])
        contrib_opencv = installed_package_version("opencv-contrib-python")

    if plain_opencv and contrib_opencv:
        log(
            "OpenCV: opencv-python ist fuer ultralytics registriert; "
            "opencv-contrib-python wird als aktive OpenCV-Variante gesetzt."
        )
        pip_install_no_upgrade(log, ["opencv-contrib-python"], ["--force-reinstall", "--no-deps"])

    if opencv_ximgproc_available():
        log("OpenCV: ximgproc ist verfuegbar.")
    else:
        raise RuntimeError("OpenCV ximgproc fehlt trotz opencv-contrib-python Installation.")


def ensure_corridorkey_module(log):
    module_dir = PROJECT_DIR / "CorridorKeyModule"
    init_file = module_dir / "__init__.py"
    if init_file.exists():
        log(f"CorridorKeyModule gefunden: {module_dir}")
        return

    log("Lade CorridorKeyModule aus dem offiziellen CorridorKey-Projekt ...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "corridorkey.zip"
        urllib.request.urlretrieve(CORRIDORKEY_SOURCE_ZIP_URL, zip_path)
        extract_dir = temp_path / "extract"
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        candidates = list(extract_dir.glob("*/CorridorKeyModule"))
        if not candidates:
            raise RuntimeError("CorridorKeyModule wurde im heruntergeladenen Archiv nicht gefunden.")
        source_dir = candidates[0]
        if module_dir.exists():
            shutil.rmtree(module_dir)
        shutil.copytree(source_dir, module_dir)
    log(f"CorridorKeyModule installiert: {module_dir}")


def missing_python_packages(packages):
    code = (
        "import importlib.util, json\n"
        f"package_imports = {json.dumps(PACKAGE_IMPORTS)}\n"
        f"packages = {json.dumps(packages)}\n"
        "missing = []\n"
        "for package in packages:\n"
        "    import_name = package_imports.get(package, package.replace('-', '_'))\n"
        "    if importlib.util.find_spec(import_name) is None:\n"
        "        missing.append(package)\n"
        "print(json.dumps(missing))\n"
    )
    result = _run_quiet([project_python(), "-c", code], timeout=30)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return list(packages)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return list(packages)
    return [package for package in data if package in packages]


def install_missing_packages(log, packages):
    missing = missing_python_packages(packages)
    if not missing:
        log("Basispakete: alles bereits vorhanden, keine Nachinstallation noetig.")
        return []

    regular_missing = [package for package in missing if package != "ultralytics"]
    if regular_missing:
        log("Installiere nur fehlende Basispakete: " + ", ".join(regular_missing))
        pip_install(log, regular_missing)

    if "ultralytics" in missing:
        support_missing = missing_python_packages(ULTRALYTICS_SUPPORT_PACKAGES)
        if support_missing:
            log("Installiere fehlende YOLO-Abhaengigkeiten ohne OpenCV-Doppelinstallation: " + ", ".join(support_missing))
            pip_install(log, support_missing)
        log("Installiere ultralytics ohne automatische Zusatzpakete; opencv-contrib-python bleibt erhalten.")
        pip_install_no_upgrade(log, ["ultralytics"], ["--no-deps"])
    return missing


def installed_torch_status():
    if not project_python().exists():
        return {"installed": False, "cuda_available": False, "version": None, "cuda_version": None, "device": None}
    code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({\n"
        "        'installed': True,\n"
        "        'version': torch.__version__,\n"
        "        'cuda_available': bool(torch.cuda.is_available()),\n"
        "        'cuda_version': torch.version.cuda,\n"
        "        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "    }))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'installed': False, 'error': str(exc)}))\n"
    )
    result = _run_quiet([project_python(), "-c", code], timeout=20)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return {"installed": False, "cuda_available": False, "version": None, "cuda_version": None, "device": None}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"installed": False, "cuda_available": False, "version": None, "cuda_version": None, "device": None}


def log_torch_status(log, prefix="PyTorch"):
    status = installed_torch_status()
    if not status.get("installed"):
        log(f"{prefix}: nicht installiert")
        return status
    cuda_label = status.get("cuda_version") or "keine CUDA-Runtime"
    device_label = status.get("device") or "CPU"
    log(
        f"{prefix}: {status.get('version')} | CUDA verfuegbar: "
        f"{status.get('cuda_available')} | Runtime: {cuda_label} | Geraet: {device_label}"
    )
    return status


def download_mediapipe_model(log):
    if MEDIAPIPE_MODEL.exists() and MEDIAPIPE_MODEL.stat().st_size > 100_000:
        log("MediaPipe-Modell ist bereits vorhanden.")
        return
    log("Lade MediaPipe Selfie-Modell ...")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(MEDIAPIPE_MODEL_URL, context=ctx, timeout=120) as response:
        data = response.read()
    MEDIAPIPE_MODEL.write_bytes(data)
    log(f"MediaPipe-Modell gespeichert: {MEDIAPIPE_MODEL.name} ({len(data) // 1024} KiB)")


def warmup_models(log):
    warmup_script = PROJECT_DIR / "_installer_model_warmup.py"
    code = r"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("NO_COLOR", "1")
os.environ.setdefault("TERM", "dumb")
import logging
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

print("Pruefe PyTorch/CUDA ...", flush=True)
import torch
print("Torch:", torch.__version__, flush=True)
print("CUDA verfuegbar:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Lade BiRefNet in den HuggingFace-Cache ...", flush=True)
from transformers import AutoModelForImageSegmentation
AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)

print("Lade RVM ByteDance in den Torch-Cache ...", flush=True)
torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", pretrained=True, trust_repo=True, verbose=False)

print("Lade YOLO in den Ultralytics-Cache ...", flush=True)
from ultralytics import YOLO
YOLO("yolo11n-seg.pt")
YOLO("yolo11s-seg.pt")
YOLO("yolo11n.pt")

print("Lade CorridorKey-Modell in den lokalen Cache ...", flush=True)
from pathlib import Path
from huggingface_hub import hf_hub_download
from CorridorKeyModule import CorridorKeyEngine
corridor_dir = Path(__file__).resolve().parent / "CorridorKeyModule" / "checkpoints"
corridor_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id="nikopueringer/CorridorKey_v1.0",
    filename="CorridorKey_v1.0.safetensors",
    local_dir=str(corridor_dir),
)
print("Modelle sind vorbereitet.", flush=True)
"""
    warmup_script.write_text(code, encoding="utf-8")
    try:
        run_command([project_python(), str(warmup_script)], log)
    finally:
        try:
            warmup_script.unlink()
        except OSError:
            pass


def write_launchers(log):
    start_bat = PROJECT_DIR / "start_programm.bat"
    install_bat = PROJECT_DIR / "installer_starten.bat"
    start_bat.write_text(
        """@echo off
cd /d %~dp0
if not exist .venv\\Scripts\\python.exe (
    echo Die virtuelle Umgebung fehlt. Bitte zuerst installer_starten.bat ausfuehren.
    pause
    exit /b 1
)
.venv\\Scripts\\python.exe script.py
pause
""",
        encoding="utf-8",
    )
    install_bat.write_text(
        """@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0

set "PYTHON_CMD="

if not exist .venv\\Scripts\\python.exe (
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
if not exist .venv\\Scripts\\python.exe (
    echo Python wurde nicht gefunden. Bitte Python 3.10 oder neuer installieren.
    pause
    exit /b 1
)
.venv\\Scripts\\python.exe -m pip install --upgrade pip customtkinter
.venv\\Scripts\\python.exe install_windows.py
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
    "%LocalAppData%\\Programs\\Python\\Python312\\python.exe"
    "%LocalAppData%\\Programs\\Python\\Python311\\python.exe"
    "%LocalAppData%\\Programs\\Python\\Python310\\python.exe"
    "%ProgramFiles%\\Python312\\python.exe"
    "%ProgramFiles%\\Python311\\python.exe"
    "%ProgramFiles%\\Python310\\python.exe"
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
""",
        encoding="utf-8",
    )
    log(f"Starter geschrieben: {start_bat.name}")
    log(f"Installer-Starter geschrieben: {install_bat.name}")


class InstallerWorker:
    def __init__(self, log, set_status, use_cuda=True, preload_models=True):
        self.log = log
        self.set_status = set_status
        self.use_cuda = use_cuda
        self.preload_models = preload_models

    def run(self):
        scan = hardware_scan()
        gpu = detect_nvidia_gpu()
        self.log("Hardware-Erkennung:")
        self.log(format_hardware_report(scan))

        self.set_status("Installiere Python-Umgebung ...")
        ensure_venv(self.log)
        run_command([project_python(), "-m", "pip", "install", "--upgrade", "pip", "wheel"], self.log)
        run_command([project_python(), "-m", "pip", "install", "--upgrade", "setuptools<82"], self.log)

        self.set_status("Installiere PyTorch ...")
        torch_before = log_torch_status(self.log, "PyTorch vor Installation")
        if scan["nvidia_available"] and self.use_cuda:
            self.log(f"Installiere CUDA-PyTorch fuer RTX/NVIDIA ueber {CUDA_TORCH_INDEX}")
            torch_args = ["--index-url", CUDA_TORCH_INDEX]
            if torch_before.get("installed") and not torch_before.get("cuda_available"):
                self.log("Vorhandenes CPU-PyTorch wird durch CUDA-PyTorch ersetzt.")
                torch_args.append("--force-reinstall")
            pip_install(self.log, ["torch", "torchvision"], torch_args)
        else:
            self.log("Installiere CPU-PyTorch.")
            pip_install(self.log, ["torch", "torchvision"], ["--index-url", CPU_TORCH_INDEX])
        log_torch_status(self.log, "PyTorch nach Installation")

        self.set_status("Pruefe Basispakete ...")
        install_missing_packages(self.log, BASE_PACKAGES)
        ensure_opencv_contrib_last(self.log)

        self.set_status("Lade Modelle ...")
        download_mediapipe_model(self.log)
        ensure_corridorkey_module(self.log)
        if self.preload_models:
            try:
                warmup_models(self.log)
            except Exception as exc:
                self.log(f"WARNUNG: Modelle konnten nicht vollstaendig vorgeladen werden: {exc}")
                self.log("Die Installation wird fortgesetzt. Die Modelle koennen beim ersten Auswaehlen erneut laden.")
        else:
            self.log("Modell-Vorladen uebersprungen. Modelle laden beim ersten Auswaehlen nach.")

        self.set_status("Pruefe Installation ...")
        verify_code = (
            "import cv2, customtkinter, mediapipe, PIL, torch, ultralytics, CorridorKeyModule; "
            "print('OpenCV', cv2.__version__); "
            "print('OpenCV ximgproc', hasattr(cv2, 'ximgproc')); "
            "print('CUDA verfuegbar', torch.cuda.is_available()); "
            "print('Geraet', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        )
        run_command([project_python(), "-c", verify_code], self.log)
        run_command([project_python(), "-m", "pip", "check"], self.log)
        write_launchers(self.log)
        self.set_status("Fertig. Das Hauptprogramm kann gestartet werden.")


if ctk:
    class InstallerApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("AI Segmenter - Windows Installer")
            self.geometry("980x680")
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

            self.messages = queue.Queue()
            self.worker_thread = None
            self.hardware_scan = hardware_scan()
            self.gpu = detect_nvidia_gpu()

            self.left = ctk.CTkFrame(self, fg_color="#2b2b2b")
            self.left.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            self.right = ctk.CTkScrollableFrame(self, width=330)
            self.right.pack(side="right", fill="y", padx=(0, 20), pady=20)

            ctk.CTkLabel(self.left, text="Installationsprotokoll", font=ctk.CTkFont(size=18, weight="bold")).pack(
                anchor="w", padx=16, pady=(14, 8)
            )
            self.logbox = ctk.CTkTextbox(self.left, font=ctk.CTkFont(family="Consolas", size=12), wrap="word")
            self.logbox.pack(fill="both", expand=True, padx=16, pady=(0, 16))

            ctk.CTkLabel(self.right, text="Setup", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(12, 8))
            ctk.CTkLabel(self.right, text="Hardware", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10)
            self.hardware_label = ctk.CTkLabel(
                self.right,
                text=format_hardware_report(self.hardware_scan),
                justify="left",
                wraplength=290,
            )
            self.hardware_label.pack(anchor="w", padx=10, pady=(0, 14), fill="x")
            self.scan_button = ctk.CTkButton(self.right, text="Hardware neu scannen", command=self.refresh_hardware)
            self.scan_button.pack(fill="x", padx=10, pady=(0, 12))

            self.use_cuda = ctk.BooleanVar(value=bool(self.hardware_scan["nvidia_available"]))
            self.preload_models = ctk.BooleanVar(value=True)
            self.cuda_box = ctk.CTkCheckBox(
                self.right,
                text="RTX/CUDA fuer PyTorch nutzen",
                variable=self.use_cuda,
                state="normal" if self.hardware_scan["nvidia_available"] else "disabled",
            )
            self.cuda_box.pack(anchor="w", padx=10, pady=(4, 8))
            self.model_box = ctk.CTkCheckBox(
                self.right,
                text="AI-Modelle vorladen",
                variable=self.preload_models,
            )
            self.model_box.pack(anchor="w", padx=10, pady=(0, 14))

            self.status = ctk.CTkLabel(self.right, text="Bereit", justify="left", wraplength=290)
            self.status.pack(anchor="w", padx=10, pady=(0, 12), fill="x")
            self.progress = ctk.CTkProgressBar(self.right, mode="indeterminate")
            self.progress.pack(fill="x", padx=10, pady=(0, 12))

            self.install_button = ctk.CTkButton(self.right, text="Installation starten", command=self.start_install)
            self.install_button.pack(fill="x", padx=10, pady=5)
            self.launch_button = ctk.CTkButton(self.right, text="Hauptprogramm starten", command=self.launch_main, state="disabled")
            self.launch_button.pack(fill="x", padx=10, pady=5)

            self.after(100, self.drain_messages)
            self.log("Projekt: " + str(PROJECT_DIR))
            self.log("PyTorch CUDA-Wheels: " + CUDA_TORCH_INDEX)
            self.log("Initialer Hardware-Scan:")
            self.log(format_hardware_report(self.hardware_scan))

        def log(self, text):
            self.messages.put(("log", text))

        def set_status(self, text):
            self.messages.put(("status", text))

        def start_install(self):
            self.install_button.configure(state="disabled")
            self.launch_button.configure(state="disabled")
            self.progress.start()
            worker = InstallerWorker(self.log, self.set_status, self.use_cuda.get(), self.preload_models.get())

            def target():
                try:
                    worker.run()
                    self.messages.put(("done", None))
                except Exception as exc:
                    self.messages.put(("error", str(exc)))

            self.worker_thread = threading.Thread(target=target, daemon=True)
            self.worker_thread.start()

        def refresh_hardware(self):
            self.hardware_scan = hardware_scan()
            self.gpu = detect_nvidia_gpu()
            self.hardware_label.configure(text=format_hardware_report(self.hardware_scan))
            self.use_cuda.set(bool(self.hardware_scan["nvidia_available"]))
            self.cuda_box.configure(state="normal" if self.hardware_scan["nvidia_available"] else "disabled")
            self.log("Hardware-Scan aktualisiert:")
            self.log(format_hardware_report(self.hardware_scan))

        def launch_main(self):
            exe = project_pythonw() if project_pythonw().exists() else project_python()
            subprocess.Popen([str(exe), str(SCRIPT_PATH)], cwd=str(PROJECT_DIR))

        def drain_messages(self):
            try:
                while True:
                    kind, value = self.messages.get_nowait()
                    if kind == "log":
                        self.logbox.insert("end", value + "\n")
                        self.logbox.see("end")
                    elif kind == "status":
                        self.status.configure(text=value)
                    elif kind == "done":
                        self.progress.stop()
                        self.install_button.configure(state="normal")
                        self.launch_button.configure(state="normal")
                        self.status.configure(text="Fertig. Installation erfolgreich.")
                    elif kind == "error":
                        self.progress.stop()
                        self.install_button.configure(state="normal")
                        self.status.configure(text="Fehler: " + value)
                        self.logbox.insert("end", "\nFEHLER: " + value + "\n")
                        self.logbox.see("end")
            except queue.Empty:
                pass
            self.after(100, self.drain_messages)


else:
    class InstallerApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("AI Segmenter - Windows Installer")
            self.geometry("980x680")
            self.configure(bg="#1f1f1f")
            self.messages = queue.Queue()
            self.hardware_scan = hardware_scan()
            self.use_cuda = tk.BooleanVar(value=bool(self.hardware_scan["nvidia_available"]))
            self.preload_models = tk.BooleanVar(value=True)

            self.logbox = tk.Text(self, bg="#2b2b2b", fg="#f0f0f0", insertbackground="#f0f0f0")
            self.logbox.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            panel = tk.Frame(self, bg="#1f1f1f", width=330)
            panel.pack(side="right", fill="y", padx=(0, 20), pady=20)
            tk.Label(panel, text="Setup", bg="#1f1f1f", fg="#f0f0f0", font=("Segoe UI", 18, "bold")).pack(pady=10)
            tk.Label(
                panel,
                text=format_hardware_report(self.hardware_scan),
                bg="#1f1f1f",
                fg="#f0f0f0",
                wraplength=290,
                justify="left",
            ).pack(fill="x", pady=8)
            tk.Checkbutton(panel, text="RTX/CUDA nutzen", variable=self.use_cuda, bg="#1f1f1f", fg="#f0f0f0").pack(anchor="w")
            tk.Checkbutton(panel, text="Modelle vorladen", variable=self.preload_models, bg="#1f1f1f", fg="#f0f0f0").pack(anchor="w")
            self.status = tk.Label(panel, text="Bereit", bg="#1f1f1f", fg="#f0f0f0", wraplength=290, justify="left")
            self.status.pack(fill="x", pady=12)
            self.progress = ttk.Progressbar(panel, mode="indeterminate")
            self.progress.pack(fill="x", pady=8)
            self.install_button = ttk.Button(panel, text="Installation starten", command=self.start_install)
            self.install_button.pack(fill="x", pady=5)
            self.after(100, self.drain_messages)

        def log(self, text):
            self.messages.put(("log", text))

        def set_status(self, text):
            self.messages.put(("status", text))

        def start_install(self):
            self.install_button.configure(state="disabled")
            self.progress.start()
            worker = InstallerWorker(self.log, self.set_status, self.use_cuda.get(), self.preload_models.get())
            threading.Thread(target=lambda: self._run_worker(worker), daemon=True).start()

        def _run_worker(self, worker):
            try:
                worker.run()
                self.messages.put(("done", None))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        def drain_messages(self):
            try:
                while True:
                    kind, value = self.messages.get_nowait()
                    if kind == "log":
                        self.logbox.insert("end", value + "\n")
                        self.logbox.see("end")
                    elif kind == "status":
                        self.status.configure(text=value)
                    elif kind == "done":
                        self.progress.stop()
                        self.install_button.configure(state="normal")
                        self.status.configure(text="Fertig. Installation erfolgreich.")
                    elif kind == "error":
                        self.progress.stop()
                        self.install_button.configure(state="normal")
                        self.status.configure(text="Fehler: " + value)
                        self.logbox.insert("end", "\nFEHLER: " + value + "\n")
            except queue.Empty:
                pass
            self.after(100, self.drain_messages)


if __name__ == "__main__":
    if not sys.platform.startswith("win"):
        print("Dieser Installer ist fuer Windows gedacht.")
    app = InstallerApp()
    app.mainloop()
