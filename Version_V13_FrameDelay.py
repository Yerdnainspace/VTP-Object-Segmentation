import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
import threading
import importlib.util
import sys
import urllib.request
import ssl
import subprocess
import shutil
import time
import logging
import re
import warnings
import contextlib
import ctypes
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")


@contextlib.contextmanager
def quiet_terminal_output():
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "w") as devnull:
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _select_torch_device(torch):
    """
    Pick the best available torch device.

    Order:
      1) CUDA (NVIDIA)
      2) DirectML (Windows, AMD/Intel/NVIDIA via torch-directml) if installed
      3) CPU

    Returns: (device_obj, label, hint)
      - device_obj: torch.device-like object for .to(...)
      - label: short human-readable label
      - hint: optional install hint if we're on CPU due to missing GPU backend
    """
    hint = None

    try:
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA", None
    except Exception:
        pass

    # Optional: DirectML backend on Windows (torch-directml).
    if sys.platform.startswith("win"):
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "DirectML", None
        except Exception:
            # Keep going to CPU.
            hint = (
                "Kein CUDA/DirectML-Backend verfuegbar, falle auf CPU zurueck. "
                "NVIDIA: installiere ein CUDA-faehiges PyTorch (siehe https://pytorch.org). "
                "AMD/Intel (Windows): 'pip install torch-directml' probieren."
            )

    return torch.device("cpu"), "CPU", hint


def _ensure_odd_ksize(k: int, min_k: int = 1) -> int:
    k = int(k)
    if k < min_k:
        k = min_k
    if k % 2 == 0:
        k += 1
    return k


def _keep_largest_component(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    mask_u8: 0/255 uint8
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8, copy=False)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_u8

    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    if int(stats[best_idx, cv2.CC_STAT_AREA]) < int(min_area):
        return mask_u8

    out = np.zeros_like(mask_u8)
    out[labels == best_idx] = 255
    return out


def _guided_refine_alpha(rgb_u8: np.ndarray, alpha_f32: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Edge-aware refine of alpha using guided filter (OpenCV ximgproc).
    """
    try:
        guide = rgb_u8.astype(np.float32) / 255.0
        src = alpha_f32.astype(np.float32, copy=False)
        refined = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=int(radius), eps=float(eps))
        return np.clip(refined, 0.0, 1.0).astype(np.float32, copy=False)
    except Exception:
        # Fallback: mild bilateral filter on alpha only (still helps with shimmering).
        a = (alpha_f32 * 255.0).astype(np.uint8)
        a = cv2.bilateralFilter(a, d=5, sigmaColor=40, sigmaSpace=6)
        return (a.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _despill_green(rgb_u8: np.ndarray, alpha_f32: np.ndarray) -> np.ndarray:
    """
    Simple green-spill suppression near edges (for preview/greenscreen mode).
    """
    rgb = rgb_u8.astype(np.float32)
    a = alpha_f32.astype(np.float32)

    # Edge region: where alpha is neither fully foreground nor background.
    edge = (a > 0.15) & (a < 0.95)
    if not np.any(edge):
        return rgb_u8

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Only reduce excessive green; clamp towards max(R,B).
    rb_max = np.maximum(r, b)
    excess = g - rb_max
    excess = np.maximum(excess, 0.0)

    strength = 0.65  # conservative
    g2 = g.copy()
    g2[edge] = g[edge] - excess[edge] * strength
    rgb[:, :, 1] = np.clip(g2, 0.0, 255.0)
    return rgb.astype(np.uint8)


def get_camera_backends():
    if sys.platform == "darwin":
        return [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
    if sys.platform.startswith("win"):
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
    return [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Auto"),
    ]


def open_camera(index):
    for backend_id, backend_name in get_camera_backends():
        cap = cv2.VideoCapture(index, backend_id)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                return cap, backend_name
            time.sleep(0.05)

        cap.release()

    return None, None


def measure_camera_input_fps(cap, sample_seconds=1.2, max_frames=90):
    """
    Estimate the actual incoming camera FPS from blocking read intervals.
    CAP_PROP_FPS is often unreliable for capture devices such as BMD/Blackmagic.
    """
    timestamps = []
    deadline = time.perf_counter() + float(sample_seconds)

    while len(timestamps) < int(max_frames) and time.perf_counter() < deadline:
        ok, _ = cap.read()
        if not ok:
            break
        timestamps.append(time.perf_counter())

    if len(timestamps) < 3:
        return 0.0

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0

    return (len(timestamps) - 1) / elapsed


def get_windows_camera_names():
    if not sys.platform.startswith("win"):
        return []

    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { "
        "$_.PNPClass -eq 'Camera' -or "
        "($_.PNPClass -eq 'Image' -and $_.Name -match 'camera|webcam|capture|video|BMD|Blackmagic') "
        "} | Select-Object -ExpandProperty Name"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    names = []
    seen = set()
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_camera_choice(index, device_names):
    if index < len(device_names):
        return f"{device_names[index]} (Kamera {index})"
    if len(device_names) == 1:
        return f"{device_names[0]} (Kamera {index})"
    return f"Kamera {index}"


def parse_camera_index(choice):
    match = re.search(r"\(Kamera\s+(\d+)\)\s*$", choice)
    if match:
        return int(match.group(1))
    match = re.search(r"Kamera\s+(\d+)", choice)
    if match:
        return int(match.group(1))
    raise ValueError(f"Kamera-Index nicht gefunden: {choice}")


def get_available_cameras():
    cameras = []
    device_names = get_windows_camera_names()
    for i in range(10):
        cap, _ = open_camera(i)
        if cap is not None:
            cameras.append(format_camera_choice(i, device_names))
            cap.release()
    return cameras if cameras else ["Keine Kamera gefunden"]


MODEL_OPTIONS = ["MediaPipe Selfie", "BiRefNet", "RVM ByteDance", "YOLO"]
MAIN_AI_DEVICE_OPTIONS = ["Automatisch", "CUDA", "CPU"]
YOLO_MODEL_OPTIONS = ["yolo11n-seg.pt", "yolo11s-seg.pt"]
YOLO_DEVICE_OPTIONS = ["Automatisch", "CUDA", "CPU"]
YOLO_POSTPROCESS_MODEL = "yolo11n.pt"
CORRIDORKEY_DEVICE_OPTIONS = ["Automatisch", "CUDA", "CPU"]
CORRIDORKEY_CHECKPOINT_REPO = "nikopueringer/CorridorKey_v1.0"
CORRIDORKEY_CHECKPOINT_FILE = "CorridorKey_v1.0.safetensors"
CORRIDORKEY_IMG_SIZE = 2048

DECKLINK_SDK_DLL_PATHS = [
    r"C:\Program Files (x86)\Blackmagic Design\Blackmagic Desktop Video\DeckLinkAPI.dll",
    r"C:\Program Files\Blackmagic Design\Blackmagic Desktop Video\DeckLinkAPI.dll",
    r"C:\Windows\System32\DeckLinkAPI.dll",
]

DECKLINK_OUTPUT_MODES = {
    "1080p25 - 1920x1080": ("bmdModeHD1080p25", 1920, 1080, 25.0),
    "1080p29.97 - 1920x1080": ("bmdModeHD1080p2997", 1920, 1080, 30000.0 / 1001.0),
    "1080p30 - 1920x1080": ("bmdModeHD1080p30", 1920, 1080, 30.0),
    "1080p50 - 1920x1080": ("bmdModeHD1080p50", 1920, 1080, 50.0),
    "1080p59.94 - 1920x1080": ("bmdModeHD1080p5994", 1920, 1080, 60000.0 / 1001.0),
    "1080p60 - 1920x1080": ("bmdModeHD1080p6000", 1920, 1080, 60.0),
    "720p50 - 1280x720": ("bmdModeHD720p50", 1280, 720, 50.0),
    "720p59.94 - 1280x720": ("bmdModeHD720p5994", 1280, 720, 60000.0 / 1001.0),
    "720p60 - 1280x720": ("bmdModeHD720p60", 1280, 720, 60.0),
}


def load_decklink_api():
    try:
        from comtypes import client
    except ImportError as exc:
        raise RuntimeError("DeckLink-Ausgabe benoetigt comtypes. Installiere es mit pip install comtypes.") from exc

    errors = []
    for path in ["DeckLinkAPI.dll", *DECKLINK_SDK_DLL_PATHS]:
        try:
            client.GetModule(path)
            import comtypes.gen.DeckLinkAPI as decklink_api
            return decklink_api
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise RuntimeError(
        "DeckLinkAPI.dll konnte nicht geladen werden. Blackmagic Desktop Video muss installiert sein.\n"
        + "\n".join(errors[-3:])
    )


def get_decklink_output_devices():
    try:
        return DeckLinkLiveOutput.list_devices()
    except Exception:
        return ["Keine DeckLink-Ausgabe gefunden"]


def get_decklink_input_devices():
    try:
        return DeckLinkLiveInput.list_devices()
    except Exception:
        return []


def get_live_input_sources():
    sources = get_available_cameras()
    if sources == ["Keine Kamera gefunden"]:
        sources = []
    sources.extend([f"DeckLink: {name}" for name in get_decklink_input_devices()])
    return sources if sources else ["Keine Live-Quelle gefunden"]


def run_with_timeout(func, fallback, timeout=6.0):
    result = {"value": fallback}

    def worker():
        try:
            result["value"] = func()
        except Exception:
            result["value"] = fallback

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout))
    return result["value"] if not thread.is_alive() else fallback


class DeckLinkLiveOutput:
    def __init__(self, device_name, mode_label, status_callback=None):
        self.device_name = device_name
        self.mode_label = mode_label
        self.status_callback = status_callback
        _, self.width, self.height, self.fps = DECKLINK_OUTPUT_MODES[mode_label]
        self._lock = threading.Lock()
        self._latest_rgb_or_rgba = None
        self._last_rgb_or_rgba = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._error = None

    @staticmethod
    def list_devices():
        import comtypes
        from comtypes.client import CreateObject

        comtypes.CoInitialize()
        try:
            decklink_api = load_decklink_api()
            iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
            devices = []
            while True:
                try:
                    decklink = iterator.Next()
                except Exception:
                    break
                if not decklink:
                    break

                try:
                    attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                    io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                    if not (io_support & int(decklink_api.bmdDeviceSupportsPlayback)):
                        continue
                    devices.append(str(decklink.GetDisplayName()))
                except Exception:
                    continue

            return devices if devices else ["Keine DeckLink-Ausgabe gefunden"]
        finally:
            comtypes.CoUninitialize()

    def start(self):
        self._thread = threading.Thread(target=self._run, name="DeckLinkLiveOutput", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=3.0)
        if self._error:
            raise self._error

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def write(self, frame_rgb_or_rgba):
        if self._thread is None or not self._thread.is_alive():
            return
        if frame_rgb_or_rgba is None:
            return
        with self._lock:
            self._latest_rgb_or_rgba = np.ascontiguousarray(frame_rgb_or_rgba)

    def _set_status(self, text):
        if self.status_callback:
            self.status_callback(text)

    def _select_device(self, decklink_api):
        from comtypes.client import CreateObject

        iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
        first_playback_device = None
        while True:
            try:
                decklink = iterator.Next()
            except Exception:
                break
            if not decklink:
                break

            try:
                attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                if not (io_support & int(decklink_api.bmdDeviceSupportsPlayback)):
                    continue
                display_name = str(decklink.GetDisplayName())
            except Exception:
                continue

            if first_playback_device is None:
                first_playback_device = decklink
            if display_name == self.device_name:
                return decklink, display_name

        if first_playback_device is not None:
            return first_playback_device, str(first_playback_device.GetDisplayName())
        raise RuntimeError("Keine DeckLink-Ausgabekarte gefunden.")

    def _frame_to_bgra(self, frame):
        frame = np.asarray(frame)
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError("DeckLink-Ausgabe erwartet RGB oder RGBA Frames.")

        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)

    def _write_decklink_frame(self, decklink_output, frame):
        bgra = np.ascontiguousarray(self._frame_to_bgra(frame), dtype=np.uint8)
        video_frame = decklink_output.CreateVideoFrame(
            self.width,
            self.height,
            self.width * 4,
            self._decklink_api.bmdFormat8BitBGRA,
            self._decklink_api.bmdFrameFlagDefault,
        )
        buffer_ptr = video_frame.GetBytes()
        if hasattr(buffer_ptr, "value"):
            buffer_ptr = buffer_ptr.value
        ctypes.memmove(buffer_ptr, bgra.ctypes.data, bgra.nbytes)
        decklink_output.DisplayVideoFrameSync(video_frame)

    def _run(self):
        import comtypes

        comtypes.CoInitialize()
        decklink_output = None
        try:
            self._decklink_api = load_decklink_api()
            decklink, actual_device_name = self._select_device(self._decklink_api)
            decklink_output = decklink.QueryInterface(self._decklink_api.IDeckLinkOutput_v14_2_1)
            mode_name, _, _, _ = DECKLINK_OUTPUT_MODES[self.mode_label]
            display_mode = getattr(self._decklink_api, mode_name)

            actual_mode, supported = decklink_output.DoesSupportVideoMode(
                self._decklink_api.bmdVideoConnectionUnspecified,
                display_mode,
                self._decklink_api.bmdFormat8BitBGRA,
                self._decklink_api.bmdNoVideoOutputConversion,
                self._decklink_api.bmdSupportedVideoModeDefault,
            )
            if not supported:
                raise RuntimeError(f"{actual_device_name} unterstuetzt {self.mode_label} mit BGRA nicht.")

            decklink_output.EnableVideoOutput(display_mode, self._decklink_api.bmdVideoOutputFlagDefault)
            self._set_status(f"DeckLink aktiv: {actual_device_name}\n{self.mode_label}")
            self._ready_event.set()

            black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._last_rgb_or_rgba = black
            period = 1.0 / max(1.0, float(self.fps))
            next_frame_time = time.perf_counter()

            while not self._stop_event.is_set():
                with self._lock:
                    frame = self._latest_rgb_or_rgba
                    self._latest_rgb_or_rgba = None
                if frame is None:
                    frame = self._last_rgb_or_rgba
                else:
                    self._last_rgb_or_rgba = frame

                self._write_decklink_frame(decklink_output, frame)
                next_frame_time += period
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
                else:
                    next_frame_time = time.perf_counter()
        except Exception as exc:
            self._error = exc
            self._set_status(f"DeckLink Fehler: {exc}")
            self._ready_event.set()
        finally:
            if decklink_output is not None:
                try:
                    decklink_output.DisableVideoOutput()
                except Exception:
                    pass
            comtypes.CoUninitialize()


def create_decklink_input_callback(owner, decklink_api):
    from comtypes import COMObject

    class _DeckLinkInputCallback(COMObject):
        _com_interfaces_ = [decklink_api.IDeckLinkInputCallback_v14_2_1]

        def VideoInputFormatChanged(self, notificationEvents, newDisplayMode, detectedSignalFlags):
            owner._on_video_format_changed(newDisplayMode, detectedSignalFlags)
            return 0

        def VideoInputFrameArrived(self, videoFrame, audioPacket):
            if videoFrame:
                owner._on_video_frame(videoFrame)
            return 0

    return _DeckLinkInputCallback()


class DeckLinkLiveInput:
    def __init__(self, device_name, mode_label):
        self.device_name = device_name
        self.mode_label = mode_label
        _, self.width, self.height, self.fps = DECKLINK_OUTPUT_MODES[mode_label]
        self._decklink_api = None
        self._decklink_input = None
        self._callback = None
        self._lock = threading.Lock()
        self._format_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_rgb = None
        self._opened = False
        self._com_initialized = False
        self._input_pixel_format = None
        self._input_flags = 0

    @staticmethod
    def list_devices():
        import comtypes
        from comtypes.client import CreateObject

        comtypes.CoInitialize()
        try:
            decklink_api = load_decklink_api()
            iterator = CreateObject(decklink_api.CDeckLinkIterator, interface=decklink_api.IDeckLinkIterator)
            devices = []
            while True:
                try:
                    decklink = iterator.Next()
                except Exception:
                    break
                if not decklink:
                    break

                try:
                    attributes = decklink.QueryInterface(decklink_api.IDeckLinkProfileAttributes)
                    io_support = int(attributes.GetInt(decklink_api.BMDDeckLinkVideoIOSupport))
                    if not (io_support & int(decklink_api.bmdDeviceSupportsCapture)):
                        continue
                    devices.append(str(decklink.GetDisplayName()))
                except Exception:
                    continue

            return devices
        finally:
            comtypes.CoUninitialize()

    def open(self):
        import comtypes
        from comtypes.client import CreateObject

        if self._opened:
            return True

        comtypes.CoInitialize()
        self._com_initialized = True
        self._decklink_api = load_decklink_api()
        iterator = CreateObject(self._decklink_api.CDeckLinkIterator, interface=self._decklink_api.IDeckLinkIterator)

        decklink = None
        while True:
            try:
                candidate = iterator.Next()
            except Exception:
                break
            if not candidate:
                break
            try:
                name = str(candidate.GetDisplayName())
                attributes = candidate.QueryInterface(self._decklink_api.IDeckLinkProfileAttributes)
                io_support = int(attributes.GetInt(self._decklink_api.BMDDeckLinkVideoIOSupport))
                if name == self.device_name and (io_support & int(self._decklink_api.bmdDeviceSupportsCapture)):
                    decklink = candidate
                    break
            except Exception:
                continue

        if decklink is None:
            raise RuntimeError(f"DeckLink Input nicht gefunden: {self.device_name}")

        self._decklink_input = decklink.QueryInterface(self._decklink_api.IDeckLinkInput_v14_2_1)
        mode_name, _, _, _ = DECKLINK_OUTPUT_MODES[self.mode_label]
        display_mode = getattr(self._decklink_api, mode_name)
        pixel_format = self._decklink_api.bmdFormat8BitYUV
        actual_mode, supported = self._decklink_input.DoesSupportVideoMode(
            self._decklink_api.bmdVideoConnectionUnspecified,
            display_mode,
            pixel_format,
            self._decklink_api.bmdNoVideoInputConversion,
            self._decklink_api.bmdSupportedVideoModeDefault,
        )
        if not supported:
            raise RuntimeError(f"{self.device_name} unterstuetzt {self.mode_label} als DeckLink Input nicht.")

        self._callback = create_decklink_input_callback(self, self._decklink_api)
        self._decklink_input.SetCallback(self._callback)
        self._input_pixel_format = pixel_format
        self._input_flags = getattr(
            self._decklink_api,
            "bmdVideoInputEnableFormatDetection",
            self._decklink_api.bmdVideoInputFlagDefault,
        )
        self._decklink_input.EnableVideoInput(
            display_mode,
            pixel_format,
            self._input_flags,
        )
        self._decklink_input.StartStreams()
        self._opened = True
        return True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def read(self, timeout=1.0):
        deadline = time.perf_counter() + float(timeout)
        with self._frame_ready:
            while self._opened and self._latest_rgb is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False, None
                self._frame_ready.wait(remaining)
            if self._latest_rgb is None:
                return False, None
            frame = self._latest_rgb
            self._latest_rgb = None
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self._opened = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._decklink_input is not None:
            try:
                self._decklink_input.StopStreams()
            except Exception:
                pass
            try:
                self._decklink_input.SetCallback(None)
            except Exception:
                pass
            try:
                self._decklink_input.DisableVideoInput()
            except Exception:
                pass
        self._decklink_input = None
        self._callback = None
        if self._com_initialized:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass
            self._com_initialized = False

    def _on_video_frame(self, video_frame):
        try:
            try:
                no_signal_flag = getattr(self._decklink_api, "bmdFrameHasNoInputSource", 0)
                if int(video_frame.GetFlags()) & int(no_signal_flag):
                    return
            except Exception:
                pass
            width = int(video_frame.GetWidth())
            height = int(video_frame.GetHeight())
            row_bytes = int(video_frame.GetRowBytes())
            try:
                pixel_format = video_frame.GetPixelFormat()
            except Exception:
                pixel_format = self._input_pixel_format
            buffer_ptr = video_frame.GetBytes()
            if hasattr(buffer_ptr, "value"):
                buffer_ptr = buffer_ptr.value
            raw = ctypes.string_at(buffer_ptr, row_bytes * height)
            if pixel_format == self._decklink_api.bmdFormat8BitBGRA:
                bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, row_bytes // 4, 4))[:, :width, :]
                rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB).copy()
            elif pixel_format == self._decklink_api.bmdFormat8BitYUV:
                uyvy = np.frombuffer(raw, dtype=np.uint8).reshape((height, row_bytes // 2, 2))[:, :width, :]
                rgb = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY).copy()
            else:
                return
        except Exception:
            return

        with self._frame_ready:
            self._latest_rgb = rgb
            self._frame_ready.notify()

    def _on_video_format_changed(self, new_display_mode, detected_signal_flags):
        if self._decklink_input is None or self._decklink_api is None:
            return

        with self._format_lock:
            try:
                display_mode = new_display_mode.GetDisplayMode()
                self.width = int(new_display_mode.GetWidth())
                self.height = int(new_display_mode.GetHeight())
                try:
                    frame_duration, time_scale = new_display_mode.GetFrameRate()
                    if frame_duration:
                        self.fps = float(time_scale) / float(frame_duration)
                except Exception:
                    pass

                rgb_flag = getattr(self._decklink_api, "bmdDetectedVideoInputRGB444", 0)
                if int(detected_signal_flags) & int(rgb_flag):
                    pixel_format = self._decklink_api.bmdFormat8BitBGRA
                else:
                    pixel_format = self._decklink_api.bmdFormat8BitYUV

                self._decklink_input.StopStreams()
                try:
                    self._decklink_input.FlushStreams()
                except Exception:
                    pass
                self._decklink_input.DisableVideoInput()
                self._decklink_input.EnableVideoInput(display_mode, pixel_format, self._input_flags)
                self._input_pixel_format = pixel_format
                with self._frame_ready:
                    self._latest_rgb = None
                    self._frame_ready.notify_all()
                self._decklink_input.StartStreams()
            except Exception:
                pass


def _center_crop_resize_rgb(image_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize by cover-fit and center crop. This preserves aspect ratio instead of squeezing.
    """
    src_h, src_w = image_rgb.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    return resized[y0:y0 + target_h, x0:x0 + target_w].copy()


def _generate_checker_background(width: int, height: int, tile: int = 40) -> np.ndarray:
    y_idx, x_idx = np.indices((height, width))
    pattern = ((x_idx // tile + y_idx // tile) % 2).astype(np.uint8)
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([90, 90, 90], dtype=np.uint8)
    return np.where(pattern[..., None] == 0, light, dark)


class MediaPipeSelfieModel:
    def __init__(self, model_path):
        self.device_label = "CPU"
        self.device_hint = "MediaPipe Selfie laeuft in diesem Programm CPU-basiert."
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        with quiet_terminal_output():
            self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def predict_mask(self, rgb_frame):
        ai_frame = cv2.resize(rgb_frame, (512, 288))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ai_frame)
        result = self.segmenter.segment(mp_image)
        mask_raw = result.category_mask.numpy_view()
        return (mask_raw > 0).astype(np.uint8) * 255


class BiRefNetModel:
    def __init__(self, force_device=None):
        required_modules = ["torch", "torchvision", "transformers", "timm", "safetensors"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "BiRefNet benoetigt zusaetzliche Python-Pakete. "
                f"Fehlend: {', '.join(missing_modules)}. "
                f"Installiere sie im aktiven Python mit: \"{sys.executable}\" -m pip install "
                "torch torchvision transformers timm safetensors"
            )

        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "BiRefNet benoetigt PyTorch und transformers. Installiere z.B.: "
                "pip install torch torchvision transformers timm safetensors"
            ) from exc

        self.torch = torch
        if force_device == "cpu":
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "BiRefNet wurde manuell auf CPU gesetzt. Das entlastet die GPU, ist aber langsamer."
        elif force_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer BiRefNet gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            self.device = torch.device("cuda")
            self.device_label = "CUDA"
            self.device_hint = None
        else:
            self.device, self.device_label, self.device_hint = _select_torch_device(torch)

        # Small global perf knobs (safe no-ops on CPU).
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            # Better matmul kernels on newer PyTorch.
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            from transformers import logging as transformers_logging
            transformers_logging.set_verbosity_error()
        except Exception:
            pass
        with quiet_terminal_output():
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True
            )
        self.model.to(self.device)
        self.model.eval()

        # Prefer FP16 on GPU backends to speed up inference.
        self.use_autocast = False
        self.autocast_dtype = None
        if self.device_label in ("CUDA", "DirectML"):
            try:
                self.model = self.model.half()
                self.use_autocast = self.device_label == "CUDA"
                self.autocast_dtype = torch.float16
            except Exception:
                pass

        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32

        # Pre-create normalization tensors once (avoid per-frame allocations).
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
            dtype=self.model_dtype
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
            dtype=self.model_dtype
        ).view(1, 3, 1, 1)

        # Warmup to reduce "first frame takes forever" effect.
        try:
            dummy = torch.zeros((1, 3, 512, 512), device=self.device, dtype=self.model_dtype)
            with torch.inference_mode():
                _ = self.model(dummy)
        except Exception:
            pass

    def _extract_prediction(self, output):
        torch = self.torch
        if isinstance(output, dict):
            for key in ("logits", "preds", "prediction", "out"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if not torch.is_tensor(output):
            raise RuntimeError("BiRefNet hat kein Tensor-Ergebnis geliefert.")
        return output

    def predict_mask(self, rgb_frame):
        torch = self.torch
        input_size = 512
        resized = cv2.resize(rgb_frame, (input_size, input_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.model_dtype) / 255.0
        tensor = (tensor - self.mean) / self.std

        # inference_mode is faster than no_grad.
        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    output = self.model(tensor)
            else:
                output = self.model(tensor)
            pred = self._extract_prediction(output).sigmoid()

        mask = pred[0].detach().float().cpu().squeeze().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


class RVMByteDanceModel:
    def __init__(self, force_device=None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RVM benoetigt PyTorch. Installiere z.B.: "
                "pip install torch torchvision"
            ) from exc

        self.torch = torch
        if force_device == "cpu":
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "RVM wurde manuell auf CPU gesetzt. Das entlastet die GPU, ist aber langsamer."
        elif force_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer RVM gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            self.device = torch.device("cuda")
            self.device_label = "CUDA"
            self.device_hint = None
        else:
            self.device, self.device_label, self.device_hint = _select_torch_device(torch)
        self.rec = [None, None, None, None]
        with quiet_terminal_output():
            self.model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3",
                pretrained=True,
                trust_repo=True,
                verbose=False
            )
        self.model.to(self.device)
        self.model.eval()

        self.use_autocast = self.device_label == "CUDA"
        if self.device_label in ("CUDA", "DirectML"):
            try:
                self.model = self.model.half()
            except Exception:
                pass

    def predict_mask(self, rgb_frame):
        torch = self.torch
        h, w = rgb_frame.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        model_w = max(32, int(w * scale) // 32 * 32)
        model_h = max(32, int(h * scale) // 32 * 32)
        resized = cv2.resize(rgb_frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=0.25)
            else:
                _, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=0.25)

        mask = pha[0, 0].detach().float().cpu().numpy()
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def create_segmentation_model(model_name, mediapipe_model_path, force_device=None):
    if model_name == "MediaPipe Selfie":
        return MediaPipeSelfieModel(mediapipe_model_path)
    if model_name == "BiRefNet":
        return BiRefNetModel(force_device=force_device)
    if model_name == "RVM ByteDance":
        return RVMByteDanceModel(force_device=force_device)
    raise ValueError(f"Unbekanntes Modell: {model_name}")


class YoloObjectDetector:
    def __init__(self, model_name="yolo11n-seg.pt", force_device=None):
        required_modules = ["ultralytics", "torch"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "YOLO benoetigt zusaetzliche Python-Pakete. "
                f"Fehlend: {', '.join(missing_modules)}. "
                f"Installiere sie mit: \"{sys.executable}\" -m pip install ultralytics"
            )

        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.model_name = model_name
        self.force_device = force_device
        if force_device == "cpu":
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "YOLO laeuft auf CPU. Das entlastet die GPU, ist aber langsamer als CUDA."
        else:
            self.device, self.device_label, self.device_hint = _select_torch_device(torch)
        with quiet_terminal_output():
            self.model = YOLO(model_name)
        self.predict_device = "cpu" if force_device == "cpu" else (0 if self.device_label == "CUDA" else "cpu")
        self.names = getattr(self.model, "names", {}) or {}

    def detect(self, rgb_frame, conf=0.35, max_det=12, imgsz=416):
        with quiet_terminal_output():
            results = self.model.predict(
                source=rgb_frame,
                conf=float(conf),
                max_det=int(max_det),
                imgsz=int(imgsz),
                device=self.predict_device,
                verbose=False,
            )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        masks_obj = getattr(result, "masks", None)
        masks = None
        if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
            try:
                masks = masks_obj.data.detach().float().cpu().numpy()
            except Exception:
                masks = None
        detections = []
        h, w = rgb_frame.shape[:2]
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(float)
            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w, round(x2))))
            y2 = int(max(0, min(h, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls[i])
            name = str(self.names.get(class_id, class_id))
            mask = None
            if masks is not None and i < len(masks):
                mask = masks[i]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
            detections.append(
                {
                    "name": name,
                    "class_id": class_id,
                    "conf": float(confs[i]),
                    "box": (x1, y1, x2, y2),
                    "mask": mask,
                }
            )

        detections.sort(key=lambda item: (item["name"], item["box"][0], item["box"][1]))
        counts = {}
        for det in detections:
            counts[det["name"]] = counts.get(det["name"], 0) + 1
            det["key"] = f"{det['name']} {counts[det['name']]}"
        return detections


class CorridorKeyRefiner:
    def __init__(self, device_mode="Automatisch", img_size=CORRIDORKEY_IMG_SIZE):
        required_modules = ["torch", "timm", "safetensors", "huggingface_hub", "CorridorKeyModule"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "CorridorKey benoetigt zusaetzliche Python-Pakete/Module. "
                f"Fehlend: {', '.join(missing_modules)}. "
                "Bitte den Windows-Installer erneut ausfuehren."
            )

        import torch
        from huggingface_hub import hf_hub_download
        from CorridorKeyModule import CorridorKeyEngine
        from CorridorKeyModule.core import color_utils as corridor_color

        self.torch = torch
        self.corridor_color = corridor_color
        self.device_mode = device_mode
        self.device = self._resolve_device(torch, device_mode)
        self.device_label = "CUDA" if self.device == "cuda" else "CPU"
        self.img_size = int(img_size)

        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "CorridorKeyModule",
            "checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, CORRIDORKEY_CHECKPOINT_FILE)
        if not os.path.exists(checkpoint_path):
            checkpoint_path = hf_hub_download(
                repo_id=CORRIDORKEY_CHECKPOINT_REPO,
                filename=CORRIDORKEY_CHECKPOINT_FILE,
                local_dir=checkpoint_dir,
            )

        os.environ.setdefault("CORRIDORKEY_SKIP_COMPILE", "1")
        with quiet_terminal_output():
            self.engine = CorridorKeyEngine(
                checkpoint_path=checkpoint_path,
                device=self.device,
                img_size=self.img_size,
                mixed_precision=self.device == "cuda",
            )

    @staticmethod
    def _resolve_device(torch, device_mode):
        if device_mode == "CPU":
            return "cpu"
        if device_mode == "CUDA":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer CorridorKey gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def refine(self, rgb_frame, alpha_2d, despill_strength=0.7, despeckle_size=400):
        mask = np.clip(alpha_2d.astype(np.float32, copy=False), 0.0, 1.0)
        despill_strength = float(np.clip(float(despill_strength), 0.0, 1.0))
        despeckle_size = max(0, int(despeckle_size))
        result = self.engine.process_frame(
            rgb_frame,
            mask,
            input_is_linear=False,
            despill_strength=despill_strength,
            auto_despeckle=True,
            despeckle_size=despeckle_size,
            generate_comp=False,
            post_process_on_gpu=self.device == "cuda",
            screen_channel=1,
        )
        processed_rgba = result.get("processed")
        refined_alpha = result.get("alpha")
        refined_fg = result.get("fg")
        if processed_rgba is not None:
            processed_rgba = np.nan_to_num(processed_rgba, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
            if processed_rgba.ndim == 3 and processed_rgba.shape[2] >= 4:
                processed_alpha = np.clip(processed_rgba[:, :, 3], 0.0, 1.0)
                premul_linear_rgb = np.clip(processed_rgba[:, :, :3], 0.0, None)
                straight_linear_rgb = premul_linear_rgb / np.maximum(processed_alpha[:, :, np.newaxis], 1e-4)
                refined_fg = self.corridor_color.linear_to_srgb(np.clip(straight_linear_rgb, 0.0, 1.0))
                refined_alpha = processed_alpha
        if refined_alpha is None:
            return rgb_frame, alpha_2d
        if refined_alpha.ndim == 3:
            refined_alpha = refined_alpha[:, :, 0]
        refined_alpha = np.nan_to_num(refined_alpha, nan=0.0, posinf=1.0, neginf=0.0)
        refined_alpha = np.clip(refined_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        if refined_fg is None:
            return rgb_frame, refined_alpha
        refined_fg = np.nan_to_num(refined_fg, nan=0.0, posinf=1.0, neginf=0.0)
        if refined_fg.dtype != np.uint8:
            refined_fg = np.clip(refined_fg * 255.0, 0, 255).astype(np.uint8)
        return refined_fg, refined_alpha


class FoolproofSyncApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Segmenter - Multi Model")
        self.root.geometry("1250x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.model_path = "selfie_multiclass.tflite"
        self.check_and_download_model()

        self.ui_w, self.ui_h = 800, 450

        self.model_name = ctk.StringVar(value=MODEL_OPTIONS[0])
        self.main_ai_device_mode = ctk.StringVar(value=MAIN_AI_DEVICE_OPTIONS[0])
        self.model_lock = threading.Lock()
        self.segmenter = create_segmentation_model(
            self.model_name.get(),
            self.model_path,
            self._resolve_main_ai_force_device()
        )
        self.loaded_model_name = self.model_name.get()
        self.model_status = self._format_model_status(self.segmenter, self.loaded_model_name)

        self.cap = None
        self.is_running = False
        self.current_camera_index = 0
        self.current_live_source = None

        self.bg_mode = ctk.StringVar(value="Checker")
        self.view_mode = ctk.StringVar(value="Processed")
        self.custom_background_image = None
        self.custom_background_source = None
        self.checker_background_source = None
        self.force_bg_update = False
        self.yolo_enabled = ctk.BooleanVar(value=False)
        self.yolo_model_name = ctk.StringVar(value=YOLO_MODEL_OPTIONS[0])
        self.yolo_device_mode = ctk.StringVar(value=YOLO_DEVICE_OPTIONS[0])
        self.yolo_confidence = ctk.DoubleVar(value=0.25)
        self.yolo_detector = None
        self.yolo_post_detector = None
        self.yolo_model_lock = threading.Lock()
        self.yolo_status = ctk.StringVar(value="YOLO aus / als Modell waehlbar")
        self.yolo_detections = []
        self.yolo_selected_keys = set()
        self.yolo_seen_keys = set()
        self.yolo_detection_vars = {}
        self.yolo_frame_counter = 0
        self.yolo_detect_every = 3
        self.yolo_last_detect_time = 0.0
        self.yolo_min_detect_interval = 0.12
        self.yolo_last_detection_signature = None
        self.yolo_last_ui_update_time = 0.0
        self.yolo_ui_update_interval = 0.2
        self.yolo_last_analysis_ms = 0.0
        self.yolo_last_analysis_time = 0.0
        self.yolo_async_lock = threading.Lock()
        self.yolo_async_running = False
        self.yolo_async_pending_frame = None
        self.yolo_async_stop = False
        self.yolo_async_thread = None
        self.yolo_select_all = ctk.BooleanVar(value=True)
        self.yolo_sync_postprocess = ctk.BooleanVar(value=False)
        self.corridor_enabled = ctk.BooleanVar(value=False)
        self.corridor_device_mode = ctk.StringVar(value=CORRIDORKEY_DEVICE_OPTIONS[0])
        self.corridor_despill_strength = ctk.DoubleVar(value=0.45)
        self.corridor_despeckle_size = ctk.IntVar(value=400)
        self.corridor_refiner = None
        self.corridor_lock = threading.Lock()
        self.corridor_status = ctk.StringVar(value="CorridorKey aus")

        self.edge_erode = ctk.IntVar(value=3)
        self.edge_soft = ctk.IntVar(value=7)

        self.latest_pil_image = None
        self.latest_display_payload = None
        self.empty_dummy_image = ctk.CTkImage(light_image=Image.new("RGB", (1, 1)), size=(1, 1))

        self.metrics_lock = threading.Lock()
        self.metrics_text = "Performance\nKamera gestoppt"
        self.metrics_last_gui_text = None
        self._reset_perf_metrics()
        self.app_mode = ctk.StringVar(value="Live")
        self.post_input_path = ctk.StringVar(value="Keine Datei gewaehlt")
        self.post_output_path = ctk.StringVar(value="Kein Ziel gewaehlt")
        self.post_status = ctk.StringVar(value="Bereit")
        self.post_is_processing = False
        decklink_devices = run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0)
        self.live_output_enabled = ctk.BooleanVar(value=False)
        self.live_output_device = ctk.StringVar(value=decklink_devices[0])
        self.live_key_output_enabled = ctk.BooleanVar(value=False)
        key_device_default = decklink_devices[1] if len(decklink_devices) > 1 else decklink_devices[0]
        self.live_key_output_device = ctk.StringVar(value=key_device_default)
        self.live_output_mode = ctk.StringVar(value=next(iter(DECKLINK_OUTPUT_MODES)))
        self.live_output_status = ctk.StringVar(value="DeckLink Output aus")
        self.decklink_output = None
        self.decklink_key_output = None
        self.sync_overlay_mode = ctk.StringVar(value="Aus")
        self.fill_delay_frames = ctk.IntVar(value=0)
        self.matte_delay_frames = ctk.IntVar(value=0)
        self.live_output_frame_counter = 0
        self.fill_delay_buffer = deque()
        self.matte_delay_buffer = deque()

        self.setup_gui()
        self.update_gui_loop()

    def _resolve_main_ai_force_device(self):
        mode = self.main_ai_device_mode.get()
        if mode == "CPU":
            return "cpu"
        if mode == "CUDA":
            return "cuda"
        return None

    def _main_ai_device_mode_note(self):
        return self.main_ai_device_mode.get()

    def _format_model_status(self, segmenter, choice):
        if choice == "YOLO":
            with self.yolo_model_lock:
                detector = self.yolo_detector
            if detector is not None:
                base = f"Modell bereit: YOLO ({detector.device_label})"
                if detector.device_hint:
                    return base + "\n" + str(detector.device_hint)
                return base
            return "YOLO wird ueber die Objektauswahl geladen"
        device_label = getattr(segmenter, "device_label", None)
        device_hint = getattr(segmenter, "device_hint", None)
        if device_label:
            base = f"Modell bereit: {choice} ({device_label})"
        else:
            base = f"Modell bereit: {choice}"
        if device_hint:
            return base + "\n" + str(device_hint)
        return base

    def check_and_download_model(self):
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 100000:
            print("Lade KI-Modell...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                with urllib.request.urlopen(url, context=ctx) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Fehler beim Modell-Download: {e}")
                sys.exit(1)

    def setup_gui_legacy(self):
        self.video_label = ctk.CTkLabel(self.root, text="Kamera gestoppt", width=self.ui_w, height=self.ui_h,
                                        fg_color="#2b2b2b")
        self.video_label.pack(side="left", padx=20, pady=20, expand=True, fill="both")

        control_panel = ctk.CTkFrame(self.root, width=320)
        control_panel.pack(side="right", fill="y", padx=20, pady=20)

        title = ctk.CTkLabel(control_panel, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=15)

        ctk.CTkLabel(control_panel, text="AI-Modell", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 5))
        self.model_select = ctk.CTkOptionMenu(control_panel, values=MODEL_OPTIONS,
                                              variable=self.model_name,
                                              command=self.change_model)
        self.model_select.pack(pady=5, padx=10, fill="x")

        self.model_status_label = ctk.CTkLabel(control_panel, text=self.model_status, wraplength=280,
                                               justify="left")
        self.model_status_label.pack(pady=(0, 10), padx=10, fill="x")

        self.camera_select = ctk.CTkOptionMenu(control_panel, values=run_with_timeout(get_available_cameras, ["Keine Kamera gefunden"], timeout=6.0),
                                               command=self.change_camera)
        self.camera_select.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_cameras = ctk.CTkButton(control_panel, text="Kameras neu suchen",
                                                command=self.refresh_cameras)
        self.btn_refresh_cameras.pack(pady=5, padx=10, fill="x")

        self.btn_toggle = ctk.CTkButton(control_panel, text="Kamera Starten", command=self.toggle_camera)
        self.btn_toggle.pack(pady=10, padx=10, fill="x")

        metrics_header = ctk.CTkFrame(control_panel, fg_color="transparent")
        metrics_header.pack(pady=(10, 5), padx=10, fill="x")
        ctk.CTkLabel(metrics_header, text="Live-Metriken", font=ctk.CTkFont(size=14, weight="bold")).pack(
            side="left")
        self.btn_metrics_info = ctk.CTkButton(
            metrics_header,
            text="i",
            width=26,
            height=24,
            command=self.show_metrics_info
        )
        self.btn_metrics_info.pack(side="right")
        self.metrics_label = ctk.CTkLabel(
            control_panel,
            text=self.metrics_text,
            font=ctk.CTkFont(family="Consolas", size=12),
            justify="left",
            anchor="w",
            wraplength=280
        )
        self.metrics_label.pack(pady=(0, 8), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Hintergrund-Effekt", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(20, 5))

        rb_checker = ctk.CTkRadioButton(control_panel, text="Karo-Muster (Transparenz)", variable=self.bg_mode,
                                        value="Checker")
        rb_checker.pack(pady=5, anchor="w", padx=20)

        rb_transparent = ctk.CTkRadioButton(control_panel, text="Echt Transparent", variable=self.bg_mode,
                                            value="Transparent")
        rb_transparent.pack(pady=5, anchor="w", padx=20)

        rb_green = ctk.CTkRadioButton(control_panel, text="Virtueller Greenscreen", variable=self.bg_mode,
                                      value="Green")
        rb_green.pack(pady=5, anchor="w", padx=20)

        rb_custom = ctk.CTkRadioButton(control_panel, text="Eigenes Hintergrundbild", variable=self.bg_mode,
                                       value="CustomImage")
        rb_custom.pack(pady=5, anchor="w", padx=20)

        self.btn_load_bg = ctk.CTkButton(control_panel, text="Finder öffnen & Bild laden",
                                         command=self.trigger_background_load, state="disabled")
        self.btn_load_bg.pack(pady=(5, 10), padx=20, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Schrumpfung", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(25, 0))
        self.slider_erode = ctk.CTkSlider(control_panel, from_=0, to=10, number_of_steps=10, variable=self.edge_erode)
        self.slider_erode.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Weichheit", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(15, 0))
        self.slider_soft = ctk.CTkSlider(control_panel, from_=1, to=21, number_of_steps=10, variable=self.edge_soft)
        self.slider_soft.pack(pady=5, padx=10, fill="x")

        self.bg_mode.trace_add("write", self.update_bg_button_state)

    def setup_gui(self):
        self.video_label = ctk.CTkLabel(self.root, text="Kamera gestoppt", width=self.ui_w, height=self.ui_h,
                                        fg_color="#2b2b2b")
        self.video_label.pack(side="left", padx=20, pady=20, expand=True, fill="both")

        control_panel = ctk.CTkScrollableFrame(self.root, width=340)
        control_panel.pack(side="right", fill="y", padx=20, pady=20)

        title = ctk.CTkLabel(control_panel, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=15)

        ctk.CTkLabel(control_panel, text="AI-Modell", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 5))
        self.model_select = ctk.CTkOptionMenu(control_panel, values=MODEL_OPTIONS,
                                              variable=self.model_name,
                                              command=self.change_model)
        self.model_select.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Main AI Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.main_ai_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=MAIN_AI_DEVICE_OPTIONS,
            variable=self.main_ai_device_mode,
            command=self.change_main_ai_device
        )
        self.main_ai_device_select.pack(pady=(3, 6), padx=10, fill="x")

        self.model_status_label = ctk.CTkLabel(control_panel, text=self.model_status, wraplength=300,
                                               justify="left")
        self.model_status_label.pack(pady=(0, 10), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Fensteransicht", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(10, 5))
        self.view_switch = ctk.CTkSegmentedButton(
            control_panel,
            values=["Input", "Alpha Matte", "Processed"],
            variable=self.view_mode
        )
        self.view_switch.pack(pady=(0, 10), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Hintergrund-Effekt", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(10, 5))

        rb_checker = ctk.CTkRadioButton(control_panel, text="Karo-Muster (Transparenz)", variable=self.bg_mode,
                                        value="Checker")
        rb_checker.pack(pady=5, anchor="w", padx=20)

        rb_transparent = ctk.CTkRadioButton(control_panel, text="Echt Transparent", variable=self.bg_mode,
                                            value="Transparent")
        rb_transparent.pack(pady=5, anchor="w", padx=20)

        rb_green = ctk.CTkRadioButton(control_panel, text="Virtueller Greenscreen", variable=self.bg_mode,
                                      value="Green")
        rb_green.pack(pady=5, anchor="w", padx=20)

        rb_custom = ctk.CTkRadioButton(control_panel, text="Eigenes Hintergrundbild", variable=self.bg_mode,
                                       value="CustomImage")
        rb_custom.pack(pady=5, anchor="w", padx=20)

        self.btn_load_bg = ctk.CTkButton(control_panel, text="Finder oeffnen & Bild laden",
                                         command=self.trigger_background_load, state="disabled")
        self.btn_load_bg.pack(pady=(5, 10), padx=20, fill="x")

        ctk.CTkLabel(control_panel, text="YOLO Objektauswahl", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        self.chk_yolo_enabled = ctk.CTkCheckBox(
            control_panel,
            text="YOLO-Nachbearbeitung aktiv",
            variable=self.yolo_enabled,
            command=self.toggle_yolo
        )
        self.chk_yolo_enabled.pack(pady=(0, 6), padx=20, anchor="w")
        self.yolo_model_select = ctk.CTkOptionMenu(
            control_panel,
            values=YOLO_MODEL_OPTIONS,
            variable=self.yolo_model_name,
            command=self.change_yolo_model
        )
        self.yolo_model_select.pack(pady=(0, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="YOLO Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.yolo_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=YOLO_DEVICE_OPTIONS,
            variable=self.yolo_device_mode,
            command=self.change_yolo_device
        )
        self.yolo_device_select.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="YOLO Confidence", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.yolo_conf_slider = ctk.CTkSlider(
            control_panel,
            from_=0.1,
            to=0.8,
            number_of_steps=14,
            variable=self.yolo_confidence
        )
        self.yolo_conf_slider.pack(pady=(3, 6), padx=10, fill="x")
        self.chk_yolo_sync_postprocess = ctk.CTkCheckBox(
            control_panel,
            text="YOLO synchron nach Main AI",
            variable=self.yolo_sync_postprocess
        )
        self.chk_yolo_sync_postprocess.pack(pady=(0, 6), padx=20, anchor="w")
        self.chk_yolo_select_all = ctk.CTkCheckBox(
            control_panel,
            text="Neue Objekte automatisch auswaehlen",
            variable=self.yolo_select_all
        )
        self.chk_yolo_select_all.pack(pady=(0, 6), padx=20, anchor="w")
        self.yolo_status_label = ctk.CTkLabel(
            control_panel,
            textvariable=self.yolo_status,
            wraplength=300,
            justify="left"
        )
        self.yolo_status_label.pack(pady=(0, 6), padx=10, fill="x")
        self.yolo_objects_frame = ctk.CTkFrame(control_panel, fg_color="transparent")

        self.corridor_header = ctk.CTkLabel(
            control_panel,
            text="CorridorKey Greenscreen",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.corridor_header.pack(
            pady=(12, 5))
        self.chk_corridor_enabled = ctk.CTkCheckBox(
            control_panel,
            text="CorridorKey-Refinement aktiv",
            variable=self.corridor_enabled,
            command=self.toggle_corridor_key
        )
        self.chk_corridor_enabled.pack(pady=(0, 6), padx=20, anchor="w")
        ctk.CTkLabel(control_panel, text="CorridorKey Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=CORRIDORKEY_DEVICE_OPTIONS,
            variable=self.corridor_device_mode,
            command=self.change_corridor_device
        )
        self.corridor_device_select.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="CorridorKey Despill", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_despill_slider = ctk.CTkSlider(
            control_panel,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self.corridor_despill_strength,
            command=lambda _value: self._update_corridor_status_settings()
        )
        self.corridor_despill_slider.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="CorridorKey Despeckle", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_despeckle_slider = ctk.CTkSlider(
            control_panel,
            from_=0,
            to=1200,
            number_of_steps=24,
            variable=self.corridor_despeckle_size,
            command=lambda _value: self._update_corridor_status_settings()
        )
        self.corridor_despeckle_slider.pack(pady=(3, 6), padx=10, fill="x")
        self.corridor_status_label = ctk.CTkLabel(
            control_panel,
            textvariable=self.corridor_status,
            wraplength=300,
            justify="left"
        )
        self.corridor_status_label.pack(pady=(0, 6), padx=10, fill="x")

        self.metrics_header = ctk.CTkFrame(control_panel, fg_color="transparent")
        self.metrics_header.pack(pady=(18, 5), padx=10, fill="x")
        ctk.CTkLabel(self.metrics_header, text="Status / Metriken", font=ctk.CTkFont(size=14, weight="bold")).pack(
            side="left")
        self.btn_metrics_info = ctk.CTkButton(
            self.metrics_header,
            text="i",
            width=26,
            height=24,
            command=self.show_metrics_info
        )
        self.btn_metrics_info.pack(side="right")
        self.metrics_label = ctk.CTkLabel(
            control_panel,
            text=self.metrics_text,
            font=ctk.CTkFont(family="Consolas", size=12),
            justify="left",
            anchor="w",
            wraplength=300
        )
        self.metrics_label.pack(pady=(0, 8), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Optimierung", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        ctk.CTkLabel(control_panel, text="Kanten-Schrumpfung", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0))
        self.slider_erode = ctk.CTkSlider(control_panel, from_=0, to=10, number_of_steps=10, variable=self.edge_erode)
        self.slider_erode.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Weichheit", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(10, 0))
        self.slider_soft = ctk.CTkSlider(control_panel, from_=1, to=21, number_of_steps=10, variable=self.edge_soft)
        self.slider_soft.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Arbeitsmodus", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        self.mode_switch = ctk.CTkSegmentedButton(
            control_panel,
            values=["Live", "Postproduktion"],
            variable=self.app_mode,
            command=self.change_app_mode
        )
        self.mode_switch.pack(pady=(0, 10), padx=10, fill="x")

        self.live_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        live_sources = run_with_timeout(get_live_input_sources, ["Keine Live-Quelle gefunden"], timeout=8.0)
        self.current_live_source = live_sources[0]
        self.camera_select = ctk.CTkOptionMenu(self.live_frame, values=live_sources,
                                               command=self.change_camera)
        self.camera_select.set(self.current_live_source)
        self.camera_select.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_cameras = ctk.CTkButton(self.live_frame, text="Kameras neu suchen",
                                                command=self.refresh_cameras)
        self.btn_refresh_cameras.pack(pady=5, padx=10, fill="x")

        self.btn_toggle = ctk.CTkButton(self.live_frame, text="Kamera Starten", command=self.toggle_camera)
        self.btn_toggle.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(self.live_frame, text="Live Output DeckLink", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(14, 5), padx=10, anchor="w")
        self.decklink_device_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0),
            variable=self.live_output_device,
            command=lambda _choice: self.restart_live_output_if_needed()
        )
        self.decklink_device_select.pack(pady=(4, 4), padx=10, fill="x")
        ctk.CTkLabel(self.live_frame, text="Alpha Matte SDI", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0), padx=10, anchor="w")
        self.decklink_key_device_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0),
            variable=self.live_key_output_device,
            command=lambda _choice: self.restart_live_output_if_needed()
        )
        self.decklink_key_device_select.pack(pady=(4, 4), padx=10, fill="x")
        self.decklink_mode_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=list(DECKLINK_OUTPUT_MODES.keys()),
            variable=self.live_output_mode,
            command=lambda _choice: self.restart_decklink_io_if_needed()
        )
        self.decklink_mode_select.pack(pady=4, padx=10, fill="x")
        self.btn_refresh_decklink = ctk.CTkButton(
            self.live_frame,
            text="DeckLink Geraete neu suchen",
            command=self.refresh_decklink_devices
        )
        self.btn_refresh_decklink.pack(pady=4, padx=10, fill="x")
        self.chk_live_output = ctk.CTkCheckBox(
            self.live_frame,
            text="Live Output aktiv",
            variable=self.live_output_enabled,
            command=self.toggle_live_output
        )
        self.chk_live_output.pack(pady=(8, 4), padx=20, anchor="w")
        self.chk_live_key_output = ctk.CTkCheckBox(
            self.live_frame,
            text="Alpha Matte auf zweitem SDI",
            variable=self.live_key_output_enabled,
            command=self.toggle_live_output
        )
        self.chk_live_key_output.pack(pady=(2, 4), padx=20, anchor="w")
        ctk.CTkLabel(self.live_frame, text="Output Sync", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0), padx=10, anchor="w")
        self.sync_overlay_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=["Aus", "Frame", "Timecode", "Beides"],
            variable=self.sync_overlay_mode
        )
        self.sync_overlay_select.pack(pady=(4, 4), padx=10, fill="x")
        fill_sync_row = ctk.CTkFrame(self.live_frame, fg_color="transparent")
        fill_sync_row.pack(pady=(2, 2), padx=10, fill="x")
        ctk.CTkLabel(fill_sync_row, text="Fill Delay", width=90, anchor="w").pack(side="left")
        ctk.CTkButton(fill_sync_row, text="-", width=32, command=lambda: self.adjust_output_delay("fill", -1)).pack(side="left", padx=2)
        self.fill_delay_label = ctk.CTkLabel(fill_sync_row, textvariable=self.fill_delay_frames, width=34)
        self.fill_delay_label.pack(side="left", padx=4)
        ctk.CTkButton(fill_sync_row, text="+", width=32, command=lambda: self.adjust_output_delay("fill", 1)).pack(side="left", padx=2)
        matte_sync_row = ctk.CTkFrame(self.live_frame, fg_color="transparent")
        matte_sync_row.pack(pady=(2, 6), padx=10, fill="x")
        ctk.CTkLabel(matte_sync_row, text="Matte Delay", width=90, anchor="w").pack(side="left")
        ctk.CTkButton(matte_sync_row, text="-", width=32, command=lambda: self.adjust_output_delay("matte", -1)).pack(side="left", padx=2)
        self.matte_delay_label = ctk.CTkLabel(matte_sync_row, textvariable=self.matte_delay_frames, width=34)
        self.matte_delay_label.pack(side="left", padx=4)
        ctk.CTkButton(matte_sync_row, text="+", width=32, command=lambda: self.adjust_output_delay("matte", 1)).pack(side="left", padx=2)
        self.live_output_status_label = ctk.CTkLabel(
            self.live_frame,
            textvariable=self.live_output_status,
            wraplength=300,
            justify="left"
        )
        self.live_output_status_label.pack(pady=(0, 8), padx=10, fill="x")

        self.post_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        self.btn_post_input = ctk.CTkButton(self.post_frame, text="Quelldatei waehlen", command=self.select_post_input)
        self.btn_post_input.pack(pady=(8, 4), padx=10, fill="x")
        self.post_input_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_input_path,
                                             wraplength=300, justify="left")
        self.post_input_label.pack(pady=(0, 8), padx=10, fill="x")

        self.btn_post_output = ctk.CTkButton(self.post_frame, text="Speicherziel waehlen", command=self.select_post_output)
        self.btn_post_output.pack(pady=4, padx=10, fill="x")
        self.post_output_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_output_path,
                                              wraplength=300, justify="left")
        self.post_output_label.pack(pady=(0, 8), padx=10, fill="x")

        self.post_progress = ctk.CTkProgressBar(self.post_frame)
        self.post_progress.set(0)
        self.post_progress.pack(pady=(6, 4), padx=10, fill="x")
        self.post_status_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_status,
                                              wraplength=300, justify="left")
        self.post_status_label.pack(pady=(0, 8), padx=10, fill="x")

        self.btn_post_process = ctk.CTkButton(self.post_frame, text="Datei verarbeiten",
                                              command=self.start_post_processing)
        self.btn_post_process.pack(pady=(4, 12), padx=10, fill="x")

        self.bg_mode.trace_add("write", self.update_bg_button_state)
        self.view_mode.trace_add("write", self.refresh_display_view)
        self.change_app_mode(self.app_mode.get())

    def _resolve_yolo_force_device(self, for_postprocess=False):
        mode = self.yolo_device_mode.get()
        if mode == "CPU":
            return "cpu"
        if mode == "CUDA":
            return None
        if for_postprocess:
            return "cpu"
        return None

    def _yolo_device_mode_note(self, for_postprocess=False):
        mode = self.yolo_device_mode.get()
        if mode == "Automatisch":
            return "Auto: CPU" if for_postprocess else "Auto: CUDA"
        return mode

    def _reset_yolo_runtime_state(self, clear_post_detector=False, clear_primary_detector=False):
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        with self.yolo_model_lock:
            if clear_post_detector:
                self.yolo_post_detector = None
            if clear_primary_detector:
                self.yolo_detector = None
        self.yolo_detections = []
        self.yolo_selected_keys.clear()
        self.yolo_seen_keys.clear()
        self.yolo_last_detection_signature = None
        self.yolo_last_ui_update_time = 0.0
        self.yolo_frame_counter = 0
        self.yolo_last_detect_time = -999.0
        self.yolo_last_analysis_ms = 0.0
        self.yolo_last_analysis_time = 0.0
        self._refresh_yolo_detection_list()
        self.refresh_display_view()

    def toggle_yolo(self):
        if not self.yolo_enabled.get():
            if self.model_name.get() == "YOLO" or self.loaded_model_name == "YOLO":
                self.yolo_status.set("YOLO als Hauptmodell aktiv")
            else:
                self.yolo_status.set("YOLO-Nachbearbeitung aus")
                self._reset_yolo_runtime_state(clear_post_detector=True)
            return
        self.yolo_frame_counter = 0
        self.yolo_last_detect_time = -999.0
        self.yolo_status.set(f"YOLO-Nachbearbeitung wird geladen ({self._yolo_device_mode_note(True)}) ...")
        threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()

    def change_yolo_model(self, choice):
        self._reset_yolo_runtime_state(clear_post_detector=self.yolo_enabled.get(), clear_primary_detector=True)
        if self.model_name.get() == "YOLO" or self.loaded_model_name == "YOLO":
            self.yolo_status.set(f"YOLO wird geladen: {choice} ({self._yolo_device_mode_note(False)})")
            threading.Thread(target=self._load_yolo_worker, args=(choice,), daemon=True).start()
        elif self.yolo_enabled.get():
            self.yolo_status.set(f"YOLO-Nachbearbeitung nutzt {YOLO_POSTPROCESS_MODEL} ({self._yolo_device_mode_note(True)})")
            threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()

    def change_yolo_device(self, choice):
        self._reset_yolo_runtime_state(clear_post_detector=True, clear_primary_detector=True)
        if self.model_name.get() == "YOLO" or self.loaded_model_name == "YOLO":
            model_name = self.yolo_model_name.get()
            self.yolo_status.set(f"YOLO wird auf {self._yolo_device_mode_note(False)} geladen: {model_name}")
            threading.Thread(target=self._load_yolo_worker, args=(model_name,), daemon=True).start()
        elif self.yolo_enabled.get():
            self.yolo_status.set(f"YOLO-Nachbearbeitung wird auf {self._yolo_device_mode_note(True)} geladen ...")
            threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()
        else:
            self.yolo_status.set(f"YOLO aus / Hardware: {choice}")

    def _load_yolo_worker(self, model_name):
        try:
            detector = YoloObjectDetector(model_name, force_device=self._resolve_yolo_force_device(False))
            with self.yolo_model_lock:
                self.yolo_detector = detector
            status = f"YOLO bereit: {model_name} ({detector.device_label})"
            if detector.device_hint:
                status += "\n" + str(detector.device_hint)
            self.model_status = f"Modell bereit: YOLO ({detector.device_label})"
            self.root.after(0, lambda: self.yolo_status.set(status))
            self.root.after(0, lambda: self.model_status_label.configure(text=self.model_status))
        except Exception as exc:
            with self.yolo_model_lock:
                self.yolo_detector = None
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))

    def _load_yolo_post_worker(self):
        try:
            detector = YoloObjectDetector(
                YOLO_POSTPROCESS_MODEL,
                force_device=self._resolve_yolo_force_device(True)
            )
            with self.yolo_model_lock:
                self.yolo_post_detector = detector
            status = f"YOLO-Nachbearbeitung bereit: {YOLO_POSTPROCESS_MODEL} ({detector.device_label})"
            if detector.device_hint:
                status += "\n" + str(detector.device_hint)
            self.root.after(0, lambda: self.yolo_status.set(status))
            self.root.after(0, self._refresh_yolo_detection_list)
        except Exception as exc:
            with self.yolo_model_lock:
                self.yolo_post_detector = None
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Nachbearbeitung Fehler: {e}"))

    def toggle_corridor_key(self):
        if not self.corridor_enabled.get():
            with self.corridor_lock:
                self.corridor_refiner = None
            self.corridor_status.set("CorridorKey aus")
            return
        self.corridor_status.set(f"CorridorKey wird geladen ({self.corridor_device_mode.get()}) ...")
        threading.Thread(target=self._load_corridor_worker, daemon=True).start()

    def change_corridor_device(self, choice):
        with self.corridor_lock:
            self.corridor_refiner = None
        if self.corridor_enabled.get():
            self.corridor_status.set(f"CorridorKey wird auf {choice} geladen ...")
            threading.Thread(target=self._load_corridor_worker, daemon=True).start()
        else:
            self.corridor_status.set(f"CorridorKey aus / Hardware: {choice}")

    def _update_corridor_status_settings(self):
        if not self.corridor_enabled.get():
            self.corridor_status.set(
                f"CorridorKey aus / Despill {self.corridor_despill_strength.get():.2f} | "
                f"Despeckle {int(self.corridor_despeckle_size.get())}"
            )
            return
        with self.corridor_lock:
            refiner = self.corridor_refiner
        if refiner is None:
            return
        self.corridor_status.set(
            f"CorridorKey bereit ({refiner.device_label}, {refiner.img_size}px)\n"
            f"Despill {self.corridor_despill_strength.get():.2f} | "
            f"Despeckle {int(self.corridor_despeckle_size.get())}"
        )

    def _load_corridor_worker(self):
        try:
            refiner = CorridorKeyRefiner(device_mode=self.corridor_device_mode.get())
            with self.corridor_lock:
                self.corridor_refiner = refiner
            self.root.after(
                0,
                lambda: self.corridor_status.set(
                    f"CorridorKey bereit ({refiner.device_label}, {refiner.img_size}px)\n"
                    f"Despill {self.corridor_despill_strength.get():.2f} | "
                    f"Despeckle {int(self.corridor_despeckle_size.get())}"
                )
            )
        except Exception as exc:
            with self.corridor_lock:
                self.corridor_refiner = None
            self.root.after(0, lambda e=exc: self.corridor_status.set(f"CorridorKey Fehler: {e}"))

    def _apply_corridor_key(self, rgb_frame, alpha_2d):
        if not self.corridor_enabled.get():
            return rgb_frame, alpha_2d
        with self.corridor_lock:
            refiner = self.corridor_refiner
        if refiner is None:
            return rgb_frame, alpha_2d
        try:
            return refiner.refine(
                rgb_frame,
                alpha_2d,
                despill_strength=self.corridor_despill_strength.get(),
                despeckle_size=self.corridor_despeckle_size.get(),
            )
        except Exception as exc:
            with self.corridor_lock:
                self.corridor_refiner = None
            self.root.after(0, lambda e=exc: self.corridor_status.set(f"CorridorKey Fehler: {e}"))
            return rgb_frame, alpha_2d

    def _is_yolo_primary_model(self):
        return self.model_name.get() == "YOLO" or self.loaded_model_name == "YOLO"

    def _yolo_selection_active(self):
        return self.yolo_enabled.get() or self._is_yolo_primary_model()

    def _refresh_yolo_detection_list(self):
        if not hasattr(self, "yolo_objects_frame"):
            return
        for child in self.yolo_objects_frame.winfo_children():
            child.destroy()
        self.yolo_detection_vars = {}

        if not self._yolo_selection_active():
            self.yolo_objects_frame.pack_forget()
            return
        pack_options = {"pady": (0, 8), "padx": 10, "fill": "x"}
        if hasattr(self, "corridor_header"):
            pack_options["before"] = self.corridor_header
        elif hasattr(self, "metrics_header"):
            pack_options["before"] = self.metrics_header
        self.yolo_objects_frame.pack(**pack_options)
        if not self.yolo_detections:
            with self.yolo_model_lock:
                detector_ready = self.yolo_detector is not None if self._is_yolo_primary_model() else self.yolo_post_detector is not None
            text = "Keine Objekte erkannt" if detector_ready else "YOLO wird geladen ..."
            ctk.CTkLabel(
                self.yolo_objects_frame,
                text=text,
                justify="left",
                wraplength=280
            ).pack(anchor="w", fill="x")
            return

        current_keys = {det["key"] for det in self.yolo_detections}
        self.yolo_selected_keys.intersection_update(current_keys)
        if self.yolo_select_all.get():
            new_keys = current_keys - self.yolo_seen_keys
            self.yolo_selected_keys.update(new_keys)
        self.yolo_seen_keys = current_keys

        for det in self.yolo_detections:
            key = det["key"]
            var = ctk.BooleanVar(value=key in self.yolo_selected_keys)
            self.yolo_detection_vars[key] = var
            label = f"{key} ({det['conf']:.0%})"
            box = ctk.CTkCheckBox(
                self.yolo_objects_frame,
                text=label,
                variable=var,
                command=lambda k=key: self._toggle_yolo_detection(k)
            )
            box.pack(pady=2, anchor="w")

    def _toggle_yolo_detection(self, key):
        var = self.yolo_detection_vars.get(key)
        if var is not None and bool(var.get()):
            self.yolo_selected_keys.add(key)
        else:
            self.yolo_selected_keys.discard(key)

    def _yolo_detection_signature(self, detections):
        return tuple(
            (
                det.get("key"),
                int(det.get("class_id", -1)),
            )
            for det in detections
        )

    def _apply_yolo_detections(self, detections, elapsed_ms, now=None):
        now = time.perf_counter() if now is None else now
        self.yolo_detections = detections
        self.yolo_last_analysis_ms = float(elapsed_ms)
        self.yolo_last_analysis_time = now
        signature = self._yolo_detection_signature(detections)
        current_keys = {det["key"] for det in detections}
        self.yolo_selected_keys.intersection_update(current_keys)
        if self.yolo_select_all.get():
            new_keys = current_keys - self.yolo_seen_keys
            self.yolo_selected_keys.update(new_keys)
        self.yolo_seen_keys = current_keys
        selected_count = len([det for det in detections if det["key"] in self.yolo_selected_keys])
        ui_due = now - self.yolo_last_ui_update_time >= float(self.yolo_ui_update_interval)
        try:
            list_visible = bool(self.yolo_objects_frame.winfo_ismapped())
        except Exception:
            list_visible = True
        if signature != self.yolo_last_detection_signature or not list_visible:
            self.yolo_last_detection_signature = signature
            self.yolo_last_ui_update_time = now
            self.root.after(0, self._refresh_yolo_detection_list)
            ui_due = False
        if ui_due:
            self.yolo_last_ui_update_time = now
            self.root.after(
                0,
                lambda count=len(detections), ms=elapsed_ms, selected=selected_count: self.yolo_status.set(
                    f"YOLO aktiv: {count} Objekte, {selected} ausgewaehlt\nAnalyse: {ms:.1f} ms"
                )
            )
        return elapsed_ms

    def _run_yolo_primary(self, rgb_frame):
        with self.yolo_model_lock:
            detector = self.yolo_detector
        if detector is None:
            return 0.0
        self.yolo_frame_counter += 1
        now = time.perf_counter()
        self.yolo_last_detect_time = now
        start = time.perf_counter()
        detections = detector.detect(rgb_frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._apply_yolo_detections(detections, elapsed_ms, now=now)

    def _schedule_yolo_postprocess(self, rgb_frame):
        if not self.yolo_enabled.get() or self._is_yolo_primary_model():
            return 0.0
        with self.yolo_model_lock:
            detector = self.yolo_post_detector
        if detector is None:
            return 0.0

        self.yolo_frame_counter += 1
        now = time.perf_counter()
        if self.yolo_frame_counter % int(self.yolo_detect_every) != 1:
            return 0.0
        if now - self.yolo_last_detect_time < float(self.yolo_min_detect_interval):
            return 0.0
        self.yolo_last_detect_time = now

        with self.yolo_async_lock:
            self.yolo_async_pending_frame = rgb_frame.copy()
            if self.yolo_async_running:
                return 0.0
            self.yolo_async_running = True

        self.yolo_async_thread = threading.Thread(target=self._yolo_postprocess_worker, daemon=True)
        self.yolo_async_thread.start()
        return 0.0

    def _run_yolo_postprocess_sync(self, rgb_frame):
        if not self.yolo_enabled.get() or self._is_yolo_primary_model():
            return 0.0
        with self.yolo_model_lock:
            detector = self.yolo_post_detector
        if detector is None:
            return 0.0
        self.yolo_frame_counter += 1
        now = time.perf_counter()
        self.yolo_last_detect_time = now
        start = time.perf_counter()
        try:
            detections = detector.detect(rgb_frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
        except Exception as exc:
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))
            detections = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._apply_yolo_detections(detections, elapsed_ms, now=now)

    def _yolo_postprocess_worker(self):
        while True:
            with self.yolo_async_lock:
                frame = self.yolo_async_pending_frame
                self.yolo_async_pending_frame = None
            if frame is None:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return
            if self.yolo_async_stop:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return

            with self.yolo_model_lock:
                detector = self.yolo_post_detector
            if detector is None:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return

            start = time.perf_counter()
            try:
                detections = detector.detect(frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))
                detections = []
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.root.after(0, lambda d=detections, ms=elapsed_ms: self._apply_yolo_detections(d, ms))

    def _make_yolo_roi(self, shape):
        h, w = shape[:2]
        if not self.yolo_enabled.get():
            return None
        with self.yolo_model_lock:
            detector_ready = self.yolo_post_detector is not None
        if not detector_ready:
            return None
        if not self.yolo_detections:
            return None
        mask = np.zeros((h, w), dtype=np.float32)
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            if det["key"] not in selected:
                continue
            x1, y1, x2, y2 = det["box"]
            pad_x = max(8, int((x2 - x1) * 0.06))
            pad_y = max(8, int((y2 - y1) * 0.06))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            mask[y1:y2, x1:x2] = 1.0
        if np.max(mask) <= 0:
            return mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return np.clip(mask, 0.0, 1.0)

    def _apply_yolo_roi_to_alpha(self, alpha_2d):
        roi = self._make_yolo_roi(alpha_2d.shape)
        if roi is None:
            return alpha_2d
        return np.clip(alpha_2d * roi, 0.0, 1.0).astype(np.float32, copy=False)

    def _make_yolo_alpha(self, shape):
        h, w = shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            if det["key"] not in selected:
                continue
            det_mask = det.get("mask")
            if det_mask is not None:
                if det_mask.shape[:2] != (h, w):
                    det_mask = cv2.resize(det_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.maximum(alpha, det_mask.astype(np.float32, copy=False))
                continue

            x1, y1, x2, y2 = det["box"]
            pad_x = max(8, int((x2 - x1) * 0.04))
            pad_y = max(8, int((y2 - y1) * 0.04))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            alpha[y1:y2, x1:x2] = 1.0

        if np.max(alpha) > 0:
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)

    def _draw_yolo_overlay(self, frame_rgb):
        if not self._yolo_selection_active() or not self.yolo_detections:
            return frame_rgb
        out = frame_rgb.copy()
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            x1, y1, x2, y2 = det["box"]
            is_selected = det["key"] in selected
            color = (60, 220, 80) if is_selected else (235, 170, 60)
            thickness = 3 if is_selected else 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{det['key']} {det['conf']:.0%}"
            cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return out

    def _postprocess_to_alpha(self, rgb_frame: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        Convert raw model mask to a stable, refined alpha matte (float32 [0,1]).
        Operates in UI resolution.
        """
        h, w = rgb_frame.shape[:2]
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha_raw = mask_u8.astype(np.float32) / 255.0

        # 1) Build a stable binary core for cleanup.
        #    For soft masks (BiRefNet/RVM) we threshold mid-range; for binary it's already fine.
        bin_thresh = 128
        bin_mask = ((mask_u8 >= bin_thresh).astype(np.uint8) * 255)

        # Fill small holes / connect thin structures (helps green flecks).
        k_close = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Optional "shrink edges" on the cleaned binary core (more predictable than eroding the raw mask).
        erode_size = int(self.edge_erode.get())
        if erode_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            bin_mask = cv2.erode(bin_mask, k, iterations=1)

        # 2) Create a soft alpha "core" from the cleaned binary mask.
        soft_size = _ensure_odd_ksize(int(self.edge_soft.get()), min_k=1)
        alpha_core = cv2.GaussianBlur(bin_mask, (soft_size, soft_size), 0).astype(np.float32) / 255.0

        # Combine: keep the model's soft details, but clamp them to the cleaned core region.
        # This removes speckles/holes while preserving edge nuance.
        core_gate = (alpha_core > 0.03).astype(np.float32)
        alpha = np.clip(alpha_raw * core_gate, 0.0, 1.0)
        alpha = np.maximum(alpha, alpha_core * 0.85)

        return alpha

    def update_bg_button_state(self, *args):
        if self.bg_mode.get() == "CustomImage":
            self.btn_load_bg.configure(state="normal")
        else:
            self.btn_load_bg.configure(state="disabled")

    def _get_checker_background(self, width: int, height: int) -> np.ndarray:
        if self.checker_background_source is None:
            self.checker_background_source = _generate_checker_background(1920, 1080)
        return _center_crop_resize_rgb(self.checker_background_source, width, height)

    def _get_custom_background(self, width: int, height: int) -> np.ndarray:
        if self.custom_background_source is None:
            return np.zeros((height, width, 3), dtype=np.uint8) + 30
        return _center_crop_resize_rgb(self.custom_background_source, width, height)

    def _alpha_to_u8(self, alpha_2d: np.ndarray) -> np.ndarray:
        return np.clip(alpha_2d * 255.0, 0, 255).astype(np.uint8)

    def _make_alpha_preview(self, alpha_2d: np.ndarray) -> Image.Image:
        alpha_u8 = self._alpha_to_u8(alpha_2d)
        return Image.fromarray(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))

    def _make_display_image(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray) -> Image.Image:
        view_mode = self.view_mode.get()
        if view_mode == "Input":
            return Image.fromarray(self._draw_yolo_overlay(rgb_frame))
        if view_mode == "Alpha Matte":
            return self._make_alpha_preview(alpha_2d)
        if processed_frame.shape[2] == 4:
            return Image.fromarray(processed_frame)
        return Image.fromarray(processed_frame)

    def _make_view_frame(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray) -> np.ndarray:
        view_mode = self.view_mode.get()
        if view_mode == "Input":
            return rgb_frame
        if view_mode == "Alpha Matte":
            alpha_u8 = self._alpha_to_u8(alpha_2d)
            return cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB)
        return processed_frame

    def _set_latest_display(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray):
        self.latest_display_payload = (rgb_frame, alpha_2d, processed_frame)
        self.latest_pil_image = self._make_display_image(rgb_frame, alpha_2d, processed_frame)

    def refresh_display_view(self, *args):
        if self.latest_display_payload is None:
            return
        rgb_frame, alpha_2d, processed_frame = self.latest_display_payload
        self.latest_pil_image = self._make_display_image(rgb_frame, alpha_2d, processed_frame)

    def change_app_mode(self, mode):
        if mode == "Postproduktion":
            if self.is_running:
                self._stop_camera_internal()
            self.stop_live_output()
            self.live_frame.pack_forget()
            self.post_frame.pack(pady=(0, 8), padx=0, fill="x")
            self.video_label.configure(image=self.empty_dummy_image, text="Postproduktion bereit")
            with self.metrics_lock:
                if not self.post_is_processing:
                    self.metrics_text = "Performance\nPostproduktion bereit"
            return

        self.post_frame.pack_forget()
        self.live_frame.pack(pady=(0, 8), padx=0, fill="x")
        if not self.is_running:
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")
            self._reset_perf_metrics()

    def refresh_decklink_devices(self):
        if getattr(self, "_decklink_refresh_running", False):
            return
        self._decklink_refresh_running = True
        try:
            self.btn_refresh_decklink.configure(text="Suche laeuft...", state="disabled")
        except Exception:
            pass
        self.live_output_status.set("DeckLink Geraete werden gesucht ...")

        def worker():
            try:
                values = run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=6.0)
                inputs = run_with_timeout(get_decklink_input_devices, [], timeout=6.0)
                error = None
            except Exception as exc:
                values = None
                inputs = []
                error = exc
            self.root.after(0, lambda: self._finish_decklink_refresh(values, inputs, error))

        threading.Thread(target=worker, name="DeckLinkRefresh", daemon=True).start()

    def _finish_decklink_refresh(self, values, inputs=None, error=None):
        self._decklink_refresh_running = False
        try:
            self.btn_refresh_decklink.configure(text="DeckLink Geraete neu suchen", state="normal")
        except Exception:
            pass
        if error is not None:
            self.live_output_status.set(f"DeckLink Suche Fehler: {error}")
            return
        if not values:
            values = ["Keine DeckLink-Ausgabe gefunden"]
        self.decklink_device_select.configure(values=values)
        self.decklink_key_device_select.configure(values=values)
        if self.live_output_device.get() not in values:
            self.live_output_device.set(values[0])
        if self.live_key_output_device.get() not in values:
            self.live_key_output_device.set(values[1] if len(values) > 1 else values[0])
        existing_sources = []
        try:
            existing_sources = list(self.camera_select.cget("values") or [])
        except Exception:
            existing_sources = []
        non_decklink_sources = [
            source for source in existing_sources
            if source and not str(source).startswith("DeckLink: ") and source != "Keine Live-Quelle gefunden"
        ]
        live_sources = [*non_decklink_sources, *[f"DeckLink: {name}" for name in (inputs or [])]]
        if not live_sources:
            live_sources = ["Keine Live-Quelle gefunden"]
        self.camera_select.configure(values=live_sources)
        if self.current_live_source not in live_sources:
            self.current_live_source = live_sources[0]
            self.camera_select.set(self.current_live_source)
        self.live_output_status.set("DeckLink Geraeteliste aktualisiert.")

    def toggle_live_output(self):
        if self.live_output_enabled.get() or self.live_key_output_enabled.get():
            self.start_live_output()
        else:
            self.stop_live_output()

    def restart_live_output_if_needed(self):
        if self.live_output_enabled.get() or self.live_key_output_enabled.get():
            self.stop_live_output()
            self.start_live_output()

    def restart_decklink_io_if_needed(self):
        self.restart_live_output_if_needed()
        if self.is_running and (self.current_live_source or "").startswith("DeckLink: "):
            self._stop_camera_internal(preserve_preview=True)
            self.root.after(150, lambda: self._start_camera_internal(preserve_preview=True))

    def adjust_output_delay(self, target, delta):
        if target == "fill":
            self.fill_delay_frames.set(max(0, min(120, int(self.fill_delay_frames.get()) + int(delta))))
            self.fill_delay_buffer.clear()
        elif target == "matte":
            self.matte_delay_frames.set(max(0, min(120, int(self.matte_delay_frames.get()) + int(delta))))
            self.matte_delay_buffer.clear()
        self._update_live_output_sync_status()

    def _reset_output_sync_buffers(self):
        self.live_output_frame_counter = 0
        self.fill_delay_buffer.clear()
        self.matte_delay_buffer.clear()

    def _update_live_output_sync_status(self):
        base = self.live_output_status.get()
        lines = [line for line in base.splitlines() if not line.startswith("Sync:")]
        lines.append(
            f"Sync: Overlay {self.sync_overlay_mode.get()} | "
            f"Fill +{int(self.fill_delay_frames.get())}F | Matte +{int(self.matte_delay_frames.get())}F"
        )
        self.live_output_status.set("\n".join(lines))

    def _current_output_fps(self):
        mode = self.live_output_mode.get()
        if mode in DECKLINK_OUTPUT_MODES:
            return float(DECKLINK_OUTPUT_MODES[mode][3])
        return 25.0

    def _format_output_timecode(self, frame_index):
        fps = max(1, int(round(self._current_output_fps())))
        frame_index = max(0, int(frame_index))
        frames = frame_index % fps
        total_seconds = frame_index // fps
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _sync_overlay_text(self, frame_index, stream_label):
        mode = self.sync_overlay_mode.get()
        if mode == "Aus":
            return None
        parts = [stream_label]
        if mode in ("Frame", "Beides"):
            parts.append(f"F{int(frame_index):06d}")
        if mode in ("Timecode", "Beides"):
            parts.append(self._format_output_timecode(frame_index))
        if self.yolo_enabled.get() and not self._is_yolo_primary_model():
            parts.append("YOLO-S" if self.yolo_sync_postprocess.get() else "YOLO-A")
        return "  ".join(parts)

    def _draw_output_sync_overlay(self, frame_rgb, frame_index, stream_label):
        text = self._sync_overlay_text(frame_index, stream_label)
        if not text:
            return frame_rgb
        out = frame_rgb.copy()
        h, w = out.shape[:2]
        scale = max(0.7, min(2.2, w / 960.0))
        thickness = max(2, int(round(scale * 2)))
        margin = max(14, int(18 * scale))
        y = h - margin
        x = margin
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            out,
            (x - 8, y - th - baseline - 8),
            (x + tw + 8, y + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    def _delay_output_frame(self, buffer, frame, delay_frames):
        delay_frames = max(0, int(delay_frames))
        if delay_frames <= 0:
            buffer.clear()
            return frame
        buffer.append(frame.copy())
        if len(buffer) <= delay_frames:
            return buffer[0]
        return buffer.popleft()

    def start_live_output(self):
        self.stop_live_output()
        self._reset_output_sync_buffers()

        device_name = self.live_output_device.get()
        if device_name == "Keine DeckLink-Ausgabe gefunden":
            self.live_output_enabled.set(False)
            self.live_output_status.set("Keine DeckLink-Ausgabe gefunden.")
            return

        mode_label = self.live_output_mode.get()
        if mode_label not in DECKLINK_OUTPUT_MODES:
            self.live_output_enabled.set(False)
            self.live_output_status.set("Ungueltiger DeckLink-Modus.")
            return

        try:
            started = []
            if self.live_output_enabled.get():
                self.decklink_output = DeckLinkLiveOutput(
                    device_name,
                    mode_label,
                    status_callback=lambda text: self.root.after(0, lambda: self.live_output_status.set(text))
                )
                self.decklink_output.start()
                started.append(f"Fill: {device_name}")

            if self.live_key_output_enabled.get():
                key_device_name = self.live_key_output_device.get()
                if key_device_name == "Keine DeckLink-Ausgabe gefunden":
                    raise RuntimeError("Keine zweite DeckLink-Ausgabe fuer Alpha Matte gefunden.")
                if key_device_name == device_name and self.live_output_enabled.get():
                    raise RuntimeError("Fill und Alpha Matte muessen auf verschiedene DeckLink-Ausgaenge gelegt werden.")
                self.decklink_key_output = DeckLinkLiveOutput(
                    key_device_name,
                    mode_label,
                    status_callback=None
                )
                self.decklink_key_output.start()
                started.append(f"Key/Matte: {key_device_name}")

            if started:
                self.live_output_status.set("DeckLink aktiv:\n" + "\n".join(started) + f"\n{mode_label}")
                self._update_live_output_sync_status()
        except Exception as exc:
            if self.decklink_output is not None:
                self.decklink_output.stop()
            if self.decklink_key_output is not None:
                self.decklink_key_output.stop()
            self.decklink_output = None
            self.decklink_key_output = None
            self.live_output_enabled.set(False)
            self.live_key_output_enabled.set(False)
            self.live_output_status.set(f"DeckLink Startfehler: {exc}")

    def stop_live_output(self):
        self._reset_output_sync_buffers()
        if self.decklink_output is not None:
            self.decklink_output.stop()
            self.decklink_output = None
        if self.decklink_key_output is not None:
            self.decklink_key_output.stop()
            self.decklink_key_output = None
        if not self.live_output_enabled.get() and not self.live_key_output_enabled.get():
            self.live_output_status.set("DeckLink Output aus")

    def write_live_output_frame(self, rgb_frame, alpha_2d, processed_frame):
        if self.decklink_output is None and self.decklink_key_output is None:
            return
        try:
            frame_index = self.live_output_frame_counter
            self.live_output_frame_counter += 1
            if self.decklink_output is not None:
                fill_frame = self._make_view_frame(rgb_frame, alpha_2d, processed_frame)
                if fill_frame.shape[2] == 4:
                    fill_frame = fill_frame[:, :, :3]
                fill_frame = self._draw_output_sync_overlay(fill_frame, frame_index, "FILL")
                fill_frame = self._delay_output_frame(
                    self.fill_delay_buffer,
                    fill_frame,
                    self.fill_delay_frames.get()
                )
                self.decklink_output.write(fill_frame)
            if self.decklink_key_output is not None:
                alpha_u8 = self._alpha_to_u8(alpha_2d)
                alpha_rgb = cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB)
                alpha_rgb = self._draw_output_sync_overlay(alpha_rgb, frame_index, "MATTE")
                alpha_rgb = self._delay_output_frame(
                    self.matte_delay_buffer,
                    alpha_rgb,
                    self.matte_delay_frames.get()
                )
                self.decklink_key_output.write(alpha_rgb)
        except Exception as exc:
            self.live_output_status.set(f"DeckLink Schreibfehler: {exc}")

    def select_post_input(self):
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Datei fuer Postproduktion waehlen",
            filetypes=[
                ("Video/Bild", "*.mp4;*.mov;*.avi;*.mkv;*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("Videos", "*.mp4;*.mov;*.avi;*.mkv"),
                ("Bilder", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("Alle Dateien", "*.*"),
            ]
        )
        if not file_path:
            return

        self.post_input_path.set(file_path)
        if self.post_output_path.get() == "Kein Ziel gewaehlt":
            self.post_output_path.set(self._default_post_output_path(file_path))

    def select_post_output(self):
        from tkinter import filedialog

        input_path = self.post_input_path.get()
        initial = self._default_post_output_path(input_path) if os.path.exists(input_path) else "processed_output.mp4"
        ext = os.path.splitext(initial)[1].lower()
        filetypes = [
            ("Apple ProRes 4444 MOV", "*.mov"),
            ("MP4 Video", "*.mp4"),
            ("PNG Bild", "*.png"),
            ("JPEG Bild", "*.jpg"),
            ("Alle Dateien", "*.*"),
        ]
        file_path = filedialog.asksaveasfilename(
            title="Speicherziel waehlen",
            initialfile=os.path.basename(initial),
            initialdir=os.path.dirname(initial) if os.path.dirname(initial) else None,
            defaultextension=ext if ext else ".mp4",
            filetypes=filetypes
        )
        if file_path:
            self.post_output_path.set(file_path)

    def _default_post_output_path(self, input_path):
        base, ext = os.path.splitext(input_path)
        ext = ext.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            return base + "_processed.png"
        if self.bg_mode.get() == "Transparent":
            return base + "_processed.mov"
        return base + "_processed.mp4"

    def start_post_processing(self):
        if self.post_is_processing:
            return

        input_path = self.post_input_path.get()
        output_path = self.post_output_path.get()
        if not os.path.exists(input_path):
            self.post_status.set("Bitte zuerst eine gueltige Quelldatei waehlen.")
            return
        if output_path == "Kein Ziel gewaehlt":
            output_path = self._default_post_output_path(input_path)
            self.post_output_path.set(output_path)
        input_ext = os.path.splitext(input_path)[1].lower()
        output_ext = os.path.splitext(output_path)[1].lower()
        if self.bg_mode.get() == "Transparent":
            if input_ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and output_ext != ".png":
                output_path = os.path.splitext(output_path)[0] + ".png"
                self.post_output_path.set(output_path)
            elif input_ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and output_ext != ".mov":
                output_path = os.path.splitext(output_path)[0] + ".mov"
                self.post_output_path.set(output_path)

        self.post_is_processing = True
        self.post_progress.set(0)
        self.post_status.set("Verarbeitung startet ...")
        self.btn_post_process.configure(state="disabled")
        self.model_select.configure(state="disabled")
        threading.Thread(target=self._post_processing_worker, args=(input_path, output_path), daemon=True).start()

    def _set_post_progress(self, progress, status):
        self.post_progress.set(max(0.0, min(1.0, float(progress))))
        self.post_status.set(status)

    def _finish_post_processing(self, message, error=False):
        self.post_is_processing = False
        self.btn_post_process.configure(state="normal")
        self.model_select.configure(state="normal")
        self.post_status.set(message)
        if error:
            self.video_label.configure(image=self.empty_dummy_image, text=message)

    def _post_processing_worker(self, input_path, output_path):
        try:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self._process_post_image(input_path, output_path)
            else:
                self._process_post_video(input_path, output_path)
            self.root.after(0, lambda: self._finish_post_processing(f"Fertig gespeichert:\n{output_path}"))
        except Exception as exc:
            self.root.after(0, lambda e=exc: self._finish_post_processing(f"Fehler: {e}", error=True))

    def _compose_processed_frame(self, rgb_frame, alpha_2d, bg_cache=None, last_bg_mode=None):
        h, w = rgb_frame.shape[:2]
        current_mode = self.bg_mode.get()

        if current_mode == "Transparent":
            alpha_u8 = self._alpha_to_u8(alpha_2d)
            output_rgba = np.dstack((rgb_frame, alpha_u8))
            self.force_bg_update = False
            return output_rgba, None, current_mode

        if current_mode != last_bg_mode or self.force_bg_update or bg_cache is None:
            if current_mode == "Checker":
                bg_cache = self._get_checker_background(w, h)
            elif current_mode == "Green":
                bg_cache = np.zeros((h, w, 3), dtype=np.uint8)
                bg_cache[:] = (0, 255, 0)
            elif current_mode == "CustomImage":
                bg_cache = self._get_custom_background(w, h)
            else:
                bg_cache = np.zeros((h, w, 3), dtype=np.uint8) + 30
            last_bg_mode = current_mode
            self.force_bg_update = False

        fg_rgb = rgb_frame
        if current_mode == "Green":
            fg_rgb = _despill_green(fg_rgb, alpha_2d)

        alpha = alpha_2d[..., np.newaxis]
        output_rgb = (fg_rgb * alpha + bg_cache * (1.0 - alpha)).astype(np.uint8)
        return output_rgb, bg_cache, last_bg_mode

    def _process_post_rgb_frame(self, rgb_frame, segmenter, bg_cache=None, last_bg_mode=None):
        if self._is_yolo_primary_model():
            self._run_yolo_primary(rgb_frame)
            alpha_2d = self._make_yolo_alpha(rgb_frame.shape)
        else:
            mask_binary = segmenter.predict_mask(rgb_frame)
            alpha_2d = self._postprocess_to_alpha(rgb_frame, mask_binary)
            if self.yolo_sync_postprocess.get():
                self._run_yolo_postprocess_sync(rgb_frame)
            else:
                self._schedule_yolo_postprocess(rgb_frame)
            alpha_2d = self._apply_yolo_roi_to_alpha(alpha_2d)
        compose_rgb_frame, alpha_2d = self._apply_corridor_key(rgb_frame, alpha_2d)
        output_frame, bg_cache, last_bg_mode = self._compose_processed_frame(
            compose_rgb_frame,
            alpha_2d,
            bg_cache,
            last_bg_mode
        )
        return output_frame, alpha_2d, bg_cache, last_bg_mode

    def _process_post_image(self, input_path, output_path):
        image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("Bild konnte nicht gelesen werden.")

        rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self.model_lock:
            segmenter = self.segmenter
        output_frame, alpha_2d, _, _ = self._process_post_rgb_frame(rgb_frame, segmenter)
        if output_frame.shape[2] == 4:
            output_to_write = cv2.cvtColor(output_frame, cv2.COLOR_RGBA2BGRA)
        else:
            output_to_write = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, output_to_write):
            raise RuntimeError("Ausgabebild konnte nicht geschrieben werden.")

        preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
        preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
        preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
        self._set_latest_display(preview_frame, preview_alpha, preview_output)
        self.root.after(0, lambda: self._set_post_progress(1.0, "Bild fertig verarbeitet."))

    def _find_ffmpeg_executable(self):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        app_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(app_dir, "ffmpeg.exe"),
            os.path.join(app_dir, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(app_dir, "bin", "ffmpeg.exe"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _open_prores_4444_writer(self, output_path, width, height, fps):
        ffmpeg_path = self._find_ffmpeg_executable()
        if not ffmpeg_path:
            raise RuntimeError(
                "FFmpeg wurde nicht gefunden. Fuer transparente MOV-Dateien bitte FFmpeg installieren "
                "oder ffmpeg.exe in den Projektordner bzw. in einen ffmpeg\\bin-Unterordner legen."
            )

        command = [
            ffmpeg_path,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s:v", f"{int(width)}x{int(height)}",
            "-r", f"{float(fps):.6f}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "prores_ks",
            "-profile:v", "4",
            "-pix_fmt", "yuva444p10le",
            "-alpha_bits", "16",
            "-vendor", "apl0",
            output_path,
        ]
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except Exception as exc:
            raise RuntimeError(f"FFmpeg konnte nicht gestartet werden: {exc}") from exc

    def _process_post_transparent_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Video konnte nicht geoeffnet werden.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video-Aufloesung konnte nicht gelesen werden.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        writer = self._open_prores_4444_writer(output_path, width, height, fps)

        with self.model_lock:
            segmenter = self.segmenter

        bg_cache = None
        last_bg_mode = None
        processed = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output_frame, alpha_2d, bg_cache, last_bg_mode = self._process_post_rgb_frame(
                    rgb_frame, segmenter, bg_cache, last_bg_mode
                )
                if output_frame.shape[2] != 4:
                    alpha_u8 = self._alpha_to_u8(alpha_2d)
                    output_frame = np.dstack((rgb_frame, alpha_u8))

                try:
                    writer.stdin.write(np.ascontiguousarray(output_frame).tobytes())
                except Exception as exc:
                    raise RuntimeError(f"FFmpeg konnte keinen Frame schreiben: {exc}") from exc

                processed += 1

                if processed == 1 or processed % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - start)
                    proc_fps = processed / elapsed
                    progress = (processed / total_frames) if total_frames > 0 else 0.0
                    status = f"Exportiere ProRes 4444 Frame {processed}"
                    if total_frames > 0:
                        status += f"/{total_frames}"
                    status += f" ({proc_fps:.1f} FPS)"
                    preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
                    preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    self._set_latest_display(preview_frame, preview_alpha, preview_output)
                    self.root.after(0, lambda p=progress, s=status: self._set_post_progress(p, s))
        finally:
            cap.release()
            if writer.stdin:
                try:
                    writer.stdin.close()
                except Exception:
                    pass

        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg ProRes-Export fehlgeschlagen: {stderr.strip() or return_code}")

        self.root.after(0, lambda: self._set_post_progress(1.0, f"ProRes 4444 MOV fertig: {processed} Frames."))

    def _process_post_video(self, input_path, output_path):
        if self.bg_mode.get() == "Transparent":
            self._process_post_transparent_video(input_path, output_path)
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Video konnte nicht geoeffnet werden.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video-Aufloesung konnte nicht gelesen werden.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Ausgabevideo konnte nicht erstellt werden. Nutze als Ziel am besten .mp4.")

        with self.model_lock:
            segmenter = self.segmenter

        bg_cache = None
        last_bg_mode = None
        processed = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output_frame, alpha_2d, bg_cache, last_bg_mode = self._process_post_rgb_frame(
                    rgb_frame, segmenter, bg_cache, last_bg_mode
                )
                writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                processed += 1

                if processed == 1 or processed % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - start)
                    proc_fps = processed / elapsed
                    progress = (processed / total_frames) if total_frames > 0 else 0.0
                    status = f"Verarbeite Frame {processed}"
                    if total_frames > 0:
                        status += f"/{total_frames}"
                    status += f" ({proc_fps:.1f} FPS)"
                    preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
                    preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    self._set_latest_display(preview_frame, preview_alpha, preview_output)
                    self.root.after(0, lambda p=progress, s=status: self._set_post_progress(p, s))
        finally:
            cap.release()
            writer.release()

        self.root.after(0, lambda: self._set_post_progress(1.0, f"Video fertig: {processed} Frames."))


    def show_metrics_info(self):
        info = ctk.CTkToplevel(self.root)
        info.title("Live-Metriken")
        info.geometry("520x560")
        info.transient(self.root)
        info.grab_set()

        text = (
            "Bedeutung der Messwerte\n\n"
            "Modell: Aktuell geladener Segmentierer.\n\n"
            "Backend: OpenCV-Kamera-Backend, z.B. DirectShow oder Media Foundation.\n\n"
            "Quelle: Von der Kamera gelieferte Aufloesung und gemessene Eingangs-FPS. "
            "Der OpenCV-FPS-Wert wird nicht blind verwendet, weil Capture-Hardware wie "
            "BMD/Blackmagic haeufig 30 FPS meldet, obwohl das Signal z.B. 25 FPS hat.\n\n"
            "Verarbeitet: Bilder pro Sekunde, die komplett durch Kamera, Modell, "
            "Postprocessing und Anzeige-Pipeline laufen.\n\n"
            "Verworfen: Geschaetzte Differenz zwischen Quell-FPS und verarbeiteten FPS. "
            "OpenCV liefert meist keine echten Drop-Frame-Zaehler, daher ist das ein "
            "Vergleichswert fuer Modell-Benchmarks.\n\n"
            "Latenz gesamt: Durchschnittliche Zeit fuer einen kompletten Frame von Kamera-Read "
            "bis fertigem Ausgabebild.\n\n"
            "Main AI: Reine Modellzeit fuer die Haupt-Maskenberechnung, z.B. RVM, "
            "BiRefNet, MediaPipe oder YOLO als Hauptmodell.\n\n"

            "YOLO sync: YOLO-Nachbearbeitung auf demselben Frame nach der Main AI. "
            "Diese Zeit ist framegenau und Teil der Gesamt-Latenz.\n\n"
            "YOLO async: YOLO-Nachbearbeitung im Neben-Thread. Diese Zeit wird nicht zur "
            "Frame-Latenz addiert, weil die Hauptpipeline mit der zuletzt verfuegbaren "
            "Objektauswahl weiterlaeuft. Das Alter zeigt, wie alt die aktuelle YOLO-Erkennung ist.\n\n"
            "Eingehend: Geschaetzte Rohdatenrate der Kamera auf Basis von "
            "Aufloesung x 3 Farbkanalen x Quell-FPS.\n\n"
            "Verarbeitet: Geschaetzte Rohdatenrate der Frames, die die Pipeline wirklich "
            "fertig verarbeitet.\n\n"
            "Frames: Anzahl der seit Kamerastart fertig verarbeiteten Frames.\n\n"
            "Lesefehler: Fehlgeschlagene Kamera-Reads."
        )

        label = ctk.CTkLabel(info, text=text, justify="left", anchor="nw", wraplength=470)
        label.pack(padx=20, pady=20, fill="both", expand=True)
        ctk.CTkButton(info, text="Schliessen", command=info.destroy).pack(pady=(0, 16))

    def _reset_perf_metrics(self, source_fps=0.0, backend_name="-", capture_size=(0, 0)):
        now = time.perf_counter()
        self._perf_window_start = now
        self._perf_total_frames = 0
        self._perf_window_frames = 0
        self._perf_read_failures = 0
        self._perf_source_fps = float(source_fps or 0.0)
        self._perf_backend_name = backend_name or "-"
        self._perf_capture_size = capture_size
        self._perf_last_text_update = 0.0
        self._perf_ema = {
            "total_ms": 0.0,
            "infer_ms": 0.0,
            "post_ms": 0.0,
            "compose_ms": 0.0,
        }
        with self.metrics_lock:
            if self.is_running:
                self.metrics_text = "Performance\nMesse Daten ..."
            else:
                self.metrics_text = "Performance\nKamera gestoppt"

    def _update_perf_metrics(self, frame_shape, total_ms, infer_ms, post_ms, compose_ms):
        now = time.perf_counter()
        self._perf_total_frames += 1
        self._perf_window_frames += 1

        alpha = 0.18
        values = {
            "total_ms": total_ms,
            "infer_ms": infer_ms,
            "post_ms": post_ms,
            "compose_ms": compose_ms,
        }
        for key, value in values.items():
            previous = self._perf_ema.get(key, 0.0)
            self._perf_ema[key] = value if previous <= 0.0 else (previous * (1.0 - alpha) + value * alpha)

        elapsed = now - self._perf_window_start
        if elapsed < 0.5:
            return

        processed_fps = self._perf_window_frames / elapsed if elapsed > 0 else 0.0
        source_fps = self._perf_source_fps if self._perf_source_fps > 1.0 else 0.0
        dropped_fps = max(0.0, source_fps - processed_fps)
        dropped_percent = (dropped_fps / source_fps * 100.0) if source_fps > 0 else 0.0

        h, w = frame_shape[:2]
        bytes_per_frame = int(w * h * 3)
        incoming_mib = (bytes_per_frame * source_fps) / (1024 * 1024)
        incoming_mbit = incoming_mib * 8.0
        processed_mib = (bytes_per_frame * processed_fps) / (1024 * 1024)
        processed_mbit = processed_mib * 8.0
        source_label = f"{source_fps:.1f} FPS gemessen" if source_fps > 0 else "unbekannt"
        dropped_label = (
            f"{dropped_fps:.1f} FPS ({dropped_percent:.0f}%)"
            if source_fps > 0
            else "unbekannt"
        )

        text = (
            "Performance\n"
            f"Modell: {self.loaded_model_name}\n"
            f"Backend: {self._perf_backend_name}\n"
            f"Quelle: {int(self._perf_capture_size[0])}x{int(self._perf_capture_size[1])} @ {source_label}\n"
            f"Verarbeitet: {processed_fps:.1f} FPS\n"
            f"Verworfen: {dropped_label}\n"
            f"Latenz gesamt: {self._perf_ema['total_ms']:.1f} ms\n"
            f"Main AI: {self._perf_ema['infer_ms']:.1f} ms\n"
            f"Eingehend: {incoming_mib:.1f} MiB/s ({incoming_mbit:.0f} Mbit/s)\n"
            f"Verarbeitet: {processed_mib:.1f} MiB/s ({processed_mbit:.0f} Mbit/s)\n"
            f"Frames: {self._perf_total_frames}"
        )
        if self._yolo_selection_active():
            age_ms = (now - self.yolo_last_analysis_time) * 1000.0 if self.yolo_last_analysis_time > 0 else 0.0
            if self.yolo_last_analysis_ms <= 0:
                text += "\nYOLO Analyse: wartet"
            elif self._is_yolo_primary_model():
                text += f"\nYOLO Analyse: {self.yolo_last_analysis_ms:.1f} ms"
            elif self.yolo_sync_postprocess.get():
                text += f"\nYOLO sync: {self.yolo_last_analysis_ms:.1f} ms"
            else:
                text += f"\nYOLO async: {self.yolo_last_analysis_ms:.1f} ms, Alter {age_ms:.0f} ms"
        if self._perf_read_failures:
            text += f"\nLesefehler: {self._perf_read_failures}"

        with self.metrics_lock:
            self.metrics_text = text

        self._perf_window_start = now
        self._perf_window_frames = 0

    def change_main_ai_device(self, choice):
        if self.loaded_model_name == "YOLO" or self.model_name.get() == "YOLO":
            self.model_status = "YOLO nutzt die separate YOLO-Hardwareauswahl."
            self.model_status_label.configure(text=self.model_status)
            return
        self.change_model(self.loaded_model_name)

    def change_model(self, choice):
        was_running = self.is_running
        if was_running:
            self._stop_camera_internal()

        self.model_select.configure(state="disabled")
        if hasattr(self, "main_ai_device_select"):
            self.main_ai_device_select.configure(state="disabled")
        self.video_label.configure(image=self.empty_dummy_image, text=f"Lade Modell: {choice}...")
        self.model_status_label.configure(text=f"Lade Modell: {choice} ({self._main_ai_device_mode_note()})...")
        threading.Thread(target=self._load_model_worker, args=(choice, was_running), daemon=True).start()

    def _load_model_worker(self, choice, restart_camera):
        try:
            if choice == "YOLO":
                detector = YoloObjectDetector(
                    self.yolo_model_name.get(),
                    force_device=self._resolve_yolo_force_device(False)
                )
                with self.yolo_model_lock:
                    self.yolo_detector = detector
                with self.model_lock:
                    self.segmenter = None
                status = f"Modell bereit: YOLO ({detector.device_label})"
                if detector.device_hint:
                    status += "\n" + str(detector.device_hint)
                self.model_status = status
                self.root.after(
                    0,
                    lambda: self.yolo_status.set(
                        f"YOLO als Hauptmodell aktiv: {self.yolo_model_name.get()} ({detector.device_label})"
                    )
                )
            else:
                new_segmenter = create_segmentation_model(
                    choice,
                    self.model_path,
                    self._resolve_main_ai_force_device()
                )
                with self.model_lock:
                    self.segmenter = new_segmenter
                self.model_status = self._format_model_status(new_segmenter, choice)
            self.root.after(0, lambda: self._finish_model_load(choice, restart_camera, None))
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._finish_model_load(choice, False, error))

    def _finish_model_load(self, choice, restart_camera, error):
        self.model_select.configure(state="normal")
        if hasattr(self, "main_ai_device_select"):
            self.main_ai_device_select.configure(state="normal")
        if error is None:
            self.model_name.set(choice)
            self.loaded_model_name = choice
            self.model_status_label.configure(text=self.model_status)
            if choice != "YOLO" and not self.yolo_enabled.get():
                self.yolo_detections = []
                self.yolo_selected_keys.clear()
                self.yolo_seen_keys.clear()
                self.yolo_status.set("YOLO-Nachbearbeitung aus")
                self._refresh_yolo_detection_list()
            if self.app_mode.get() == "Postproduktion":
                self.video_label.configure(image=self.empty_dummy_image, text="Postproduktion bereit")
            else:
                self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")
            if restart_camera:
                self.root.after(300, self._start_camera_internal)
            return

        self.model_name.set(self.loaded_model_name)
        self.model_status = f"Fehler beim Laden von {choice}: {error}"
        self.model_status_label.configure(text=self.model_status)
        self.video_label.configure(image=self.empty_dummy_image, text=self.model_status)

    def refresh_cameras(self):
        if getattr(self, "_camera_refresh_running", False):
            return
        self._camera_refresh_running = True
        try:
            self.btn_refresh_cameras.configure(text="Suche laeuft...", state="disabled")
        except Exception:
            pass

        def worker():
            try:
                cameras = run_with_timeout(get_available_cameras, ["Keine Kamera gefunden"], timeout=8.0)
                if cameras == ["Keine Kamera gefunden"]:
                    cameras = []
                decklink_inputs = run_with_timeout(get_decklink_input_devices, [], timeout=5.0)
                values = [*cameras, *[f"DeckLink: {name}" for name in decklink_inputs]]
                if not values:
                    values = ["Keine Live-Quelle gefunden"]
                error = None
            except Exception as exc:
                values = None
                error = exc
            self.root.after(0, lambda: self._finish_camera_refresh(values, error))

        threading.Thread(target=worker, name="CameraRefresh", daemon=True).start()

    def _finish_camera_refresh(self, values, error=None):
        self._camera_refresh_running = False
        try:
            self.btn_refresh_cameras.configure(text="Kameras neu suchen", state="normal")
        except Exception:
            pass
        if error is not None:
            self.video_label.configure(text=f"Kamerasuche Fehler: {error}")
            return
        if not values:
            values = ["Keine Live-Quelle gefunden"]

        previous_source = self.current_live_source or self.camera_select.get()
        self.camera_select.configure(values=values)
        selected_source = previous_source if previous_source in values else values[0]
        self.camera_select.set(selected_source)
        self.current_live_source = selected_source
        if selected_source != "Keine Live-Quelle gefunden" and not selected_source.startswith("DeckLink: "):
            try:
                self.current_camera_index = parse_camera_index(selected_source)
            except Exception:
                self.current_camera_index = 0

    def trigger_background_load(self):
        threading.Thread(target=self._applescript_worker, daemon=True).start()

    def _applescript_worker(self):
        # Automatische Erkennung ob Mac oder Windows!
        if sys.platform == "darwin":
            script = '''
            set f to choose file with prompt "Hintergrundbild wählen"
            POSIX path of f
            '''
            try:
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                if result.returncode == 0:
                    file_path = result.stdout.strip()
                    self._load_image_from_path(file_path)
            except Exception as e:
                print(f"macOS Finder Fehler: {e}")
        else:
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.jpg;*.jpeg;*.png")])
            if file_path:
                self._load_image_from_path(file_path)

    def _load_image_from_path(self, file_path):
        if file_path and os.path.exists(file_path):
            try:
                pil_img = Image.open(file_path).convert('RGB')
                bg_rgb = np.array(pil_img)
                self.custom_background_source = bg_rgb
                self.custom_background_image = _center_crop_resize_rgb(bg_rgb, self.ui_w, self.ui_h)
                self.force_bg_update = True
                self.root.after(0, lambda: self.bg_mode.set("CustomImage"))
            except Exception as img_e:
                print(f"Fehler beim Verarbeiten des Bildes: {img_e}")

    def change_camera(self, choice):
        try:
            if choice == self.current_live_source:
                return

            was_running = self.is_running
            if was_running:
                self._stop_camera_internal(preserve_preview=True)

            self.current_live_source = choice
            if choice != "Keine Live-Quelle gefunden" and not choice.startswith("DeckLink: "):
                self.current_camera_index = parse_camera_index(choice)

            if was_running:
                self.root.after(150, lambda: self._start_camera_internal(preserve_preview=True))
        except Exception as e:
            print(f"Fehler beim Kamerawechsel: {e}")

    def toggle_camera(self):
        if not self.is_running:
            self._start_camera_internal()
        else:
            self._stop_camera_internal()

    def _start_camera_internal(self, preserve_preview=False):
        if self.is_running: return

        self.yolo_async_stop = False
        if not preserve_preview:
            self.latest_pil_image = None
            self.latest_display_payload = None
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera startet...")

        source = self.current_live_source or self.camera_select.get()
        if source == "Keine Live-Quelle gefunden":
            self.video_label.configure(image=self.empty_dummy_image, text="Keine Live-Quelle gefunden")
            return
        if source.startswith("DeckLink: "):
            device_name = source.split("DeckLink: ", 1)[1]
            self.cap = DeckLinkLiveInput(device_name, self.live_output_mode.get())
            try:
                self.cap.open()
            except Exception as exc:
                self.cap = None
                self.video_label.configure(text=f"DeckLink Input Fehler: {exc}")
                return
            backend_name = "DeckLink SDK"
        else:
            self.cap, backend_name = open_camera(self.current_camera_index)

        if self.cap is not None and self.cap.isOpened():
            if not isinstance(self.cap, DeckLinkLiveInput):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            measured_fps = 0.0 if isinstance(self.cap, DeckLinkLiveInput) else measure_camera_input_fps(self.cap)
            if measured_fps > 1.0:
                actual_fps = measured_fps
            elif reported_fps > 1.0 and reported_fps <= 240.0:
                actual_fps = reported_fps
            else:
                actual_fps = 0.0
            self._reset_perf_metrics(source_fps=actual_fps, backend_name=backend_name, capture_size=(actual_w, actual_h))

            self.is_running = True
            self.btn_toggle.configure(text="Kamera Stoppen", fg_color="#d32f2f", hover_color="#9a0007")
            self.video_label.configure(text="")

            threading.Thread(target=self.video_worker_loop, daemon=True).start()
        else:
            self.video_label.configure(text="Fehler: Kamera besetzt oder nicht verfuegbar")

    def _stop_camera_internal(self, preserve_preview=False):
        self.is_running = False
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_toggle.configure(text="Kamera Starten", fg_color=["#3a7ebf", "#1f538d"])
        if not preserve_preview:
            self.latest_pil_image = None
            self.latest_display_payload = None
        self._reset_perf_metrics()
        if not preserve_preview:
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")

    def update_gui_loop(self):
        if (self.is_running or self.app_mode.get() == "Postproduktion") and self.latest_pil_image is not None:
            img_tk = ctk.CTkImage(light_image=self.latest_pil_image, size=(self.ui_w, self.ui_h))
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk
        with self.metrics_lock:
            metrics_text = self.metrics_text
        if metrics_text != self.metrics_last_gui_text:
            self.metrics_label.configure(text=metrics_text)
            self.metrics_last_gui_text = metrics_text
        self.root.after(33, self.update_gui_loop)

    def video_worker_loop(self):
        bg_cache = None
        last_bg_mode = ""

        while self.is_running:
            if self.cap is None: break
            frame_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                self._perf_read_failures += 1
                if isinstance(self.cap, DeckLinkLiveInput):
                    time.sleep(0.02)
                    continue
                break

            ui_frame = cv2.resize(frame, (self.ui_w, self.ui_h))
            rgb_frame = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)

            try:
                infer_start = time.perf_counter()
                if self._is_yolo_primary_model():
                    self._run_yolo_primary(rgb_frame)
                    mask_binary = None
                    alpha_2d = self._make_yolo_alpha(rgb_frame.shape)
                else:
                    with self.model_lock:
                        segmenter = self.segmenter
                    mask_binary = segmenter.predict_mask(rgb_frame)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
            except Exception as model_error:
                self.is_running = False
                self.root.after(0, lambda e=model_error: self.video_label.configure(
                    image=self.empty_dummy_image,
                    text=f"Modellfehler: {e}"
                ))
                break

            post_start = time.perf_counter()
            if not self._is_yolo_primary_model():
                alpha_2d = self._postprocess_to_alpha(rgb_frame, mask_binary)
                if self.yolo_sync_postprocess.get():
                    self._run_yolo_postprocess_sync(rgb_frame)
                else:
                    self._schedule_yolo_postprocess(rgb_frame)
                alpha_2d = self._apply_yolo_roi_to_alpha(alpha_2d)
            compose_rgb_frame, alpha_2d = self._apply_corridor_key(rgb_frame, alpha_2d)
            post_ms = (time.perf_counter() - post_start) * 1000.0

            compose_start = time.perf_counter()
            output_final, bg_cache, last_bg_mode = self._compose_processed_frame(
                compose_rgb_frame, alpha_2d, bg_cache, last_bg_mode
            )
            self._set_latest_display(rgb_frame, alpha_2d, output_final)
            self.write_live_output_frame(rgb_frame, alpha_2d, output_final)
            compose_ms = (time.perf_counter() - compose_start) * 1000.0
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            self._update_perf_metrics(frame.shape, total_ms, infer_ms, post_ms, compose_ms)

        if self.cap:
            self.cap.release()
        self.cap = None

    def on_closing(self):
        self.yolo_async_stop = True
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        self._stop_camera_internal()
        self.stop_live_output()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = FoolproofSyncApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
