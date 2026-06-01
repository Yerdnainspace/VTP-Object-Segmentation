import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
import threading
import importlib.util
import os
import sys
import urllib.request
import ssl
import subprocess
import time
import traceback

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
                print(f"Kamera {index} geoeffnet mit Backend: {backend_name}")
                return cap, backend_name
            time.sleep(0.05)

        cap.release()

    return None, None


def get_available_cameras():
    cameras = []
    for i in range(10):
        cap, _ = open_camera(i)
        if cap is not None:
            cameras.append(f"Kamera {i}")
            cap.release()
    return cameras if cameras else ["Keine Kamera gefunden"]


MODEL_OPTIONS = ["MediaPipe Selfie", "BiRefNet", "RVM ByteDance"]


class MediaPipeSelfieModel:
    def __init__(self, model_path):
        print("Initialisiere MediaPipe Engine...")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        print("MediaPipe Engine bereit.")

    def predict_mask(self, rgb_frame):
        ai_frame = cv2.resize(rgb_frame, (512, 288))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ai_frame)
        result = self.segmenter.segment(mp_image)
        mask_raw = result.category_mask.numpy_view()
        return (mask_raw > 0).astype(np.uint8) * 255


class BiRefNetModel:
    def __init__(self):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Lade BiRefNet auf {self.device}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        print("BiRefNet bereit.")

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
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = self._extract_prediction(output)
            pred = pred.sigmoid()

        mask = pred[0].detach().float().cpu().squeeze().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


class RVMByteDanceModel:
    def __init__(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RVM benoetigt PyTorch. Installiere z.B.: "
                "pip install torch torchvision"
            ) from exc

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec = [None, None, None, None]
        print(f"Lade RVM ByteDance auf {self.device}...")
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "mobilenetv3",
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()
        print("RVM ByteDance bereit.")

    def predict_mask(self, rgb_frame):
        torch = self.torch
        h, w = rgb_frame.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        model_w = max(32, int(w * scale) // 32 * 32)
        model_h = max(32, int(h * scale) // 32 * 32)
        resized = cv2.resize(rgb_frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            _, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=0.25)

        mask = pha[0, 0].detach().float().cpu().numpy()
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def create_segmentation_model(model_name, mediapipe_model_path):
    if model_name == "MediaPipe Selfie":
        return MediaPipeSelfieModel(mediapipe_model_path)
    if model_name == "BiRefNet":
        return BiRefNetModel()
    if model_name == "RVM ByteDance":
        return RVMByteDanceModel()
    raise ValueError(f"Unbekanntes Modell: {model_name}")


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
        self.model_lock = threading.Lock()
        self.segmenter = create_segmentation_model(self.model_name.get(), self.model_path)
        self.loaded_model_name = self.model_name.get()
        self.model_status = "Modell bereit: MediaPipe Selfie"

        self.cap = None
        self.is_running = False
        self.current_camera_index = 0

        self.bg_mode = ctk.StringVar(value="Blur")
        self.custom_background_image = None
        self.force_bg_update = False

        self.edge_erode = ctk.IntVar(value=3)
        self.edge_soft = ctk.IntVar(value=7)

        self.latest_pil_image = None
        self.empty_dummy_image = ctk.CTkImage(light_image=Image.new("RGB", (1, 1)), size=(1, 1))

        self.setup_gui()
        self.update_gui_loop()

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

    def setup_gui(self):
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

        self.camera_select = ctk.CTkOptionMenu(control_panel, values=get_available_cameras(),
                                               command=self.change_camera)
        self.camera_select.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_cameras = ctk.CTkButton(control_panel, text="Kameras neu suchen",
                                                command=self.refresh_cameras)
        self.btn_refresh_cameras.pack(pady=5, padx=10, fill="x")

        self.btn_toggle = ctk.CTkButton(control_panel, text="Kamera Starten", command=self.toggle_camera)
        self.btn_toggle.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Hintergrund-Effekt", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(20, 5))

        rb_blur = ctk.CTkRadioButton(control_panel, text="Weichzeichnen (Teams)", variable=self.bg_mode, value="Blur")
        rb_blur.pack(pady=5, anchor="w", padx=20)

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

    def update_bg_button_state(self, *args):
        if self.bg_mode.get() == "CustomImage":
            self.btn_load_bg.configure(state="normal")
        else:
            self.btn_load_bg.configure(state="disabled")

    def change_model(self, choice):
        was_running = self.is_running
        if was_running:
            self._stop_camera_internal()

        self.model_select.configure(state="disabled")
        self.video_label.configure(image=self.empty_dummy_image, text=f"Lade Modell: {choice}...")
        self.model_status_label.configure(text=f"Lade Modell: {choice}...")
        threading.Thread(target=self._load_model_worker, args=(choice, was_running), daemon=True).start()

    def _load_model_worker(self, choice, restart_camera):
        try:
            new_segmenter = create_segmentation_model(choice, self.model_path)
            with self.model_lock:
                self.segmenter = new_segmenter
            self.model_status = f"Modell bereit: {choice}"
            self.root.after(0, lambda: self._finish_model_load(choice, restart_camera, None))
        except Exception as exc:
            traceback.print_exc()
            self.root.after(0, lambda error=exc: self._finish_model_load(choice, False, error))

    def _finish_model_load(self, choice, restart_camera, error):
        self.model_select.configure(state="normal")
        if error is None:
            self.model_name.set(choice)
            self.loaded_model_name = choice
            self.model_status_label.configure(text=self.model_status)
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")
            if restart_camera:
                self.root.after(300, self._start_camera_internal)
            return

        self.model_name.set(self.loaded_model_name)
        self.model_status = f"Fehler beim Laden von {choice}: {error}"
        self.model_status_label.configure(text=self.model_status)
        self.video_label.configure(image=self.empty_dummy_image, text=self.model_status)

    def refresh_cameras(self):
        values = get_available_cameras()
        self.camera_select.configure(values=values)
        self.camera_select.set(values[0])
        if values[0] != "Keine Kamera gefunden":
            try:
                self.current_camera_index = int(values[0].split(' ')[1])
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
                self.custom_background_image = cv2.resize(bg_rgb, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                self.force_bg_update = True
                self.root.after(0, lambda: self.bg_mode.set("CustomImage"))
            except Exception as img_e:
                print(f"Fehler beim Verarbeiten des Bildes: {img_e}")

    def change_camera(self, choice):
        try:
            new_index = int(choice.split(' ')[1])
            if new_index == self.current_camera_index:
                return

            was_running = self.is_running
            if was_running:
                self._stop_camera_internal()

            self.current_camera_index = new_index

            if was_running:
                self.root.after(300, self._start_camera_internal)
        except Exception as e:
            print(f"Fehler beim Kamerawechsel: {e}")

    def toggle_camera(self):
        if not self.is_running:
            self._start_camera_internal()
        else:
            self._stop_camera_internal()

    def _start_camera_internal(self):
        if self.is_running: return

        self.latest_pil_image = None
        self.video_label.configure(image=self.empty_dummy_image, text="Kamera startet...")

        self.cap, backend_name = open_camera(self.current_camera_index)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.is_running = True
            self.btn_toggle.configure(text="Kamera Stoppen", fg_color="#d32f2f", hover_color="#9a0007")
            self.video_label.configure(text="")

            threading.Thread(target=self.video_worker_loop, daemon=True).start()
        else:
            self.video_label.configure(text="Fehler: Kamera besetzt oder nicht verfuegbar")

    def _stop_camera_internal(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

        self.btn_toggle.configure(text="Kamera Starten", fg_color=["#3a7ebf", "#1f538d"])
        self.latest_pil_image = None
        self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")

    def update_gui_loop(self):
        if self.is_running and self.latest_pil_image is not None:
            img_tk = ctk.CTkImage(light_image=self.latest_pil_image, size=(self.ui_w, self.ui_h))
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk
        self.root.after(33, self.update_gui_loop)

    def video_worker_loop(self):
        bg_cache = None
        last_bg_mode = ""

        while self.is_running:
            if self.cap is None: break
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ui_frame = cv2.resize(frame, (self.ui_w, self.ui_h))
            rgb_frame = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)

            try:
                with self.model_lock:
                    segmenter = self.segmenter
                mask_binary = segmenter.predict_mask(rgb_frame)
            except Exception as model_error:
                self.is_running = False
                self.root.after(0, lambda e=model_error: self.video_label.configure(
                    image=self.empty_dummy_image,
                    text=f"Modellfehler: {e}"
                ))
                break

            erode_size = int(self.edge_erode.get())
            if erode_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
                mask_binary = cv2.erode(mask_binary, kernel, iterations=1)

            soft_size = int(self.edge_soft.get())
            if soft_size % 2 == 0: soft_size += 1
            mask_blurred = cv2.GaussianBlur(mask_binary, (soft_size, soft_size), 0)

            mask_ui = cv2.resize(mask_blurred, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
            alpha = (mask_ui / 255.0)[..., np.newaxis]

            current_mode = self.bg_mode.get()

            if current_mode != last_bg_mode or current_mode == "Blur" or self.force_bg_update:
                if current_mode == "Blur":
                    bg_cache = cv2.GaussianBlur(rgb_frame, (41, 41), 0)
                elif current_mode == "Green":
                    bg_cache = np.zeros((self.ui_h, self.ui_w, 3), dtype=np.uint8)
                    bg_cache[:] = (0, 255, 0)
                elif current_mode == "CustomImage":
                    if self.custom_background_image is not None:
                        bg_cache = self.custom_background_image.copy()
                    else:
                        bg_cache = np.zeros((self.ui_h, self.ui_w, 3), dtype=np.uint8) + 30

                last_bg_mode = current_mode
                self.force_bg_update = False

            if bg_cache is None:
                bg_cache = np.zeros((self.ui_h, self.ui_w, 3), dtype=np.uint8) + 30

            output_final = (rgb_frame * alpha + bg_cache * (1.0 - alpha)).astype(np.uint8)
            self.latest_pil_image = Image.fromarray(output_final)

        if self.cap:
            self.cap.release()
        self.cap = None

    def on_closing(self):
        self._stop_camera_internal()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = FoolproofSyncApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
