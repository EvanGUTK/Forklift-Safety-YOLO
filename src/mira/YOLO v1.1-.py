import os
import time
import math
import uuid
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Optional, Union, Tuple

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np
import torch
import torch.cuda as cuda
from ultralytics import YOLO

def cuda_available() -> bool:
    return cuda.is_available()

def gpu_name() -> str:
    return cuda.get_device_name(0) if cuda_available() else "CPU"

def device_arg(use_gpu: bool):
    return 0 if (use_gpu and cuda_available()) else "cpu"

DEF_ALL_YOLO_DET = [
    # YOLOv8
    "yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt",
    # YOLOv9
    "yolov9t.pt","yolov9s.pt","yolov9m.pt","yolov9c.pt","yolov9e.pt",
    # YOLOv10
    "yolov10n.pt","yolov10s.pt","yolov10m.pt","yolov10l.pt","yolov10x.pt",
]

def discover_local_yolo_models(search_roots=None):
    if search_roots is None:
        search_roots = [".", str(Path.home() / ".cache"), str(Path.home() / "models")]
    seen, found = set(), []
    for root in search_roots:
        try:
            for p in Path(root).rglob("*.pt"):
                name = p.name
                if name not in seen and name.lower().startswith("yolo"):
                    seen.add(name)
                    found.append(name)
        except Exception:
            pass
    found_yolo = [n for n in found if n.lower().startswith("yolo")]
    found_other = [n for n in found if not n.lower().startswith("yolo")]
    return found_yolo + found_other

def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def build_model_list(env_value: Optional[str]) -> list:
    if not env_value or env_value.strip().upper() == "ALL":
        local = discover_local_yolo_models()
        return dedup_keep_order(DEF_ALL_YOLO_DET + local)
    return [t.strip() for t in env_value.split(",") if t.strip()]

def int_or_str(x: str):
    return int(x) if x.isdigit() else x

def parse_list_env(name: str, default: str):
    return [t.strip() for t in os.getenv(name, default).split(",") if t.strip()]

def fit_font_scale(text: str, box_w: int, box_h: int, base_scale=0.9):
    scale = base_scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    if tw > box_w * 0.92:
        scale *= (box_w * 0.92) / max(1, tw)
    if th > box_h * 0.82:
        scale *= (box_h * 0.82) / max(1, th)
    return max(scale, 0.55)

def put_text_sharp(img, text, org, box_wh=None, color=(255,255,255), base_scale=0.9):
    if box_wh:
        bw, bh = box_wh
        scale = fit_font_scale(text, bw, bh, base_scale=base_scale)
    else:
        h, w = img.shape[:2]
        scale = max(0.65, 0.9 * (w / 1920))
    thickness = max(2, int(scale * 2.2))
    ox, oy = org
    cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_button(img, rect, label, active=False, palette=None):
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    bg = (42, 42, 42) if not active else (78, 78, 78)
    if palette:
        bg = palette.get("btn", bg) if not active else palette.get("active", (78, 78, 78))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)
    bw, bh = x2 - x1, y2 - y1
    fs = fit_font_scale(label, bw, bh)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    tx = x1 + (bw - tw) // 2
    ty = y1 + (bh + th) // 2 - 3
    col = (255, 255, 255) if not palette else palette.get("text", (255, 255, 255))
    put_text_sharp(img, label, (tx, ty), (bw, bh), color=col, base_scale=fs)

def point_in_rect(pt, rect):
    if pt is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_bottom_right_block(img, lines, pad=10, palette=None):
    if not lines: return
    h, w = img.shape[:2]
    fs_list, sizes = [], []
    for t in lines:
        fs = fit_font_scale(t, int(w * 0.38), 46)
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        fs_list.append(fs); sizes.append((tw, th))
    block_w = max(s[0] for s in sizes) + pad * 2
    block_h = sum(s[1] for s in sizes) + pad * (len(lines) + 1)
    x1 = w - block_w - 14; y1 = h - block_h - 14; x2 = w - 14; y2 = h - 14
    ov = img.copy()
    bg = (28, 28, 28) if not palette else palette.get("hud_bg", (28, 28, 28))
    cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    col = (255, 255, 255) if not palette else palette.get("text", (255, 255, 255))
    y = y1 + pad + sizes[0][1]
    for i, text in enumerate(lines):
        tw, th = sizes[i]
        tx = x2 - pad - tw
        put_text_sharp(img, text, (tx, y), (tw + pad, th + pad), color=col, base_scale=fs_list[i])
        y += th + pad

def draw_toast(img, text, until_ts, pad=8, palette=None):
    if time.time() > until_ts: return
    h, w = img.shape[:2]
    fs = fit_font_scale(text, int(w * 0.62), 44)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    x1, y1 = 12, 12; x2, y2 = x1 + tw + pad * 2, y1 + th + pad * 2
    ov = img.copy()
    bg = (18, 18, 18) if not palette else palette.get("toast_bg", (18, 18, 18))
    cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(ov, 0.86, img, 0.14, 0, img)
    col = (255, 255, 255) if not palette else palette.get("text", (255, 255, 255))
    put_text_sharp(img, text, (x1 + pad, y2 - pad), (tw, th), color=col, base_scale=fs)

THEMES = [
    ("Classic", {"btn": (40, 40, 40), "active": (70, 70, 70), "text": (255, 255, 255), "head": (35, 35, 35), "hud_bg": (30, 30, 30), "toast_bg": (20, 20, 20)}),
    ("Vols Orange", {"btn": (30, 60, 150), "active": (45, 90, 210), "text": (255, 255, 255), "head": (30, 80, 200), "accent": (130, 130, 255), "hud_bg": (35, 35, 35), "toast_bg": (25, 25, 25)}),
    ("Smokey Gray", {"btn": (60, 60, 60), "active": (90, 90, 90), "text": (255, 255, 255), "head": (80, 80, 80), "accent": (180, 180, 180), "hud_bg": (40, 40, 40), "toast_bg": (30, 30, 30)}),
    ("Neyland Checkerboard", {"btn": (40, 40, 40), "active": (70, 70, 70), "text": (255, 255, 255), "head": (35, 35, 35), "hud_bg": (30, 30, 30), "toast_bg": (20, 20, 20)}),
    ("Vol Navy", {"btn": (90, 70, 40), "active": (120, 95, 60), "text": (255, 255, 255), "head": (70, 50, 30), "hud_bg": (40, 35, 30), "toast_bg": (25, 20, 18)}),
]

def draw_header(img, rect, theme_name):
    x1, y1, x2, y2 = rect
    if theme_name == "Neyland Checkerboard":
        tile = max(20, (x2 - x1) // 16)
        for i in range((x2 - x1) // tile + 2):
            c = (255, 255, 255) if i % 2 == 0 else (0, 130, 255)
            cv2.rectangle(img, (x1 + i * tile, y1), (x1 + (i + 1) * tile, y2), c, -1)
        overlay = img.copy()
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    else:
        palette = THEMES[[t[0] for t in THEMES].index(theme_name)][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), palette.get("head", (35, 35, 35)), -1)

SESSION_ID = str(uuid.uuid4())[:6]
NOTES_DIR = Path("notes"); NOTES_DIR.mkdir(exist_ok=True)
NOTES_FILE = NOTES_DIR / (datetime.now().strftime("log_%Y%m%d.txt"))
NOTES_RING = deque(maxlen=6)

def note(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print("NOTE:", line)
    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def add_note(text):
    NOTES_RING.appendleft(text)
    note(text)

def draw_notes(img):
    if not NOTES_RING: return
    h, w = img.shape[:2]
    x, y = 12, 60
    for line in list(NOTES_RING):
        put_text_sharp(img, line, (x, y), (int(w*0.6), 32), color=(240,240,240), base_scale=0.8)
        y += 28

def point_in_poly(pt: Tuple[int, int], poly: List[Tuple[int, int]]):
    x, y = pt; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1):
            inside = not inside
    return inside

def draw_zones(img, zones: List[Tuple[List[Tuple[int, int]], str]], current_zone: List[Tuple[int, int]]):
    for poly, name in zones:
        if len(poly) < 2: continue
        cv2.polylines(img, [np.int32(poly)], True, (60, 200, 255), 2, cv2.LINE_AA)
        cx = int(sum(p[0] for p in poly) / len(poly)); cy = int(sum(p[1] for p in poly) / len(poly))
        put_text_sharp(img, name, (cx - 20, cy), (100, 28), color=(255,255,255), base_scale=0.6)
    if current_zone:
        cv2.polylines(img, [np.int32(current_zone)], False, (60, 200, 255), 1, cv2.LINE_AA)

class Trails:
    def __init__(self, max_len=40, fade=0.90):
        self.points = []
        self.max_len = max_len
        self.fade = fade

    def push(self, centers: List[Tuple[int, int]]):
        for (x, y) in centers:
            self.points.append([x, y, 1.0])
        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len:]

    def draw(self, img, color=(0, 180, 255)):
        if not self.points: return
        for i in range(1, len(self.points)):
            x1, y1, a1 = self.points[i - 1]
            x2, y2, a2 = self.points[i]
            a = min(a1, a2)
            c = (int(color[0] * a), int(color[1] * a), int(color[2] * a))
            cv2.line(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        for p in self.points:
            p[2] *= self.fade
        self.points = [p for p in self.points if p[2] > 0.08]

class Heatmap:
    def __init__(self, decay=0.94):
        self.decay = decay
        self.map = None

    def update(self, frame_shape, boxes: List[Tuple[int, int, int, int]]):
        h, w = frame_shape[:2]
        if self.map is None or self.map.shape[:2] != (h, w):
            self.map = np.zeros((h, w), dtype=np.float32)
        self.map *= self.decay
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                self.map[y1:y2, x1:x2] += 0.8
        np.clip(self.map, 0, 8.0, out=self.map)

    def draw(self, img):
        if self.map is None: return
        norm = (self.map / (self.map.max() + 1e-6) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.addWeighted(colored, 0.18, img, 0.82, 0, img)

def capture_size_from_imgsz(imgsz: int):
    if imgsz >= 1280: return (1280, 720)
    if imgsz >= 960:  return (960, 540)
    return (640, 480)

def open_cap(src: Union[int, str], imgsz: int) -> Optional[cv2.VideoCapture]:
    if isinstance(src, int): backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
    else: backends = [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(src, be)
        if not cap.isOpened():
            cap.release(); continue
        w, h = capture_size_from_imgsz(imgsz)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok = True
        for _ in range(3):
            ret, _ = cap.read()
            if not ret: ok = False; break
        if ok: return cap
        cap.release()
    return None

def probe_cap(src: Union[int, str], imgsz: int, tries=5) -> bool:
    cap = open_cap(src, imgsz)
    if cap is None: return False
    ok = True
    for _ in range(tries):
        ret, _ = cap.read()
        if not ret: ok = False; break
    cap.release()
    return ok

def release_caps(caps: List[cv2.VideoCapture]):
    for c in caps:
        try: c.release()
        except Exception: pass

def load_model_on_device(weights: str, use_gpu: bool) -> YOLO:
    """Create a fresh model on the requested device (GPU 0 or CPU)."""
    m = YOLO(weights)
    try:
        if use_gpu and cuda_available():
            m.to(0)  # cuda:0
        else:
            m.to("cpu")
    except Exception:
        pass
    return m

def switch_device_inplace(current_model: YOLO, weights: str, use_gpu: bool) -> YOLO:
    """Rebuild the model on target device and free old CUDA memory when leaving GPU."""
    if not use_gpu and cuda_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return load_model_on_device(weights, use_gpu)

def run_predict(model: YOLO, frame, use_gpu_flag: bool, imgsz: int, conf: float, iou: float):
    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=[0],
        device=device_arg(use_gpu_flag),
        verbose=False,
    )
    r = results[0]
    people_count = 0
    centers = []
    boxes_xyxy = []
    if getattr(r, "boxes", None) is not None:
        people_count = len(r.boxes)
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xyxy:
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                centers.append((cx, cy))
                boxes_xyxy.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            pass
    return r.plot(line_width=2, labels=True), people_count, centers, boxes_xyxy

def make_grid(images, cols=2):
    if not images: return None
    ih, iw = images[0].shape[:2]
    tgt_w = 640; tgt_h = int(ih * (tgt_w / iw))
    rs = [cv2.resize(im, (tgt_w, tgt_h)) for im in images]
    rows = math.ceil(len(rs) / cols)
    while len(rs) < rows * cols: rs.append(np.zeros_like(rs[0]))
    return cv2.vconcat([cv2.hconcat(rs[r * cols: (r + 1) * cols]) for r in range(rows)])

def run_app():
    weights_list = build_model_list(os.getenv("YOLO_MODELS", "ALL"))
    cams_list = [int_or_str(x) for x in parse_list_env("CAMS", "0,1,2,3")]
    weights = os.getenv("YOLO_WEIGHTS", weights_list[0] if weights_list else "yolov8s.pt")
    imgsz = int(os.getenv("IMG", "960"))
    conf = 0.35
    iou = 0.60
    cam = int_or_str(os.getenv("CAM", "0"))

# Ensure CUDA is used if user has CUDA hardware
    tile_mode = False
    use_gpu = cuda_available()
    show_fps = True
    fps_cap = 0 

    settings_open = False
    temp_open = False
    render_open = False
    style_open = False
    notes_open = False

    click_xy = None

    theme_idx = 0
    hud_mode = "Verbose"
    trails_on = True
    heatmap_on = False

    zones_enabled = False
    zones: List[Tuple[List[Tuple[int, int]], str]] = []
    current_zone: List[Tuple[int, int]] = []
    zone_names = ["Rocky Top A", "Rocky Top B", "Vol Navy C", "Smokey D"]

    trails = Trails(max_len=40, fade=0.90)
    heatmap = Heatmap(decay=0.94)

    def on_mouse(event, x, y, flags, param):
        nonlocal click_xy, current_zone, zones
        if event == cv2.EVENT_LBUTTONDOWN:
            click_xy = (x, y)
            if zones_enabled:
                current_zone.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and zones_enabled:
            if len(current_zone) >= 3:
                name = zone_names[len(zones) % len(zone_names)]
                zones.append((current_zone.copy(), name))
                add_note(f"Committed zone: {name} ({len(current_zone)} pts)")
            current_zone = []

    shots_dir = Path("shots"); shots_dir.mkdir(exist_ok=True)
    toast_text, toast_until = "", 0.0

    window = "YOLOv8 - Human Detection (v1.9 UTK Senior Edition)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    print("torch:", torch.__version__)
    print("cuda available:", cuda_available())
    if cuda_available():
        print("device:", gpu_name(), "cuda:", torch.version.cuda)
    else:
        print("NOTE: CUDA not available. Install the matching cu wheel for your GPU.")

    add_note(f"Session {SESSION_ID} started • weights={weights} • imgsz={imgsz} • cam={cam}")

    running = True
    restart_stream = True
    model = load_model_on_device(weights, use_gpu)
    caps: List[cv2.VideoCapture] = []
    last_t = None
    fps = 0.0

    while running:
        if restart_stream:
            release_caps(caps); caps = []
            sources = cams_list[:4] if (tile_mode and cams_list) else [cam]
            for s in sources:
                c = open_cap(s, imgsz)
                if c is not None: caps.append(c)
            if not caps:
                c = open_cap(cam, imgsz)
                if c is not None: caps.append(c)
                tile_mode = False
            model = load_model_on_device(weights, use_gpu)
            restart_stream = False
            last_t, fps = None, 0.0
            add_note(f"Stream restarted • tile={tile_mode} • cams={len(caps)}")

        if not caps:
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            draw_toast(black, "No camera available", time.time() + 2.0)
            cv2.imshow(window, black)
            if cv2.waitKey(10) & 0xFF == 27: break
            time.sleep(0.1); restart_stream = True; continue

        palette = THEMES[theme_idx][1]
        theme_name = THEMES[theme_idx][0]

        if tile_mode and len(caps) > 1:
            frames_for_grid = []; total_people = 0; all_ok = True
            all_centers = []; all_boxes = []
            for c in caps:
                ok, frame = c.read()
                if not ok: all_ok = False; break
                plotted, cnt, centers, boxes = run_predict(model, frame, use_gpu, imgsz, conf, iou)
                frames_for_grid.append(plotted); total_people += cnt
                all_centers.extend(centers); all_boxes.extend(boxes)
            if not all_ok:
                toast_text, toast_until = "A camera failed; restarting...", time.time() + 1.5
                restart_stream = True; continue
            disp = make_grid(frames_for_grid, cols=2)
            people = total_people; centers = all_centers; boxes = all_boxes
        else:
            ok, frame = caps[0].read()
            if not ok:
                toast_text, toast_until = f"Cam '{cam}' not delivering frames; restarting...", time.time() + 1.5
                restart_stream = True; continue
            disp, people, centers, boxes = run_predict(model, frame, use_gpu, imgsz, conf, iou)

        if trails_on:
            trails.push(centers)
            trails.draw(disp, color=palette.get("accent", (0, 180, 255)))
        if heatmap_on:
            heatmap.update(disp.shape, boxes)
            heatmap.draw(disp)

        h, w = disp.shape[:2]
        gap = max(8, int(0.01 * w))
        bar_h = max(44, int(0.08 * h))
        bw = int((w - gap * 6) / 5)
        r_settings = (gap, 0, gap + bw, bar_h)
        r_temp = (gap * 2 + bw, 0, gap * 2 + bw * 2, bar_h)
        r_render = (gap * 3 + bw * 2, 0, gap * 3 + bw * 3, bar_h)
        r_style = (gap * 4 + bw * 3, 0, gap * 4 + bw * 4, bar_h)
        r_shot = (gap * 5 + bw * 4, 0, gap * 5 + bw * 5, bar_h)

        draw_header(disp, (0, 0, w, bar_h), theme_name)
        draw_button(disp, r_settings, "Settings", settings_open, palette)
        draw_button(disp, r_temp, "Temp", temp_open, palette)
        draw_button(disp, r_render, "Render", render_open, palette)
        draw_button(disp, r_style, "Style", style_open, palette)
        draw_button(disp, r_shot, "Screenshot", False, palette)

        this_click = click_xy
        if point_in_rect(this_click, r_settings):
            settings_open, temp_open, render_open, style_open = not settings_open, False, False, False
        elif point_in_rect(this_click, r_temp):
            temp_open, settings_open, render_open, style_open = not temp_open, False, False, False
        elif point_in_rect(this_click, r_render):
            render_open, settings_open, temp_open, style_open = not render_open, False, False, False
        elif point_in_rect(this_click, r_style):
            style_open, settings_open, temp_open, render_open = not style_open, False, False, False
        elif point_in_rect(this_click, r_shot):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stamp = disp.copy()
            put_text_sharp(stamp, f"GBO • Evan • UTK • {SESSION_ID}", (12, 36), (420, 32))
            outp = shots_dir / f"shot_{ts}.png"
            try:
                cv2.imwrite(str(outp), stamp)
                add_note(f"Screenshot saved: {outp.name}")
            except Exception:
                pass

        if settings_open:
            menu_w = bw; row_h = int(bar_h * 0.95)
            px1, py1 = r_settings[0], r_settings[3] + 4
            px2, py2 = px1 + menu_w, py1 + row_h * 4 + 8
            cv2.rectangle(disp, (px1, py1), (px2, py2), palette.get("hud_bg", (35, 35, 35)), -1)
            rows = [
                ("FPS: on" if show_fps else "FPS: off"),
                (f"Cap: {'none' if fps_cap == 0 else int(fps_cap)}"),
                (f"Res: {imgsz}"),
                (f"View: {'Tiled' if tile_mode else 'Single'}"),
            ]
            rects = []
            for i, lbl in enumerate(rows):
                rx1, ry1 = px1 + 6, py1 + 6 + i * row_h
                rx2, ry2 = px2 - 6, ry1 + row_h - 6
                draw_button(disp, (rx1, ry1, rx2, ry2), lbl, False, palette)
                rects.append((rx1, ry1, rx2, ry2))
            if point_in_rect(this_click, rects[0]):
                show_fps = not show_fps; add_note(f"Show FPS: {show_fps}")
            elif point_in_rect(this_click, rects[1]):
                fps_cap = {0: 15, 15: 30, 30: 60, 60: 0}[fps_cap]; add_note(f"FPS cap: {fps_cap if fps_cap else 'none'}")
            elif point_in_rect(this_click, rects[2]):
                imgsz = {640: 960, 960: 1280, 1280: 640}.get(imgsz, 960)
                restart_stream = True; add_note(f"Resolution -> {imgsz}")
            elif point_in_rect(this_click, rects[3]):
                tile_mode = not tile_mode; restart_stream = True; add_note(f"View -> {'Tiled' if tile_mode else 'Single'}")

        if temp_open:
            menu_w = bw; row_h = int(bar_h * 0.95)
            px1, py1 = r_temp[0], r_temp[3] + 4
            px2, py2 = px1 + menu_w, py1 + row_h * 2 + 8
            cv2.rectangle(disp, (px1, py1), (px2, py2), palette.get("hud_bg", (35, 35, 35)), -1)
            rows = [
                (f"Cam: {cam if not tile_mode else ','.join(map(str, cams_list[:4]))}"),
                (f"Model: {os.path.basename(weights)}"),
            ]
            rects = []
            for i, lbl in enumerate(rows):
                rx1, ry1 = px1 + 6, py1 + 6 + i * row_h
                rx2, ry2 = px2 - 6, ry1 + row_h - 6
                draw_button(disp, (rx1, ry1, rx2, ry2), lbl, False, palette)
                rects.append((rx1, ry1, rx2, ry2))
            if point_in_rect(this_click, rects[0]):
                if tile_mode:
                    if len(cams_list) > 1:
                        cams_list = cams_list[1:] + cams_list[:1]
                        restart_stream = True; add_note("Rotated tiled camera set")
                else:
                    if cams_list:
                        try: next_idx = (cams_list.index(cam) + 1) % len(cams_list)
                        except ValueError: next_idx = 0
                        next_cam = cams_list[next_idx]
                        if probe_cap(next_cam, imgsz):
                            cam = next_cam; restart_stream = True
                            toast_text, toast_until = f"Switched to cam: {cam}", time.time() + 1.5
                            add_note(f"Cam -> {cam}")
                        else:
                            toast_text, toast_until = f"Cam '{next_cam}' not available", time.time() + 1.5
            elif point_in_rect(this_click, rects[1]):
                if weights_list:
                    next_idx = (weights_list.index(weights) + 1) % len(weights_list) if weights in weights_list else 0
                    weights = weights_list[next_idx]
                    model = load_model_on_device(weights, use_gpu)
                    toast_text, toast_until = f"Model: {os.path.basename(weights)}", time.time() + 1.5
                    add_note(f"Model -> {os.path.basename(weights)}")

        if render_open:
            menu_w = bw; row_h = int(bar_h * 0.95)
            px1, py1 = r_render[0], r_render[3] + 4
            px2, py2 = px1 + menu_w, py1 + row_h * 2 + 8
            cv2.rectangle(disp, (px1, py1), (px2, py2), palette.get("hud_bg", (35, 35, 35)), -1)
            dev_label = f"Device: {'GPU' if (use_gpu and cuda_available()) else 'CPU'}"
            cuda_label = f"{gpu_name() if cuda_available() else 'N/A'}"
            rects = []
            for i, lbl in enumerate([dev_label, cuda_label]):
                rx1, ry1 = px1 + 6, py1 + 6 + i * row_h
                rx2, ry2 = px2 - 6, ry1 + row_h - 6
                draw_button(disp, (rx1, ry1, rx2, ry2), lbl, False, palette)
                rects.append((rx1, ry1, rx2, ry2))
            if point_in_rect(this_click, rects[0]):
                if cuda_available():
                    use_gpu = not use_gpu
                    model = switch_device_inplace(model, weights, use_gpu)
                    toast_text, toast_until = f"Render: {'GPU' if use_gpu else 'CPU'}", time.time() + 1.5
                    add_note(f"Device -> {'GPU' if use_gpu else 'CPU'} (model reloaded)")
                else:
                    use_gpu = False
                    model = switch_device_inplace(model, weights, use_gpu)
                    toast_text, toast_until = "CUDA not available; using CPU", time.time() + 2.0

        if style_open:
            menu_w = bw; row_h = int(bar_h * 0.95)
            px1, py1 = r_style[0], r_style[3] + 4
            px2, py2 = px1 + menu_w, py1 + row_h * 5 + 8
            cv2.rectangle(disp, (px1, py1), (px2, py2), palette.get("hud_bg", (35, 35, 35)), -1)
            rows = [
                (f"Theme: {theme_name}"),
                (f"HUD: {hud_mode}"),
                (f"Notes: {'open' if notes_open else 'closed'}"),
                (f"Trails: {'on' if trails_on else 'off'}"),
                (f"Heatmap: {'on' if heatmap_on else 'off'}"),
            ]
            rects = []
            for i, lbl in enumerate(rows):
                rx1, ry1 = px1 + 6, py1 + 6 + i * row_h
                rx2, ry2 = px2 - 6, ry1 + row_h - 6
                draw_button(disp, (rx1, ry1, rx2, ry2), lbl, False, palette)
                rects.append((rx1, ry1, rx2, ry2))
            if point_in_rect(this_click, rects[0]):
                theme_idx = (theme_idx + 1) % len(THEMES); add_note(f"Theme -> {THEMES[theme_idx][0]}")
            elif point_in_rect(this_click, rects[1]):
                hud_mode = "Minimal" if hud_mode == "Verbose" else "Verbose"; add_note(f"HUD -> {hud_mode}")
            elif point_in_rect(this_click, rects[2]):
                notes_open = not notes_open; add_note(f"Notes panel {'opened' if notes_open else 'closed'}")
            elif point_in_rect(this_click, rects[3]):
                trails_on = not trails_on; add_note(f"Trails -> {'on' if trails_on else 'off'}")
            elif point_in_rect(this_click, rects[4]):
                heatmap_on = not heatmap_on; add_note(f"Heatmap -> {'on' if heatmap_on else 'off'}")

        click_xy = None

        now = time.time()
        if last_t is not None:
            dt = max(now - last_t, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        if zones_enabled and (centers and zones):
            in_any_zone = any(any(point_in_poly(c, poly) for (poly, _nm) in zones) for c in centers)
            if in_any_zone:
                put_text_sharp(disp, "IN ZONE", (12, int(0.92 * h)), (240, 36), color=(0,0,255), base_scale=0.95)

        if zones_enabled:
            draw_zones(disp, zones, current_zone)

        lines = []
        if hud_mode == "Verbose":
            lines = [f"People: {people}"]
            if show_fps: lines.append(f"FPS: {fps:.1f}")
        else:
            lines = [f"P:{people}" + (f" | {fps:.0f} FPS" if show_fps else "")]
        draw_bottom_right_block(disp, lines, palette=palette)

        if notes_open: draw_notes(disp)
        draw_toast(disp, toast_text, toast_until, palette=palette)

        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('n'):
            add_note("Pressed N: quick note (edit code to add your text)")
        elif key == ord('z'):
            notes_open = not notes_open; add_note(f"Notes panel {'opened' if notes_open else 'closed'} (key)")
        elif key == ord('x'):
            zones.clear(); current_zone.clear(); add_note("Cleared all zones")
        elif key == ord('d'):
            zones_enabled = not zones_enabled
            add_note(f"Zones editor {'ENABLED' if zones_enabled else 'disabled'}")
        elif key == ord(']'):
            if weights_list:
                idx = (weights_list.index(weights) + 1) % len(weights_list) if weights in weights_list else 0
                weights = weights_list[idx]
                model = load_model_on_device(weights, use_gpu)
                toast_text, toast_until = f"Model: {os.path.basename(weights)}", time.time() + 1.5
                add_note(f"Model -> {os.path.basename(weights)}")
        elif key == ord('['):
            if weights_list:
                idx = (weights_list.index(weights) - 1) % len(weights_list) if weights in weights_list else 0
                weights = weights_list[idx]
                model = load_model_on_device(weights, use_gpu)
                toast_text, toast_until = f"Model: {os.path.basename(weights)}", time.time() + 1.5
                add_note(f"Model -> {os.path.basename(weights)}")

        if fps_cap > 0:
            loop_dt = max(time.time() - now, 0.0)
            target_dt = 1.0 / fps_cap
            if loop_dt < target_dt:
                time.sleep(target_dt - loop_dt)

    release_caps(caps)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()
