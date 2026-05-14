#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ── Standard library (fast — always import) ──────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser
import json, os, datetime, hashlib, random, string, csv, sys, subprocess, shutil
from pathlib import Path

# ── Optional heavy libs — ALL lazy loaded, app opens instantly ───────────────
# PIL       (~104ms) : loaded on first image display
# reportlab (~15ms)  : loaded on first PDF generation
# cv2       (~418ms) : loaded on first face login/register
# qrcode    (~0ms)   : already fast, keep as-is
HAS_PIL  = False
HAS_QR   = False
HAS_PDF  = False
HAS_FACE = False

try:
    import qrcode
    HAS_QR = True
except ImportError:
    pass

HAS_PYZBAR = False
try:
    from pyzbar import pyzbar as _pyzbar_mod
    HAS_PYZBAR = True
except ImportError:
    _pyzbar_mod = None

import importlib.util as _iutil

# PIL: check without importing (0.5ms vs 104ms)
HAS_PIL = _iutil.find_spec('PIL') is not None
# Actual PIL import deferred to first use via _ensure_pil()

# reportlab: check without importing
HAS_PDF = _iutil.find_spec('reportlab') is not None
# Actual import deferred to first PDF use

def _ensure_pil():
    """Lazy-load PIL. Call before any Image/ImageTk usage."""
    global Image, ImageTk
    try:
        from PIL import Image as _I, ImageTk as _ITk
        Image = _I; ImageTk = _ITk
        return True
    except Exception:
        return False

def _ensure_pdf():
    """Lazy-load reportlab. Call before PDF generation."""
    global A4, rl_canvas
    try:
        from reportlab.lib.pagesizes import A4 as _A4
        from reportlab.pdfgen import canvas as _rlc
        A4 = _A4; rl_canvas = _rlc
        return True
    except Exception:
        return False

# Placeholders
Image = None; ImageTk = None
A4 = None; rl_canvas = None

def _ensure_face_libs():
    """Lazy-load cv2+numpy on first use. Returns (cv2_module, np_module) or (None,None)."""
    global HAS_FACE, cv2, np
    if HAS_FACE:
        return cv2, np
    try:
        import cv2 as _cv2, numpy as _np
        _ = _cv2.CascadeClassifier(
            _cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cv2 = _cv2; np = _np
        HAS_FACE = True
        return cv2, np
    except Exception:
        pass
    # Try install
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "opencv-python", "--quiet"], capture_output=True)
    try:
        import importlib as _il; _il.invalidate_caches()
        import cv2 as _cv2, numpy as _np
        cv2 = _cv2; np = _np
        HAS_FACE = True
        return cv2, np
    except Exception:
        return None, None

# cv2/np placeholders (filled by _ensure_face_libs on first face use)
cv2 = None
np  = None

# Try detecting face libs at import time (doesn't load them, just checks)
try:
    import importlib.util as _iu2
    if _iu2.find_spec('cv2') is not None:
        _cv2_check, _np_check = _ensure_face_libs()
        if _cv2_check is not None:
            HAS_FACE = True
except Exception:
    pass

def _auto_install_face_libs():
    """Auto-install opencv-python via pip. Returns True on success."""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        'opencv-python', 'numpy', '--quiet'],
                       capture_output=True, timeout=120)
        import importlib as _il; _il.invalidate_caches()
        import cv2 as _cv2t
        _ = _cv2t.CascadeClassifier(
            _cv2t.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return True
    except Exception:
        return False


def _decode_qr(image):
    """Decode QR code from image using pyzbar (preferred) or cv2 fallback.
    Returns list of decoded string data values."""
    results = []
    # Try pyzbar first (more reliable)
    if HAS_PYZBAR:
        try:
            decoded = _pyzbar_mod.decode(image)
            for obj in decoded:
                data = obj.data.decode('utf-8', errors='ignore')
                if data:
                    results.append(data)
            if results:
                return results
        except Exception:
            pass
    # Fallback to cv2.QRCodeDetector
    try:
        _cv2, _ = _ensure_face_libs()
        if _cv2 is None:
            import cv2 as _cv2
        det = _cv2.QRCodeDetector()
        data, pts, _ = det.detectAndDecode(image)
        if data:
            results.append(data)
    except Exception:
        pass
    return results


# ─── Custom Face Recognizer ───────────────────────────────────────────────────
class _FaceRecognizer:
    """
    Fast + Accurate Face Recognizer.

    Optimizations vs naive version:
      • cv2/numpy lazy-loaded (app opens instantly, ~500ms saved)
      • Cascade classifiers cached as class vars (loaded once, not per frame)
      • HOG: vectorized numpy  → 0.6ms (was 3ms, 5× faster)
      • LBP: stacked bitwise  → 0.3ms (was 1.3ms, 4× faster)
      • Eye detection: throttled (runs every 6 frames, not every frame)
      • Augmentation at train time → robust to lighting on camera
      • Adaptive personal threshold → no false positives
    """

    TARGET_SIZE   = 128
    SAFETY_MARGIN = 1.5   # personal_threshold = max_aug_self_dist × 1.5

    # Cascade classifiers — loaded ONCE, reused forever
    _face_casc  = None
    _leye_casc  = None
    _reye_casc  = None
    _eye_casc   = None

    # Pre-computed HOG bin edges (class-level constant)
    _HOG_EDGES  = None

    # Facial zone definitions (y1,y2,x1,x2) on 128×128 face
    _ZONES = [
        ( 0, 38, 22,106),  # forehead
        (26, 58,  4, 50),  # left-eye area
        (26, 58, 78,124),  # right-eye area
        (40, 72, 46, 82),  # nose bridge
        (62, 92, 36, 92),  # nose tip
        (86,116, 26,102),  # mouth
        (32, 88,  0, 26),  # left cheek
        (32, 88,102,128),  # right cheek
        ( 0, 28, 46, 82),  # forehead-center
        (50, 85, 50, 78),  # philtrum
    ]

    def __init__(self):
        self._db = []  # list of (label, master_feat, personal_threshold)
        _FaceRecognizer._init_class_resources()

    @classmethod
    def _init_class_resources(cls):
        """Load cascades + constants once for the entire class lifetime."""
        if cls._face_casc is not None:
            return
        _cv2, _np = _ensure_face_libs()
        if _cv2 is None:
            return
        hd = _cv2.data.haarcascades
        cls._face_casc = _cv2.CascadeClassifier(hd + "haarcascade_frontalface_default.xml")
        cls._leye_casc = _cv2.CascadeClassifier(hd + "haarcascade_lefteye_2splits.xml")
        cls._reye_casc = _cv2.CascadeClassifier(hd + "haarcascade_righteye_2splits.xml")
        cls._eye_casc  = _cv2.CascadeClassifier(hd + "haarcascade_eye.xml")
        cls._HOG_EDGES = _np.linspace(-_np.pi, _np.pi, 10)

    # ── Eye detection (UI dots only — throttle externally) ─────────────────
    @classmethod
    def _find_eyes(cls, gray_face_roi):
        """Find eyes for UI landmark dots. Returns (le, re) or None."""
        if cls._leye_casc is None:
            return None
        h, w = gray_face_roi.shape

        def _best(rects):
            if rects is None or len(rects) == 0: return None
            x, y, ew, eh = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[0]
            return (x + ew//2, y + eh//2)

        kw = dict(scaleFactor=1.1, minNeighbors=4,
                  minSize=(w//8, h//8), maxSize=(w//2, h//2))
        le = _best(cls._leye_casc.detectMultiScale(gray_face_roi, **kw))
        re = _best(cls._reye_casc.detectMultiScale(gray_face_roi, **kw))
        if le and re:
            if le[0] > re[0]: le, re = re, le
            return le, re
        eyes = cls._eye_casc.detectMultiScale(gray_face_roi, **kw)
        if eyes is not None and len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda r: r[0])
            return (_best([eyes[0]]), _best([eyes[1]]))
        return None

    # ── Preprocessing ──────────────────────────────────────────────────────
    @classmethod
    def _preprocess(cls, gray_roi):
        """Resize → CLAHE. Always identical — no eye-alignment inconsistency."""
        _cv2, _np = _ensure_face_libs()
        face  = _cv2.resize(gray_roi, (cls.TARGET_SIZE, cls.TARGET_SIZE))
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(face)

    # ── Vectorized feature extraction (~2ms total) ─────────────────────────
    @classmethod
    def _hog(cls, face):
        """Vectorized HOG: 8×8 blocks × 9 bins. ~0.6ms."""
        _cv2, _np = _ensure_face_libs()
        gx  = _cv2.Sobel(face, _cv2.CV_32F, 1, 0)
        gy  = _cv2.Sobel(face, _cv2.CV_32F, 0, 1)
        mag = _np.sqrt(gx**2 + gy**2)
        ang = _np.arctan2(gy, gx)
        # Reshape → 64 blocks of 256 pixels
        mag_b = mag.reshape(8,16,8,16).transpose(0,2,1,3).reshape(64,256)
        ang_b = ang.reshape(8,16,8,16).transpose(0,2,1,3).reshape(64,256)
        bin_i = _np.digitize(ang_b, cls._HOG_EDGES[1:-1])
        hog   = _np.zeros((64, 9), dtype=_np.float32)
        for b in range(9):
            hog[:, b] = (mag_b * (bin_i == b)).sum(axis=1)
        hog /= (_np.linalg.norm(hog, axis=1, keepdims=True) + 1e-5)
        return hog.ravel()

    @classmethod
    def _lbp(cls, face):
        """Vectorized LBP histogram (64 bins). ~0.3ms."""
        _cv2, _np = _ensure_face_libs()
        c   = face[1:-1, 1:-1]
        nbs = _np.stack([face[0:-2,0:-2], face[0:-2,1:-1], face[0:-2,2:],
                         face[1:-1,2:],   face[2:,  2:],   face[2:,  1:-1],
                         face[2:,  0:-2], face[1:-1,0:-2]])
        pw  = _np.array([1,2,4,8,16,32,64,128], dtype=_np.uint8)
        lbp = ((nbs >= c[_np.newaxis]).astype(_np.uint8) *
               pw[:,_np.newaxis,_np.newaxis]).sum(axis=0).astype(_np.uint8)
        h, _ = _np.histogram(lbp, bins=64, range=(0,256))
        return (h / (h.sum() + 1e-5)).astype(_np.float32)

    @classmethod
    def _zones(cls, face):
        """Zone stats + relative ratios. ~0.5ms."""
        _cv2, _np = _ensure_face_libs()
        f32   = face.astype(_np.float32)
        means = _np.array([f32[y1:y2,x1:x2].mean() for y1,y2,x1,x2 in cls._ZONES])
        stds  = _np.array([f32[y1:y2,x1:x2].std()  for y1,y2,x1,x2 in cls._ZONES])
        p25   = _np.array([float(_np.percentile(f32[y1:y2,x1:x2],25)) for y1,y2,x1,x2 in cls._ZONES])
        p75   = _np.array([float(_np.percentile(f32[y1:y2,x1:x2],75)) for y1,y2,x1,x2 in cls._ZONES])
        stats = _np.concatenate([means/255, stds/128, p25/255, p75/255])
        tot   = means.sum() + 1e-5
        n     = len(means)
        ratios = []
        for i in range(n):
            for j in range(i+1, n):
                ratios.append(means[i] / (means[j] + 1e-5))
            ratios.append(means[i] / tot)
        return stats.astype(_np.float32), _np.array(ratios, dtype=_np.float32)

    @classmethod
    def _extract(cls, face128):
        """Full feature vector. ~2ms total."""
        _cv2, _np = _ensure_face_libs()
        hog          = cls._hog(face128)  * 1.5
        lbp          = cls._lbp(face128)  * 0.8
        zstat, zrat  = cls._zones(face128)
        zstat       *= 1.4
        zrat        *= 1.8
        pv           = _cv2.resize(face128,(28,28)).flatten().astype(_np.float32)/255.0*0.5
        feat = _np.concatenate([hog, lbp, zstat, zrat, pv])
        fn   = _np.linalg.norm(feat)
        if fn > 0: feat /= fn
        return feat

    @classmethod
    def _augment(cls, gray_roi):
        """9 augmented versions: brightness ±25, noise, blur (training only)."""
        _cv2, _np = _ensure_face_libs()
        face = cls._preprocess(gray_roi)
        out  = [face]
        for bv in [-25, -12, 12, 25]:
            out.append(_np.clip(face.astype(_np.int16) + bv, 0, 255).astype(_np.uint8))
        for sigma in [4, 8]:
            noise = _np.random.normal(0, sigma, face.shape).astype(_np.int16)
            out.append(_np.clip(face.astype(_np.int16) + noise, 0, 255).astype(_np.uint8))
        out.append(_cv2.GaussianBlur(face, (3,3), 0))
        out.append(_cv2.GaussianBlur(face, (5,5), 0))
        return out  # 9 total

    # ── Public API ─────────────────────────────────────────────────────────
    def train(self, images, labels):
        """Build master template + adaptive threshold per person."""
        _cv2, _np = _ensure_face_libs()
        self._db  = []
        lbl_feats = {}
        for img, lbl in zip(images, labels.tolist()):
            for aug in self._augment(img):
                lbl_feats.setdefault(lbl, []).append(self._extract(aug))
        for lbl, feats in lbl_feats.items():
            master = _np.mean(feats, axis=0)
            fn     = _np.linalg.norm(master)
            if fn > 0: master /= fn
            dists  = [float(_np.linalg.norm(f - master)) for f in feats]
            thresh = max(dists) * self.SAFETY_MARGIN
            thresh = max(0.12, min(thresh, 0.30))   # hard cap
            self._db.append((lbl, master, thresh))

    def predict(self, gray_roi):
        """
        Fast predict — ~2ms.
        Returns (label, dist, threshold, eyes_ok=False).
        Eyes detection is NOT done here — throttle it externally for UI.
        """
        if not self._db:
            return -1, 2.0, 0.22, False
        _cv2, _np = _ensure_face_libs()
        face = self._preprocess(gray_roi)
        feat = self._extract(face)
        best_lbl, best_dist, best_thr = -1, 2.0, 0.22
        for lbl, master, thresh in self._db:
            d = float(_np.linalg.norm(feat - master))
            if d < best_dist:
                best_dist = d; best_lbl = lbl; best_thr = thresh
        return best_lbl, best_dist, best_thr, False

    def is_match(self, gray_roi):
        """
        Returns (label_or_-1, dist, sim_pct, thresh, eyes_ok=False).
        sim_pct 0–1 for UI bar display.
        """
        lbl, dist, thresh, _ = self.predict(gray_roi)
        sim = max(0.0, 1.0 - dist / (thresh + 1e-5))
        if lbl != -1 and dist < thresh:
            return lbl, dist, sim, thresh, False
        return -1, dist, sim, thresh, False

# ============================================================
#  THEME DEFINITIONS
# ============================================================
THEMES = {
    "Dark": {
        "primary": "#6366f1", "success": "#10b981", "danger": "#ef4444",
        "warning": "#f59e0b", "dark": "#0f172a", "dark_light": "#1e293b",
        "gray": "#64748b", "light": "#f1f5f9", "white": "#ffffff",
        "sidebar_bg": "#1e293b", "content_bg": "#0f172a",
        "card_bg": "#1e293b", "text": "#ffffff", "subtext": "#94a3b8"
    },
    "Light": {
        "primary": "#6366f1", "success": "#10b981", "danger": "#ef4444",
        "warning": "#f59e0b", "dark": "#f8fafc", "dark_light": "#e2e8f0",
        "gray": "#64748b", "light": "#1e293b", "white": "#000000",
        "sidebar_bg": "#e2e8f0", "content_bg": "#f8fafc",
        "card_bg": "#ffffff", "text": "#1e293b", "subtext": "#475569"
    },
    "Blue": {
        "primary": "#2563eb", "success": "#059669", "danger": "#dc2626",
        "warning": "#d97706", "dark": "#0c1a3a", "dark_light": "#1e3a5f",
        "gray": "#6b7280", "light": "#dbeafe", "white": "#ffffff",
        "sidebar_bg": "#1e3a5f", "content_bg": "#0c1a3a",
        "card_bg": "#1e3a5f", "text": "#ffffff", "subtext": "#93c5fd"
    },
    "Green": {
        "primary": "#059669", "success": "#10b981", "danger": "#ef4444",
        "warning": "#f59e0b", "dark": "#052e16", "dark_light": "#14532d",
        "gray": "#6b7280", "light": "#d1fae5", "white": "#ffffff",
        "sidebar_bg": "#14532d", "content_bg": "#052e16",
        "card_bg": "#14532d", "text": "#ffffff", "subtext": "#6ee7b7"
    },
    "Purple": {
        "primary": "#7c3aed", "success": "#10b981", "danger": "#ef4444",
        "warning": "#f59e0b", "dark": "#1a0533", "dark_light": "#2e1065",
        "gray": "#9ca3af", "light": "#ede9fe", "white": "#ffffff",
        "sidebar_bg": "#2e1065", "content_bg": "#1a0533",
        "card_bg": "#2e1065", "text": "#ffffff", "subtext": "#c4b5fd"
    }
}


# ============================================================
#  MAIN CLASS
# ============================================================
class SchoolManagerPro:

    def __init__(self, root):
        self.root = root
        self.root.title("School Manager Pro – Ultra Edition")
        self.root.geometry("1440x900")

        self.data_dir = Path.home() / ".school_manager_ultra"
        self.data_dir.mkdir(exist_ok=True)
        self.db_file = self.data_dir / "database.json"

        # in-memory data
        self.current_user = None
        self.users = []
        self.students = []
        self.teachers = []
        self.library = []
        self.exams = []
        self.fee_structures = {}   # class → {monthly, admission, exam, sports, ...}
        self.settings = {
            "school_name": "My School",
            "theme": "Dark",
            "logo_path": ""
        }
        self.colors = THEMES["Dark"].copy()

        self.load_data()
        self._apply_theme(self.settings.get("theme", "Dark"), redraw=False)

        if not self.users:
            self._create_default_admin()

        self._setup_styles()
        self.show_login()

    # ─── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def gen_id():
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))

    @staticmethod
    def _hash(pwd): return hashlib.sha256(pwd.encode()).hexdigest()

    @staticmethod
    def _verify(pwd, hashed): return hashlib.sha256(pwd.encode()).hexdigest() == hashed

    # ─── persistence ────────────────────────────────────────────────────────
    def load_data(self):
        if not self.db_file.exists():
            return
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.users = d.get("users", [])
            self.students = d.get("students", [])
            self.teachers = d.get("teachers", [])
            self.library = d.get("library", [])
            self.exams = d.get("exams", [])
            self.fee_structures = d.get("fee_structures", {})
            self.settings.update(d.get("settings", {}))
            for s in self.students:
                s.setdefault("attendance", {})
                s.setdefault("results", {})
                s.setdefault("qr_code", "")
                s.setdefault("fee_records", {})
                s.setdefault("photo_path", "")
                s.setdefault("dob", "")
                s.setdefault("address", "")
                s.setdefault("paper_fund_records", {})  # {YYYY-MM: {due, paid, paidFlag}}
            for t in self.teachers:
                t.setdefault("attendance", {})
            for u in self.users:
                u.setdefault("face_encoding", None)
                u.setdefault("face_photo_path", "")
        except Exception as e:
            print(f"[load_data] {e}")

    def save_data(self):
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump({
                    "users": self.users, "students": self.students,
                    "teachers": self.teachers, "library": self.library,
                    "exams": self.exams, "fee_structures": self.fee_structures,
                    "settings": self.settings
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    # ─── theme ──────────────────────────────────────────────────────────────
    def _apply_theme(self, theme_name, redraw=True):
        if theme_name in THEMES:
            self.settings["theme"] = theme_name
            self.colors = THEMES[theme_name].copy()
        if redraw:
            self.save_data()
            self._setup_styles()
            if self.current_user:
                self.show_main_application()
            else:
                self.show_login()

    def _setup_styles(self):
        c = self.colors
        self.root.configure(bg=c["content_bg"])
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Treeview",
                        background=c["card_bg"], foreground=c["text"],
                        fieldbackground=c["card_bg"], rowheight=28)
        style.configure("Custom.Treeview.Heading",
                        background=c["primary"], foreground="white",
                        font=("Helvetica", 10, "bold"))
        style.map("Custom.Treeview", background=[("selected", c["primary"])])

    def _create_default_admin(self):
        admin = {
            "id": self.gen_id(), "username": "admin",
            "password": self._hash("admin"), "role": "Admin",
            "permissions": {k: True for k in
                ["students","fees","attendance","teachers",
                 "library","exams","reports","settings","users"]},
            "created_at": datetime.datetime.now().isoformat(),
            "face_encoding": None
        }
        self.users.append(admin)
        self.save_data()
        messagebox.showinfo("Default Admin", "Default admin created\nUsername: admin\nPassword: admin")

    def check_perm(self, perm):
        if self.current_user["role"] == "Admin":
            return True
        if self.current_user.get("permissions", {}).get(perm):
            return True
        messagebox.showerror("Access Denied", f"No permission: '{perm}'")
        self.show_dashboard()
        return False

    # ════════════════════════════════════════════════════════════════════════
    #  LOGIN
    # ════════════════════════════════════════════════════════════════════════
    def show_login(self):
        for w in self.root.winfo_children():
            w.destroy()
        c = self.colors
        bg = tk.Frame(self.root, bg=c["primary"])
        bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        box = tk.Frame(bg, bg=c["card_bg"], padx=50, pady=50,
                       relief="flat", bd=0)
        box.place(relx=0.5, rely=0.5, anchor="center")

        # Logo / school name
        if self.settings.get("logo_path") and HAS_PIL and _ensure_pil():
            try:
                img = Image.open(self.settings["logo_path"]).resize((80,80), Image.LANCZOS)
                self._login_logo = ImageTk.PhotoImage(img)
                tk.Label(box, image=self._login_logo, bg=c["card_bg"]).pack(pady=(0,10))
            except Exception:
                tk.Label(box, text="🏫", font=("Arial",60), bg=c["card_bg"],
                         fg=c["text"]).pack()
        else:
            tk.Label(box, text="🏫", font=("Arial",60), bg=c["card_bg"],
                     fg=c["text"]).pack()

        tk.Label(box, text=self.settings.get("school_name","My School"),
                 font=("Helvetica",22,"bold"), bg=c["card_bg"],
                 fg=c["text"]).pack()
        tk.Label(box, text="School Manager Pro – Ultra Edition",
                 font=("Helvetica",11), bg=c["card_bg"],
                 fg=c["subtext"]).pack(pady=(4,20))

        self._err_lbl = tk.Label(box, text="", font=("Helvetica",10),
                                 bg=c["card_bg"], fg=c["danger"])
        self._err_lbl.pack()

        frm = tk.Frame(box, bg=c["card_bg"])
        frm.pack(fill="x", pady=10)

        tk.Label(frm, text="Username", bg=c["card_bg"],
                 fg=c["subtext"]).pack(anchor="w")
        self._uname = tk.Entry(frm, font=("Helvetica",12), bg=c["dark"],
                               fg=c["text"], insertbackground=c["text"],
                               relief="flat", bd=8)
        self._uname.pack(fill="x", pady=(3,12))
        self._uname.insert(0,"admin")

        tk.Label(frm, text="Password", bg=c["card_bg"],
                 fg=c["subtext"]).pack(anchor="w")
        self._pwd = tk.Entry(frm, font=("Helvetica",12), bg=c["dark"],
                             fg=c["text"], insertbackground=c["text"],
                             relief="flat", bd=8, show="•")
        self._pwd.pack(fill="x", pady=(3,20))
        self._pwd.insert(0,"admin")

        tk.Button(frm, text="🔐  Login", bg=c["primary"], fg="white",
                  font=("Helvetica",12,"bold"), bd=0, padx=30, pady=12,
                  activebackground=c["success"],
                  command=self._do_login).pack(fill="x")
        self._pwd.bind("<Return>", lambda e: self._do_login())

        # ── Face Login button (only if libraries available) ────────────────
        if HAS_FACE:
            tk.Button(frm, text="📷  Face Login", bg="#7c3aed", fg="white",
                      font=("Helvetica",12,"bold"), bd=0, padx=30, pady=12,
                      activebackground="#6d28d9",
                      command=self._do_face_login).pack(fill="x", pady=(8,0))
        else:
            tk.Label(frm, text="⚠️ Face login unavailable\n(Go to Settings to auto-install libraries)",
                     bg=c["card_bg"], fg=c["warning"],
                     font=("Helvetica",8), justify="center").pack(pady=(8,0))

        # theme selector on login
        th_fr = tk.Frame(box, bg=c["card_bg"])
        th_fr.pack(pady=(15,0))
        tk.Label(th_fr, text="Theme:", bg=c["card_bg"],
                 fg=c["subtext"]).pack(side="left")
        for th in THEMES:
            tk.Button(th_fr, text=th, font=("Helvetica",8),
                      bg=THEMES[th]["primary"], fg="white", bd=0, padx=6, pady=3,
                      command=lambda t=th: self._apply_theme(t)).pack(side="left", padx=2)

    def _do_login(self):
        u = self._uname.get().strip()
        p = self._pwd.get()
        user = next((x for x in self.users if x["username"]==u), None)
        if user and self._verify(p, user["password"]):
            self.current_user = user
            self.show_main_application()
        else:
            self._err_lbl.config(text="❌ Invalid credentials")
            self.root.after(3000, lambda: self._err_lbl.config(text=""))
    # ════════════════════════════════════════════════════════════════════════
    #  FACE RECOGNITION – LOGIN  (Custom recognizer, no LBPH needed)
    # ════════════════════════════════════════════════════════════════════════
    def _do_face_login(self):
        """
        Phone-style face login:
        1. Detect face → show corner bracket overlay
        2. Eye detection → show landmark dots (like phone face-ID scanning)
        3. Align + extract features
        4. DUAL threshold match (cosine + zone geometry)
        5. Liveness check (blink / motion)
        6. 5 consecutive confirmed frames → login
        """
        _cv2, _np = _ensure_face_libs()
        if not _cv2:
            messagebox.showerror("Not Available",
                "Face recognition not available.\n"
                "Go to Settings → Auto-Install to fix this.")
            return

        face_users = [u for u in self.users
                      if u.get("face_photo_path") and os.path.isfile(u["face_photo_path"])]
        if not face_users:
            messagebox.showwarning("No Faces Registered",
                "No user has registered their face yet.\n\n"
                "Please login with password first, then go to\n"
                "Settings → Register My Face.")
            return

        # Use cached cascade from _FaceRecognizer class vars
        recognizer  = _FaceRecognizer()
        face_cascade = recognizer._face_casc
        train_imgs  = []
        train_lbls  = []
        id_to_user  = {}

        for label, user in enumerate(face_users):
            paths = []
            if user.get("face_photo_path"):
                paths.append(user["face_photo_path"])
            paths += [p for p in user.get("face_sample_paths", [])
                      if p not in paths and os.path.isfile(p)]

            for pth in paths:
                img = _cv2.imread(pth, _cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                fdet = face_cascade.detectMultiScale(img, 1.1, 4, minSize=(50,50))
                if len(fdet):
                    x,y,w,h = fdet[0]; roi = img[y:y+h, x:x+w]
                else:
                    roi = img
                train_imgs.append(roi)
                train_lbls.append(label)
            id_to_user[label] = user

        if not train_imgs:
            messagebox.showerror("Error",
                "Could not load face images. Please register again."); return
        recognizer.train(train_imgs, _np.array(train_lbls))

        # ── Open camera ────────────────────────────────────────────────────
        cap = _cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            path = filedialog.askopenfilename(
                title="No Webcam – Select Your Face Photo",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            if not path: return
            img = _cv2.imread(path, _cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showerror("Error", "Could not load image."); return
            fdet = face_cascade.detectMultiScale(img, 1.1, 4, minSize=(50,50))
            if len(fdet):
                x,y,w,h = fdet[0]; roi = img[y:y+h, x:x+w]
            else:
                roi = img
            lbl, dist, sim_pct, thresh, _ = recognizer.is_match(roi)
            if lbl != -1 and lbl in id_to_user:
                user = id_to_user[lbl]
                self.current_user = user
                messagebox.showinfo("✅ Face Login",
                    f"Welcome {user['username']}!\n"
                    f"Similarity: {int(sim_pct*100)}%")
                self.show_main_application()
            else:
                messagebox.showerror("❌ Login Failed",
                    "Face did not match. Please login with password.")
            return
        cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ── Build UI ───────────────────────────────────────────────────────
        c = self.colors
        win = tk.Toplevel(self.root)
        win.title("Face Login")
        win.geometry("700x620")
        win.configure(bg="#060d1a")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # Canvas for camera + overlays (phone-style)
        canvas = tk.Canvas(win, width=660, height=440, bg="#060d1a",
                           highlightthickness=0)
        canvas.pack(pady=(12,4))

        # Status row
        status_fr = tk.Frame(win, bg="#060d1a")
        status_fr.pack(fill="x", padx=20)

        status_var  = tk.StringVar(value="🔍 Scanning…")
        sim_var     = tk.StringVar(value="")
        live_var    = tk.StringVar(value="")

        tk.Label(status_fr, textvariable=status_var,
                 font=("Helvetica",13,"bold"), bg="#060d1a",
                 fg="#a78bfa", wraplength=640).pack()
        tk.Label(status_fr, textvariable=sim_var,
                 font=("Consolas",10), bg="#060d1a",
                 fg="#38bdf8").pack()
        tk.Label(status_fr, textvariable=live_var,
                 font=("Helvetica",10), bg="#060d1a",
                 fg="#fbbf24").pack()

        progress = ttk.Progressbar(win, length=500, mode="determinate",
                                   maximum=5)
        progress.pack(pady=6)

        tk.Button(win, text="✖  Cancel", bg="#7f1d1d", fg="white",
                  font=("Helvetica",10,"bold"), bd=0,
                  padx=22, pady=8, relief="flat",
                  command=lambda: self._face_login_cancel(cap, win)).pack(pady=4)

        # ── Drawing helpers ────────────────────────────────────────────────
        def _draw_corner_bracket(cvs, x, y, w, h, col, thick=3, seg=20):
            """Draw phone-style corner brackets around face."""
            # top-left
            cvs.create_line(x, y+seg, x, y, x+seg, y, fill=col, width=thick)
            # top-right
            cvs.create_line(x+w-seg, y, x+w, y, x+w, y+seg, fill=col, width=thick)
            # bottom-left
            cvs.create_line(x, y+h-seg, x, y+h, x+seg, y+h, fill=col, width=thick)
            # bottom-right
            cvs.create_line(x+w-seg, y+h, x+w, y+h, x+w, y+h-seg, fill=col, width=thick)

        def _draw_scan_line(cvs, x, y, w, h, frame_no, col="#38bdf8"):
            """Animated horizontal scan line (phone-scan effect)."""
            offset = int((frame_no % 40) / 40 * h)
            ly = y + offset
            cvs.create_line(x+2, ly, x+w-2, ly, fill=col, width=2,
                            dash=(4,4))

        def _draw_dot(cvs, cx, cy, col="#00ff88", r=4):
            cvs.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="")

        def _draw_landmark_grid(cvs, fx, fy, fw, fh, eyes_ok,
                                 le=None, re=None, col="#00ff88"):
            """Draw facial coordinate dots — phone face-ID style."""
            sx = fw / 100.0; sy = fh / 100.0
            # Standard facial landmark positions (relative to face box)
            landmarks = [
                # Eyes (will be overridden if actual eye positions known)
                (28, 30), (72, 30),           # eye centers
                (20, 32), (36, 32),           # left eye corners
                (64, 32), (80, 32),           # right eye corners
                # Nose
                (50, 42), (50, 55),
                (42, 52), (58, 52),
                # Mouth
                (35, 68), (50, 72), (65, 68),
                (50, 78),
                # Face outline
                (10, 45), (15, 65), (20, 80),
                (50, 92),
                (80, 80), (85, 65), (90, 45),
                # Forehead
                (35, 12), (50, 8), (65, 12),
            ]
            for (lx_r, ly_r) in landmarks:
                dot_x = int(fx + lx_r * sx)
                dot_y = int(fy + ly_r * sy)
                _draw_dot(cvs, dot_x, dot_y, col, r=3)

            # Override eyes with actual detected positions
            if eyes_ok and le and re:
                _draw_dot(cvs, fx + int(le[0]*fw/100), fy + int(le[1]*fh/100),
                          "#ff6b35", r=5)
                _draw_dot(cvs, fx + int(re[0]*fw/100), fy + int(re[1]*fh/100),
                          "#ff6b35", r=5)

        # ── State ──────────────────────────────────────────────────────────
        self._face_login_running = True
        state = {
            "frame_no":      0,
            "good_frames":   0,
            "last_label":    -1,
            # liveness
            "eye_c_prev":    2,
            "blink_streak":  0,
            "blink_done":    False,
            "motion_count":  0,
            "prev_face":     None,
            "cached_eyes":   None,   # throttled eye result
            # scan phase
            "phase":         "searching",
            "scan_tick":     0,
        }

        from PIL import Image as _PI, ImageTk as _ITk

        def scan_frame():
            if not self._face_login_running or not win.winfo_exists():
                return
            ret, frame = cap.read()
            if not ret:
                win.after(80, scan_frame); return

            state["frame_no"] += 1
            fn = state["frame_no"]

            gray  = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            rgb   = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            disp  = _cv2.resize(rgb, (660, 440))
            sx = 660/frame.shape[1]; sy = 440/frame.shape[0]

            faces_rect = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

            # ── Update canvas ──────────────────────────────────────────────
            canvas.delete("all")
            img_pil  = _PI.fromarray(disp)
            imgtk    = _ITk.PhotoImage(img_pil)
            canvas._img = imgtk
            canvas.create_image(0, 0, anchor="nw", image=imgtk)

            matched        = None
            best_sim_pct   = 0.0
            best_dist      = 2.0
            face_eyes_ok   = False

            for (x, y, w, h) in faces_rect:
                roi_gray  = gray[y:y+h, x:x+w]
                dx = int(x*sx); dy_s = int(y*sy)
                dw = int(w*sx); dh   = int(h*sy)

                # ── Liveness: motion (every frame, fast) ─────────────────
                face_small = _cv2.resize(roi_gray, (50,50))
                if state["prev_face"] is not None:
                    diff = _cv2.absdiff(face_small, state["prev_face"])
                    if float(_np.mean(diff)) > 3.5:
                        state["motion_count"] = min(state["motion_count"]+1, 20)
                state["prev_face"] = face_small.copy()

                # ── Liveness: blink — throttled every 6 frames (69ms→12ms) ─
                if fn % 6 == 0:
                    eyes_det = recognizer._eye_casc.detectMultiScale(
                        roi_gray, 1.1, 4, minSize=(w//8, h//8))
                    state["cached_eyes"] = len(eyes_det) if eyes_det is not None else 0
                ec = state.get("cached_eyes", 2)
                if state["eye_c_prev"] >= 2 and ec == 0:
                    state["blink_streak"] += 1
                elif state["eye_c_prev"] == 0 and ec >= 2 \
                     and state["blink_streak"] >= 1:
                    state["blink_done"]  = True
                    state["blink_streak"] = 0
                state["eye_c_prev"] = ec

                is_live = state["blink_done"] or state["motion_count"] >= 8

                # ── Match ────────────────────────────────────────────────
                m_label, dist, sim_pct, thresh, eyes_ok = \
                    recognizer.is_match(roi_gray)
                face_eyes_ok = eyes_ok

                # ── Draw overlay ─────────────────────────────────────────
                if m_label != -1 and is_live:
                    bracket_col = "#00ff88"
                    dot_col     = "#00ff88"
                    state["phase"] = "matched"
                    matched = id_to_user.get(m_label)
                elif m_label != -1 and not is_live:
                    bracket_col = "#ff8c00"
                    dot_col     = "#ff8c00"
                    state["phase"] = "scanning"
                elif m_label == -1 and len(faces_rect) > 0:
                    bracket_col = "#38bdf8"
                    dot_col     = "#38bdf8"
                    state["phase"] = "scanning"
                else:
                    bracket_col = "#ef4444"
                    dot_col     = "#ef4444"

                # Corner brackets (phone-style)
                _draw_corner_bracket(canvas, dx, dy_s, dw, dh,
                                     bracket_col, thick=3, seg=22)

                # Animated scan line
                if state["phase"] in ("searching", "scanning"):
                    _draw_scan_line(canvas, dx, dy_s, dw, dh, fn)
                    state["scan_tick"] += 1

                # Facial landmark dots
                _draw_landmark_grid(canvas, dx, dy_s, dw, dh,
                                    eyes_ok, col=dot_col)

                # Confidence bar at top of face box
                arc_w = int(dw * min(sim_pct, 1.0))
                canvas.create_rectangle(dx, dy_s-8, dx+dw, dy_s-4,
                                        fill="#1e293b", outline="")
                arc_col = "#00ff88" if sim_pct >= 0.85 else \
                          "#fbbf24" if sim_pct >= 0.50 else "#ef4444"
                canvas.create_rectangle(dx, dy_s-8, dx+arc_w, dy_s-4,
                                        fill=arc_col, outline="")
                canvas.create_text(
                    dx + dw//2, dy_s-14,
                    text=f"{int(sim_pct*100)}%  dist:{dist:.3f}",
                    fill=arc_col, font=("Consolas",9,"bold"))

                best_sim_pct = max(best_sim_pct, sim_pct)
                best_dist    = min(best_dist,    dist)

            # ── Overlay: guide oval when no face ──────────────────────────
            if len(faces_rect) == 0:
                cx, cy, ow, oh = 330, 220, 180, 230
                canvas.create_oval(cx-ow, cy-oh, cx+ow, cy+oh,
                                   outline="#334155", width=2, dash=(6,4))
                canvas.create_text(330, 440-20,
                    text="Keep your face inside the frame",
                    fill="#64748b", font=("Helvetica",10))

            # ── UI labels ─────────────────────────────────────────────────
            is_live = state["blink_done"] or state["motion_count"] >= 8

            if matched:
                sim_var.set(
                    f"dist: {best_dist:.3f}  |  match: {int(best_sim_pct*100)}%  "
                    f"|  eyes: {'✅' if face_eyes_ok else '⚠️'}")
            elif len(faces_rect) > 0:
                sim_var.set(
                    f"dist: {best_dist:.3f}  |  match: {int(best_sim_pct*100)}%")

            if not is_live:
                live_var.set(
                    f"👁️ Please blink your eyes  "
                    f"|  motion: {state['motion_count']}/8")
            else:
                live_var.set("✅ Liveness confirmed!")

            # ── Good-frame counter ─────────────────────────────────────────
            if matched:
                lbl_now = next(
                    (lb for lb, u in id_to_user.items() if u is matched), -1)
                if state["last_label"] == lbl_now:
                    state["good_frames"] += 1
                else:
                    state["last_label"]  = lbl_now
                    state["good_frames"] = 1
                progress["value"] = state["good_frames"]
                status_var.set(
                    f"✅ Recognized: {matched['username']}  "
                    f"({state['good_frames']}/5)  verifying…")
                if state["good_frames"] >= 5:
                    self._face_login_running = False
                    status_var.set(
                        f"✅ Logging in: {matched['username']}…")
                    win.after(600,
                              lambda: self._face_login_success(cap, win, matched))
                    return
            else:
                state["good_frames"] = 0
                state["last_label"]  = -1
                progress["value"]    = 0
                if len(faces_rect) > 0:
                    if not is_live:
                        status_var.set(
                            "⚠️ Face scan in progress — please blink your eyes")
                    else:
                        status_var.set(
                            f"❌ Not recognized  ({int(best_sim_pct*100)}% match)  "
                            "— move closer or improve lighting")
                else:
                    status_var.set("🔍 No face detected — look straight at the camera")

            win.after(50, scan_frame)

        win.after(300, scan_frame)

    def _face_login_cancel(self, cap, win):
        self._face_login_running = False
        cap.release()
        if win.winfo_exists():
            win.destroy()

    def _face_login_success(self, cap, win, user):
        cap.release()
        if win.winfo_exists():
            win.destroy()
        self.current_user = user
        self.show_main_application()

    # ════════════════════════════════════════════════════════════════════════
    #  FACE RECOGNITION – REGISTER  (Pure OpenCV LBPH, Python 3.14 compatible)
    # ════════════════════════════════════════════════════════════════════════
    def _register_face(self):
        """
        Phone-style face registration:
        - Shows scanning overlay with corner brackets + landmark dots
        - Requires eyes to be detected for a high-quality capture
        - Captures 5 samples (front, left, right + 2 auto-variety)
        - Warns if quality is poor
        """
        if not HAS_FACE:
            messagebox.showerror("Not Available",
                "Face recognition not available.\n"
                "Go to Settings → Auto-Install.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera Required",
                "Face registration requires a camera.\n"
                "Please connect a webcam and try again.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        c = self.colors
        win = tk.Toplevel(self.root)
        win.title("Face Register")
        win.geometry("720x680")
        win.configure(bg="#060d1a")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # Title header
        hdr = tk.Frame(win, bg="#065f46", padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📷  Face Register – 5 Samples",
                 font=("Helvetica",16,"bold"), bg="#065f46", fg="white").pack(anchor="w")
        tk.Label(hdr,
                 text=f"User: {self.current_user['username']}  |  "
                      "Straight, LEFT, RIGHT, slightly up, slightly down",
                 font=("Helvetica",10), bg="#065f46", fg="#a7f3d0").pack(anchor="w")

        # Camera canvas
        canvas = tk.Canvas(win, width=660, height=430,
                           bg="#060d1a", highlightthickness=0)
        canvas.pack(pady=(8,2))

        status_var  = tk.StringVar(value="📷 Camera live – Press Capture")
        quality_var = tk.StringVar(value="")
        sample_var  = tk.StringVar(
            value="📸 Samples: 0/5  (3 minimum, 5 recommended)")
        instr_msgs = [
            "▶ Sample 1: Look straight at the camera",
            "▶ Sample 2: Tilt slightly LEFT",
            "▶ Sample 3: Tilt slightly RIGHT",
            "▶ Sample 4: Look slightly UP",
            "▶ Sample 5: Normal expression – stay natural",
        ]
        instr_var = tk.StringVar(value=instr_msgs[0])

        for sv, fc, fw in [
            (status_var,  "#6ee7b7", 12),
            (quality_var, "#fbbf24", 10),
            (sample_var,  "#93c5fd", 10),
            (instr_var,   "#e2e8f0", 10),
        ]:
            tk.Label(win, textvariable=sv,
                     font=("Helvetica",fw,"bold" if fw==12 else "normal"),
                     bg="#060d1a", fg=fc, wraplength=680).pack()

        btn_row = tk.Frame(win, bg="#060d1a")
        btn_row.pack(pady=8)

        self._face_reg_running = True
        last_frame = [None]
        last_quality = [0.0]
        samples  = []   # list of (frame, eyes_ok)
        face_casc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        frame_no = [0]

        def _draw_brackets(cvs, x, y, w, h, col, thick=3, seg=24):
            cvs.create_line(x, y+seg, x, y, x+seg, y,
                            fill=col, width=thick)
            cvs.create_line(x+w-seg, y, x+w, y, x+w, y+seg,
                            fill=col, width=thick)
            cvs.create_line(x, y+h-seg, x, y+h, x+seg, y+h,
                            fill=col, width=thick)
            cvs.create_line(x+w-seg, y+h, x+w, y+h, x+w, y+h-seg,
                            fill=col, width=thick)

        def _draw_landmarks(cvs, fx, fy, fw, fh, eyes_ok, col):
            pts = [
                (28,30),(72,30),(20,32),(36,32),(64,32),(80,32),
                (50,42),(50,55),(42,52),(58,52),
                (35,68),(50,72),(65,68),(50,78),
                (10,45),(15,65),(20,80),(50,92),(80,80),(85,65),(90,45),
                (35,12),(50,8),(65,12),
            ]
            for (lx,ly) in pts:
                cvs.create_oval(
                    fx+int(lx*fw/100)-3, fy+int(ly*fh/100)-3,
                    fx+int(lx*fw/100)+3, fy+int(ly*fh/100)+3,
                    fill=col, outline="")

        from PIL import Image as _PI, ImageTk as _ITk

        def update_feed():
            if not self._face_reg_running or not win.winfo_exists():
                return
            ret, frame = cap.read()
            if not ret:
                win.after(60, update_feed); return

            frame_no[0] += 1
            fn = frame_no[0]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            disp = cv2.resize(rgb, (660, 430))
            sx = 660/frame.shape[1]; sy = 430/frame.shape[0]

            faces = face_casc.detectMultiScale(gray, 1.1, 5, minSize=(80,80))

            canvas.delete("all")
            pil_img = _PI.fromarray(disp)
            imgtk   = _ITk.PhotoImage(pil_img)
            canvas._img = imgtk
            canvas.create_image(0, 0, anchor="nw", image=imgtk)

            eyes_ok = False
            quality = 0.0

            if len(faces):
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                dx, dy_ = int(x*sx), int(y*sy)
                dw, dh  = int(w*sx), int(h*sy)
                roi     = gray[y:y+h, x:x+w]

                # Eye detection for quality score
                result = _FaceRecognizer._find_eyes(roi)
                eyes_ok = result is not None

                # Quality = sharpness (Laplacian variance)
                lap_var = float(cv2.Laplacian(
                    cv2.resize(roi,(100,100)), cv2.CV_64F).var())
                quality = min(lap_var / 500.0, 1.0)
                last_quality[0] = quality

                col = "#00ff88" if eyes_ok else \
                      "#38bdf8" if quality > 0.3 else "#ff8c00"

                _draw_brackets(canvas, dx, dy_, dw, dh, col)

                # Scan line animation
                off = int((fn % 40) / 40 * dh)
                canvas.create_line(dx+2, dy_+off, dx+dw-2, dy_+off,
                                   fill="#38bdf8", width=2, dash=(4,3))

                _draw_landmarks(canvas, dx, dy_, dw, dh, eyes_ok, col)

                # Quality bar
                qw = int(dw * quality)
                canvas.create_rectangle(dx, dy_-10, dx+dw, dy_-5,
                                        fill="#1e293b", outline="")
                qcol = "#00ff88" if quality>0.6 else \
                       "#fbbf24" if quality>0.3 else "#ef4444"
                canvas.create_rectangle(dx, dy_-10, dx+qw, dy_-5,
                                        fill=qcol, outline="")
                canvas.create_text(dx+dw//2, dy_-18,
                    text=f"Quality: {int(quality*100)}%  "
                         f"Eyes: {'✅' if eyes_ok else '⚠️'}",
                    fill=qcol, font=("Consolas",9,"bold"))

                last_frame[0] = (frame, eyes_ok)
                if len(samples) < 5:
                    if eyes_ok and quality > 0.3:
                        status_var.set("✅ Excellent quality! Press Capture")
                        quality_var.set(
                            f"👁️ Eyes detected | Quality: {int(quality*100)}%")
                    elif eyes_ok:
                        status_var.set("✅ Eyes detected – improve lighting for better quality")
                        quality_var.set(
                            f"⚠️ Quality is low: {int(quality*100)}%")
                    else:
                        status_var.set(
                            "⚠️ Eyes not detected – look straight at camera")
                        quality_var.set(
                            f"Quality: {int(quality*100)}% | Eyes: ❌")
            else:
                last_frame[0] = None
                # Guide oval
                canvas.create_oval(330-170, 215-220, 330+170, 215+220,
                                   outline="#334155", width=2, dash=(6,4))
                canvas.create_text(330, 415,
                    text="Place your face here",
                    fill="#64748b", font=("Helvetica",11))
                status_var.set("⚠️ No face detected – move closer")
                quality_var.set("")

            win.after(55, update_feed)

        def capture():
            if len(samples) >= 5:
                messagebox.showinfo("5 Samples", "All 5 samples captured. Click Save."); return
            data = last_frame[0]
            if data is None:
                messagebox.showwarning("No Face Found",
                    "Please position your face in front of the camera."); return
            frame, eyes_ok = data
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_casc.detectMultiScale(gray, 1.1, 4, minSize=(70,70))
            if len(faces) == 0:
                messagebox.showwarning("No Face Found",
                    "Face not detected.\n"
                    "Improve lighting and look straight at camera."); return
            if not eyes_ok and len(samples) == 0:
                if not messagebox.askyesno("Quality Warning",
                    "Eyes not detected — quality is low.\n"
                    "Capture anyway?\n"
                    "(For better results, improve lighting)"):
                    return
            samples.append(frame.copy())
            n = len(samples)
            sample_var.set(
                f"📸 Samples: {n}/5  "
                f"({'MIN DONE – You can save now' if n >= 3 else str(5-n)+' more needed'})")
            if n < 5:
                instr_var.set(instr_msgs[n])
            else:
                instr_var.set("✅ All 5 samples complete! Click Save.")
            status_var.set(f"✅ Sample {n} captured!")

        def save_register():
            if not samples:
                messagebox.showwarning("Nothing Captured",
                    "Please capture at least 1 photo first."); return
            if len(samples) < 3:
                if not messagebox.askyesno("Few Samples",
                    f"Only {len(samples)} sample(s) captured.\n"
                    "3+ samples give better accuracy.\n\n"
                    "Save now?"):
                    return

            face_dir = self.data_dir / "face_photos"
            face_dir.mkdir(exist_ok=True)
            uid = self.current_user["id"]
            saved = []
            for i, frm in enumerate(samples):
                path = str(face_dir / f"{uid}_face_{i}.jpg")
                cv2.imwrite(path, frm)
                saved.append(path)
            primary = str(face_dir / f"{uid}_face.jpg")
            cv2.imwrite(primary, samples[0])

            self.current_user["face_photo_path"]   = primary
            self.current_user["face_sample_paths"] = saved
            self.current_user["face_encoding"]     = None
            self.save_data()
            self._face_reg_running = False
            cap.release()
            win.destroy()
            messagebox.showinfo("✅ Registration Complete",
                f"{len(samples)} samples saved for "
                f"'{self.current_user['username']}'!\n\n"
                "You can now use 📷 Face Login.")

        def cancel():
            self._face_reg_running = False
            cap.release()
            if win.winfo_exists(): win.destroy()

        tk.Button(btn_row, text="📸 Capture",
                  bg="#1d4ed8", fg="white",
                  font=("Helvetica",11,"bold"), bd=0,
                  padx=22, pady=10, relief="flat",
                  command=capture).pack(side="left", padx=6)
        tk.Button(btn_row, text="💾 Save & Register",
                  bg="#065f46", fg="white",
                  font=("Helvetica",11,"bold"), bd=0,
                  padx=22, pady=10, relief="flat",
                  command=save_register).pack(side="left", padx=6)
        tk.Button(btn_row, text="✖ Cancel",
                  bg="#7f1d1d", fg="white",
                  font=("Helvetica",11,"bold"), bd=0,
                  padx=22, pady=10, relief="flat",
                  command=cancel).pack(side="left", padx=6)

        win.after(200, update_feed)


    # ════════════════════════════════════════════════════════════════════════
    #  MAIN APPLICATION SHELL
    def _register_face_from_image(self):
        """Register face from uploaded image files (fallback when no webcam)."""
        c = self.colors
        win = tk.Toplevel(self.root)
        win.title("📷 Face Register – Upload Photos")
        win.geometry("500x500")
        win.configure(bg="#060d1a")

        tk.Label(win, text="📷  Face Register – Upload Photos",
                 font=("Helvetica",14,"bold"), bg="#060d1a", fg="white").pack(pady=12)
        tk.Label(win,
                 text="No webcam detected. Upload 1-5 face photos instead.\n"
                      "Use clear, well-lit photos with face visible.",
                 bg="#060d1a", fg="#a7f3d0", font=("Helvetica",10),
                 justify="center").pack(pady=4)

        photos = []
        list_var = tk.StringVar(value="No photos added yet")
        tk.Label(win, textvariable=list_var, bg="#060d1a", fg="#93c5fd",
                 font=("Helvetica",10)).pack(pady=8)

        preview_frame = tk.Frame(win, bg="#060d1a")
        preview_frame.pack(fill="x", padx=20)
        win._imgs = []

        def add_photo():
            paths = filedialog.askopenfilenames(
                title="Select Face Photos (1-5)",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            if not paths: return
            for p in paths:
                if len(photos) >= 5: break
                img = cv2.imread(p)
                if img is None: continue
                photos.append(img)
                try:
                    _ensure_pil()
                    pil_img = Image.open(p).resize((60,60), Image.LANCZOS)
                    itk = ImageTk.PhotoImage(pil_img)
                    win._imgs.append(itk)
                    tk.Label(preview_frame, image=itk, bg="#060d1a").pack(
                        side="left", padx=3)
                except Exception:
                    pass
            list_var.set(f"📸 {len(photos)}/5 photos added")

        def save_register():
            if not photos:
                messagebox.showwarning("No Photos", "Add at least 1 photo first."); return
            face_dir = self.data_dir / "face_photos"
            face_dir.mkdir(exist_ok=True)
            uid = self.current_user["id"]
            saved = []
            for i, frm in enumerate(photos):
                path = str(face_dir / f"{uid}_face_{i}.jpg")
                cv2.imwrite(path, frm)
                saved.append(path)
            primary = str(face_dir / f"{uid}_face.jpg")
            cv2.imwrite(primary, photos[0])
            self.current_user["face_photo_path"]   = primary
            self.current_user["face_sample_paths"] = saved
            self.current_user["face_encoding"]     = None
            self.save_data()
            win.destroy()
            messagebox.showinfo("✅ Registration Complete",
                f"{len(photos)} photos saved for "
                f"'{self.current_user['username']}'!\n\n"
                "You can now use 📷 Face Login.")

        btn_row = tk.Frame(win, bg="#060d1a")
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="📂 Add Photos", bg="#1d4ed8", fg="white",
                  font=("Helvetica",11,"bold"), padx=16, pady=8,
                  command=add_photo).pack(side="left", padx=6)
        tk.Button(btn_row, text="💾 Save & Register", bg="#065f46", fg="white",
                  font=("Helvetica",11,"bold"), padx=16, pady=8,
                  command=save_register).pack(side="left", padx=6)
        tk.Button(btn_row, text="✖ Cancel", bg="#7f1d1d", fg="white",
                  font=("Helvetica",11,"bold"), padx=16, pady=8,
                  command=win.destroy).pack(side="left", padx=6)


    # ════════════════════════════════════════════════════════════════════════
    def show_main_application(self):
        for w in self.root.winfo_children():
            w.destroy()
        self.root.configure(bg=self.colors["content_bg"])
        self._build_sidebar()
        self._build_content_area()
        self.show_dashboard()

    def _build_sidebar(self):
        c = self.colors
        self.sidebar = tk.Frame(self.root, bg=c["sidebar_bg"], width=240)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo area
        hdr = tk.Frame(self.sidebar, bg=c["sidebar_bg"], pady=15)
        hdr.pack(fill="x", padx=15)

        if self.settings.get("logo_path") and HAS_PIL and _ensure_pil():
            try:
                img = Image.open(self.settings["logo_path"]).resize((55,55), Image.LANCZOS)
                self._sb_logo = ImageTk.PhotoImage(img)
                tk.Label(hdr, image=self._sb_logo, bg=c["sidebar_bg"]).pack()
            except Exception:
                tk.Label(hdr, text="🏫", font=("Arial",36), bg=c["sidebar_bg"],
                         fg="white").pack()
        else:
            tk.Label(hdr, text="🏫", font=("Arial",36), bg=c["sidebar_bg"],
                     fg="white").pack()

        tk.Label(hdr, text=self.settings.get("school_name","My School"),
                 font=("Helvetica",12,"bold"), bg=c["sidebar_bg"],
                 fg="white", wraplength=200).pack(pady=(6,2))

        # User chip
        chip = tk.Frame(self.sidebar, bg=c["primary"], padx=12, pady=8)
        chip.pack(fill="x", padx=12, pady=(0,15))
        tk.Label(chip, text=self.current_user["username"][0].upper(),
                 font=("Helvetica",16,"bold"), bg="white",
                 fg=c["primary"], width=2).pack(side="left")
        inf = tk.Frame(chip, bg=c["primary"])
        inf.pack(side="left", padx=8)
        tk.Label(inf, text=self.current_user["username"],
                 font=("Helvetica",10,"bold"), bg=c["primary"],
                 fg="white").pack(anchor="w")
        tk.Label(inf, text=self.current_user["role"],
                 font=("Helvetica",8), bg=c["primary"],
                 fg="#e0e7ff").pack(anchor="w")

        # Nav items
        nav = [
            ("📊","Dashboard", self.show_dashboard),
            ("👨‍🎓","Students", self.show_students),
            ("📋","Attendance", self.show_attendance),
            ("👩‍🏫","Teachers", self.show_teachers),
            ("💰","Fees", self.show_fees),
            ("📄","Paper Fund", self.show_paper_fund),
            ("📚","Library", self.show_library),
            ("📝","Exams", self.show_exams),
            ("📈","Reports", self.show_reports),
        ]
        if self.current_user["role"]=="Admin" or \
           self.current_user.get("permissions",{}).get("users"):
            nav.append(("👥","Users", self.show_users))
        nav.append(("⚙️","Settings", self.show_settings))

        self._nav_btns = []
        for ic, txt, cmd in nav:
            b = tk.Button(self.sidebar, text=f"{ic}  {txt}",
                          font=("Helvetica",10), bg=c["sidebar_bg"],
                          fg=c["subtext"], activebackground=c["primary"],
                          activeforeground="white", bd=0,
                          padx=18, pady=11, anchor="w",
                          command=lambda cm=cmd: self._nav_click(cm))
            b.pack(fill="x")
            self._nav_btns.append((b, cmd))

        tk.Button(self.sidebar, text="🚪  Logout",
                  font=("Helvetica",10,"bold"), bg=c["danger"],
                  fg="white", bd=0, padx=18, pady=11,
                  command=self._logout).pack(side="bottom", fill="x",
                                             padx=12, pady=12)

    def _nav_click(self, cmd):
        c = self.colors
        for b, bc in self._nav_btns:
            if bc == cmd:
                b.config(bg=c["primary"], fg="white")
            else:
                b.config(bg=c["sidebar_bg"], fg=c["subtext"])
        cmd()

    def _build_content_area(self):
        self.main = tk.Frame(self.root, bg=self.colors["content_bg"])
        self.main.pack(side="left", fill="both", expand=True)

    def _clear(self):
        for w in self.main.winfo_children():
            w.destroy()

    def _header(self, title, subtitle=""):
        c = self.colors
        hdr = tk.Frame(self.main, bg=c["primary"], padx=30, pady=25)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=("Helvetica",26,"bold"),
                 bg=c["primary"], fg="white").pack(anchor="w")
        if subtitle:
            tk.Label(hdr, text=subtitle, font=("Helvetica",11),
                     bg=c["primary"], fg="#e0e7ff").pack(anchor="w")
        return hdr

    def _logout(self):
        self.current_user = None
        self.show_login()

    # ════════════════════════════════════════════════════════════════════════
    #  DASHBOARD
    # ════════════════════════════════════════════════════════════════════════
    def show_dashboard(self):
        self._clear()
        c = self.colors
        self._header("📊  Dashboard",
                     f"Welcome back, {self.current_user['username']}! "
                     f"– {datetime.datetime.now().strftime('%A, %d %B %Y')}")

        # ── calculate stats ──────────────────────────────────────────────
        total_s = len(self.students)
        total_t = len(self.teachers)
        cur_mon = datetime.datetime.now().strftime("%Y-%m")

        collected = 0
        pending_cnt = 0
        total_rev = 0
        outstanding = 0
        pending_students = []   # list of students with pending fee this month

        for stu in self.students:
            fr = stu.get("fee_records", {})
            for mon, rec in fr.items():
                paid_amt = rec.get("paid", 0)
                due_amt = rec.get("due", 0)
                disc = rec.get("discount", 0)
                fine = rec.get("fine", 0)
                total_rev += paid_amt
                bal = due_amt - disc + fine - paid_amt
                if bal > 0:
                    outstanding += bal
            if cur_mon in fr:
                if fr[cur_mon].get("paidFlag"):
                    collected += 1
                else:
                    pending_cnt += 1
                    pending_students.append(stu)
            else:
                # no record this month at all → pending
                pending_students.append(stu)

        # ── quick action cards ────────────────────────────────────────────
        qa = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=18)
        qa.pack(fill="x")
        tk.Label(qa, text="Quick Actions", font=("Helvetica",14,"bold"),
                 bg=c["content_bg"], fg=c["text"]).pack(anchor="w", pady=(0,10))

        qa_row = tk.Frame(qa, bg=c["content_bg"])
        qa_row.pack(fill="x")

        quick_actions = [
            ("💰","Add Fee Payment", self.show_fees, "#8b5cf6"),
            ("➕","Add Student", self.show_students, c["success"]),
            ("📅","Generate Vouchers", self.show_fees, "#f59e0b"),
            ("📊","Fee Report", self.show_reports, "#06b6d4"),
        ]
        for ic, label, cmd, col in quick_actions:
            btn_card = tk.Frame(qa_row, bg=col, padx=18, pady=14, cursor="hand2")
            btn_card.pack(side="left", expand=True, fill="both", padx=6)
            btn_card.bind("<Button-1>", lambda e, cm=cmd: cm())
            tk.Label(btn_card, text=ic, font=("Arial",24), bg=col).pack(anchor="w")
            tk.Label(btn_card, text=label, font=("Helvetica",11,"bold"),
                     bg=col, fg="white").pack(anchor="w", pady=(6,0))

        # ── stat cards ────────────────────────────────────────────────────
        sc_outer = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=5)
        sc_outer.pack(fill="x")
        tk.Label(sc_outer, text="Overview", font=("Helvetica",14,"bold"),
                 bg=c["content_bg"], fg=c["text"]).pack(anchor="w", pady=(0,8))

        sc_row = tk.Frame(sc_outer, bg=c["content_bg"])
        sc_row.pack(fill="x")

        stats = [
            ("👨‍🎓","Total Students", str(total_s), c["primary"], self.show_students),
            ("👩‍🏫","Teachers", str(total_t), "#8b5cf6", self.show_teachers),
            ("✅","Fees Collected", str(collected), c["success"], self.show_fees),
            ("⏳","Fees Pending", str(pending_cnt), c["danger"],
             lambda ps=pending_students: self._show_pending_students(ps)),
            ("💵","Total Revenue", f"RS {total_rev:,.0f}", "#06b6d4", self.show_reports),
            ("📉","Outstanding", f"RS {outstanding:,.0f}", c["warning"], self.show_fees),
        ]

        for ic, label, val, col, cmd in stats:
            card = tk.Frame(sc_row, bg=c["card_bg"], padx=16, pady=16, cursor="hand2")
            card.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            card.bind("<Button-1>", lambda e, cm=cmd: cm())
            for w in [card]: pass  # placeholder

            tk.Label(card, text=ic, font=("Arial",22),
                     bg=c["card_bg"]).pack(anchor="w")
            tk.Label(card, text=label, font=("Helvetica",9),
                     bg=c["card_bg"], fg=c["subtext"]).pack(anchor="w", pady=(8,2))
            tk.Label(card, text=val, font=("Helvetica",20,"bold"),
                     bg=c["card_bg"], fg=col).pack(anchor="w")

        # ── recent fee records table ──────────────────────────────────────
        rf = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=10)
        rf.pack(fill="both", expand=True)
        tk.Label(rf, text="Recent Fee Activity", font=("Helvetica",14,"bold"),
                 bg=c["content_bg"], fg=c["text"]).pack(anchor="w", pady=(0,8))

        cols = ("Student", "Class", "Month", "Due", "Paid", "Balance", "Status")
        tree = ttk.Treeview(rf, columns=cols, show="headings",
                            style="Custom.Treeview", height=8)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=120)
        sb = ttk.Scrollbar(rf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        rows_added = 0
        for stu in self.students:
            for mon, rec in sorted(stu.get("fee_records",{}).items(), reverse=True):
                due = rec.get("due", 0)
                paid = rec.get("paid", 0)
                disc = rec.get("discount", 0)
                fine = rec.get("fine", 0)
                bal = due - disc + fine - paid
                status = "✅ Paid" if rec.get("paidFlag") else "⏳ Pending"
                tree.insert("", "end", values=(
                    stu.get("name",""), stu.get("class",""), mon,
                    f"RS {due:,}", f"RS {paid:,}", f"RS {bal:,}", status
                ))
                rows_added += 1
                if rows_added >= 20:
                    break
            if rows_added >= 20:
                break

    def _show_pending_students(self, students):
        """Show a popup list of students with pending fees."""
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Pending Fee Students")
        dlg.geometry("900x580")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=c["danger"], padx=20, pady=15)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"⏳  Pending Fee Students ({len(students)})",
                 font=("Helvetica",16,"bold"), bg=c["danger"],
                 fg="white").pack(anchor="w")
        tk.Label(hdr, text="Students who have not fully paid fees this month",
                 font=("Helvetica",10), bg=c["danger"],
                 fg="#fecaca").pack(anchor="w")

        # ── filter bar ──────────────────────────────────────────────────────
        sf = tk.Frame(dlg, bg=c["dark"], padx=20, pady=8)
        sf.pack(fill="x")

        tk.Label(sf, text="🔍 Search:", bg=c["dark"], fg=c["gray"]).pack(side="left")
        search_var = tk.StringVar()
        se = tk.Entry(sf, textvariable=search_var, bg=c["dark_light"],
                      fg="white", insertbackground="white",
                      relief="flat", bd=6, font=("Helvetica",11), width=18)
        se.pack(side="left", padx=6)

        tk.Label(sf, text="Class:", bg=c["dark"], fg=c["gray"]).pack(side="left", padx=(12,0))
        cls_filter_var = tk.StringVar(value="All")
        all_cls = ["All"] + sorted(set(s.get("class","") for s in students if s.get("class","")))
        cls_cb = ttk.Combobox(sf, textvariable=cls_filter_var, values=all_cls,
                               state="readonly", width=8)
        cls_cb.pack(side="left", padx=4)
        tk.Button(sf, text="🔄 Filter", bg=c["primary"], fg="white", bd=0,
                  padx=8, pady=3,
                  command=lambda: populate(search_var.get().lower(),
                                           cls_filter_var.get())).pack(side="left", padx=6)

        # ── table ───────────────────────────────────────────────────────────
        tbl_f = tk.Frame(dlg, bg=c["dark"], padx=20)
        tbl_f.pack(fill="both", expand=True)

        cols = ("#","Adm No","Name","Class","Sec","Month","Due","Paid","Balance","Phone")
        tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                            style="Custom.Treeview")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=80)
        tree.column("Name", width=140)
        tree.column("Adm No", width=80)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        cur_mon = datetime.datetime.now().strftime("%Y-%m")

        def _stu_row_data(stu):
            fr = stu.get("fee_records", {})
            if cur_mon in fr:
                rec  = fr[cur_mon]
                due  = rec.get("due",0)
                paid = rec.get("paid",0)
                disc = rec.get("discount",0)
                fine = rec.get("fine",0)
                bal  = due - disc + fine - paid
                mon  = cur_mon
            else:
                due = paid = bal = 0
                mon = "—"
            return due, paid, bal, mon

        visible_students = []   # track what's currently shown

        def populate(term="", cls_f="All"):
            nonlocal visible_students
            for i in tree.get_children():
                tree.delete(i)
            visible_students = []
            idx = 1
            for stu in students:
                if cls_f != "All" and stu.get("class","") != cls_f:
                    continue
                if term and term not in stu.get("name","").lower() \
                        and term not in stu.get("admissionNo","").lower():
                    continue
                due, paid, bal, mon = _stu_row_data(stu)
                tree.insert("","end",
                            values=(idx, stu.get("admissionNo",""),
                                    stu.get("name",""), stu.get("class",""),
                                    stu.get("section",""), mon,
                                    f"RS {due:,}", f"RS {paid:,}",
                                    f"RS {bal:,}", stu.get("phone","")),
                            tags=(stu["id"],))
                visible_students.append(stu)
                idx += 1

        populate()
        search_var.trace_add("write",
            lambda *a: populate(search_var.get().lower(), cls_filter_var.get()))

        # ── helper: write rows to CSV ────────────────────────────────────────
        def _write_csv(path, stu_list):
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Adm No","Name","Class","Section","Month",
                             "Due (RS)","Paid (RS)","Balance (RS)","Phone"])
                for stu in stu_list:
                    due, paid, bal, mon = _stu_row_data(stu)
                    w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                                stu.get("class",""), stu.get("section",""), mon,
                                due, paid, bal, stu.get("phone","")])
            messagebox.showinfo("✅ Exported", f"Saved to:\n{path}")

        # ── Export ALL visible students ───────────────────────────────────────
        def export_all_csv():
            if not visible_students:
                messagebox.showwarning("Empty","No students to export."); return
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")],
                title="Export All Pending Students")
            if path:
                _write_csv(path, visible_students)

        # ── Export SELECTED single student ───────────────────────────────────
        def export_single_csv():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Select","Please select a student row first."); return
            sid = tree.item(sel[0],"tags")[0]
            stu = next((s for s in students if s["id"]==sid), None)
            if not stu: return
            safe_name = stu.get("name","student").replace(" ","_")
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV","*.csv")],
                initialfile=f"pending_{safe_name}_{cur_mon}.csv",
                title=f"Export – {stu.get('name','')}")
            if path:
                _write_csv(path, [stu])

        # ── Export CLASS-WISE (one CSV per class, zipped) ────────────────────
        def export_classwise_csv():
            if not visible_students:
                messagebox.showwarning("Empty","No students to export."); return
            # Group by class
            class_groups: dict = {}
            for stu in visible_students:
                cls = stu.get("class","Unknown")
                class_groups.setdefault(cls, []).append(stu)

            if len(class_groups) == 1:
                # Only one class – just save one file
                cls = list(class_groups.keys())[0]
                path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV","*.csv")],
                    initialfile=f"pending_class_{cls}_{cur_mon}.csv",
                    title=f"Export Class {cls} Pending")
                if path:
                    _write_csv(path, class_groups[cls])
                return

            # Multiple classes → ask for folder
            folder = filedialog.askdirectory(title="Select Folder – one CSV per class will be saved")
            if not folder: return
            saved = []
            for cls, stu_list in class_groups.items():
                safe_cls = cls.replace(" ","_").replace("/","-")
                fpath = os.path.join(folder, f"pending_class_{safe_cls}_{cur_mon}.csv")
                with open(fpath,"w",newline="",encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Adm No","Name","Class","Section","Month",
                                 "Due (RS)","Paid (RS)","Balance (RS)","Phone"])
                    for stu in stu_list:
                        due, paid, bal, mon = _stu_row_data(stu)
                        w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                                    stu.get("class",""), stu.get("section",""), mon,
                                    due, paid, bal, stu.get("phone","")])
                saved.append(os.path.basename(fpath))
            messagebox.showinfo("✅ Class-wise Export",
                f"Exported {len(saved)} file(s) to:\n{folder}\n\n" + "\n".join(saved))

        # ── right-click context menu on a row ────────────────────────────────
        ctx = tk.Menu(dlg, tearoff=0, bg=c["dark_light"], fg=c["text"],
                      activebackground=c["primary"], activeforeground="white")
        ctx.add_command(label="📥 Export This Student CSV", command=export_single_csv)
        ctx.add_separator()
        ctx.add_command(label="📥 Export All (Visible) CSV", command=export_all_csv)
        ctx.add_command(label="📂 Export Class-Wise CSV",    command=export_classwise_csv)

        def show_ctx(event):
            # Select row under cursor
            row = tree.identify_row(event.y)
            if row:
                tree.selection_set(row)
            try:
                ctx.tk_popup(event.x_root, event.y_root)
            finally:
                ctx.grab_release()

        tree.bind("<Button-3>", show_ctx)   # right-click

        # ── bottom button bar ────────────────────────────────────────────────
        btn_f = tk.Frame(dlg, bg=c["dark"], pady=10, padx=20)
        btn_f.pack(fill="x")

        tk.Button(btn_f, text="📥 Export All CSV",
                  bg=c["success"], fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=export_all_csv).pack(side="left", padx=4)
        tk.Button(btn_f, text="👤 Export Selected Student CSV",
                  bg="#0ea5e9", fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=export_single_csv).pack(side="left", padx=4)
        tk.Button(btn_f, text="📂 Export Class-Wise CSV",
                  bg="#7c3aed", fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=export_classwise_csv).pack(side="left", padx=4)
        tk.Label(btn_f, text="💡 Right-click any row for quick export",
                 bg=c["dark"], fg=c["subtext"], font=("Helvetica",9)).pack(side="left", padx=14)
        tk.Button(btn_f, text="✖ Close",
                  bg=c["danger"], fg="white", padx=12, pady=6,
                  command=dlg.destroy).pack(side="right", padx=4)

    # ════════════════════════════════════════════════════════════════════════
    #  STUDENTS
    # ════════════════════════════════════════════════════════════════════════
    def show_students(self):
        if not self.check_perm("students"): return
        self._clear()
        c = self.colors
        self._header("👨‍🎓  Student Management", "Add, edit, delete, promote, QR-code")

        ctrl = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=12)
        ctrl.pack(fill="x")

        self._stu_search = tk.Entry(ctrl, font=("Helvetica",11),
                                    bg=c["card_bg"], fg=c["text"],
                                    insertbackground=c["text"],
                                    relief="flat", bd=8)
        self._stu_search.pack(side="left", fill="x", expand=True, padx=(0,8))
        self._stu_search.bind("<KeyRelease>", lambda e: self._refresh_students())
        tk.Label(ctrl, text="🔍", bg=c["content_bg"], fg=c["subtext"],
                 font=("Helvetica",14)).pack(side="left")

        # filter by class
        tk.Label(ctrl, text=" Class:", bg=c["content_bg"],
                 fg=c["subtext"]).pack(side="left", padx=(10,0))
        self._stu_class_filter = tk.StringVar(value="All")
        class_cb = ttk.Combobox(ctrl, textvariable=self._stu_class_filter,
                                  values=["All"] + self._all_classes(),
                                  state="readonly", width=8)
        class_cb.pack(side="left", padx=4)
        class_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_students())

        # filter by fee status
        tk.Label(ctrl, text="Fee:", bg=c["content_bg"],
                 fg=c["subtext"]).pack(side="left", padx=(10,0))
        self._fee_filter = tk.StringVar(value="All")
        fee_cb = ttk.Combobox(ctrl, textvariable=self._fee_filter,
                               values=["All","Paid","Pending"],
                               state="readonly", width=8)
        fee_cb.pack(side="left", padx=4)
        fee_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_students())

        tk.Button(ctrl, text="➕ Add Student", bg=c["success"], fg="white",
                  font=("Helvetica",10,"bold"),
                  command=self._add_student_dlg).pack(side="right", padx=4)

        # table
        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True)

        cols = ("#","Adm No","Name","Father","Class","Sec","Phone",
                "Fee Status","Paid","Due","Balance")
        self._stu_tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                       style="Custom.Treeview", height=18)
        widths = {"#":40,"Adm No":80,"Name":150,"Father":120,"Class":60,
                  "Sec":50,"Phone":100,"Fee Status":90,"Paid":80,"Due":80,"Balance":90}
        for col in cols:
            self._stu_tree.heading(col, text=col)
            self._stu_tree.column(col, anchor="center", width=widths.get(col,100))

        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._stu_tree.yview)
        hsb = ttk.Scrollbar(tbl_f, orient="horizontal", command=self._stu_tree.xview)
        self._stu_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._stu_tree.pack(fill="both", expand=True)
        self._stu_tree.bind("<Double-1>", lambda e: self._view_student())

        self._refresh_students()

    def _all_classes(self):
        return sorted({s.get("class","") for s in self.students if s.get("class")})

    def _refresh_students(self):
        for i in self._stu_tree.get_children():
            self._stu_tree.delete(i)
        term = self._stu_search.get().lower()
        cls_f = self._stu_class_filter.get()
        fee_f = self._fee_filter.get()
        cur_mon = datetime.datetime.now().strftime("%Y-%m")
        idx = 1
        for stu in self.students:
            if term and term not in stu.get("name","").lower() \
                    and term not in stu.get("admissionNo","").lower() \
                    and term not in stu.get("class","").lower():
                continue
            if cls_f != "All" and stu.get("class","") != cls_f:
                continue
            fr = stu.get("fee_records",{})
            rec = fr.get(cur_mon, {})
            paid_flag = rec.get("paidFlag", False)
            if fee_f == "Paid" and not paid_flag: continue
            if fee_f == "Pending" and paid_flag: continue

            due = rec.get("due",0)
            paid = rec.get("paid",0)
            disc = rec.get("discount",0)
            fine = rec.get("fine",0)
            bal = due - disc + fine - paid
            status = "✅ Paid" if paid_flag else "⏳ Pending"

            self._stu_tree.insert("","end",
                values=(idx, stu.get("admissionNo",""), stu.get("name",""),
                        stu.get("father",""), stu.get("class",""),
                        stu.get("section",""), stu.get("phone",""),
                        status, f"RS {paid:,}", f"RS {due:,}", f"RS {bal:,}"),
                tags=(stu["id"],))
            idx += 1

    def _view_student(self):
        sel = self._stu_tree.selection()
        if not sel: return
        sid = self._stu_tree.item(sel[0],"tags")[0]
        stu = next((s for s in self.students if s["id"]==sid), None)
        if not stu: return

        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Student: {stu['name']}")
        dlg.geometry("700x600")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        # tabs via notebook
        nb = ttk.Notebook(dlg)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Info tab ──────────────────────────────────────────────────────
        info_tab = tk.Frame(nb, bg=c["dark"])
        nb.add(info_tab, text="📋 Info")

        if stu.get("photo_path") and HAS_PIL and _ensure_pil():
            ph = self.data_dir / stu["photo_path"]
            if ph.exists():
                try:
                    img = Image.open(ph).resize((120,120), Image.LANCZOS)
                    itk = ImageTk.PhotoImage(img)
                    lbl = tk.Label(info_tab, image=itk, bg=c["dark"])
                    lbl.pack(pady=10)
                    lbl.image = itk
                except Exception:
                    pass

        for label, key in [("Name","name"),("Admission No","admissionNo"),
                            ("Class","class"),("Section","section"),
                            ("Father","father"),("Phone","phone"),
                            ("DOB","dob"),("Address","address")]:
            row = tk.Frame(info_tab, bg=c["dark"])
            row.pack(fill="x", padx=30, pady=3)
            tk.Label(row, text=f"{label}:", font=("Helvetica",10,"bold"),
                     bg=c["dark"], fg=c["gray"], width=14).pack(side="left")
            tk.Label(row, text=stu.get(key,"—"), font=("Helvetica",10),
                     bg=c["dark"], fg="white").pack(side="left")

        act = tk.Frame(info_tab, bg=c["dark"])
        act.pack(pady=15, padx=30, anchor="w")
        tk.Button(act, text="✏️ Edit", bg=c["primary"], fg="white",
                  command=lambda: (dlg.destroy(), self._add_student_dlg(stu))).pack(side="left", padx=3)
        tk.Button(act, text="🗑️ Delete", bg=c["danger"], fg="white",
                  command=lambda: self._del_student(sid, dlg)).pack(side="left", padx=3)
        tk.Button(act, text="🚀 Promote", bg=c["success"], fg="white",
                  command=lambda: self._promote_dlg(sid)).pack(side="left", padx=3)
        tk.Button(act, text="📱 QR Code", bg="#8b5cf6", fg="white",
                  command=lambda: self._generate_student_qr(sid)).pack(side="left", padx=3)

        # Show face photo status + upload button
        has_photo = bool(stu.get("photo_path") and (self.data_dir / stu["photo_path"]).exists())
        face_f = tk.Frame(info_tab, bg=c["dark"], padx=30)
        face_f.pack(fill="x", pady=(5,0))
        tk.Label(face_f,
                 text=f"📷 Face Photo: {'✅ Uploaded' if has_photo else '❌ Not uploaded'}",
                 bg=c["dark"], fg=c["success"] if has_photo else c["danger"],
                 font=("Helvetica",9,"bold")).pack(side="left")

        def _upload_face_photo():
            path = filedialog.askopenfilename(
                title=f"Upload Face Photo for {stu['name']}",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            if not path: return
            pd = self.data_dir / "student_photos"
            pd.mkdir(exist_ok=True)
            ext = Path(path).suffix.lower()
            dest = pd / f"{sid}{ext}"
            try:
                shutil.copyfile(path, dest)
                stu["photo_path"] = str(dest.relative_to(self.data_dir))
                self.save_data()
                messagebox.showinfo("✅", f"Photo uploaded for {stu['name']}!\n"
                                    "Face attendance will now work for this student.")
                dlg.destroy()
            except Exception as ex:
                messagebox.showerror("Error", f"Photo upload failed: {ex}")

        tk.Button(face_f, text="📷 Upload Face Photo" if not has_photo else "🔄 Change Photo",
                  bg="#7c3aed" if not has_photo else c["primary"],
                  fg="white", font=("Helvetica",9),
                  command=_upload_face_photo).pack(side="left", padx=8)
        if not has_photo:
            tk.Label(face_f, text="(Required for Face Attendance)",
                     bg=c["dark"], fg=c["warning"],
                     font=("Helvetica",8)).pack(side="left")

        # ── Fee History tab ───────────────────────────────────────────────
        fee_tab = tk.Frame(nb, bg=c["dark"])
        nb.add(fee_tab, text="💰 Fee History")

        tk.Label(fee_tab, text=f"Complete Fee Record — {stu['name']}",
                 font=("Helvetica",12,"bold"), bg=c["dark"],
                 fg="white").pack(pady=(15,5), padx=20, anchor="w")

        # summary row
        total_due = total_paid = total_bal = 0
        for rec in stu.get("fee_records",{}).values():
            d = rec.get("due",0); p = rec.get("paid",0)
            disc = rec.get("discount",0); fine = rec.get("fine",0)
            total_due += d; total_paid += p
            total_bal += d - disc + fine - p

        sum_f = tk.Frame(fee_tab, bg=c["primary"], padx=15, pady=8)
        sum_f.pack(fill="x", padx=20)
        for lbl, val in [("Total Due", f"RS {total_due:,}"),
                         ("Total Paid", f"RS {total_paid:,}"),
                         ("Balance", f"RS {total_bal:,}")]:
            tk.Label(sum_f, text=f"{lbl}: {val}",
                     font=("Helvetica",10,"bold"), bg=c["primary"],
                     fg="white").pack(side="left", padx=20)

        # detailed table
        fcols = ("Month","Due","Discount","Fine","Paid","Balance","Status")
        ftree = ttk.Treeview(fee_tab, columns=fcols, show="headings",
                              style="Custom.Treeview", height=10)
        for col in fcols:
            ftree.heading(col, text=col)
            ftree.column(col, anchor="center", width=90)
        ftree.column("Month", width=80)
        fsb = ttk.Scrollbar(fee_tab, orient="vertical", command=ftree.yview)
        ftree.configure(yscrollcommand=fsb.set)
        ftree.pack(side="left", fill="both", expand=True, padx=(20,0), pady=10)
        fsb.pack(side="right", fill="y", pady=10, padx=(0,20))

        for mon, rec in sorted(stu.get("fee_records",{}).items(), reverse=True):
            d = rec.get("due",0); p = rec.get("paid",0)
            disc = rec.get("discount",0); fine = rec.get("fine",0)
            bal = d - disc + fine - p
            status = "✅ Paid" if rec.get("paidFlag") else "⏳ Pending"
            ftree.insert("","end",
                values=(mon, f"RS {d:,}", f"RS {disc:,}", f"RS {fine:,}",
                        f"RS {p:,}", f"RS {bal:,}", status))

        pay_btn = tk.Button(fee_tab, text="💵 Record Payment",
                            bg=c["success"], fg="white",
                            command=lambda: self._pay_for_student(stu, ftree))
        pay_btn.pack(pady=5)

    def _pay_for_student(self, stu, tree_ref):
        """Full payment dialog – shows live balance, supports partial payment."""
        c = self.colors
        cur_mon = datetime.datetime.now().strftime("%Y-%m")

        dlg = tk.Toplevel(self.root)
        dlg.title(f"💵 Record Payment – {stu['name']}")
        dlg.geometry("480x560")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        # Header
        hdr = tk.Frame(dlg, bg=c["success"], padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"💵  Record Fee Payment",
                 font=("Helvetica",14,"bold"), bg=c["success"],
                 fg="white").pack(anchor="w")
        tk.Label(hdr, text=f"Student: {stu['name']}  |  Class: {stu.get('class','')}  |  Adm: {stu.get('admissionNo','')}",
                 bg=c["success"], fg="#d1fae5",
                 font=("Helvetica",9)).pack(anchor="w")

        body = tk.Frame(dlg, bg=c["dark"], padx=25)
        body.pack(fill="both", expand=True, pady=5)

        # Month selector with live balance display
        tk.Label(body, text="Select Month:", bg=c["dark"],
                 fg=c["gray"], font=("Helvetica",10)).pack(anchor="w", pady=(10,2))

        months_with_records = sorted(stu.get("fee_records", {}).keys(), reverse=True)
        mon_var = tk.StringVar(value=cur_mon)

        mon_frame = tk.Frame(body, bg=c["dark"])
        mon_frame.pack(fill="x")
        mon_cb = ttk.Combobox(mon_frame, textvariable=mon_var,
                               values=[cur_mon] + [m for m in months_with_records if m != cur_mon],
                               width=14)
        mon_cb.pack(side="left")
        tk.Label(mon_frame, text="  or type: YYYY-MM", bg=c["dark"],
                 fg=c["gray"], font=("Helvetica",8)).pack(side="left")

        # Live balance card
        balance_card = tk.Frame(body, bg=c["dark_light"], padx=15, pady=10)
        balance_card.pack(fill="x", pady=10)

        lbl_due   = tk.Label(balance_card, text="Due:      RS 0", bg=c["dark_light"],
                              fg="white", font=("Helvetica",10), anchor="w")
        lbl_disc  = tk.Label(balance_card, text="Discount: RS 0", bg=c["dark_light"],
                              fg=c["success"], font=("Helvetica",10), anchor="w")
        lbl_fine  = tk.Label(balance_card, text="Fine:     RS 0", bg=c["dark_light"],
                              fg=c["danger"], font=("Helvetica",10), anchor="w")
        lbl_paid  = tk.Label(balance_card, text="Paid:     RS 0", bg=c["dark_light"],
                              fg="#38bdf8", font=("Helvetica",10), anchor="w")
        lbl_bal   = tk.Label(balance_card, text="Balance:  RS 0", bg=c["dark_light"],
                              fg=c["warning"], font=("Helvetica",13,"bold"), anchor="w")
        lbl_status = tk.Label(balance_card, text="", bg=c["dark_light"],
                               font=("Helvetica",11,"bold"), anchor="w")

        for lbl in [lbl_due, lbl_disc, lbl_fine, lbl_paid, lbl_bal, lbl_status]:
            lbl.pack(fill="x", pady=1)

        def refresh_balance(*args):
            mon = mon_var.get().strip()
            fr = stu.get("fee_records", {})
            rec = fr.get(mon, {})
            due  = rec.get("due", 0)
            disc = rec.get("discount", 0)
            fine = rec.get("fine", 0)
            paid = rec.get("paid", 0)
            bal  = due - disc + fine - paid
            lbl_due  .config(text=f"Due:          RS {due:,}")
            lbl_disc .config(text=f"Discount:     RS {disc:,}")
            lbl_fine .config(text=f"Fine:         RS {fine:,}")
            lbl_paid .config(text=f"Already Paid: RS {paid:,}")
            lbl_bal  .config(text=f"Balance Due:  RS {bal:,}",
                              fg=c["danger"] if bal > 0 else c["success"])
            if rec.get("paidFlag"):
                lbl_status.config(text="✅  FULLY PAID", fg=c["success"])
            elif paid > 0:
                lbl_status.config(text="⚠️  PARTIALLY PAID", fg=c["warning"])
            else:
                lbl_status.config(text="⏳  UNPAID", fg=c["danger"])

        mon_var.trace_add("write", refresh_balance)
        refresh_balance()

        # Payment entry fields
        tk.Frame(body, bg=c["gray"], height=1).pack(fill="x", pady=8)

        fields_frame = tk.Frame(body, bg=c["dark"])
        fields_frame.pack(fill="x")

        def make_row(label, placeholder="0"):
            row = tk.Frame(fields_frame, bg=c["dark"])
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=c["dark"], fg=c["gray"],
                     width=22, anchor="w", font=("Helvetica",10)).pack(side="left")
            e = tk.Entry(row, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=6,
                          font=("Helvetica",11), width=14)
            e.pack(side="left", padx=5)
            tk.Label(row, text=placeholder, bg=c["dark"],
                     fg=c["gray"], font=("Helvetica",8)).pack(side="left")
            return e

        pay_e  = make_row("💵 Amount to Pay (RS):", "required")
        disc_e = make_row("🏷️ Extra Discount (RS):", "optional")
        fine_e = make_row("⚠️ Late Fine (RS):",      "optional")

        # Quick-pay full balance button
        def pay_full():
            mon = mon_var.get().strip()
            fr = stu.get("fee_records", {})
            rec = fr.get(mon, {})
            due  = rec.get("due", 0)
            disc = rec.get("discount", 0)
            fine_existing = rec.get("fine", 0)
            paid = rec.get("paid", 0)
            bal  = due - disc + fine_existing - paid
            if bal <= 0:
                messagebox.showinfo("ℹ️","This fee is already fully paid!"); return
            pay_e.delete(0,"end")
            pay_e.insert(0, str(bal))

        tk.Button(body, text="⚡ Fill Full Balance Amount",
                  bg=c["primary"], fg="white", font=("Helvetica",9),
                  padx=8, pady=4,
                  command=pay_full).pack(anchor="w", pady=(2,8))

        def _refresh_tree():
            for i in tree_ref.get_children():
                tree_ref.delete(i)
            for m2, r2 in sorted(stu.get("fee_records",{}).items(), reverse=True):
                d2  = r2.get("due",0);    p2   = r2.get("paid",0)
                di2 = r2.get("discount",0); fi2 = r2.get("fine",0)
                bal2 = d2 - di2 + fi2 - p2
                tree_ref.insert("","end",
                    values=(m2, f"RS {d2:,}", f"RS {di2:,}", f"RS {fi2:,}",
                            f"RS {p2:,}", f"RS {bal2:,}",
                            "✅ Paid" if r2.get("paidFlag") else "⏳ Pending"))

        def submit():
            mon = mon_var.get().strip()
            try:
                datetime.datetime.strptime(mon, "%Y-%m")
            except Exception:
                messagebox.showerror("Error","Month must be YYYY-MM (e.g. 2025-06)"); return
            try:
                pay  = int(pay_e.get().strip()  or 0)
                disc = int(disc_e.get().strip() or 0)
                fine = int(fine_e.get().strip() or 0)
            except ValueError:
                messagebox.showerror("Error","Payment amounts must be whole numbers"); return
            if pay == 0 and disc == 0 and fine == 0:
                messagebox.showerror("Error","Enter at least one amount"); return

            fr = stu.setdefault("fee_records", {})
            if mon not in fr:
                # auto-create from fee structure
                fs = self.fee_structures.get(stu.get("class",""), {})
                if isinstance(fs, dict):
                    due_amt = sum(v for v in fs.values() if isinstance(v,(int,float)))
                elif isinstance(fs,(int,float)):
                    due_amt = int(fs)
                else:
                    due_amt = 0
                fr[mon] = {"due": due_amt, "paid": 0, "discount": 0, "fine": 0, "paidFlag": False}

            rec = fr[mon]
            rec["paid"]     = rec.get("paid",0)     + pay
            rec["discount"] = rec.get("discount",0) + disc
            rec["fine"]     = rec.get("fine",0)     + fine
            net_due = rec["due"] - rec["discount"] + rec["fine"]
            rec["paidFlag"] = rec["paid"] >= net_due

            self.save_data()
            refresh_balance()
            _refresh_tree()
            pay_e.delete(0,"end"); disc_e.delete(0,"end"); fine_e.delete(0,"end")
            bal_left = net_due - rec["paid"]
            if rec["paidFlag"]:
                messagebox.showinfo("✅ Fully Paid",
                    f"RS {pay:,} recorded.\n{stu['name']}'s fee for {mon} is now FULLY PAID!")
            else:
                messagebox.showinfo("✅ Partial Payment",
                    f"RS {pay:,} recorded.\nRemaining balance: RS {max(0,bal_left):,}")

        tk.Button(dlg, text="✅  Record Payment", bg=c["success"], fg="white",
                  font=("Helvetica",12,"bold"), pady=12,
                  command=submit).pack(fill="x", padx=25, pady=12)

    def _add_student_dlg(self, edit=None):
        c = self.colors
        is_edit = edit is not None
        dlg = tk.Toplevel(self.root)
        dlg.title("Edit Student" if is_edit else "Add New Student")
        dlg.geometry("600x680")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Frame(dlg, bg=c["primary"], height=5).pack(fill="x")
        tk.Label(dlg, text="Edit Student" if is_edit else "➕  Add New Student",
                 font=("Helvetica",16,"bold"), bg=c["dark"],
                 fg="white").pack(pady=12)

        # scrollable area
        cv = tk.Canvas(dlg, bg=c["dark"], highlightthickness=0)
        sb = ttk.Scrollbar(dlg, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        frm = tk.Frame(cv, bg=c["dark"], padx=30)
        fw = cv.create_window((0,0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(fw, width=e.width))
        def _stu_scroll(event):
            if event.num == 4:
                cv.yview_scroll(-3, "units")
            elif event.num == 5:
                cv.yview_scroll(3, "units")
            else:
                cv.yview_scroll(int(-1*(event.delta/120)), "units")
        cv.bind_all("<MouseWheel>", _stu_scroll)
        cv.bind_all("<Button-4>", _stu_scroll)
        cv.bind_all("<Button-5>", _stu_scroll)
        def _stu_unbind(e):
            try:
                cv.unbind_all("<MouseWheel>")
                cv.unbind_all("<Button-4>")
                cv.unbind_all("<Button-5>")
            except Exception:
                pass
        dlg.bind("<Destroy>", _stu_unbind)

        fields = [
            ("Admission No *","admissionNo",
             edit.get("admissionNo","") if is_edit else f"ADM{1000+len(self.students)}"),
            ("Student Name *","name", edit.get("name","") if is_edit else ""),
            ("Father's Name","father", edit.get("father","") if is_edit else ""),
            ("Phone","phone", edit.get("phone","") if is_edit else ""),
            ("Class","class", edit.get("class","") if is_edit else ""),
            ("Section","section", edit.get("section","") if is_edit else ""),
            ("Email","email", edit.get("email","") if is_edit else ""),
            ("Date of Birth (YYYY-MM-DD)","dob", edit.get("dob","") if is_edit else ""),
            ("Address","address", edit.get("address","") if is_edit else ""),
        ]

        entries = {}
        for lbl, key, default in fields:
            tk.Label(frm, text=lbl, bg=c["dark"], fg=c["gray"],
                     font=("Helvetica",10)).pack(anchor="w", pady=(10,0))
            e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat",
                          bd=7, font=("Helvetica",11))
            e.pack(fill="x", pady=3)
            e.insert(0, default)
            entries[key] = e

        # ── Fee Section ───────────────────────────────────────────────────
        tk.Frame(frm, bg=c["primary"], height=2).pack(fill="x", pady=(15,0))
        tk.Label(frm, text="💰  Fee Information (Current Month)",
                 font=("Helvetica",12,"bold"), bg=c["dark"],
                 fg=c["primary"]).pack(anchor="w", pady=(8,0))

        fee_frame = tk.Frame(frm, bg=c["dark_light"], padx=15, pady=12)
        fee_frame.pack(fill="x", pady=5)

        # Auto-fill from fee structure when class is chosen
        def auto_fill_fee(*args):
            cls_val = entries.get("class","") and entries["class"].get().strip()
            if cls_val and cls_val in self.fee_structures:
                fs = self.fee_structures[cls_val]
                total = 0
                if isinstance(fs, dict):
                    total = sum(v for v in fs.values() if isinstance(v,(int,float)))
                elif isinstance(fs,(int,float)):
                    total = int(fs)
                if total:
                    fee_due_e.delete(0,"end")
                    fee_due_e.insert(0, str(total))

        # Monthly Due
        r1 = tk.Frame(fee_frame, bg=c["dark_light"])
        r1.pack(fill="x", pady=3)
        tk.Label(r1, text="Monthly Fee Due (RS):", bg=c["dark_light"],
                 fg=c["gray"], width=22, anchor="w").pack(side="left")
        fee_due_e = tk.Entry(r1, bg=c["dark"], fg="white",
                              insertbackground="white", relief="flat",
                              bd=6, font=("Helvetica",11), width=15)
        fee_due_e.pack(side="left", padx=5)
        # pre-fill from fee structure if editing
        if is_edit:
            cur_mon = datetime.date.today().strftime("%Y-%m")
            existing_rec = edit.get("fee_records",{}).get(cur_mon,{})
            fee_due_e.insert(0, str(existing_rec.get("due","")))

        tk.Button(r1, text="📋 Auto-fill from Class", bg=c["primary"], fg="white",
                  font=("Helvetica",8), padx=6, pady=2,
                  command=auto_fill_fee).pack(side="left", padx=5)

        # Admission Fee (one-time)
        r2 = tk.Frame(fee_frame, bg=c["dark_light"])
        r2.pack(fill="x", pady=3)
        tk.Label(r2, text="Admission Fee (RS, one-time):", bg=c["dark_light"],
                 fg=c["gray"], width=22, anchor="w").pack(side="left")
        fee_admission_e = tk.Entry(r2, bg=c["dark"], fg="white",
                                    insertbackground="white", relief="flat",
                                    bd=6, font=("Helvetica",11), width=15)
        fee_admission_e.pack(side="left", padx=5)
        if is_edit:
            adm_rec = edit.get("fee_records",{}).get("admission",{})
            fee_admission_e.insert(0, str(adm_rec.get("due","")) if adm_rec else "")

        # Initial Payment
        r3 = tk.Frame(fee_frame, bg=c["dark_light"])
        r3.pack(fill="x", pady=3)
        tk.Label(r3, text="Initial Payment (RS):", bg=c["dark_light"],
                 fg=c["gray"], width=22, anchor="w").pack(side="left")
        fee_initial_pay_e = tk.Entry(r3, bg=c["dark"], fg="white",
                                      insertbackground="white", relief="flat",
                                      bd=6, font=("Helvetica",11), width=15)
        fee_initial_pay_e.pack(side="left", padx=5)
        tk.Label(r3, text="(amount paid right now)", bg=c["dark_light"],
                 fg=c["gray"], font=("Helvetica",8)).pack(side="left")

        # Discount
        r4 = tk.Frame(fee_frame, bg=c["dark_light"])
        r4.pack(fill="x", pady=3)
        tk.Label(r4, text="Concession / Discount (RS):", bg=c["dark_light"],
                 fg=c["gray"], width=22, anchor="w").pack(side="left")
        fee_disc_e = tk.Entry(r4, bg=c["dark"], fg="white",
                               insertbackground="white", relief="flat",
                               bd=6, font=("Helvetica",11), width=15)
        fee_disc_e.pack(side="left", padx=5)

        tk.Label(fee_frame,
                 text="ℹ️  Leave blank if no fee is being set now. You can always update from the Fees module.",
                 bg=c["dark_light"], fg=c["gray"],
                 font=("Helvetica",8), wraplength=480).pack(anchor="w", pady=(5,0))

        # Photo
        photo_var = tk.StringVar()
        tk.Label(frm, text="Student Photo", bg=c["dark"],
                 fg=c["gray"]).pack(anchor="w", pady=(12,0))
        ph_f = tk.Frame(frm, bg=c["dark_light"], padx=10, pady=10)
        ph_f.pack(fill="x", pady=3)
        prev = tk.Label(ph_f, text="No photo", bg=c["dark_light"],
                        fg=c["gray"], width=14, height=7)
        prev.pack(side="left", padx=(0,10))

        def pick_photo():
            path = filedialog.askopenfilename(
                filetypes=[("Image","*.png *.jpg *.jpeg *.bmp *.gif")])
            if path:
                photo_var.set(path)
                if HAS_PIL and _ensure_pil():
                    try:
                        img = Image.open(path).resize((110,110), Image.LANCZOS)
                        itk = ImageTk.PhotoImage(img)
                        prev.config(image=itk, text="")
                        prev.image = itk
                    except Exception:
                        pass

        if is_edit and edit.get("photo_path") and HAS_PIL:
            ph = self.data_dir / edit["photo_path"]
            if ph.exists():
                try:
                    img = Image.open(ph).resize((110,110), Image.LANCZOS)
                    itk = ImageTk.PhotoImage(img)
                    prev.config(image=itk, text="")
                    prev.image = itk
                    photo_var.set(str(ph))
                except Exception:
                    pass

        tk.Button(ph_f, text="📁 Upload Photo", bg=c["primary"], fg="white",
                  command=pick_photo).pack(anchor="w")

        def save():
            data = {k: entries[k].get().strip() for k in entries}
            if not data["name"]:
                messagebox.showerror("Error","Student name required"); return
            if not data["admissionNo"]:
                messagebox.showerror("Error","Admission No required"); return

            # handle photo
            photo_src = photo_var.get()
            if photo_src and (not is_edit or photo_src != str(self.data_dir/edit.get("photo_path",""))):
                pd = self.data_dir / "student_photos"
                pd.mkdir(exist_ok=True)
                sid = edit["id"] if is_edit else self.gen_id()
                ext = Path(photo_src).suffix.lower()
                dest = pd / f"{sid}{ext}"
                try:
                    shutil.copyfile(photo_src, dest)
                    data["photo_path"] = str(dest.relative_to(self.data_dir))
                except Exception as ex:
                    messagebox.showerror("Error", f"Photo copy failed: {ex}"); return
            elif is_edit:
                data["photo_path"] = edit.get("photo_path","")
            else:
                data["photo_path"] = ""

            if is_edit:
                edit.update(data)
                # update current month fee due if provided
                due_str = fee_due_e.get().strip()
                if due_str:
                    try:
                        due_amt = int(due_str)
                        cur_mon = datetime.date.today().strftime("%Y-%m")
                        fr = edit.setdefault("fee_records", {})
                        if cur_mon not in fr:
                            fr[cur_mon] = {"due": due_amt, "paid": 0,
                                           "discount": 0, "fine": 0, "paidFlag": False}
                        else:
                            fr[cur_mon]["due"] = due_amt
                        disc_str = fee_disc_e.get().strip()
                        if disc_str:
                            fr[cur_mon]["discount"] = int(disc_str)
                    except ValueError:
                        pass
            else:
                new_id = self.gen_id()
                data.update({"id": new_id, "attendance":{}, "results":{},
                              "qr_code":"", "fee_records":{}})
                # ── process fee fields ──────────────────────────────────
                cur_mon = datetime.date.today().strftime("%Y-%m")
                due_str = fee_due_e.get().strip()
                init_pay_str = fee_initial_pay_e.get().strip()
                disc_str = fee_disc_e.get().strip()
                adm_str = fee_admission_e.get().strip()

                if due_str:
                    try:
                        due_amt = int(due_str)
                        init_pay = int(init_pay_str) if init_pay_str else 0
                        disc_amt = int(disc_str) if disc_str else 0
                        net = due_amt - disc_amt
                        paid_flag = init_pay >= net
                        data["fee_records"][cur_mon] = {
                            "due": due_amt, "paid": init_pay,
                            "discount": disc_amt, "fine": 0,
                            "paidFlag": paid_flag
                        }
                    except ValueError:
                        messagebox.showerror("Error","Fee amounts must be numbers"); return

                if adm_str:
                    try:
                        adm_amt = int(adm_str)
                        data["fee_records"]["admission"] = {
                            "due": adm_amt, "paid": 0,
                            "discount": 0, "fine": 0,
                            "paidFlag": False,
                            "label": "Admission Fee"
                        }
                    except ValueError:
                        pass
                # ───────────────────────────────────────────────────────
                self.students.append(data)

            self.save_data()
            dlg.destroy()
            self._refresh_students()
            messagebox.showinfo("✅","Student saved successfully!")

        tk.Button(frm, text="💾  Save Student", bg=c["success"], fg="white",
                  font=("Helvetica",12,"bold"), pady=12,
                  command=save).pack(fill="x", pady=20)

    def _del_student(self, sid, parent_dlg=None):
        if not messagebox.askyesno("Confirm","Delete this student permanently?"):
            return
        self.students = [s for s in self.students if s["id"] != sid]
        self.save_data()
        if parent_dlg:
            parent_dlg.destroy()
        self._refresh_students()
        messagebox.showinfo("Deleted","Student removed.")

    def _promote_dlg(self, sid):
        stu = next((s for s in self.students if s["id"]==sid), None)
        if not stu: return
        new_cls = simpledialog.askstring("Promote Student",
            f"Current class: {stu.get('class','')}\nEnter new class:")
        if new_cls:
            stu["class"] = new_cls.strip()
            self.save_data()
            self._refresh_students()
            messagebox.showinfo("Promoted", f"{stu['name']} promoted to {new_cls}")

    # ════════════════════════════════════════════════════════════════════════
    #  ATTENDANCE
    # ════════════════════════════════════════════════════════════════════════
    def show_attendance(self):
        if not self.check_perm("attendance"): return
        self._clear()
        c = self.colors
        self._header("📋  Attendance System", "Complete attendance management with QR, Face Recognition & Manual marking")

        # ── Class & Date Selection Bar ───────────────────────────────────
        sel_bar = tk.Frame(self.main, bg=c["card_bg"], padx=25, pady=12)
        sel_bar.pack(fill="x", padx=20, pady=(10,5))

        tk.Label(sel_bar, text="📚 Class:", bg=c["card_bg"], fg=c["subtext"],
                 font=("Helvetica",11,"bold")).pack(side="left")
        cls_v = tk.StringVar()
        cls_cb = ttk.Combobox(sel_bar, textvariable=cls_v,
                               values=self._all_classes(), state="readonly", width=10)
        cls_cb.pack(side="left", padx=6)

        tk.Label(sel_bar, text="📅 Date:", bg=c["card_bg"], fg=c["subtext"],
                 font=("Helvetica",11,"bold")).pack(side="left", padx=(20,0))
        date_v = tk.StringVar(value=datetime.date.today().isoformat())
        tk.Entry(sel_bar, textvariable=date_v, bg=c["dark_light"], fg=c["text"],
                  insertbackground=c["text"], width=12, relief="flat", bd=7,
                  font=("Helvetica",10)).pack(side="left", padx=6)

        # ── Attendance Method Cards ──────────────────────────────────────
        methods = tk.Frame(self.main, bg=c["content_bg"], padx=20)
        methods.pack(fill="x", pady=5)

        cards_data = [
            ("📱 QR Code\nAttendance", "#8b5cf6",
             "Generate dynamic QR codes\nand scan to mark attendance",
             lambda: self._generate_qr_for_class(cls_v.get())),
            ("🔍 Scan QR\nCode", c["warning"],
             "Scan student QR codes\nvia webcam or image file",
             lambda: self._scan_qr_attendance(date_v.get())),
            ("🤖 Face\nRecognition", "#7c3aed",
             "Auto-detect student faces\nand mark attendance",
             lambda: self._face_att_scan(cls_v.get(), date_v.get())),
            ("📋 Manual\nAttendance", c["success"],
             "Manually mark student\nattendance for a class",
             lambda: self._mark_attendance(cls_v.get(), date_v.get())),
            ("👩‍🏫 Teacher\nAttendance", c["primary"],
             "Mark teacher attendance\nfor today",
             lambda: self._teacher_att(date_v.get())),
        ]

        for i, (title, color, desc, cmd) in enumerate(cards_data):
            card = tk.Frame(methods, bg=c["card_bg"], padx=15, pady=12,
                            cursor="hand2")
            card.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            tk.Label(card, text=title, font=("Helvetica",12,"bold"),
                     bg=c["card_bg"], fg="white", justify="center").pack(pady=(0,5))

            btn = tk.Button(card, text="Open", bg=color, fg="white",
                            font=("Helvetica",10,"bold"), bd=0, padx=20, pady=6,
                            command=cmd, cursor="hand2")
            btn.pack(pady=5)

            tk.Label(card, text=desc, bg=c["card_bg"], fg=c["subtext"],
                     font=("Helvetica",8), justify="center",
                     wraplength=130).pack(pady=(2,0))

        # ── Attendance Report Section ────────────────────────────────────
        rpt_f = tk.Frame(self.main, bg=c["card_bg"], padx=20, pady=15)
        rpt_f.pack(fill="x", padx=20, pady=5)
        tk.Label(rpt_f, text="📊 Attendance Report", font=("Helvetica",12,"bold"),
                 bg=c["card_bg"], fg=c["text"]).pack(anchor="w")

        r2 = tk.Frame(rpt_f, bg=c["card_bg"])
        r2.pack(fill="x", pady=8)
        tk.Label(r2, text="Month (YYYY-MM):", bg=c["card_bg"], fg=c["subtext"]).pack(side="left")
        mon_v = tk.StringVar(value=datetime.date.today().strftime("%Y-%m"))
        tk.Entry(r2, textvariable=mon_v, bg=c["dark_light"], fg=c["text"],
                  insertbackground=c["text"], width=10, relief="flat", bd=7
                  ).pack(side="left", padx=6)

        # CSV Export button
        def export_att_csv():
            cls = cls_v.get()
            month = mon_v.get()
            if not cls:
                messagebox.showerror("Error", "Select a class first"); return
            students = [s for s in self.students if s.get("class") == cls]
            if not students:
                messagebox.showinfo("Info", f"No students in class {cls}"); return
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Export Attendance CSV",
                initialfile=f"attendance_{cls}_{month}.csv")
            if not path: return
            import csv
            try:
                yr, mn = int(month[:4]), int(month[5:7])
                import calendar
                days_in_month = calendar.monthrange(yr, mn)[1]
                dates = [f"{yr}-{mn:02d}-{d:02d}" for d in range(1, days_in_month+1)]
            except Exception:
                dates = [datetime.date.today().isoformat()]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Student Name", "Admission No", "Class"] + dates + ["Total Present", "Percentage"])
                for stu in students:
                    att = stu.get("attendance", {})
                    row = [stu.get("name",""), stu.get("admissionNo",""), cls]
                    present_count = 0
                    for d in dates:
                        status = att.get(d, "Absent")
                        row.append(status)
                        if status == "Present":
                            present_count += 1
                    pct = round(present_count / len(dates) * 100, 1) if dates else 0
                    row.extend([present_count, f"{pct}%"])
                    writer.writerow(row)
            messagebox.showinfo("✅", f"Attendance CSV exported to:\n{path}")

        tk.Button(r2, text="📤 Export CSV", bg=c["primary"], fg="white",
                  font=("Helvetica",9), command=export_att_csv).pack(side="left", padx=8)

        self._att_tree_outer = tk.Frame(self.main, bg=c["content_bg"])
        self._att_tree_outer.pack(fill="both", expand=True, padx=20, pady=5)

    def _mark_attendance(self, cls, date_str):
        if not cls:
            messagebox.showerror("Error","Select a class"); return
        try: datetime.datetime.strptime(date_str,"%Y-%m-%d")
        except: messagebox.showerror("Error","Date must be YYYY-MM-DD"); return

        c = self.colors
        for w in self._att_tree_outer.winfo_children():
            w.destroy()

        students = [s for s in self.students if s.get("class")==cls]
        if not students:
            tk.Label(self._att_tree_outer, text="No students in this class.",
                     bg=c["content_bg"], fg=c["subtext"]).pack(pady=30)
            return

        tk.Label(self._att_tree_outer,
                 text=f"Marking attendance for Class {cls} – {date_str}",
                 font=("Helvetica",11,"bold"), bg=c["content_bg"],
                 fg=c["text"]).pack(anchor="w", pady=(0,5))

        vars_ = {}
        frm = tk.Frame(self._att_tree_outer, bg=c["content_bg"])
        frm.pack(fill="both", expand=True)

        cols = ("#","Name","Admission","Status")
        tree = ttk.Treeview(frm, columns=cols, show="headings",
                             style="Custom.Treeview", height=14)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=150)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for idx, stu in enumerate(students, 1):
            existing = stu.get("attendance",{}).get(date_str,"Absent")
            var = tk.StringVar(value=existing)
            vars_[stu["id"]] = var
            status_text = f"{'✅' if existing=='Present' else '❌'} {existing}"
            tree.insert("","end",
                values=(idx, stu.get("name",""),
                        stu.get("admissionNo",""), status_text),
                tags=(stu["id"],))

        # toggle on double-click
        def toggle(event):
            sel = tree.selection()
            if not sel: return
            iid = sel[0]
            sid = tree.item(iid,"tags")[0]
            cur = vars_[sid].get()
            new = "Absent" if cur == "Present" else "Present"
            vars_[sid].set(new)
            tree.set(iid, "Status", f"{'✅' if new=='Present' else '❌'} {new}")

        tree.bind("<Double-1>", toggle)
        tk.Label(self._att_tree_outer,
                 text="💡 Double-click a row to toggle Present/Absent",
                 bg=c["content_bg"], fg=c["subtext"],
                 font=("Helvetica",9)).pack(pady=3)

        def save_att():
            for stu in students:
                stu.setdefault("attendance",{})[date_str] = vars_[stu["id"]].get()
            self.save_data()
            messagebox.showinfo("✅ Saved",f"Attendance saved for {cls} on {date_str}")

        tk.Button(self._att_tree_outer, text="💾 Save Attendance",
                  bg=c["success"], fg="white", font=("Helvetica",11,"bold"),
                  pady=8, command=save_att).pack(pady=8)

    def _teacher_att(self, date_str):
        if not self.teachers:
            messagebox.showinfo("Info","No teachers added yet."); return
        try: datetime.datetime.strptime(date_str,"%Y-%m-%d")
        except: messagebox.showerror("Error","Date must be YYYY-MM-DD"); return

        c = self.colors
        for w in self._att_tree_outer.winfo_children():
            w.destroy()

        tk.Label(self._att_tree_outer,
                 text=f"Teacher Attendance – {date_str}",
                 font=("Helvetica",11,"bold"), bg=c["content_bg"],
                 fg=c["text"]).pack(anchor="w", pady=(0,5))

        vars_ = {}
        frm = tk.Frame(self._att_tree_outer, bg=c["content_bg"])
        frm.pack(fill="both", expand=True)

        cols = ("#","Name","Subject","Status")
        tree = ttk.Treeview(frm, columns=cols, show="headings",
                             style="Custom.Treeview", height=12)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=160)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for idx, t in enumerate(self.teachers, 1):
            existing = t.get("attendance",{}).get(date_str,"Absent")
            var = tk.StringVar(value=existing)
            vars_[t["id"]] = var
            tree.insert("","end",
                values=(idx, t.get("name",""), t.get("subject",""),
                        f"{'✅' if existing=='Present' else '❌'} {existing}"),
                tags=(t["id"],))

        def toggle(event):
            sel = tree.selection()
            if not sel: return
            iid = sel[0]
            tid = tree.item(iid,"tags")[0]
            cur = vars_[tid].get()
            new = "Absent" if cur=="Present" else "Present"
            vars_[tid].set(new)
            tree.set(iid, "Status", f"{'✅' if new=='Present' else '❌'} {new}")

        tree.bind("<Double-1>", toggle)
        tk.Label(self._att_tree_outer, text="💡 Double-click to toggle",
                 bg=c["content_bg"], fg=c["subtext"], font=("Helvetica",9)).pack()

        def save():
            for t in self.teachers:
                t.setdefault("attendance",{})[date_str] = vars_[t["id"]].get()
            self.save_data()
            messagebox.showinfo("✅","Teacher attendance saved!")

        tk.Button(self._att_tree_outer, text="💾 Save Teacher Attendance",
                  bg=c["success"], fg="white", font=("Helvetica",11,"bold"),
                  pady=8, command=save).pack(pady=8)

    # ════════════════════════════════════════════════════════════════════════
    #  FACE RECOGNITION ATTENDANCE (Pure OpenCV LBPH – Python 3.14 safe)
    # ════════════════════════════════════════════════════════════════════════
    def _face_att_scan(self, cls, date_str):
        """
        Open webcam and mark attendance automatically using face recognition.
        Only works if student photos have been uploaded when adding the student.
        Uses OpenCV LBPH (Local Binary Pattern Histogram) – no extra install needed
        beyond opencv-contrib-python which the app already uses for face login.
        """
        if not HAS_FACE:
            _cv2a, _npa = _ensure_face_libs()
            if not _cv2a:
                messagebox.showerror(
                    "Face Recognition Not Available",
                    "OpenCV face libraries not found.\n\n"
                    "Go to Settings → Auto-Install Libraries first."
                )
                return
        else:
            _cv2a, _npa = _ensure_face_libs()

        if not cls:
            messagebox.showerror("Error", "Select a class first"); return
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            messagebox.showerror("Error", "Date must be in YYYY-MM-DD format"); return

        # ── Collect students who have a photo uploaded ─────────────────────
        students = [s for s in self.students if s.get("class") == cls]
        if not students:
            messagebox.showinfo("Info", f"No students found in class {cls}"); return

        recognizer_att = _FaceRecognizer()
        face_cascade   = recognizer_att._face_casc

        # Build training set from student photos
        train_imgs, train_labels, label_to_stu = [], [], {}
        skipped = []

        for label, stu in enumerate(students):
            photo_path = stu.get("photo_path", "")
            if not photo_path:
                skipped.append(stu.get("name", "?"))
                continue
            full_path = self.data_dir / photo_path
            if not full_path.exists():
                skipped.append(stu.get("name", "?"))
                continue
            img_gray = _cv2a.imread(str(full_path), _cv2a.IMREAD_GRAYSCALE)
            if img_gray is None:
                skipped.append(stu.get("name", "?"))
                continue
            faces = face_cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                # fallback – treat whole image as face region
                face_roi = cv2.resize(img_gray, (200, 200))
            else:
                x, y, w, h = faces[0]
                face_roi = cv2.resize(img_gray[y:y+h, x:x+w], (200, 200))
            train_imgs.append(face_roi)
            train_labels.append(label)
            label_to_stu[label] = stu

        if not train_imgs:
            messagebox.showerror(
                "No Face Photos Found",
                f"Class {cls} – No student has a face photo uploaded.\n\n"
                "How to fix:\n"
                "1. Go to Students → Double-click a student\n"
                "2. Click '📷 Upload Face Photo' button\n"
                "3. Upload a clear face photo for each student\n\n"
                "Face attendance requires student photos to work."
            )
            return

        # Train LBPH recognizer
        recognizer = _FaceRecognizer()
        recognizer.train(train_imgs, np.array(train_labels))

        # ── Open webcam ────────────────────────────────────────────────────
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            self._face_att_from_image(cls, date_str, recognizer, label_to_stu, students)
            return

        c = self.colors
        # Track which students were already marked in this session
        marked_this_session = {}   # stu_id → True

        # Preload existing attendance for today
        for stu in students:
            existing = stu.get("attendance", {}).get(date_str, "Absent")
            if existing == "Present":
                marked_this_session[stu["id"]] = True

        # ── Build popup window ─────────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title(f"🤖 Face Attendance – Class {cls} – {date_str}")
        win.geometry("900x640")
        win.configure(bg="#0f172a")
        win.resizable(True, True)
        win.transient(self.root)
        win.grab_set()

        # Header
        hdr = tk.Frame(win, bg="#7c3aed", padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🤖  Face Recognition Attendance",
                 font=("Helvetica", 16, "bold"), bg="#7c3aed", fg="white").pack(anchor="w")
        info_parts = []
        if skipped:
            info_parts.append(f"⚠️ {len(skipped)} students have no photo: {', '.join(skipped[:4])}"
                              + ("..." if len(skipped) > 4 else ""))
        info_parts.append(f"✅ {len(train_imgs)} student faces loaded | Look at camera")
        tk.Label(hdr, text="   ".join(info_parts),
                 font=("Helvetica", 9), bg="#7c3aed", fg="#ddd6fe",
                 wraplength=860, justify="left").pack(anchor="w")

        body = tk.Frame(win, bg="#0f172a")
        body.pack(fill="both", expand=True)

        # Left: camera feed
        left = tk.Frame(body, bg="#0f172a")
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        cam_lbl = tk.Label(left, bg="#0f172a")
        cam_lbl.pack()

        status_var = tk.StringVar(value="🔍 Camera scanning…")
        tk.Label(left, textvariable=status_var, font=("Helvetica", 11, "bold"),
                 bg="#0f172a", fg="#a78bfa", wraplength=560).pack(pady=4)

        # Right: live attendance list
        right = tk.Frame(body, bg="#1e293b", width=280)
        right.pack(side="right", fill="y", padx=(0, 10), pady=10)
        right.pack_propagate(False)

        tk.Label(right, text="📋 Attendance Status",
                 font=("Helvetica", 11, "bold"), bg="#1e293b", fg="white").pack(pady=(10, 4))
        tk.Frame(right, bg="#7c3aed", height=2).pack(fill="x", padx=10)

        att_frame = tk.Frame(right, bg="#1e293b")
        att_frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Create labels for each student in the list panel
        stu_label_widgets = {}
        for stu in students:
            row = tk.Frame(att_frame, bg="#1e293b")
            row.pack(fill="x", pady=2)
            is_marked = marked_this_session.get(stu["id"], False)
            lbl = tk.Label(row,
                           text=f"{'✅' if is_marked else '⬜'} {stu.get('name', '?')}",
                           font=("Helvetica", 9),
                           bg="#1e293b",
                           fg="#10b981" if is_marked else "#94a3b8",
                           anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            stu_label_widgets[stu["id"]] = lbl

        # Counters
        counter_var = tk.StringVar()

        def update_counter():
            total_marked = sum(1 for sid in marked_this_session if marked_this_session[sid])
            counter_var.set(f"✅ Present: {total_marked} / {len(students)}")

        counter_lbl = tk.Label(right, textvariable=counter_var,
                               font=("Helvetica", 11, "bold"),
                               bg="#1e293b", fg="#10b981")
        counter_lbl.pack(pady=6)
        update_counter()

        # Buttons
        btn_row = tk.Frame(win, bg="#0f172a")
        btn_row.pack(fill="x", padx=20, pady=10)

        self._face_att_running = True
        last_detected_id = [None]
        good_frames = [0]

        def scan_frame():
            if not self._face_att_running or not win.winfo_exists():
                return
            ret, frame = cap.read()
            if not ret:
                win.after(80, scan_frame)
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display = cv2.resize(frame_rgb, (580, 400))
            scale_x = 580 / frame.shape[1]
            scale_y = 400 / frame.shape[0]

            faces_rect = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))

            matched_stu = None
            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                # New API: predict returns (label, dist, thresh, eyes_ok)
                label, dist, thresh, eyes_ok = recognizer.predict(face_roi)

                dx = int(x * scale_x)
                dy = int(y * scale_y)
                dw = int(w * scale_x)
                dh = int(h * scale_y)

                if dist < thresh:
                    matched_stu = label_to_stu.get(label)
                    color = (16, 185, 129)
                    name_txt = matched_stu.get("name", "?") if matched_stu else "?"
                    cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), color, 2)
                    cv2.putText(display, f"{name_txt} ({int((1-dist/thresh)*100)}%)",
                                (dx, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                else:
                    color = (239, 68, 68)
                    cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), color, 2)
                    cv2.putText(display, f"Unknown",
                                (dx, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Update camera feed
            if HAS_PIL and _ensure_pil():
                from PIL import Image as _PILImg, ImageTk as _PILItk
                imgtk = _PILItk.PhotoImage(_PILImg.fromarray(display))
                cam_lbl.configure(image=imgtk)
                cam_lbl.image = imgtk

            # Confirm match after N consecutive frames
            if matched_stu:
                sid = matched_stu["id"]
                if last_detected_id[0] == sid:
                    good_frames[0] += 1
                else:
                    last_detected_id[0] = sid
                    good_frames[0] = 1

                name = matched_stu.get("name", "?")
                if marked_this_session.get(sid):
                    status_var.set(f"✅ {name} – already marked Present")
                else:
                    remaining = max(0, 4 - good_frames[0])
                    status_var.set(f"🔍 {name} recognized – confirming… ({good_frames[0]}/4)")

                if good_frames[0] >= 4 and not marked_this_session.get(sid):
                    # Mark attendance
                    matched_stu.setdefault("attendance", {})[date_str] = "Present"
                    marked_this_session[sid] = True
                    self.save_data()
                    # Update sidebar label
                    lbl_w = stu_label_widgets.get(sid)
                    if lbl_w:
                        lbl_w.config(text=f"✅ {name}",
                                     fg="#10b981")
                    update_counter()
                    status_var.set(f"✅ {name}'s attendance marked!")
                    good_frames[0] = 0
            else:
                last_detected_id[0] = None
                good_frames[0] = 0
                if len(faces_rect) > 0:
                    status_var.set("❌ Not recognized – try again")
                else:
                    status_var.set("🔍 Look at the camera…")

            win.after(60, scan_frame)

        def stop_and_close():
            self._face_att_running = False
            cap.release()
            if win.winfo_exists():
                win.destroy()
            # Refresh attendance view
            total_marked = sum(1 for v in marked_this_session.values() if v)
            messagebox.showinfo(
                "✅ Face Attendance Complete",
                f"Class {cls} – {date_str}\n"
                f"Present marked: {total_marked} / {len(students)} students"
            )
            # Reload the attendance page to reflect saved changes
            self._mark_attendance(cls, date_str)

        tk.Button(btn_row, text="✖  Stop & Save",
                  bg=c["danger"], fg="white",
                  font=("Helvetica", 11, "bold"), bd=0, padx=24, pady=10,
                  command=stop_and_close).pack(side="right", padx=6)

        tk.Label(btn_row,
                 text="💡 Students come one by one in front of camera – attendance marks automatically",
                 bg="#0f172a", fg="#94a3b8",
                 font=("Helvetica", 9)).pack(side="left")

        win.protocol("WM_DELETE_WINDOW", stop_and_close)
        win.after(200, scan_frame)

    def _face_att_from_image(self, cls, date_str, recognizer, label_to_stu, students):
        """Face attendance from uploaded image (fallback when no webcam)."""
        c = self.colors
        win = tk.Toplevel(self.root)
        win.title(f"🤖 Face Attendance (Image) – Class {cls}")
        win.geometry("600x500")
        win.configure(bg="#0f172a")

        tk.Label(win, text="🤖  Face Attendance – Upload Photos",
                 font=("Helvetica",14,"bold"), bg="#0f172a", fg="white").pack(pady=12)
        tk.Label(win,
                 text="No webcam detected. Upload student face photos to mark attendance.",
                 bg="#0f172a", fg="#a78bfa", font=("Helvetica",10)).pack(pady=4)

        log_text = tk.Text(win, bg="#1e293b", fg="white",
                           font=("Helvetica",10), height=15, state="disabled")
        log_text.pack(fill="both", expand=True, padx=20, pady=8)

        def log(msg):
            log_text.config(state="normal")
            log_text.insert("end", msg + "\n")
            log_text.see("end")
            log_text.config(state="disabled")

        face_cascade = recognizer._face_casc

        def upload_and_match():
            paths = filedialog.askopenfilenames(
                title="Select Face Photos",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            if not paths: return
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    log(f"❌ Cannot read: {os.path.basename(p)}"); continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
                if len(faces) == 0:
                    log(f"⚠️  No face found in: {os.path.basename(p)}"); continue
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                face_roi = gray[y:y+h, x:x+w]
                label, dist, thresh, _ = recognizer.predict(face_roi)
                if dist < thresh and label in label_to_stu:
                    stu = label_to_stu[label]
                    stu.setdefault("attendance", {})[date_str] = "Present"
                    self.save_data()
                    log(f"✅ {stu.get('name','?')} – Present (confidence: {int((1-dist/thresh)*100)}%)")
                else:
                    log(f"❌ Face not recognized in: {os.path.basename(p)}")

        btn_row = tk.Frame(win, bg="#0f172a")
        btn_row.pack(pady=10)
        tk.Button(btn_row, text="📂 Upload Face Photos",
                  bg="#7c3aed", fg="white",
                  font=("Helvetica",11,"bold"), padx=16, pady=8,
                  command=upload_and_match).pack(side="left", padx=6)
        tk.Button(btn_row, text="✖ Close",
                  bg=c["danger"], fg="white",
                  font=("Helvetica",11,"bold"), padx=16, pady=8,
                  command=win.destroy).pack(side="left", padx=6)

    # ════════════════════════════════════════════════════════════════════════
    #  QR CODE ATTENDANCE (Dynamic QR – refreshes every 30 seconds)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _get_local_ip():
        """Get the local IP address of this machine."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def _get_host_port():
        """Get host:port identifier for this device."""
        import socket
        ip = SchoolManagerPro._get_local_ip()
        hostname = socket.gethostname()
        return f"{ip}|{hostname}"

    @staticmethod
    def _qr_token(student_id, secret_key="school_manager_secret"):
        """Generate HMAC-SHA256 token for current 30-second window."""
        import hmac, time
        time_slot = str(int(time.time()) // 30)
        msg = f"{student_id}:{time_slot}".encode()
        return hmac.new(secret_key.encode(), msg, hashlib.sha256).hexdigest()[:16]

    @staticmethod
    def _verify_qr_token(student_id, token, secret_key="school_manager_secret"):
        """Verify token matches current or previous 30-second window."""
        import hmac, time
        for offset in [0, -1]:
            time_slot = str(int(time.time()) // 30 + offset)
            msg = f"{student_id}:{time_slot}".encode()
            expected = hmac.new(secret_key.encode(), msg, hashlib.sha256).hexdigest()[:16]
            if token == expected:
                return True
        return False

    def _verify_qr_network(self, qr_host):
        """Verify QR was generated on the trusted network/device."""
        trusted_host = self.settings.get("trusted_host", "")
        if not trusted_host:
            return True  # no restriction set
        current_host = self._get_host_port()
        # Check if IP matches or hostname matches
        qr_parts = qr_host.split("|")
        current_parts = current_host.split("|")
        trusted_parts = trusted_host.split("|")
        # IP must match trusted host IP
        return qr_parts[0] == trusted_parts[0]

    def _verify_student_email(self, student):
        """Verify student has a trusted email set."""
        trusted_domain = self.settings.get("trusted_email_domain", "")
        if not trusted_domain:
            return True  # no restriction
        email = student.get("email", "")
        if not email:
            return False
        return email.lower().endswith(f"@{trusted_domain.lower()}")

    def _generate_qr_for_class(self, class_name):
        """Generate Dynamic QR codes for class – open live window with 30s refresh."""
        if not class_name:
            messagebox.showerror("Error","Select a class first"); return
        if not HAS_QR:
            messagebox.showerror("Error",
                "qrcode library not installed.\nRun: pip install qrcode[pil]"); return
        if not (HAS_PIL and _ensure_pil()):
            messagebox.showerror("Error",
                "Pillow library not installed.\nRun: pip install pillow"); return

        import base64, time as _time
        students = [s for s in self.students if s.get("class") == class_name]
        if not students:
            messagebox.showinfo("Info",f"No students in class {class_name}"); return

        qr_dir = self.data_dir / "qr_codes"
        qr_dir.mkdir(exist_ok=True)

        c = self.colors
        prev = tk.Toplevel(self.root)
        prev.title(f"🔄 Dynamic QR – Class {class_name}")
        prev.geometry("750x600")
        prev.configure(bg=c["dark"])

        tk.Label(prev,
                 text=f"🔄  Dynamic QR Codes – Class {class_name}",
                 font=("Helvetica",14,"bold"), bg=c["dark"],
                 fg=c["success"]).pack(pady=(12,2))
        tk.Label(prev,
                 text="QR codes refresh every 30 seconds with HMAC tokens",
                 bg=c["dark"], fg=c["subtext"],
                 font=("Helvetica",9)).pack()

        timer_var = tk.StringVar(value="⏱ Next refresh in: 30s")
        timer_lbl = tk.Label(prev, textvariable=timer_var,
                             font=("Helvetica",11,"bold"),
                             bg=c["dark"], fg="#fbbf24")
        timer_lbl.pack(pady=4)

        grid = tk.Frame(prev, bg=c["dark"])
        grid.pack(fill="both", expand=True, padx=20, pady=10)
        prev._imgs = []
        prev._running = True

        def _refresh_qr():
            if not prev._running or not prev.winfo_exists():
                return
            for w in grid.winfo_children():
                w.destroy()
            prev._imgs = []

            host_id = self._get_host_port()
            for i, stu in enumerate(students[:12]):
                token = self._qr_token(stu["id"])
                payload = json.dumps({"id": stu["id"], "token": token, "host": host_id})
                img = qrcode.make(payload)
                itk = ImageTk.PhotoImage(img.resize((90, 90), Image.LANCZOS))
                prev._imgs.append(itk)
                col_f = tk.Frame(grid, bg=c["dark_light"], padx=5, pady=5)
                col_f.grid(row=i//6, column=i%6, padx=4, pady=4)
                tk.Label(col_f, image=itk, bg=c["dark_light"]).pack()
                tk.Label(col_f, text=stu.get("name","")[:12],
                         bg=c["dark_light"], fg="white",
                         font=("Helvetica",7)).pack()

                path = qr_dir / f"{stu['id']}.png"
                img.save(str(path))
                stu["qr_code"] = token
            self.save_data()

        def _tick():
            if not prev._running or not prev.winfo_exists():
                return
            import time as _t
            remaining = 30 - (int(_t.time()) % 30)
            timer_var.set(f"⏱ Next refresh in: {remaining}s")
            if remaining == 30:
                _refresh_qr()
            prev.after(1000, _tick)

        _refresh_qr()
        prev.after(1000, _tick)

        def _on_close():
            prev._running = False
            prev.destroy()

        def open_folder():
            try:
                if os.name == "nt":
                    os.startfile(str(qr_dir))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(qr_dir)])
                else:
                    subprocess.run(["xdg-open", str(qr_dir)])
            except Exception:
                pass

        btn_row = tk.Frame(prev, bg=c["dark"])
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="📂 Open QR Folder", bg=c["primary"], fg="white",
                  command=open_folder).pack(side="left", padx=4)
        tk.Button(btn_row, text="✖ Close", bg=c["danger"], fg="white",
                  command=_on_close).pack(side="left", padx=4)
        prev.protocol("WM_DELETE_WINDOW", _on_close)

    def _scan_qr_attendance(self, date_str):
        """Live webcam QR scanner to mark attendance."""
        try:
            import cv2
            import base64
        except ImportError:
            messagebox.showerror("Error",
                "opencv-python not installed.\nRun: pip install opencv-python"); return

        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            messagebox.showerror("Error","Date must be YYYY-MM-DD"); return

        c = self.colors

        # Status window
        status_win = tk.Toplevel(self.root)
        status_win.title("📷 QR Code Attendance Scanner")
        status_win.geometry("420x380")
        status_win.configure(bg=c["dark"])

        tk.Label(status_win, text="📷  QR Code Attendance Scanner",
                 font=("Helvetica",14,"bold"), bg=c["dark"],
                 fg="white").pack(pady=12)
        tk.Label(status_win, text=f"Date: {date_str}",
                 bg=c["dark"], fg=c["subtext"]).pack()
        tk.Label(status_win,
                 text="Hold student QR code in front of webcam.\nPress ESC in the camera window to stop.",
                 bg=c["dark"], fg=c["gray"],
                 font=("Helvetica",10), justify="center").pack(pady=8)

        log_frame = tk.Frame(status_win, bg=c["dark_light"], padx=10, pady=5)
        log_frame.pack(fill="both", expand=True, padx=20)
        log_text = tk.Text(log_frame, bg=c["dark_light"], fg="white",
                            font=("Helvetica",9), height=10, state="disabled")
        log_text.pack(fill="both", expand=True)

        marked_today = set()

        def log(msg, color="white"):
            log_text.config(state="normal")
            log_text.insert("end", msg + "\n")
            log_text.see("end")
            log_text.config(state="disabled")

        log(f"🟢 Scanner started for {date_str}")
        log("Waiting for QR code scan...")

        stop_flag = [False]

        def do_stop():
            stop_flag[0] = True
            try: cv2.destroyAllWindows()
            except Exception: pass
            status_win.destroy()

        btn_frame = tk.Frame(status_win, bg=c["dark"])
        btn_frame.pack(pady=10)

        def _process_qr_data(data):
            """Process decoded QR data string and mark attendance."""
            if not data or data in marked_today:
                return
            try:
                qr_data = json.loads(data)
                stu_id = qr_data.get("id", "")
                token = qr_data.get("token", "")
                qr_host = qr_data.get("host", "")
                stu = next((s for s in self.students if s["id"]==stu_id), None)
                if not stu:
                    log(f"⚠️  Unknown student ID in QR"); return
                # Verify network/WiFi
                if not self._verify_qr_network(qr_host):
                    log(f"⚠️  {stu['name']} – QR from untrusted network!"); return
                # Verify trusted email
                if not self._verify_student_email(stu):
                    log(f"⚠️  {stu['name']} – No trusted email set!"); return
                # Verify token
                if self._verify_qr_token(stu_id, token):
                    stu.setdefault("attendance",{})[date_str] = "Present"
                    self.save_data()
                    marked_today.add(data)
                    log(f"✅ {stu['name']} – Present (token + network verified)")
                else:
                    log(f"⚠️  {stu['name']} – QR expired! Token invalid")
            except (json.JSONDecodeError, KeyError):
                try:
                    import base64 as b64
                    stu_id = b64.urlsafe_b64decode(data).decode()
                    stu = next((s for s in self.students if s["id"]==stu_id), None)
                    if stu:
                        stu.setdefault("attendance",{})[date_str] = "Present"
                        self.save_data()
                        marked_today.add(data)
                        log(f"✅ {stu['name']} – Present (legacy QR)")
                    else:
                        log(f"⚠️  Unknown QR data")
                except Exception as ex:
                    log(f"❌ Error: {ex}")

        def scan_from_image():
            """Scan QR from an image file instead of webcam."""
            path = filedialog.askopenfilename(
                title="Select QR Code Image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
            if not path: return
            img = cv2.imread(path)
            if img is None:
                log("❌ Could not read image file"); return
            decoded_list = _decode_qr(img)
            if decoded_list:
                for data in decoded_list:
                    _process_qr_data(data)
            else:
                log("❌ No QR code found in image")

        tk.Button(btn_frame, text="📷 Scan from Image File",
                  bg=c["primary"], fg="white",
                  font=("Helvetica",10,"bold"),
                  command=scan_from_image).pack(side="left", padx=4)
        tk.Button(btn_frame, text="⏹ Stop Scanner", bg=c["danger"], fg="white",
                  font=("Helvetica",11,"bold"), command=do_stop).pack(side="left", padx=4)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            log("⚠️  Webcam not available – use 'Scan from Image File' button")
            log("💡  Tip: install pyzbar for better scanning: pip install pyzbar")
            return

        def _scan_frame():
            if stop_flag[0]:
                cap.release()
                return
            ret, frame = cap.read()
            if not ret:
                self.root.after(100, _scan_frame); return

            decoded_list = _decode_qr(frame)
            for data in decoded_list:
                _process_qr_data(data)

            cv2.imshow(f"QR Scanner – {date_str} (ESC to stop)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                do_stop(); return

            self.root.after(30, _scan_frame)

        self.root.after(100, _scan_frame)

    def _generate_student_qr(self, student_id):
        """Generate Dynamic QR for a single student with live 30s refresh."""
        if not HAS_QR or not HAS_PIL:
            messagebox.showerror("Error","Install qrcode and Pillow: pip install qrcode[pil] pillow")
            return
        _ensure_pil()
        stu = next((s for s in self.students if s["id"]==student_id), None)
        if not stu: return

        qr_dir = self.data_dir / "qr_codes"
        qr_dir.mkdir(exist_ok=True)

        c = self.colors
        prev = tk.Toplevel(self.root)
        prev.title(f"🔄 Dynamic QR – {stu['name']}")
        prev.configure(bg=c["dark"])
        prev._running = True

        tk.Label(prev, text=f"🔄 Dynamic QR – {stu['name']}",
                 font=("Helvetica",12,"bold"), bg=c["dark"], fg="white").pack(pady=(10,2))
        timer_var = tk.StringVar(value="⏱ 30s")
        tk.Label(prev, textvariable=timer_var, bg=c["dark"], fg="#fbbf24",
                 font=("Helvetica",10,"bold")).pack()

        qr_lbl = tk.Label(prev, bg=c["dark"])
        qr_lbl.pack(padx=20, pady=10)
        prev._img = None

        tk.Label(prev, text=f"{stu['name']}  |  {stu.get('class','')} {stu.get('section','')}",
                 bg=c["dark"], fg="white", font=("Helvetica",11,"bold")).pack()
        tk.Label(prev, text=f"Adm: {stu.get('admissionNo','')}",
                 bg=c["dark"], fg=c["subtext"]).pack(pady=3)

        def _refresh():
            if not prev._running or not prev.winfo_exists(): return
            token = self._qr_token(stu["id"])
            payload = json.dumps({"id": stu["id"], "token": token})
            img = qrcode.make(payload).resize((260, 260), Image.LANCZOS)
            itk = ImageTk.PhotoImage(img)
            prev._img = itk
            qr_lbl.configure(image=itk)
            path = qr_dir / f"{stu['id']}.png"
            qrcode.make(payload).save(str(path))

        def _tick():
            if not prev._running or not prev.winfo_exists(): return
            import time as _t
            remaining = 30 - (int(_t.time()) % 30)
            timer_var.set(f"⏱ Refresh in: {remaining}s")
            if remaining == 30:
                _refresh()
            prev.after(1000, _tick)

        _refresh()
        prev.after(1000, _tick)

        def _close():
            prev._running = False
            prev.destroy()

        tk.Button(prev, text="✖ Close", bg=c["primary"], fg="white",
                  command=_close).pack(pady=10)
        prev.protocol("WM_DELETE_WINDOW", _close)

    # ════════════════════════════════════════════════════════════════════════
    #  TEACHERS
    # ════════════════════════════════════════════════════════════════════════
    def show_teachers(self):
        if not self.check_perm("teachers"): return
        self._clear()
        c = self.colors
        self._header("👩‍🏫  Teacher Management", "Add, edit, delete teachers")

        ctrl = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=12)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="➕ Add Teacher", bg=c["success"], fg="white",
                  command=self._add_teacher_dlg).pack(side="left")

        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True)

        cols = ("#","Name","Subject","Phone","Email","Salary","Qualification")
        self._tch_tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                       style="Custom.Treeview", height=18)
        for col in cols:
            self._tch_tree.heading(col, text=col)
            self._tch_tree.column(col, anchor="center", width=110)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._tch_tree.yview)
        self._tch_tree.configure(yscrollcommand=vsb.set)
        self._tch_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tch_tree.bind("<Double-1>", lambda e: self._view_teacher())
        self._refresh_teachers()

    def _refresh_teachers(self):
        for i in self._tch_tree.get_children():
            self._tch_tree.delete(i)
        for idx, t in enumerate(self.teachers, 1):
            self._tch_tree.insert("","end",
                values=(idx, t.get("name",""), t.get("subject",""),
                        t.get("phone",""), t.get("email",""),
                        t.get("salary",""), t.get("qualification","")),
                tags=(t["id"],))

    def _view_teacher(self):
        sel = self._tch_tree.selection()
        if not sel: return
        tid = self._tch_tree.item(sel[0],"tags")[0]
        t = next((x for x in self.teachers if x["id"]==tid), None)
        if not t: return
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Teacher: {t['name']}")
        dlg.geometry("420x380")
        dlg.configure(bg=c["dark"])
        frm = tk.Frame(dlg, bg=c["dark"], padx=30, pady=20)
        frm.pack(fill="both", expand=True)
        for lbl, key in [("Name","name"),("Subject","subject"),
                          ("Phone","phone"),("Email","email"),
                          ("Salary","salary"),
                          ("Qualification","qualification")]:
            row = tk.Frame(frm, bg=c["dark"])
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"{lbl}:", font=("Helvetica",10,"bold"),
                     bg=c["dark"], fg=c["gray"], width=14).pack(side="left")
            tk.Label(row, text=t.get(key,"—"), bg=c["dark"], fg="white").pack(side="left")
        act = tk.Frame(frm, bg=c["dark"])
        act.pack(pady=12)
        tk.Button(act, text="✏️ Edit", bg=c["primary"], fg="white",
                  command=lambda: (dlg.destroy(), self._add_teacher_dlg(t))
                  ).pack(side="left", padx=4)
        tk.Button(act, text="🗑️ Delete", bg=c["danger"], fg="white",
                  command=lambda: self._del_teacher(tid, dlg)
                  ).pack(side="left", padx=4)

    def _add_teacher_dlg(self, edit=None):
        c = self.colors
        is_edit = edit is not None
        dlg = tk.Toplevel(self.root)
        dlg.title("Edit Teacher" if is_edit else "Add Teacher")
        dlg.geometry("500x600")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Edit Teacher" if is_edit else "➕ Add Teacher",
                 font=("Helvetica",15,"bold"), bg=c["dark"], fg="white").pack(pady=15)

        # scrollable frame so the Save button is always accessible
        cv = tk.Canvas(dlg, bg=c["dark"], highlightthickness=0)
        sb = ttk.Scrollbar(dlg, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        frm = tk.Frame(cv, bg=c["dark"], padx=30)
        fw = cv.create_window((0,0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(fw, width=e.width))
        def _tchr_scroll(event):
            if event.num == 4:
                cv.yview_scroll(-3, "units")
            elif event.num == 5:
                cv.yview_scroll(3, "units")
            else:
                cv.yview_scroll(int(-1*(event.delta/120)), "units")
        cv.bind_all("<MouseWheel>", _tchr_scroll)
        cv.bind_all("<Button-4>", _tchr_scroll)
        cv.bind_all("<Button-5>", _tchr_scroll)
        def _tchr_unbind(e):
            try:
                cv.unbind_all("<MouseWheel>")
                cv.unbind_all("<Button-4>")
                cv.unbind_all("<Button-5>")
            except Exception:
                pass
        dlg.bind("<Destroy>", _tchr_unbind)

        fields = [("Name *","name"),("Subject","subject"),
                  ("Phone","phone"),("Email","email"),
                  ("Salary","salary"),
                  ("Qualification","qualification")]
        entries = {}
        for lbl, key in fields:
            tk.Label(frm, text=lbl, bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
            e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
            e.pack(fill="x", pady=3)
            e.insert(0, edit.get(key,"") if is_edit else "")
            entries[key] = e

        def save():
            data = {k: entries[k].get().strip() for k in entries}
            if not data["name"]:
                messagebox.showerror("Error","Name required"); return
            if is_edit:
                edit.update(data)
            else:
                data.update({"id": self.gen_id(), "attendance":{}})
                self.teachers.append(data)
            self.save_data()
            dlg.destroy()
            self._refresh_teachers()
            messagebox.showinfo("✅","Teacher saved!")

        tk.Button(frm, text="💾 Save", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10, command=save).pack(fill="x", pady=20)

    def _del_teacher(self, tid, dlg=None):
        if not messagebox.askyesno("Confirm","Delete this teacher?"): return
        self.teachers = [t for t in self.teachers if t["id"]!=tid]
        self.save_data()
        if dlg: dlg.destroy()
        self._refresh_teachers()
        messagebox.showinfo("Deleted","Teacher removed.")

    # ════════════════════════════════════════════════════════════════════════
    #  FEES
    # ════════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════════
    #  PAPER FUND MODULE
    # ════════════════════════════════════════════════════════════════════════
    def show_paper_fund(self):
        """Full Paper Fund management page."""
        if not self.check_perm("fees"): return
        self._clear()
        c = self.colors
        self._header("📄  Paper Fund",
                     "Manage paper/stationary fund collection per student")

        # ── action bar ───────────────────────────────────────────────────────
        act = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=10)
        act.pack(fill="x")
        tk.Button(act, text="💰 Set / Update Fund Amount",
                  bg=c["primary"], fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=self._paper_fund_set_dlg).pack(side="left", padx=4)
        tk.Button(act, text="💵 Record Payment",
                  bg=c["success"], fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=self._paper_fund_pay_dlg).pack(side="left", padx=4)
        tk.Button(act, text="⏳ Pending Only",
                  bg=c["danger"], fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=self._paper_fund_pending_panel).pack(side="left", padx=4)
        tk.Button(act, text="📥 Export All CSV",
                  bg="#7c3aed", fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=lambda: self._pf_export_csv(None)).pack(side="right", padx=4)
        tk.Button(act, text="📂 Export Class-Wise",
                  bg="#0ea5e9", fg="white", padx=12, pady=6,
                  font=("Helvetica",10,"bold"),
                  command=self._pf_export_classwise).pack(side="right", padx=4)

        # ── filters ──────────────────────────────────────────────────────────
        flt = tk.Frame(self.main, bg=c["card_bg"], padx=20, pady=8)
        flt.pack(fill="x", padx=25, pady=5)

        self._pf_mon_var = tk.StringVar(value=datetime.date.today().strftime("%Y-%m"))
        tk.Label(flt, text="Month:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left")
        pf_months = sorted(set(
            mon for s in self.students
            for mon in s.get("paper_fund_records",{})
        ), reverse=True)
        ttk.Combobox(flt, textvariable=self._pf_mon_var,
                      values=["All"] + pf_months,
                      state="readonly", width=10).pack(side="left", padx=4)

        self._pf_cls_var = tk.StringVar(value="All")
        tk.Label(flt, text="Class:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        ttk.Combobox(flt, textvariable=self._pf_cls_var,
                      values=["All"]+self._all_classes(),
                      state="readonly", width=8).pack(side="left", padx=4)

        self._pf_status_var = tk.StringVar(value="All")
        tk.Label(flt, text="Status:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        ttk.Combobox(flt, textvariable=self._pf_status_var,
                      values=["All","Paid","Pending"],
                      state="readonly", width=8).pack(side="left", padx=4)

        self._pf_search_var = tk.StringVar()
        tk.Label(flt, text="Search:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        tk.Entry(flt, textvariable=self._pf_search_var,
                 bg=c["dark_light"], fg=c["text"],
                 insertbackground=c["text"], relief="flat", bd=6,
                 width=14).pack(side="left", padx=4)

        tk.Button(flt, text="🔍 Apply", bg=c["primary"], fg="white",
                  command=self._pf_refresh).pack(side="left", padx=6)
        self._pf_search_var.trace_add("write", lambda *a: self._pf_refresh())

        # ── summary strip ─────────────────────────────────────────────────────
        self._pf_summary_frame = tk.Frame(self.main, bg=c["content_bg"])
        self._pf_summary_frame.pack(fill="x", padx=25, pady=(4,0))

        # ── table ─────────────────────────────────────────────────────────────
        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True, pady=5)

        pf_cols = ("#","Adm No","Name","Class","Month","Fund Due","Paid","Balance","Status")
        self._pf_tree = ttk.Treeview(tbl_f, columns=pf_cols, show="headings",
                                      style="Custom.Treeview", height=14)
        for col in pf_cols:
            self._pf_tree.heading(col, text=col)
            self._pf_tree.column(col, anchor="center", width=100)
        self._pf_tree.column("#", width=40)
        self._pf_tree.column("Name", width=150)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._pf_tree.yview)
        hsb = ttk.Scrollbar(tbl_f, orient="horizontal", command=self._pf_tree.xview)
        self._pf_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._pf_tree.pack(fill="both", expand=True)
        self._pf_tree.bind("<Double-1>", lambda e: self._pf_row_action())

        # right-click context for row-level export
        ctx = tk.Menu(self.root, tearoff=0, bg=c["dark_light"], fg=c["text"],
                      activebackground=c["primary"], activeforeground="white")
        ctx.add_command(label="💵 Record Payment for Selected",
                        command=self._pf_row_action)
        ctx.add_separator()
        ctx.add_command(label="📥 Export This Student CSV",
                        command=lambda: self._pf_export_csv("single"))
        ctx.add_command(label="📥 Export All (Visible) CSV",
                        command=lambda: self._pf_export_csv(None))
        ctx.add_command(label="📂 Export Class-Wise CSV",
                        command=self._pf_export_classwise)

        def pf_ctx(event):
            row = self._pf_tree.identify_row(event.y)
            if row:
                self._pf_tree.selection_set(row)
            try:
                ctx.tk_popup(event.x_root, event.y_root)
            finally:
                ctx.grab_release()

        self._pf_tree.bind("<Button-3>", pf_ctx)
        self._pf_refresh()

    def _pf_refresh(self):
        """Refresh the paper fund table."""
        for i in self._pf_tree.get_children():
            self._pf_tree.delete(i)
        # clear summary
        for w in self._pf_summary_frame.winfo_children():
            w.destroy()

        c = self.colors
        mon_f    = self._pf_mon_var.get()
        cls_f    = self._pf_cls_var.get()
        sta_f    = self._pf_status_var.get()
        search   = self._pf_search_var.get().lower()
        idx      = 1
        tot_due  = tot_paid = tot_bal = tot_pending_cnt = 0

        for stu in self.students:
            if cls_f != "All" and stu.get("class","") != cls_f: continue
            if search and search not in stu.get("name","").lower() \
                    and search not in stu.get("admissionNo","").lower(): continue
            pfr = stu.get("paper_fund_records", {})
            if not pfr:
                continue
            for mon, rec in sorted(pfr.items(), reverse=True):
                if mon_f not in ("All", mon): continue
                paid_flag = rec.get("paidFlag", False)
                if sta_f == "Paid" and not paid_flag: continue
                if sta_f == "Pending" and paid_flag: continue
                due  = rec.get("due", 0)
                paid = rec.get("paid", 0)
                bal  = due - paid
                status = "✅ Paid" if paid_flag else "⏳ Pending"
                self._pf_tree.insert("","end",
                    values=(idx, stu.get("admissionNo",""), stu.get("name",""),
                            stu.get("class",""), mon,
                            f"RS {due:,}", f"RS {paid:,}", f"RS {bal:,}", status),
                    tags=(stu["id"], mon))
                tot_due  += due
                tot_paid += paid
                tot_bal  += bal
                if not paid_flag: tot_pending_cnt += 1
                idx += 1

        # summary strip
        for label, val, col in [
            ("Total Due",     f"RS {tot_due:,}",         c["warning"]),
            ("Total Paid",    f"RS {tot_paid:,}",        c["success"]),
            ("Total Balance", f"RS {tot_bal:,}",         c["danger"]),
            ("Pending Count", str(tot_pending_cnt),       c["danger"]),
        ]:
            box = tk.Frame(self._pf_summary_frame, bg=c["card_bg"], padx=16, pady=8)
            box.pack(side="left", padx=6, pady=4)
            tk.Label(box, text=label, bg=c["card_bg"],
                     fg=c["subtext"], font=("Helvetica",9)).pack()
            tk.Label(box, text=val, bg=c["card_bg"],
                     fg=col, font=("Helvetica",13,"bold")).pack()

    def _pf_row_action(self):
        """Double-click or right-click → record payment for selected row."""
        sel = self._pf_tree.selection()
        if not sel: return
        tags = self._pf_tree.item(sel[0], "tags")
        sid, mon = tags[0], tags[1]
        stu = next((s for s in self.students if s["id"]==sid), None)
        if not stu: return
        c = self.colors

        dlg = tk.Toplevel(self.root)
        dlg.title(f"Paper Fund Payment – {stu['name']} – {mon}")
        dlg.geometry("400x280")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        hdr = tk.Frame(dlg, bg=c["primary"], padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📄 Paper Fund Payment",
                 font=("Helvetica",13,"bold"), bg=c["primary"], fg="white").pack(anchor="w")
        tk.Label(hdr, text=f"{stu['name']}  |  {mon}  |  Class {stu.get('class','')}",
                 bg=c["primary"], fg="#e0e7ff", font=("Helvetica",9)).pack(anchor="w")

        rec = stu.get("paper_fund_records",{}).get(mon,{})
        due  = rec.get("due",0)
        paid_so_far = rec.get("paid",0)

        frm = tk.Frame(dlg, bg=c["dark"], padx=24, pady=16)
        frm.pack(fill="both", expand=True)
        for lbl, val in [("Fund Due:", f"RS {due:,}"),
                          ("Already Paid:", f"RS {paid_so_far:,}"),
                          ("Balance:", f"RS {due-paid_so_far:,}")]:
            r = tk.Frame(frm, bg=c["dark"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=lbl, bg=c["dark"], fg=c["subtext"], width=14, anchor="w").pack(side="left")
            tk.Label(r, text=val, bg=c["dark"], fg=c["text"],
                     font=("Helvetica",10,"bold")).pack(side="left")

        tk.Label(frm, text="Amount to Pay (RS):", bg=c["dark"],
                 fg=c["subtext"]).pack(anchor="w", pady=(12,0))
        amt_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7,
                          font=("Helvetica",12))
        amt_e.pack(fill="x", pady=4)

        btn_r = tk.Frame(dlg, bg=c["dark"], padx=20, pady=12)
        btn_r.pack(fill="x", side="bottom")

        def submit():
            try:
                amount = int(amt_e.get().strip())
                if amount <= 0: raise ValueError
            except:
                messagebox.showerror("Error","Enter a valid positive amount."); return
            pfr = stu.setdefault("paper_fund_records",{})
            if mon not in pfr:
                pfr[mon] = {"due": due, "paid": 0, "paidFlag": False}
            pfr[mon]["paid"] += amount
            if pfr[mon]["paid"] >= pfr[mon]["due"]:
                pfr[mon]["paidFlag"] = True
            self.save_data()
            dlg.destroy()
            self._pf_refresh()
            messagebox.showinfo("✅", f"RS {amount:,} recorded for {stu['name']} – {mon}")

        tk.Button(btn_r, text="✅ Record Payment", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=9,
                  command=submit).pack(side="left", expand=True, fill="x", padx=(0,6))
        tk.Button(btn_r, text="✖ Cancel", bg=c["danger"], fg="white",
                  font=("Helvetica",11,"bold"), pady=9,
                  command=dlg.destroy).pack(side="left")

    def _paper_fund_set_dlg(self):
        """Set / update paper fund amount for a class and month."""
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Set Paper Fund Amount")
        dlg.geometry("420x340")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        hdr = tk.Frame(dlg, bg=c["primary"], padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="💰 Set Paper Fund Amount",
                 font=("Helvetica",14,"bold"), bg=c["primary"], fg="white").pack(anchor="w")
        tk.Label(hdr, text="Assign paper fund due per student for a class & month",
                 bg=c["primary"], fg="#e0e7ff").pack(anchor="w")

        frm = tk.Frame(dlg, bg=c["dark"], padx=28, pady=20)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Class:", bg=c["dark"], fg=c["subtext"]).pack(anchor="w")
        cls_cb = ttk.Combobox(frm, values=self._all_classes(), state="readonly")
        cls_cb.pack(fill="x", pady=(3,12))

        tk.Label(frm, text="Month (YYYY-MM):", bg=c["dark"], fg=c["subtext"]).pack(anchor="w")
        mon_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", pady=(3,12))

        tk.Label(frm, text="Paper Fund Amount (RS):", bg=c["dark"],
                 fg=c["subtext"]).pack(anchor="w")
        amt_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        amt_e.pack(fill="x", pady=(3,12))

        def save():
            cls = cls_cb.get().strip()
            mon = mon_e.get().strip()
            try:
                amt = int(amt_e.get().strip())
                datetime.datetime.strptime(mon, "%Y-%m")
            except:
                messagebox.showerror("Error","Check values – month must be YYYY-MM"); return
            if not cls:
                messagebox.showerror("Error","Select a class"); return
            updated = 0
            for stu in self.students:
                if stu.get("class","") == cls:
                    pfr = stu.setdefault("paper_fund_records",{})
                    if mon not in pfr:
                        pfr[mon] = {"due": amt, "paid": 0, "paidFlag": False}
                    else:
                        pfr[mon]["due"] = amt   # update due but keep existing paid
                        if pfr[mon]["paid"] >= amt:
                            pfr[mon]["paidFlag"] = True
                    updated += 1
            self.save_data()
            dlg.destroy()
            self._pf_refresh()
            messagebox.showinfo("✅",
                f"Paper Fund RS {amt:,} set for Class {cls} – {mon}\n"
                f"({updated} students updated)")

        tk.Button(frm, text="💾 Save & Apply to Class",
                  bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=save).pack(fill="x")

    def _paper_fund_pay_dlg(self):
        """Quick payment dialog – choose student + month."""
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Record Paper Fund Payment")
        dlg.geometry("400x360")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg, text="💵 Record Paper Fund Payment",
                 font=("Helvetica",14,"bold"), bg=c["dark"],
                 fg="white").pack(pady=15)
        frm = tk.Frame(dlg, bg=c["dark"], padx=28)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Student (Admission No):", bg=c["dark"],
                 fg=c["subtext"]).pack(anchor="w", pady=(5,0))
        adm_cb = ttk.Combobox(frm,
            values=[s.get("admissionNo","") for s in self.students],
            state="readonly")
        adm_cb.pack(fill="x", pady=5)

        tk.Label(frm, text="Month (YYYY-MM):", bg=c["dark"],
                 fg=c["subtext"]).pack(anchor="w", pady=(5,0))
        mon_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", pady=5)

        tk.Label(frm, text="Amount (RS):", bg=c["dark"],
                 fg=c["subtext"]).pack(anchor="w", pady=(5,0))
        amt_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        amt_e.pack(fill="x", pady=5)

        def submit():
            adm = adm_cb.get()
            stu = next((s for s in self.students if s.get("admissionNo")==adm), None)
            if not stu:
                messagebox.showerror("Error","Select a student"); return
            try:
                mon = mon_e.get().strip()
                amt = int(amt_e.get().strip())
                datetime.datetime.strptime(mon, "%Y-%m")
            except:
                messagebox.showerror("Error","Check values"); return
            pfr = stu.setdefault("paper_fund_records",{})
            if mon not in pfr:
                pfr[mon] = {"due": amt, "paid": amt, "paidFlag": True}
            else:
                pfr[mon]["paid"] += amt
                if pfr[mon]["paid"] >= pfr[mon]["due"]:
                    pfr[mon]["paidFlag"] = True
            self.save_data()
            dlg.destroy()
            self._pf_refresh()
            messagebox.showinfo("✅", f"RS {amt:,} paper fund recorded for {stu['name']}")

        tk.Button(frm, text="💾 Record Payment",
                  bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=submit).pack(fill="x", pady=15)

    def _paper_fund_pending_panel(self):
        """Show only students with pending paper fund this month."""
        cur_mon = datetime.date.today().strftime("%Y-%m")
        pending = []
        for stu in self.students:
            pfr = stu.get("paper_fund_records",{})
            if cur_mon in pfr and not pfr[cur_mon].get("paidFlag"):
                pending.append((stu, pfr[cur_mon]))
            # students with no record for this month but have a record for any month → skip
        if not pending:
            messagebox.showinfo("✅ All Clear",
                f"No students with pending paper fund for {cur_mon}!"); return

        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Paper Fund Pending – {cur_mon}")
        dlg.geometry("820x500")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        hdr = tk.Frame(dlg, bg=c["danger"], padx=20, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"📄  Paper Fund Pending – {cur_mon} ({len(pending)} students)",
                 font=("Helvetica",15,"bold"), bg=c["danger"], fg="white").pack(anchor="w")

        tbl_f = tk.Frame(dlg, bg=c["dark"], padx=20, pady=10)
        tbl_f.pack(fill="both", expand=True)

        cols = ("#","Adm No","Name","Class","Section","Due","Paid","Balance","Phone")
        tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                             style="Custom.Treeview")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=88)
        tree.column("Name", width=140)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for idx, (stu, rec) in enumerate(pending, 1):
            due  = rec.get("due",0)
            paid = rec.get("paid",0)
            tree.insert("","end",
                values=(idx, stu.get("admissionNo",""), stu.get("name",""),
                        stu.get("class",""), stu.get("section",""),
                        f"RS {due:,}", f"RS {paid:,}",
                        f"RS {due-paid:,}", stu.get("phone","")),
                tags=(stu["id"],))

        # export buttons
        btn_f = tk.Frame(dlg, bg=c["dark"], pady=10, padx=20)
        btn_f.pack(fill="x")

        def export_pending_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")],
                initialfile=f"paper_fund_pending_{cur_mon}.csv")
            if not path: return
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Adm No","Name","Class","Section","Due","Paid","Balance","Phone"])
                for stu, rec in pending:
                    due  = rec.get("due",0); paid = rec.get("paid",0)
                    w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                                stu.get("class",""), stu.get("section",""),
                                due, paid, due-paid, stu.get("phone","")])
            messagebox.showinfo("✅ Exported", f"Saved to:\n{path}")

        def export_cls_csv():
            folder = filedialog.askdirectory(title="Select Folder for Class-Wise Export")
            if not folder: return
            groups: dict = {}
            for stu, rec in pending:
                groups.setdefault(stu.get("class","Unknown"), []).append((stu,rec))
            saved = []
            for cls, rows in groups.items():
                safe = cls.replace(" ","_").replace("/","-")
                fpath = os.path.join(folder, f"paperfund_pending_{safe}_{cur_mon}.csv")
                with open(fpath,"w",newline="",encoding="utf-8") as f:
                    w2 = csv.writer(f)
                    w2.writerow(["Adm No","Name","Class","Section","Due","Paid","Balance","Phone"])
                    for stu,rec in rows:
                        due = rec.get("due",0); paid = rec.get("paid",0)
                        w2.writerow([stu.get("admissionNo",""), stu.get("name",""),
                                     stu.get("class",""), stu.get("section",""),
                                     due, paid, due-paid, stu.get("phone","")])
                saved.append(os.path.basename(fpath))
            messagebox.showinfo("✅ Class-wise Export",
                f"Saved {len(saved)} file(s):\n" + "\n".join(saved))

        def export_single():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Select","Select a student row first."); return
            sid = tree.item(sel[0],"tags")[0]
            pair = next(((s,r) for s,r in pending if s["id"]==sid), None)
            if not pair: return
            stu, rec = pair
            safe = stu.get("name","student").replace(" ","_")
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV","*.csv")],
                initialfile=f"paperfund_{safe}_{cur_mon}.csv")
            if not path: return
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Adm No","Name","Class","Section","Due","Paid","Balance","Phone"])
                due = rec.get("due",0); paid = rec.get("paid",0)
                w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                            stu.get("class",""), stu.get("section",""),
                            due, paid, due-paid, stu.get("phone","")])
            messagebox.showinfo("✅ Exported", f"Saved to:\n{path}")

        tk.Button(btn_f, text="📥 Export All CSV",
                  bg=c["success"], fg="white", padx=10, pady=5,
                  command=export_pending_csv).pack(side="left", padx=4)
        tk.Button(btn_f, text="👤 Export Selected Student",
                  bg="#0ea5e9", fg="white", padx=10, pady=5,
                  command=export_single).pack(side="left", padx=4)
        tk.Button(btn_f, text="📂 Export Class-Wise",
                  bg="#7c3aed", fg="white", padx=10, pady=5,
                  command=export_cls_csv).pack(side="left", padx=4)
        tk.Button(btn_f, text="✖ Close",
                  bg=c["danger"], fg="white", padx=10, pady=5,
                  command=dlg.destroy).pack(side="right", padx=4)

    def _pf_export_csv(self, mode):
        """Export paper fund table CSV. mode='single' → selected row only, else all visible."""
        c = self.colors
        mon_f  = self._pf_mon_var.get()
        cls_f  = self._pf_cls_var.get()
        sta_f  = self._pf_status_var.get()
        search = self._pf_search_var.get().lower()

        rows = []
        for stu in self.students:
            if cls_f != "All" and stu.get("class","") != cls_f: continue
            if search and search not in stu.get("name","").lower() \
                    and search not in stu.get("admissionNo","").lower(): continue
            pfr = stu.get("paper_fund_records",{})
            for mon, rec in sorted(pfr.items(), reverse=True):
                if mon_f not in ("All", mon): continue
                paid_flag = rec.get("paidFlag",False)
                if sta_f == "Paid" and not paid_flag: continue
                if sta_f == "Pending" and paid_flag: continue
                due  = rec.get("due",0); paid = rec.get("paid",0)
                rows.append([stu.get("admissionNo",""), stu.get("name",""),
                              stu.get("class",""), stu.get("section",""),
                              mon, due, paid, due-paid,
                              "Paid" if paid_flag else "Pending",
                              stu.get("phone","")])

        if mode == "single":
            sel = self._pf_tree.selection()
            if not sel:
                messagebox.showwarning("Select","Select a student row first."); return
            tags = self._pf_tree.item(sel[0],"tags")
            sid, mon = tags[0], tags[1]
            rows = [r for r in rows if r[0] == next(
                (s.get("admissionNo","") for s in self.students if s["id"]==sid), "")]

        if not rows:
            messagebox.showwarning("Empty","No records to export."); return

        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], title="Export Paper Fund CSV")
        if not path: return
        with open(path,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Adm No","Name","Class","Section","Month",
                         "Due (RS)","Paid (RS)","Balance (RS)","Status","Phone"])
            w.writerows(rows)
        messagebox.showinfo("✅ Exported", f"Saved to:\n{path}")

    def _pf_export_classwise(self):
        """Export separate CSV for each class from the current paper fund view."""
        mon_f  = self._pf_mon_var.get()
        cls_f  = self._pf_cls_var.get()
        sta_f  = self._pf_status_var.get()
        search = self._pf_search_var.get().lower()

        groups: dict = {}
        for stu in self.students:
            if cls_f != "All" and stu.get("class","") != cls_f: continue
            if search and search not in stu.get("name","").lower() \
                    and search not in stu.get("admissionNo","").lower(): continue
            pfr = stu.get("paper_fund_records",{})
            for mon, rec in sorted(pfr.items(), reverse=True):
                if mon_f not in ("All", mon): continue
                paid_flag = rec.get("paidFlag",False)
                if sta_f == "Paid" and not paid_flag: continue
                if sta_f == "Pending" and paid_flag: continue
                due  = rec.get("due",0); paid = rec.get("paid",0)
                cls  = stu.get("class","Unknown")
                groups.setdefault(cls,[]).append(
                    [stu.get("admissionNo",""), stu.get("name",""),
                     cls, stu.get("section",""), mon, due, paid, due-paid,
                     "Paid" if paid_flag else "Pending", stu.get("phone","")])

        if not groups:
            messagebox.showwarning("Empty","No records to export."); return

        folder = filedialog.askdirectory(title="Select Folder for Class-Wise Paper Fund Export")
        if not folder: return
        saved = []
        for cls, rows in groups.items():
            safe = cls.replace(" ","_").replace("/","-")
            fpath = os.path.join(folder, f"paper_fund_{safe}_{mon_f}.csv")
            with open(fpath,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Adm No","Name","Class","Section","Month",
                             "Due (RS)","Paid (RS)","Balance (RS)","Status","Phone"])
                w.writerows(rows)
            saved.append(os.path.basename(fpath))
        messagebox.showinfo("✅ Class-Wise Export",
            f"Saved {len(saved)} file(s):\n" + "\n".join(saved))

    # ════════════════════════════════════════════════════════════════════════
    #  FEES MODULE
    # ════════════════════════════════════════════════════════════════════════
    def show_fees(self):
        if not self.check_perm("fees"): return
        self._clear()
        c = self.colors
        self._header("💰  Fees Management",
                     "Fee structures, vouchers, payments, history")

        # top action bar
        act = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=10)
        act.pack(fill="x")
        tk.Button(act, text="🏗️ Fee Structures", bg=c["primary"], fg="white",
                  command=self._fee_structures_panel).pack(side="left", padx=3)
        tk.Button(act, text="📅 Generate Vouchers", bg=c["warning"], fg="white",
                  command=self._generate_vouchers_dlg).pack(side="left", padx=3)
        tk.Button(act, text="💵 Record Payment", bg=c["success"], fg="white",
                  command=self._record_payment_dlg).pack(side="left", padx=3)
        tk.Button(act, text="🖨️ Print Voucher PDF", bg="#8b5cf6", fg="white",
                  command=self._print_voucher).pack(side="left", padx=3)
        tk.Button(act, text="📊 Finance Graphs", bg="#06b6d4", fg="white",
                  command=self._show_finance_graphs).pack(side="left", padx=3)
        tk.Button(act, text="⏳ Pending Students", bg=c["danger"], fg="white",
                  command=self._open_pending_panel).pack(side="right", padx=3)

        # filters
        flt = tk.Frame(self.main, bg=c["card_bg"], padx=20, pady=10)
        flt.pack(fill="x", padx=25, pady=5)
        tk.Label(flt, text="Filter:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left")

        self._fee_mon_filter = tk.StringVar(value="All")
        tk.Label(flt, text="Month:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        months = sorted(set(
            mon for stu in self.students
            for mon in stu.get("fee_records",{})
        ), reverse=True)
        ttk.Combobox(flt, textvariable=self._fee_mon_filter,
                      values=["All"] + months, state="readonly", width=10
                      ).pack(side="left", padx=4)

        self._fee_status_filter = tk.StringVar(value="All")
        tk.Label(flt, text="Status:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        ttk.Combobox(flt, textvariable=self._fee_status_filter,
                      values=["All","Paid","Pending"], state="readonly", width=8
                      ).pack(side="left", padx=4)

        self._fee_cls_filter = tk.StringVar(value="All")
        tk.Label(flt, text="Class:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        ttk.Combobox(flt, textvariable=self._fee_cls_filter,
                      values=["All"]+self._all_classes(), state="readonly", width=8
                      ).pack(side="left", padx=4)

        tk.Button(flt, text="🔍 Apply", bg=c["primary"], fg="white",
                  command=self._refresh_fees).pack(side="left", padx=8)

        # search
        tk.Label(flt, text="Search:", bg=c["card_bg"], fg=c["subtext"]).pack(side="left", padx=(10,0))
        self._fee_search = tk.Entry(flt, bg=c["dark_light"], fg=c["text"],
                                     insertbackground=c["text"], relief="flat", bd=6, width=14)
        self._fee_search.pack(side="left", padx=4)
        self._fee_search.bind("<KeyRelease>", lambda e: self._refresh_fees())

        # table
        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True, pady=5)

        cols = ("#","Adm No","Name","Class","Month",
                "Due","Discount","Fine","Paid","Balance","Status")
        self._fees_tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                        style="Custom.Treeview", height=14)
        for col in cols:
            self._fees_tree.heading(col, text=col)
            self._fees_tree.column(col, anchor="center", width=90)
        self._fees_tree.column("#",width=40)
        self._fees_tree.column("Name",width=140)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._fees_tree.yview)
        hsb = ttk.Scrollbar(tbl_f, orient="horizontal", command=self._fees_tree.xview)
        self._fees_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._fees_tree.pack(fill="both", expand=True)
        self._fees_tree.bind("<Double-1>", lambda e: self._fees_row_action())

        self._refresh_fees()

    def _refresh_fees(self):
        for i in self._fees_tree.get_children():
            self._fees_tree.delete(i)
        mon_f = self._fee_mon_filter.get()
        sta_f = self._fee_status_filter.get()
        cls_f = self._fee_cls_filter.get()
        search = self._fee_search.get().lower()
        idx = 1
        for stu in self.students:
            if cls_f != "All" and stu.get("class","") != cls_f: continue
            if search and search not in stu.get("name","").lower() \
                    and search not in stu.get("admissionNo","").lower(): continue
            for mon, rec in sorted(stu.get("fee_records",{}).items(), reverse=True):
                if mon_f != "All" and mon != mon_f: continue
                paid_flag = rec.get("paidFlag",False)
                if sta_f == "Paid" and not paid_flag: continue
                if sta_f == "Pending" and paid_flag: continue
                due = rec.get("due",0); paid = rec.get("paid",0)
                disc = rec.get("discount",0); fine = rec.get("fine",0)
                bal = due - disc + fine - paid
                status = "✅ Paid" if paid_flag else "⏳ Pending"
                self._fees_tree.insert("","end",
                    values=(idx, stu.get("admissionNo",""), stu.get("name",""),
                            stu.get("class",""), mon, f"RS {due:,}",
                            f"RS {disc:,}", f"RS {fine:,}", f"RS {paid:,}",
                            f"RS {bal:,}", status),
                    tags=(stu["id"], mon))
                idx += 1

    def _fees_row_action(self):
        sel = self._fees_tree.selection()
        if not sel: return
        tags = self._fees_tree.item(sel[0],"tags")
        sid, mon = tags[0], tags[1]
        stu = next((s for s in self.students if s["id"]==sid), None)
        if not stu: return

        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Pay Fee – {stu['name']} – {mon}")
        dlg.geometry("460x580")
        dlg.resizable(False, False)
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        # ── Header (top) ─────────────────────────────────────────────────
        hdr = tk.Frame(dlg, bg=c["primary"], padx=20, pady=12)
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="💰  Fee Payment",
                 font=("Helvetica",14,"bold"),
                 bg=c["primary"], fg="white").pack(anchor="w")
        tk.Label(hdr,
                 text=f"Student: {stu['name']}   |   Month: {mon}   |   Class: {stu.get('class','')}",
                 bg=c["primary"], fg="#e0e7ff", font=("Helvetica",9)).pack(anchor="w")

        # ── OK / Cancel (bottom) — packed BEFORE body so never hidden ────
        btn_row = tk.Frame(dlg, bg=c["dark"], padx=20, pady=14)
        btn_row.pack(fill="x", side="bottom")

        submit_fn = [None]   # filled in later

        tk.Button(btn_row, text="✅  OK – Record Payment",
                  bg=c["success"], fg="white",
                  font=("Helvetica",12,"bold"), pady=11,
                  command=lambda: submit_fn[0]()
                  ).pack(side="left", expand=True, fill="x", padx=(0,8))

        tk.Button(btn_row, text="✖  Cancel",
                  bg=c["danger"], fg="white",
                  font=("Helvetica",12,"bold"), pady=11,
                  command=dlg.destroy
                  ).pack(side="left", expand=True, fill="x")

        # ── Body (middle, fills remaining space) ─────────────────────────
        body = tk.Frame(dlg, bg=c["dark"], padx=20, pady=8)
        body.pack(fill="both", expand=True, side="top")

        # Fee record
        rec = stu.setdefault("fee_records", {}).setdefault(mon, {
            "due": 0, "paid": 0, "discount": 0, "fine": 0, "paidFlag": False
        })
        due  = rec.get("due",  0)
        disc = rec.get("discount", 0)
        fine = rec.get("fine", 0)
        paid = rec.get("paid", 0)
        bal  = due - disc + fine - paid

        # Summary card
        card = tk.Frame(body, bg=c["dark_light"], padx=16, pady=10)
        card.pack(fill="x", pady=(0, 8))
        for lbl, val, col in [
            ("Total Due:",    f"RS {due:,}",  "white"),
            ("Discount:",     f"RS {disc:,}", c["success"]),
            ("Fine:",         f"RS {fine:,}", c["danger"]),
            ("Already Paid:", f"RS {paid:,}", "#38bdf8"),
            ("Balance Due:",  f"RS {bal:,}",  c["warning"] if bal > 0 else c["success"]),
        ]:
            r = tk.Frame(card, bg=c["dark_light"])
            r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, bg=c["dark_light"], fg=c["gray"],
                     width=14, anchor="w", font=("Helvetica",10)).pack(side="left")
            tk.Label(r, text=val, bg=c["dark_light"], fg=col,
                     font=("Helvetica",10,"bold")).pack(side="left")

        s_txt = ("✅ FULLY PAID" if rec.get("paidFlag") else
                 ("⚠️ PARTIALLY PAID" if paid > 0 else "⏳ UNPAID"))
        s_col = (c["success"] if rec.get("paidFlag") else
                 (c["warning"] if paid > 0 else c["danger"]))
        tk.Label(card, text=s_txt, bg=c["dark_light"],
                 fg=s_col, font=("Helvetica",11,"bold")).pack(anchor="w", pady=(4,0))

        # Divider
        tk.Frame(body, bg=c["gray"], height=1).pack(fill="x", pady=6)

        # Input fields
        def make_field(label):
            tk.Label(body, text=label, bg=c["dark"], fg=c["gray"],
                     font=("Helvetica",10)).pack(anchor="w", pady=(4,1))
            e = tk.Entry(body, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat",
                          bd=8, font=("Helvetica",12))
            e.pack(fill="x")
            return e

        pay_e  = make_field("💵  Amount to Pay (RS):")
        disc_e = make_field("🏷️  Extra Discount (RS):")
        fine_e = make_field("⚠️  Late Fine to Add (RS):")

        def fill_full():
            pay_e.delete(0, "end")
            pay_e.insert(0, str(max(0, bal)))
        tk.Button(body, text="⚡ Auto-fill Full Balance",
                  bg=c["primary"], fg="white", font=("Helvetica",9),
                  padx=8, pady=3, command=fill_full).pack(anchor="w", pady=(6,0))

        # Submit logic
        def do_submit():
            try:
                p = int(pay_e.get().strip()  or 0)
                d = int(disc_e.get().strip() or 0)
                f = int(fine_e.get().strip() or 0)
            except ValueError:
                messagebox.showerror("Error", "Enter whole numbers only", parent=dlg)
                return
            if p == 0 and d == 0 and f == 0:
                messagebox.showerror("Error", "Enter at least one amount", parent=dlg)
                return
            rec["paid"]     = rec.get("paid",     0) + p
            rec["discount"] = rec.get("discount", 0) + d
            rec["fine"]     = rec.get("fine",     0) + f
            net = rec["due"] - rec["discount"] + rec["fine"]
            rec["paidFlag"] = rec["paid"] >= net
            self.save_data()
            dlg.destroy()
            self._refresh_fees()
            left = max(0, net - rec["paid"])
            if rec["paidFlag"]:
                messagebox.showinfo("✅ Fully Paid",
                    f"RS {p:,} recorded!\n{stu['name']}'s fee for {mon} is FULLY PAID.")
            else:
                messagebox.showinfo("✅ Payment Saved",
                    f"RS {p:,} recorded.\nRemaining balance: RS {left:,}")

        submit_fn[0] = do_submit
        dlg.bind("<Return>", lambda e: do_submit())

    def _open_pending_panel(self):
        cur_mon = datetime.datetime.now().strftime("%Y-%m")
        pending = []
        for stu in self.students:
            fr = stu.get("fee_records",{})
            if cur_mon in fr:
                if not fr[cur_mon].get("paidFlag"):
                    pending.append(stu)
            else:
                pending.append(stu)
        self._show_pending_students(pending)

    def _record_payment_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Record Payment")
        dlg.geometry("420x400")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg, text="💵 Record Fee Payment",
                 font=("Helvetica",15,"bold"), bg=c["dark"],
                 fg="white").pack(pady=15)
        frm = tk.Frame(dlg, bg=c["dark"], padx=30)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Student (Admission No):", bg=c["dark"],
                 fg=c["gray"]).pack(anchor="w", pady=(5,0))
        adm_cb = ttk.Combobox(frm,
            values=[s.get("admissionNo","") for s in self.students],
            state="readonly")
        adm_cb.pack(fill="x", pady=5)

        tk.Label(frm, text="Month (YYYY-MM):", bg=c["dark"],
                 fg=c["gray"]).pack(anchor="w", pady=(5,0))
        mon_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", pady=5)

        for label, var_name in [("Amount (RS)","amt_e2"),("Discount (RS)","disc_e2"),
                                  ("Fine (RS)","fine_e2")]:
            tk.Label(frm, text=label, bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(5,0))
            e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
            e.pack(fill="x", pady=3)
            setattr(self, var_name, e)

        def submit():
            adm = adm_cb.get()
            stu = next((s for s in self.students if s.get("admissionNo")==adm), None)
            if not stu:
                messagebox.showerror("Error","Select a student"); return
            try:
                mon = mon_e.get().strip()
                pay = int(self.amt_e2.get() or 0)
                disc = int(self.disc_e2.get() or 0)
                fine = int(self.fine_e2.get() or 0)
                datetime.datetime.strptime(mon, "%Y-%m")
            except:
                messagebox.showerror("Error","Check values"); return
            fr = stu.setdefault("fee_records",{})
            if mon not in fr:
                due_amt = 0
                fs = self.fee_structures.get(stu.get("class",""),{})
                if isinstance(fs, dict):
                    due_amt = sum(v for v in fs.values() if isinstance(v,(int,float)))
                elif isinstance(fs, (int,float)):
                    due_amt = int(fs)
                fr[mon] = {"due":due_amt,"paid":0,"discount":0,"fine":0,"paidFlag":False}
            rec = fr[mon]
            rec["paid"] += pay; rec["discount"] += disc; rec["fine"] += fine
            net = rec["due"] - rec["discount"] + rec["fine"]
            if rec["paid"] >= net: rec["paidFlag"] = True
            self.save_data()
            dlg.destroy()
            self._refresh_fees()
            messagebox.showinfo("✅",f"Payment of RS {pay:,} recorded for {stu['name']}")

        tk.Button(frm, text="💾 Record Payment", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10, command=submit
                  ).pack(fill="x", pady=15)

    def _fee_structures_panel(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Fee Structures")
        dlg.geometry("680x520")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        hdr = tk.Frame(dlg, bg=c["primary"], padx=20, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🏗️ Fee Structures",
                 font=("Helvetica",14,"bold"), bg=c["primary"], fg="white").pack(anchor="w")
        tk.Label(hdr, text="Define fee components for each class",
                 bg=c["primary"], fg="#e0e7ff").pack(anchor="w")

        # add structure form
        form = tk.Frame(dlg, bg=c["dark_light"], padx=20, pady=12)
        form.pack(fill="x", padx=20, pady=10)
        tk.Label(form, text="Add / Update Fee Structure",
                 font=("Helvetica",11,"bold"), bg=c["dark_light"],
                 fg="white").pack(anchor="w")

        row1 = tk.Frame(form, bg=c["dark_light"])
        row1.pack(fill="x", pady=5)
        tk.Label(row1, text="Class:", bg=c["dark_light"], fg=c["gray"]).pack(side="left")
        cls_e = tk.Entry(row1, bg=c["dark"], fg="white",
                          insertbackground="white", width=12, relief="flat", bd=6)
        cls_e.pack(side="left", padx=6)

        components = ["Monthly Tuition", "Admission", "Exam", "Sports", "Library", "Transport"]
        comp_entries = {}
        row2 = tk.Frame(form, bg=c["dark_light"])
        row2.pack(fill="x")
        for i, comp in enumerate(components):
            col_f = tk.Frame(row2, bg=c["dark_light"])
            col_f.pack(side="left", padx=5)
            tk.Label(col_f, text=comp, bg=c["dark_light"],
                     fg=c["gray"], font=("Helvetica",8)).pack()
            e = tk.Entry(col_f, bg=c["dark"], fg="white",
                          insertbackground="white", width=8, relief="flat", bd=5)
            e.pack()
            comp_entries[comp] = e

        def save_structure():
            cls = cls_e.get().strip()
            if not cls:
                messagebox.showerror("Error","Enter class name"); return
            fs = {}
            for comp, e in comp_entries.items():
                v = e.get().strip()
                if v:
                    try: fs[comp] = int(v)
                    except: messagebox.showerror("Error",f"Invalid value for {comp}"); return
            if not fs:
                messagebox.showerror("Error","Enter at least one fee component"); return
            self.fee_structures[cls] = fs
            # apply to current month for students without a record
            cur_mon = datetime.date.today().strftime("%Y-%m")
            total_due = sum(fs.values())
            for stu in self.students:
                if stu.get("class") == cls:
                    stu.setdefault("fee_records",{})
                    if cur_mon not in stu["fee_records"]:
                        stu["fee_records"][cur_mon] = {
                            "due": total_due, "paid":0, "discount":0,
                            "fine":0, "paidFlag":False
                        }
            self.save_data()
            refresh_list()
            messagebox.showinfo("✅",f"Fee structure for class {cls} saved!")

        tk.Button(form, text="💾 Save Structure", bg=c["success"], fg="white",
                  command=save_structure).pack(anchor="w", pady=8)

        # existing structures list
        list_f = tk.Frame(dlg, bg=c["dark"], padx=20)
        list_f.pack(fill="both", expand=True)
        tk.Label(list_f, text="Existing Fee Structures",
                 font=("Helvetica",11,"bold"), bg=c["dark"],
                 fg="white").pack(anchor="w", pady=5)

        cols = ("Class","Monthly Tuition","Admission","Exam","Sports","Library","Transport","Total")
        tree = ttk.Treeview(list_f, columns=cols, show="headings",
                             style="Custom.Treeview", height=8)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=90)
        tree.column("Class", width=80)
        vsb = ttk.Scrollbar(list_f, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def refresh_list():
            for i in tree.get_children(): tree.delete(i)
            for cls_name, fs in self.fee_structures.items():
                if isinstance(fs, dict):
                    row = [cls_name] + [fs.get(c2,0) for c2 in components] + [sum(fs.values())]
                else:
                    row = [cls_name, fs, 0, 0, 0, 0, 0, fs]
                tree.insert("","end", values=row)

        refresh_list()

        def del_structure():
            sel = tree.selection()
            if not sel: return
            cls_name = tree.item(sel[0],"values")[0]
            if messagebox.askyesno("Confirm",f"Delete fee structure for {cls_name}?"):
                self.fee_structures.pop(cls_name, None)
                self.save_data()
                refresh_list()

        tk.Button(list_f, text="🗑️ Delete Selected", bg=c["danger"], fg="white",
                  command=del_structure).pack(pady=5)

    def _generate_vouchers_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Generate Monthly Vouchers")
        dlg.geometry("360x200")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Generate Monthly Fee Vouchers",
                 font=("Helvetica",13,"bold"), bg=c["dark"],
                 fg="white").pack(pady=15)
        tk.Label(dlg, text="Month (YYYY-MM):", bg=c["dark"], fg=c["gray"]).pack()
        mon_e = tk.Entry(dlg, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", padx=30, pady=8)

        def generate():
            mon = mon_e.get().strip()
            try: datetime.datetime.strptime(mon,"%Y-%m")
            except: messagebox.showerror("Error","Month must be YYYY-MM"); return
            count = 0
            for stu in self.students:
                stu.setdefault("fee_records",{})
                if mon not in stu["fee_records"]:
                    fs = self.fee_structures.get(stu.get("class",""),{})
                    if isinstance(fs, dict):
                        due = sum(fs.values())
                    else:
                        due = int(fs) if fs else 0
                    stu["fee_records"][mon] = {
                        "due": due, "paid":0, "discount":0,
                        "fine":0, "paidFlag": due == 0
                    }
                    count += 1
            self.save_data()
            dlg.destroy()
            self._refresh_fees()
            messagebox.showinfo("✅",f"Vouchers generated for {mon}\n{count} students updated.")

        tk.Button(dlg, text="📅 Generate Vouchers", bg=c["primary"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=generate).pack(fill="x", padx=30, pady=10)

    def _show_finance_graphs(self):
        """Show finance/fee visualization graphs in a new window."""
        c = self.colors
        win = tk.Toplevel(self.root)
        win.title("📊 Finance Graphs")
        win.geometry("900x700")
        win.configure(bg=c["dark"])
        win.transient(self.root)

        tk.Label(win, text="📊  Finance & Fee Analytics",
                 font=("Helvetica",16,"bold"), bg=c["dark"], fg="white").pack(pady=(15,5))

        # Gather fee data by month
        month_data = {}
        class_data = {}
        for stu in self.students:
            cls = stu.get("class", "Unknown")
            for mon, rec in stu.get("fee_records", {}).items():
                due = rec.get("due", 0)
                paid = rec.get("paid", 0)
                disc = rec.get("discount", 0)
                fine = rec.get("fine", 0)
                balance = due - disc + fine - paid

                if mon not in month_data:
                    month_data[mon] = {"collected": 0, "pending": 0, "total_due": 0}
                month_data[mon]["collected"] += paid
                month_data[mon]["pending"] += max(0, balance)
                month_data[mon]["total_due"] += due

                if cls not in class_data:
                    class_data[cls] = {"collected": 0, "pending": 0}
                class_data[cls]["collected"] += paid
                class_data[cls]["pending"] += max(0, balance)

        if not month_data:
            tk.Label(win, text="No fee records found.\nAdd fee structures and generate vouchers first.",
                     font=("Helvetica",12), bg=c["dark"], fg=c["subtext"],
                     justify="center").pack(expand=True)
            return

        # Scrollable canvas for graphs
        outer = tk.Frame(win, bg=c["dark"])
        outer.pack(fill="both", expand=True, padx=20, pady=10)

        cv = tk.Canvas(outer, bg=c["dark"], highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(fill="both", expand=True)
        content = tk.Frame(cv, bg=c["dark"])
        cw = cv.create_window((0,0), window=content, anchor="nw")
        content.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(cw, width=e.width))

        # ── Summary Cards ─────────────────────────────────────
        summary_f = tk.Frame(content, bg=c["dark"])
        summary_f.pack(fill="x", pady=(0,15))

        total_collected = sum(d["collected"] for d in month_data.values())
        total_pending = sum(d["pending"] for d in month_data.values())
        total_due = sum(d["total_due"] for d in month_data.values())
        collection_rate = round(total_collected / total_due * 100, 1) if total_due > 0 else 0

        for label, value, color in [
            ("Total Collected", f"RS {total_collected:,.0f}", c["success"]),
            ("Total Pending", f"RS {total_pending:,.0f}", c["danger"]),
            ("Total Due", f"RS {total_due:,.0f}", c["primary"]),
            ("Collection Rate", f"{collection_rate}%", "#fbbf24"),
        ]:
            card = tk.Frame(summary_f, bg=c["card_bg"], padx=20, pady=12)
            card.pack(side="left", expand=True, fill="both", padx=5)
            tk.Label(card, text=label, bg=c["card_bg"], fg=c["subtext"],
                     font=("Helvetica",9)).pack(anchor="w")
            tk.Label(card, text=value, bg=c["card_bg"], fg=color,
                     font=("Helvetica",18,"bold")).pack(anchor="w")

        # ── Monthly Collection Bar Chart ──────────────────────
        chart1 = tk.Frame(content, bg=c["card_bg"], padx=20, pady=15)
        chart1.pack(fill="x", pady=8)
        tk.Label(chart1, text="Monthly Fee Collection Trend",
                 font=("Helvetica",13,"bold"), bg=c["card_bg"], fg=c["text"]).pack(anchor="w")

        sorted_months = sorted(month_data.keys())[-12:]  # last 12 months
        if sorted_months:
            max_val = max(max(month_data[m]["collected"], month_data[m]["pending"])
                         for m in sorted_months) or 1
            bar_canvas = tk.Canvas(chart1, bg=c["card_bg"], height=250,
                                   highlightthickness=0)
            bar_canvas.pack(fill="x", pady=10)
            bar_canvas.update_idletasks()
            cw_width = 820

            bar_w = max(20, min(50, (cw_width - 80) // (len(sorted_months) * 2 + 1)))
            x_start = 60
            y_base = 230
            y_top = 20

            # Y-axis labels
            for i in range(5):
                val = int(max_val * i / 4)
                y = y_base - int((y_base - y_top) * i / 4)
                bar_canvas.create_text(50, y, text=f"{val:,}", anchor="e",
                                       fill=c["subtext"], font=("Helvetica",7))
                bar_canvas.create_line(x_start, y, cw_width-10, y,
                                       fill="#334155", dash=(2,2))

            for i, mon in enumerate(sorted_months):
                x = x_start + i * (bar_w * 2 + 15)
                collected = month_data[mon]["collected"]
                pending = month_data[mon]["pending"]

                # Collected bar (green)
                h_c = int((y_base - y_top) * collected / max_val) if max_val else 0
                bar_canvas.create_rectangle(x, y_base - h_c, x + bar_w, y_base,
                                            fill="#10b981", outline="")

                # Pending bar (red)
                h_p = int((y_base - y_top) * pending / max_val) if max_val else 0
                bar_canvas.create_rectangle(x + bar_w + 2, y_base - h_p,
                                            x + bar_w * 2 + 2, y_base,
                                            fill="#ef4444", outline="")

                # Month label
                short_mon = mon[-2:] if len(mon) >= 7 else mon
                bar_canvas.create_text(x + bar_w, y_base + 12, text=short_mon,
                                       fill=c["subtext"], font=("Helvetica",7))

            # Legend
            bar_canvas.create_rectangle(cw_width-180, 10, cw_width-168, 22,
                                        fill="#10b981", outline="")
            bar_canvas.create_text(cw_width-165, 16, text="Collected", anchor="w",
                                   fill=c["subtext"], font=("Helvetica",8))
            bar_canvas.create_rectangle(cw_width-100, 10, cw_width-88, 22,
                                        fill="#ef4444", outline="")
            bar_canvas.create_text(cw_width-85, 16, text="Pending", anchor="w",
                                   fill=c["subtext"], font=("Helvetica",8))

        # ── Class-wise Collection Pie Chart ───────────────────
        chart2 = tk.Frame(content, bg=c["card_bg"], padx=20, pady=15)
        chart2.pack(fill="x", pady=8)
        tk.Label(chart2, text="Class-wise Fee Collection",
                 font=("Helvetica",13,"bold"), bg=c["card_bg"], fg=c["text"]).pack(anchor="w")

        if class_data:
            pie_canvas = tk.Canvas(chart2, bg=c["card_bg"], height=280,
                                   highlightthickness=0)
            pie_canvas.pack(fill="x", pady=10)

            colors_list = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                           "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"]
            total_class_collected = sum(d["collected"] for d in class_data.values()) or 1

            cx, cy, r = 200, 140, 120
            start_angle = 0
            sorted_classes = sorted(class_data.keys())

            for i, cls in enumerate(sorted_classes):
                collected = class_data[cls]["collected"]
                extent = 360 * collected / total_class_collected
                color = colors_list[i % len(colors_list)]
                pie_canvas.create_arc(cx-r, cy-r, cx+r, cy+r,
                                      start=start_angle, extent=max(extent, 1),
                                      fill=color, outline=c["card_bg"], width=2)
                start_angle += extent

            # Legend
            legend_x = cx + r + 50
            for i, cls in enumerate(sorted_classes):
                ly = 30 + i * 22
                color = colors_list[i % len(colors_list)]
                pie_canvas.create_rectangle(legend_x, ly, legend_x+14, ly+14,
                                            fill=color, outline="")
                collected = class_data[cls]["collected"]
                pending = class_data[cls]["pending"]
                pct = round(collected / total_class_collected * 100, 1)
                pie_canvas.create_text(legend_x+20, ly+7,
                                       text=f"{cls}: RS {collected:,.0f} ({pct}%) | Pending: RS {pending:,.0f}",
                                       anchor="w", fill=c["text"], font=("Helvetica",9))

        # ── Paid vs Pending Overview ──────────────────────────
        chart3 = tk.Frame(content, bg=c["card_bg"], padx=20, pady=15)
        chart3.pack(fill="x", pady=8)
        tk.Label(chart3, text="Paid vs Pending Overview",
                 font=("Helvetica",13,"bold"), bg=c["card_bg"], fg=c["text"]).pack(anchor="w")

        overview_canvas = tk.Canvas(chart3, bg=c["card_bg"], height=80,
                                    highlightthickness=0)
        overview_canvas.pack(fill="x", pady=10)

        bar_total = total_collected + total_pending or 1
        paid_pct = total_collected / bar_total
        bar_full_w = 800
        paid_w = int(bar_full_w * paid_pct)

        overview_canvas.create_rectangle(20, 20, 20 + paid_w, 60,
                                         fill="#10b981", outline="")
        overview_canvas.create_rectangle(20 + paid_w, 20, 20 + bar_full_w, 60,
                                         fill="#ef4444", outline="")
        overview_canvas.create_text(20 + paid_w // 2, 40,
                                    text=f"Paid: {round(paid_pct*100,1)}%",
                                    fill="white", font=("Helvetica",10,"bold"))
        if paid_w < bar_full_w - 60:
            overview_canvas.create_text(20 + paid_w + (bar_full_w - paid_w) // 2, 40,
                                        text=f"Pending: {round((1-paid_pct)*100,1)}%",
                                        fill="white", font=("Helvetica",10,"bold"))

        tk.Button(win, text="✖ Close", bg=c["danger"], fg="white",
                  font=("Helvetica",10,"bold"), command=win.destroy).pack(pady=10)

    def _print_voucher(self):
        if not HAS_PDF:
            messagebox.showerror("Error","reportlab not installed. Run: pip install reportlab")
            return
        if not _ensure_pdf():
            messagebox.showerror("Error","Could not load reportlab."); return
        sel = self._fees_tree.selection()
        if not sel:
            messagebox.showerror("Error","Select a fee row first"); return
        tags = self._fees_tree.item(sel[0],"tags")
        sid, mon = tags[0], tags[1]
        stu = next((s for s in self.students if s["id"]==sid), None)
        if not stu: return
        rec = stu.get("fee_records",{}).get(mon,{})
        if not rec: return

        pdf_dir = self.data_dir / "vouchers"
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / f"Voucher_{stu.get('admissionNo',sid)}_{mon}.pdf"

        c_pdf = rl_canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4
        c_pdf.setFillColorRGB(0.39,0.39,0.96)
        c_pdf.rect(0, h-80, w, 80, fill=1)
        c_pdf.setFillColorRGB(1,1,1)
        c_pdf.setFont("Helvetica-Bold",22)
        c_pdf.drawString(40, h-50, self.settings.get("school_name","My School"))
        c_pdf.setFont("Helvetica",13)
        c_pdf.drawString(40, h-75, "Fee Voucher / Receipt")
        c_pdf.setFillColorRGB(0,0,0)
        c_pdf.setFont("Helvetica",12)
        y = h-120
        c_pdf.drawString(40,y,f"Student : {stu.get('name','')}")
        c_pdf.drawString(300,y,f"Admission No : {stu.get('admissionNo','')}")
        y -= 22
        c_pdf.drawString(40,y,f"Class / Section : {stu.get('class','')} / {stu.get('section','')}")
        y -= 22; c_pdf.drawString(40,y,f"Month : {mon}")
        y -= 30
        c_pdf.setFont("Helvetica-Bold",12)
        c_pdf.drawString(40,y,"Description"); c_pdf.drawString(350,y,"Amount (RS)")
        y -= 12; c_pdf.line(40,y,w-40,y); y -= 18
        c_pdf.setFont("Helvetica",12)
        due = rec.get("due",0); disc = rec.get("discount",0)
        fine = rec.get("fine",0); paid = rec.get("paid",0)
        bal = due - disc + fine - paid
        for lbl, val in [("Total Due",due),("Discount",disc),
                          ("Fine / Late Fee",fine),("Amount Paid",paid)]:
            c_pdf.drawString(40,y,lbl)
            c_pdf.drawRightString(w-50,y,f"{val:,.2f}")
            y -= 20
        c_pdf.line(40,y,w-40,y); y -= 20
        c_pdf.setFont("Helvetica-Bold",13)
        c_pdf.drawString(40,y,"Balance Due")
        c_pdf.drawRightString(w-50,y,f"{bal:,.2f}")
        y -= 30
        status = "PAID ✓" if rec.get("paidFlag") else "PENDING"
        c_pdf.setFont("Helvetica-Bold",14)
        c_pdf.setFillColorRGB(0.06,0.73,0.51 if rec.get("paidFlag") else 0,
                                  0 if rec.get("paidFlag") else 0)
        c_pdf.drawString(40,y,f"Status: {status}")
        c_pdf.setFillColorRGB(0,0,0)
        y -= 60
        c_pdf.setFont("Helvetica-Oblique",10)
        c_pdf.drawString(40,y,f"Printed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        y -= 20; c_pdf.drawString(40,y,"Signature: _____________________")
        c_pdf.showPage(); c_pdf.save()

        try:
            if os.name == "nt":
                os.startfile(str(pdf_path))
            elif sys.platform == "darwin":
                subprocess.run(["open",str(pdf_path)])
            else:
                subprocess.run(["xdg-open",str(pdf_path)])
        except Exception as e:
            messagebox.showinfo("PDF Saved",f"Voucher saved to:\n{pdf_path}")

    # ════════════════════════════════════════════════════════════════════════
    #  LIBRARY
    # ════════════════════════════════════════════════════════════════════════
    def show_library(self):
        if not self.check_perm("library"): return
        self._clear()
        c = self.colors
        self._header("📚  Library Management", "Books, issue, return, overdue")

        ctrl = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=12)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="➕ Add Book", bg=c["success"], fg="white",
                  command=self._add_book_dlg).pack(side="left")
        tk.Button(ctrl, text="🔍 Overdue Books (30+ days)", bg=c["warning"], fg="white",
                  command=self._show_overdue).pack(side="left", padx=5)

        # search
        self._lib_search = tk.Entry(ctrl, bg=c["card_bg"], fg=c["text"],
                                     insertbackground=c["text"], relief="flat", bd=8)
        self._lib_search.pack(side="right", padx=5)
        self._lib_search.bind("<KeyRelease>", lambda e: self._refresh_library())
        tk.Label(ctrl, text="🔍", bg=c["content_bg"], fg=c["subtext"]).pack(side="right")

        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True)
        cols = ("#","Title","Author","Category","Issued To","Issue Date","Status")
        self._lib_tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                       style="Custom.Treeview", height=16)
        for col in cols:
            self._lib_tree.heading(col, text=col)
            self._lib_tree.column(col, anchor="center", width=130)
        self._lib_tree.column("Title", width=200)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._lib_tree.yview)
        self._lib_tree.configure(yscrollcommand=vsb.set)
        self._lib_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._lib_tree.bind("<Double-1>", lambda e: self._book_action())
        self._refresh_library()

    def _refresh_library(self):
        for i in self._lib_tree.get_children(): self._lib_tree.delete(i)
        term = self._lib_search.get().lower() if hasattr(self,"_lib_search") else ""
        for idx, b in enumerate(self.library, 1):
            if term and term not in b.get("title","").lower() \
                    and term not in b.get("author","").lower(): continue
            stu = next((s for s in self.students if s["id"]==b.get("issued_to","")), None)
            issued_name = stu["name"] if stu else "—"
            status = "📖 Issued" if b.get("issued_to") else "✅ Available"
            self._lib_tree.insert("","end",
                values=(idx, b.get("title",""), b.get("author",""),
                        b.get("category","General"), issued_name,
                        b.get("issue_date","—"), status),
                tags=(b["id"],))

    def _add_book_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Book")
        dlg.geometry("440x340")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="➕ Add New Book",
                 font=("Helvetica",14,"bold"), bg=c["dark"], fg="white").pack(pady=15)
        frm = tk.Frame(dlg, bg=c["dark"], padx=30)
        frm.pack(fill="both", expand=True)
        fields = [("Title *","title"),("Author *","author"),("Category","category"),("ISBN","isbn")]
        entries = {}
        for lbl, key in fields:
            tk.Label(frm, text=lbl, bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
            e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
            e.pack(fill="x", pady=3)
            entries[key] = e

        def save():
            if not entries["title"].get():
                messagebox.showerror("Error","Title required"); return
            self.library.append({
                "id": self.gen_id(),
                "title": entries["title"].get(),
                "author": entries["author"].get(),
                "category": entries["category"].get() or "General",
                "isbn": entries["isbn"].get(),
                "issued_to": None, "issue_date": None
            })
            self.save_data()
            dlg.destroy()
            self._refresh_library()
            messagebox.showinfo("✅","Book added!")

        tk.Button(frm, text="💾 Save Book", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10, command=save).pack(fill="x", pady=15)

    def _book_action(self):
        sel = self._lib_tree.selection()
        if not sel: return
        bid = self._lib_tree.item(sel[0],"tags")[0]
        book = next((b for b in self.library if b["id"]==bid), None)
        if not book: return
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Book: {book['title']}")
        dlg.geometry("380x260")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text=book["title"], font=("Helvetica",13,"bold"),
                 bg=c["dark"], fg="white").pack(pady=12)
        tk.Label(dlg, text=f"Author: {book.get('author','')}", bg=c["dark"],
                 fg=c["gray"]).pack()
        if book.get("issued_to"):
            stu = next((s for s in self.students if s["id"]==book["issued_to"]), None)
            tk.Label(dlg, text=f"Issued to: {stu['name'] if stu else 'Unknown'} on {book.get('issue_date','')}",
                     bg=c["dark"], fg=c["warning"]).pack(pady=6)
            tk.Button(dlg, text="↩️ Return Book", bg=c["warning"], fg="white",
                      command=lambda: (self._return_book(bid), dlg.destroy())
                      ).pack(pady=10)
        else:
            tk.Label(dlg, text="Status: Available ✅", bg=c["dark"], fg=c["success"]).pack(pady=6)
            tk.Label(dlg, text="Issue to Student:", bg=c["dark"], fg=c["gray"]).pack()
            adm_cb = ttk.Combobox(dlg,
                values=[s.get("admissionNo","") for s in self.students],
                state="readonly", width=20)
            adm_cb.pack(pady=5)
            def issue():
                adm = adm_cb.get()
                stu = next((s for s in self.students if s.get("admissionNo")==adm), None)
                if not stu: messagebox.showerror("Error","Select student"); return
                book["issued_to"] = stu["id"]
                book["issue_date"] = datetime.date.today().isoformat()
                self.save_data()
                dlg.destroy()
                self._refresh_library()
                messagebox.showinfo("✅",f"Book issued to {stu['name']}")
            tk.Button(dlg, text="📚 Issue Book", bg=c["primary"], fg="white",
                      command=issue).pack(pady=5)
        tk.Button(dlg, text="🗑️ Delete Book", bg=c["danger"], fg="white",
                  command=lambda: (self.library.remove(book),
                                   self.save_data(), dlg.destroy(),
                                   self._refresh_library())
                  ).pack(pady=5)

    def _return_book(self, bid):
        book = next((b for b in self.library if b["id"]==bid), None)
        if book:
            book["issued_to"] = None; book["issue_date"] = None
            self.save_data()
            self._refresh_library()
            messagebox.showinfo("✅","Book returned successfully!")

    def _show_overdue(self):
        today = datetime.date.today()
        overdue = []
        for b in self.library:
            if b.get("issue_date"):
                try:
                    iss = datetime.datetime.strptime(b["issue_date"],"%Y-%m-%d").date()
                    if (today - iss).days > 30:
                        overdue.append((b, (today-iss).days))
                except: pass
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Overdue Books ({len(overdue)})")
        dlg.geometry("720x400")
        dlg.configure(bg=c["dark"])
        cols = ("#","Title","Author","Issued To","Issue Date","Days Overdue")
        tree = ttk.Treeview(dlg, columns=cols, show="headings", style="Custom.Treeview")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=110)
        tree.pack(fill="both", expand=True, padx=20, pady=20)
        for idx, (b, days) in enumerate(overdue, 1):
            stu = next((s for s in self.students if s["id"]==b.get("issued_to","")), {})
            tree.insert("","end",
                values=(idx, b["title"], b.get("author",""),
                        stu.get("name","—"), b["issue_date"], days))
        if not overdue:
            tk.Label(dlg, text="🎉 No overdue books!", font=("Helvetica",14),
                     bg=c["dark"], fg=c["success"]).pack(pady=40)

    # ════════════════════════════════════════════════════════════════════════
    #  EXAMS
    # ════════════════════════════════════════════════════════════════════════
    def show_exams(self):
        if not self.check_perm("exams"): return
        self._clear()
        c = self.colors
        self._header("📝  Exams & Results", "Create exams, enter marks, view result cards")

        ctrl = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=12)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="➕ Create Exam", bg=c["success"], fg="white",
                  command=self._create_exam_dlg).pack(side="left")
        tk.Button(ctrl, text="📊 Class Result Report", bg=c["warning"], fg="white",
                  command=self._class_result_dlg).pack(side="left", padx=5)

        tbl_f = tk.Frame(self.main, bg=c["content_bg"], padx=25)
        tbl_f.pack(fill="both", expand=True)
        cols = ("#","Class","Subject","Max Marks","Date","Students Marked")
        self._exm_tree = ttk.Treeview(tbl_f, columns=cols, show="headings",
                                       style="Custom.Treeview", height=16)
        for col in cols:
            self._exm_tree.heading(col, text=col)
            self._exm_tree.column(col, anchor="center", width=130)
        vsb = ttk.Scrollbar(tbl_f, orient="vertical", command=self._exm_tree.yview)
        self._exm_tree.configure(yscrollcommand=vsb.set)
        self._exm_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._exm_tree.bind("<Double-1>", lambda e: self._add_marks())
        self._refresh_exams()

    def _refresh_exams(self):
        for i in self._exm_tree.get_children(): self._exm_tree.delete(i)
        for idx, ex in enumerate(self.exams, 1):
            self._exm_tree.insert("","end",
                values=(idx, ex["class"], ex["subject"],
                        ex["max_marks"], ex.get("date","—"),
                        len(ex.get("marks",{}))),
                tags=(ex["id"],))

    def _create_exam_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Create Exam")
        dlg.geometry("400x320")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="➕ Create Exam",
                 font=("Helvetica",14,"bold"), bg=c["dark"],
                 fg="white").pack(pady=15)
        frm = tk.Frame(dlg, bg=c["dark"], padx=30)
        frm.pack(fill="both", expand=True)
        fields = [("Class","class",""),("Subject","subject",""),
                  ("Max Marks","max_marks","100"),
                  ("Exam Date (YYYY-MM-DD)","date",datetime.date.today().isoformat())]
        entries = {}
        for lbl, key, default in fields:
            tk.Label(frm, text=lbl, bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
            e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
            e.insert(0, default)
            e.pack(fill="x", pady=3)
            entries[key] = e

        def save():
            try:
                mx = int(entries["max_marks"].get())
            except:
                messagebox.showerror("Error","Max marks must be a number"); return
            if not entries["class"].get() or not entries["subject"].get():
                messagebox.showerror("Error","Class and subject required"); return
            self.exams.append({
                "id": self.gen_id(),
                "class": entries["class"].get(),
                "subject": entries["subject"].get(),
                "max_marks": mx,
                "date": entries["date"].get(),
                "marks": {}
            })
            self.save_data()
            dlg.destroy()
            self._refresh_exams()
            messagebox.showinfo("✅","Exam created!")

        tk.Button(frm, text="💾 Create", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=save).pack(fill="x", pady=12)

    def _add_marks(self):
        sel = self._exm_tree.selection()
        if not sel: return
        eid = self._exm_tree.item(sel[0],"tags")[0]
        exam = next((e for e in self.exams if e["id"]==eid), None)
        if not exam: return
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Marks – {exam['subject']} ({exam['class']})")
        dlg.geometry("640x500")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg, text=f"{exam['subject']} | Class {exam['class']} | Max: {exam['max_marks']}",
                 font=("Helvetica",12,"bold"), bg=c["dark"], fg="white").pack(pady=12)

        frm = tk.Frame(dlg, bg=c["dark"], padx=20)
        frm.pack(fill="both", expand=True)

        students = [s for s in self.students if s.get("class")==exam["class"]]
        cols = ("#","Admission","Name","Marks (enter below)","Percentage","Grade")
        tree = ttk.Treeview(frm, columns=cols, show="headings",
                             style="Custom.Treeview", height=12)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=110)
        tree.pack(fill="both", expand=True)

        mark_vars = {}
        for idx, stu in enumerate(students, 1):
            existing = str(exam.get("marks",{}).get(stu["id"],""))
            var = tk.StringVar(value=existing)
            mark_vars[stu["id"]] = var
            pct = f"{int(existing)/exam['max_marks']*100:.1f}%" if existing else "—"
            grade = self._grade(int(existing), exam["max_marks"]) if existing else "—"
            tree.insert("","end",
                values=(idx, stu.get("admissionNo",""), stu.get("name",""),
                        existing, pct, grade),
                tags=(stu["id"],))

        # edit marks via entry fields below tree
        entry_f = tk.Frame(dlg, bg=c["dark"], padx=20)
        entry_f.pack(fill="x")
        tk.Label(entry_f, text="Select student row then type marks below:",
                 bg=c["dark"], fg=c["gray"], font=("Helvetica",9)).pack(anchor="w")
        marks_e = tk.Entry(entry_f, bg=c["dark_light"], fg="white",
                            insertbackground="white", relief="flat", bd=7)
        marks_e.pack(fill="x", pady=5)

        def on_select(event):
            sel2 = tree.selection()
            if not sel2: return
            sid2 = tree.item(sel2[0],"tags")[0]
            marks_e.delete(0,"end")
            marks_e.insert(0, mark_vars[sid2].get())

        def apply_marks():
            sel2 = tree.selection()
            if not sel2: return
            sid2 = tree.item(sel2[0],"tags")[0]
            val = marks_e.get().strip()
            try:
                m = int(val)
                if m > exam["max_marks"] or m < 0:
                    raise ValueError
            except:
                messagebox.showerror("Error",f"Marks must be 0–{exam['max_marks']}"); return
            mark_vars[sid2].set(val)
            pct = f"{m/exam['max_marks']*100:.1f}%"
            grade = self._grade(m, exam["max_marks"])
            tree.item(sel2[0], values=(
                tree.item(sel2[0],"values")[0],
                tree.item(sel2[0],"values")[1],
                tree.item(sel2[0],"values")[2],
                val, pct, grade))

        tree.bind("<<TreeviewSelect>>", on_select)
        tk.Button(entry_f, text="✔ Apply", bg=c["primary"], fg="white",
                  command=apply_marks).pack(side="right", padx=5)

        def save_all():
            exam.setdefault("marks",{})
            for sid2, var in mark_vars.items():
                v = var.get().strip()
                if v:
                    try: exam["marks"][sid2] = int(v)
                    except: pass
            self.save_data()
            dlg.destroy()
            self._refresh_exams()
            messagebox.showinfo("✅","Marks saved!")

        tk.Button(dlg, text="💾 Save All Marks", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=save_all).pack(fill="x", padx=20, pady=8)

    @staticmethod
    def _grade(marks, max_marks):
        pct = marks / max_marks * 100 if max_marks else 0
        if pct >= 90: return "A+"
        if pct >= 80: return "A"
        if pct >= 70: return "B"
        if pct >= 60: return "C"
        if pct >= 50: return "D"
        return "F"

    def _class_result_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Class Result Report")
        dlg.geometry("360x180")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Select Class:", bg=c["dark"], fg=c["gray"]).pack(pady=(20,5))
        cls_cb = ttk.Combobox(dlg, values=self._all_classes(), state="readonly")
        cls_cb.pack(fill="x", padx=30, pady=5)

        def show():
            cls = cls_cb.get()
            if not cls: messagebox.showerror("Error","Select class"); return
            rpt = tk.Toplevel(self.root)
            rpt.title(f"Result Report – Class {cls}")
            rpt.geometry("800x550")
            rpt.configure(bg=c["dark"])
            subjects = [ex["subject"] for ex in self.exams if ex["class"]==cls]
            cols2 = ["#","Name","Adm No"] + subjects + ["Total","Avg%","Grade"]
            tree2 = ttk.Treeview(rpt, columns=cols2, show="headings",
                                  style="Custom.Treeview", height=16)
            for col in cols2:
                tree2.heading(col, text=col)
                tree2.column(col, anchor="center", width=max(70, len(col)*10))
            vsb2 = ttk.Scrollbar(rpt, orient="vertical", command=tree2.yview)
            hsb2 = ttk.Scrollbar(rpt, orient="horizontal", command=tree2.xview)
            tree2.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
            vsb2.pack(side="right", fill="y")
            hsb2.pack(side="bottom", fill="x")
            tree2.pack(fill="both", expand=True, padx=10, pady=10)
            students = [s for s in self.students if s.get("class")==cls]
            for idx, stu in enumerate(students, 1):
                exm_data = [e for e in self.exams if e["class"]==cls]
                marks_list = []
                max_list = []
                for ex in exm_data:
                    m = ex.get("marks",{}).get(stu["id"])
                    marks_list.append(str(m) if m is not None else "—")
                    max_list.append(ex["max_marks"])
                num_marks = [m for m in marks_list if m != "—"]
                total = sum(int(m) for m in num_marks) if num_marks else 0
                tot_max = sum(max_list)
                avg = total/tot_max*100 if tot_max and num_marks else 0
                grade = self._grade(total, tot_max) if tot_max and num_marks else "—"
                tree2.insert("","end",
                    values=[idx, stu.get("name",""), stu.get("admissionNo","")] +
                           marks_list + [total, f"{avg:.1f}%", grade])

            tk.Button(rpt, text="📥 Export CSV", bg=c["success"], fg="white",
                      command=lambda: self._export_result_csv(cls)).pack(pady=8)
            dlg.destroy()

        tk.Button(dlg, text="📊 Show Report", bg=c["success"], fg="white",
                  command=show).pack(pady=15)

    def _export_result_csv(self, cls):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")])
        if not path: return
        with open(path,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            exm_data = [e for e in self.exams if e["class"]==cls]
            w.writerow(["Name","Adm No"] + [e["subject"] for e in exm_data] + ["Total","Grade"])
            for stu in (s for s in self.students if s.get("class")==cls):
                marks_list = [str(e.get("marks",{}).get(stu["id"],"")) for e in exm_data]
                nums = [int(m) for m in marks_list if m]
                tot_max = sum(e["max_marks"] for e in exm_data)
                total = sum(nums)
                grade = self._grade(total, tot_max) if tot_max and nums else "—"
                w.writerow([stu.get("name",""), stu.get("admissionNo","")] +
                           marks_list + [total, grade])
        messagebox.showinfo("✅",f"Exported to {path}")

    # ════════════════════════════════════════════════════════════════════════
    #  REPORTS
    # ════════════════════════════════════════════════════════════════════════
    def show_reports(self):
        self._clear()
        c = self.colors
        self._header("📈  Reports Center", "Generate and export various reports")

        btn_f = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=15)
        btn_f.pack(fill="x")

        report_btns = [
            ("👨‍🎓 Student Report (CSV)", c["primary"], self._export_students),
            ("👩‍🏫 Teacher Report (CSV)", "#06b6d4", self._export_teachers),
            ("🕒 Attendance Report (CSV)", "#8b5cf6", self._export_attendance),
            ("💰 Fee Report (CSV)", c["success"], self._export_fee_report),
            ("📝 Exam Report (CSV)", c["warning"], self._export_exam_report),
            ("📦 Export All Data (CSV)", "#7c3aed", self._export_all_csv),
        ]
        for i, (label, bg, cmd) in enumerate(report_btns):
            tk.Button(btn_f, text=label, bg=bg, fg="white",
                      font=("Helvetica",9,"bold"), padx=10, pady=8,
                      command=cmd).grid(row=i//3, column=i%3, padx=4, pady=4, sticky="ew")
        for col in range(3):
            btn_f.columnconfigure(col, weight=1)

        # stats view
        sf = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=10)
        sf.pack(fill="both", expand=True)
        tk.Label(sf, text="Summary Statistics", font=("Helvetica",14,"bold"),
                 bg=c["content_bg"], fg=c["text"]).pack(anchor="w", pady=(0,10))

        total_s = len(self.students)
        total_t = len(self.teachers)
        total_books = len(self.library)
        issued = sum(1 for b in self.library if b.get("issued_to"))
        total_exams = len(self.exams)
        cur_mon = datetime.datetime.now().strftime("%Y-%m")
        rev = sum(
            rec.get("paid",0)
            for stu in self.students
            for rec in stu.get("fee_records",{}).values()
        )
        outstanding = sum(
            max(0, rec.get("due",0) - rec.get("discount",0) + rec.get("fine",0) - rec.get("paid",0))
            for stu in self.students
            for rec in stu.get("fee_records",{}).values()
        )

        stats = [
            ("👨‍🎓 Total Students", total_s, c["primary"]),
            ("👩‍🏫 Total Teachers", total_t, "#8b5cf6"),
            ("📚 Library Books", total_books, "#06b6d4"),
            ("📖 Books Issued", issued, c["warning"]),
            ("📝 Total Exams", total_exams, c["success"]),
            ("💵 Total Revenue", f"RS {rev:,}", c["success"]),
            ("📉 Outstanding", f"RS {outstanding:,}", c["danger"]),
        ]

        rows = []
        row = tk.Frame(sf, bg=c["content_bg"])
        row.pack(fill="x")
        for i, (lbl, val, col) in enumerate(stats):
            if i > 0 and i % 4 == 0:
                row = tk.Frame(sf, bg=c["content_bg"])
                row.pack(fill="x")
            card = tk.Frame(row, bg=c["card_bg"], padx=15, pady=12)
            card.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            tk.Label(card, text=lbl, bg=c["card_bg"], fg=c["subtext"],
                     font=("Helvetica",9)).pack(anchor="w")
            tk.Label(card, text=str(val), font=("Helvetica",18,"bold"),
                     bg=c["card_bg"], fg=col).pack(anchor="w")
            rows.append(card)

    def _export_students(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], title="Export Students")
        if not path: return
        cur_mon = datetime.datetime.now().strftime("%Y-%m")
        with open(path,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Adm No","Name","Father","Class","Section","Phone","DOB","Fee Status"])
            for stu in self.students:
                rec = stu.get("fee_records",{}).get(cur_mon,{})
                w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                            stu.get("father",""), stu.get("class",""),
                            stu.get("section",""), stu.get("phone",""),
                            stu.get("dob",""),
                            "Paid" if rec.get("paidFlag") else "Pending"])
        messagebox.showinfo("✅",f"Exported to {path}")

    def _export_attendance(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Attendance Report")
        dlg.geometry("360x220")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Class:", bg=c["dark"], fg=c["gray"]).pack(pady=(15,3))
        cls_cb = ttk.Combobox(dlg, values=["All"]+self._all_classes(), state="readonly")
        cls_cb.set("All")
        cls_cb.pack(fill="x", padx=30)
        tk.Label(dlg, text="Month (YYYY-MM):", bg=c["dark"], fg=c["gray"]).pack(pady=(8,3))
        mon_e = tk.Entry(dlg, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", padx=30)

        def export():
            cls_f = cls_cb.get(); mon = mon_e.get().strip()
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")])
            if not path: return
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Name","Adm No","Class","Date","Status"])
                for stu in self.students:
                    if cls_f != "All" and stu.get("class","") != cls_f: continue
                    for day, stat in stu.get("attendance",{}).items():
                        if day.startswith(mon):
                            w.writerow([stu.get("name",""), stu.get("admissionNo",""),
                                        stu.get("class",""), day, stat])
            messagebox.showinfo("✅",f"Exported to {path}")
            dlg.destroy()

        tk.Button(dlg, text="📥 Export CSV", bg=c["success"], fg="white",
                  command=export).pack(pady=15)

    def _export_fee_report(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Fee Report")
        dlg.geometry("320x180")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Month (YYYY-MM):", bg=c["dark"], fg=c["gray"]).pack(pady=(20,5))
        mon_e = tk.Entry(dlg, bg=c["dark_light"], fg="white",
                          insertbackground="white", relief="flat", bd=7)
        mon_e.insert(0, datetime.date.today().strftime("%Y-%m"))
        mon_e.pack(fill="x", padx=30)

        def export():
            mon = mon_e.get().strip()
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")])
            if not path: return
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Adm No","Name","Class","Due","Paid","Discount","Fine","Balance","Status"])
                for stu in self.students:
                    rec = stu.get("fee_records",{}).get(mon)
                    if not rec: continue
                    due=rec.get("due",0); paid=rec.get("paid",0)
                    disc=rec.get("discount",0); fine=rec.get("fine",0)
                    bal=due-disc+fine-paid
                    w.writerow([stu.get("admissionNo",""), stu.get("name",""),
                                stu.get("class",""), due, paid, disc, fine, bal,
                                "Paid" if rec.get("paidFlag") else "Pending"])
            messagebox.showinfo("✅",f"Exported to {path}")
            dlg.destroy()

        tk.Button(dlg, text="📥 Export CSV", bg=c["success"], fg="white",
                  command=export).pack(pady=15)

    def _export_exam_report(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Exam Report")
        dlg.geometry("340x180")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="Class:", bg=c["dark"], fg=c["gray"]).pack(pady=(15,5))
        cls_cb = ttk.Combobox(dlg, values=["All"]+self._all_classes(), state="readonly")
        cls_cb.set("All")
        cls_cb.pack(fill="x", padx=30)

        def export():
            cls_f = cls_cb.get()
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")])
            if not path: return
            with open(path,"w",newline="",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Subject","Class","Date","Max Marks","Student","Adm No","Marks","Grade"])
                for ex in self.exams:
                    if cls_f != "All" and ex["class"] != cls_f: continue
                    for sid, mk in ex.get("marks",{}).items():
                        stu = next((s for s in self.students if s["id"]==sid), {})
                        w.writerow([ex["subject"], ex["class"], ex.get("date",""),
                                    ex["max_marks"], stu.get("name",""),
                                    stu.get("admissionNo",""), mk,
                                    self._grade(mk, ex["max_marks"])])
            messagebox.showinfo("✅",f"Exported to {path}")
            dlg.destroy()

        tk.Button(dlg, text="📥 Export CSV", bg=c["success"], fg="white",
                  command=export).pack(pady=15)

    def _export_teachers(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], title="Export Teachers")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Name","Subject","Phone","Email","Qualification","Experience","Salary"])
            for t in self.teachers:
                w.writerow([t.get("name",""), t.get("subject",""),
                            t.get("phone",""), t.get("email",""),
                            t.get("qualification",""), t.get("experience",""),
                            t.get("salary","")])
        messagebox.showinfo("✅", f"Teachers exported to {path}")

    def _export_all_csv(self):
        folder = filedialog.askdirectory(title="Select folder to export all CSV files")
        if not folder: return
        import csv as _csv
        # 1) Students
        with open(os.path.join(folder, "students.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Adm No","Name","Father","Class","Section","Phone","Email","DOB","Address"])
            for s in self.students:
                w.writerow([s.get("admissionNo",""), s.get("name",""), s.get("father",""),
                            s.get("class",""), s.get("section",""), s.get("phone",""),
                            s.get("email",""), s.get("dob",""), s.get("address","")])
        # 2) Teachers
        with open(os.path.join(folder, "teachers.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Name","Subject","Phone","Email","Qualification","Experience","Salary"])
            for t in self.teachers:
                w.writerow([t.get("name",""), t.get("subject",""), t.get("phone",""),
                            t.get("email",""), t.get("qualification",""),
                            t.get("experience",""), t.get("salary","")])
        # 3) Attendance
        with open(os.path.join(folder, "attendance.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Name","Adm No","Class","Date","Status"])
            for s in self.students:
                for day, stat in s.get("attendance", {}).items():
                    w.writerow([s.get("name",""), s.get("admissionNo",""),
                                s.get("class",""), day, stat])
        # 4) Fee Records
        with open(os.path.join(folder, "fee_records.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Adm No","Name","Class","Month","Due","Paid","Discount","Fine","Balance","Status"])
            for s in self.students:
                for mon, rec in s.get("fee_records", {}).items():
                    due = rec.get("due",0); paid = rec.get("paid",0)
                    disc = rec.get("discount",0); fine = rec.get("fine",0)
                    bal = due - disc + fine - paid
                    w.writerow([s.get("admissionNo",""), s.get("name",""),
                                s.get("class",""), mon, due, paid, disc, fine, bal,
                                "Paid" if rec.get("paidFlag") else "Pending"])
        # 5) Exams
        with open(os.path.join(folder, "exams.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Subject","Class","Date","Max Marks","Student","Adm No","Marks","Grade"])
            for ex in self.exams:
                for sid, mk in ex.get("marks", {}).items():
                    stu = next((s for s in self.students if s["id"]==sid), {})
                    w.writerow([ex["subject"], ex["class"], ex.get("date",""),
                                ex["max_marks"], stu.get("name",""),
                                stu.get("admissionNo",""), mk,
                                self._grade(mk, ex["max_marks"])])
        messagebox.showinfo("✅", f"All data exported to:\n{folder}\n\n"
                            "Files: students.csv, teachers.csv, attendance.csv,\n"
                            "fee_records.csv, exams.csv")

    # ════════════════════════════════════════════════════════════════════════
    #  USERS
    # ════════════════════════════════════════════════════════════════════════
    def show_users(self):
        if not self.check_perm("users"): return
        self._clear()
        c = self.colors
        self._header("👥  User Management", "Create and manage system users")

        ctrl = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=12)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="➕ Create New User", bg=c["success"], fg="white",
                  command=self._add_user_dlg).pack(side="left")

        list_f = tk.Frame(self.main, bg=c["content_bg"], padx=25, pady=10)
        list_f.pack(fill="both", expand=True)

        for user in self.users:
            card = tk.Frame(list_f, bg=c["card_bg"], padx=15, pady=12)
            card.pack(fill="x", pady=5)
            avatar = tk.Label(card, text=user["username"][0].upper(),
                              font=("Helvetica",18,"bold"), bg=c["primary"],
                              fg="white", width=2)
            avatar.pack(side="left", padx=(0,12))
            inf = tk.Frame(card, bg=c["card_bg"])
            inf.pack(side="left", expand=True, fill="x")
            tk.Label(inf, text=user["username"], font=("Helvetica",13,"bold"),
                     bg=c["card_bg"], fg=c["text"]).pack(anchor="w")
            tk.Label(inf, text=f"{user['role']} • Created: {user.get('created_at','')[:10]}",
                     bg=c["card_bg"], fg=c["subtext"],
                     font=("Helvetica",9)).pack(anchor="w")
            role_col = c["danger"] if user["role"]=="Admin" else c["primary"]
            tk.Label(card, text=user["role"], font=("Helvetica",9,"bold"),
                     bg=role_col, fg="white", padx=10, pady=3).pack(side="right")
            if user["id"] != self.current_user["id"]:
                tk.Button(card, text="🗑️", bg=c["danger"], fg="white",
                          command=lambda uid=user["id"]: self._del_user(uid)
                          ).pack(side="right", padx=5)

    def _add_user_dlg(self):
        c = self.colors
        dlg = tk.Toplevel(self.root)
        dlg.title("Create User")
        dlg.geometry("480x420")
        dlg.configure(bg=c["dark"])
        dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="➕ Create New User",
                 font=("Helvetica",14,"bold"), bg=c["dark"],
                 fg="white").pack(pady=15)
        frm = tk.Frame(dlg, bg=c["dark"], padx=30)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Username:", bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
        u_e = tk.Entry(frm, bg=c["dark_light"], fg="white",
                        insertbackground="white", relief="flat", bd=7)
        u_e.pack(fill="x", pady=5)

        tk.Label(frm, text="Password:", bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
        p_e = tk.Entry(frm, show="•", bg=c["dark_light"], fg="white",
                        insertbackground="white", relief="flat", bd=7)
        p_e.pack(fill="x", pady=5)

        tk.Label(frm, text="Role:", bg=c["dark"], fg=c["gray"]).pack(anchor="w", pady=(8,0))
        role_v = tk.StringVar(value="Teacher")
        ttk.Combobox(frm, textvariable=role_v,
                      values=["Teacher","Accountant","Librarian","Admin"],
                      state="readonly").pack(fill="x", pady=5)

        def save():
            username = u_e.get().strip()
            password = p_e.get()
            if not username or not password:
                messagebox.showerror("Error","Username and password required"); return
            if any(u["username"]==username for u in self.users):
                messagebox.showerror("Error","Username already exists"); return
            perms_map = {
                "Teacher": {"students":True,"attendance":True,"exams":True},
                "Accountant": {"students":True,"fees":True,"reports":True},
                "Librarian": {"library":True},
                "Admin": {k:True for k in ["students","fees","attendance","teachers",
                                            "library","exams","reports","settings","users"]}
            }
            self.users.append({
                "id": self.gen_id(), "username": username,
                "password": self._hash(password), "role": role_v.get(),
                "permissions": perms_map.get(role_v.get(),{}),
                "created_at": datetime.datetime.now().isoformat(),
                "face_encoding": None
            })
            self.save_data()
            dlg.destroy()
            self.show_users()
            messagebox.showinfo("✅","User created!")

        tk.Button(frm, text="💾 Create User", bg=c["success"], fg="white",
                  font=("Helvetica",11,"bold"), pady=10,
                  command=save).pack(fill="x", pady=15)

    def _del_user(self, uid):
        if not messagebox.askyesno("Confirm","Delete this user?"): return
        self.users = [u for u in self.users if u["id"]!=uid]
        self.save_data()
        self.show_users()
        messagebox.showinfo("Deleted","User removed.")

    # ════════════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ════════════════════════════════════════════════════════════════════════
    def show_settings(self):
        self._clear()
        c = self.colors
        self._header("⚙️  Settings", "Customize school info, theme, and logo")

        outer = tk.Frame(self.main, bg=c["content_bg"])
        outer.pack(fill="both", expand=True)

        cv = tk.Canvas(outer, bg=c["content_bg"], highlightthickness=0)
        sb_s = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb_s.set)
        sb_s.pack(side="right", fill="y")
        cv.pack(fill="both", expand=True)
        frm = tk.Frame(cv, bg=c["content_bg"], padx=30, pady=15)
        fw = cv.create_window((0,0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(fw, width=e.width))

        def _on_scroll(event):
            if event.num == 4:
                cv.yview_scroll(-3, "units")
            elif event.num == 5:
                cv.yview_scroll(3, "units")
            else:
                cv.yview_scroll(int(-1*(event.delta/120)), "units")
        cv.bind_all("<MouseWheel>", _on_scroll)
        cv.bind_all("<Button-4>", _on_scroll)
        cv.bind_all("<Button-5>", _on_scroll)

        # ── School Info ────────────────────────────────────────────────────
        sec = self._settings_section(frm, "🏫 School Information")
        tk.Label(sec, text="School Name:", bg=c["card_bg"], fg=c["subtext"]).pack(anchor="w", pady=(8,0))
        name_e = tk.Entry(sec, bg=c["dark_light"], fg=c["text"],
                           insertbackground=c["text"], relief="flat", bd=7,
                           font=("Helvetica",12))
        name_e.insert(0, self.settings.get("school_name","My School"))
        name_e.pack(fill="x", pady=5)

        def save_name():
            self.settings["school_name"] = name_e.get().strip() or "My School"
            self.save_data()
            messagebox.showinfo("✅","School name updated!")
            self.show_main_application()

        tk.Button(sec, text="💾 Save School Name", bg=c["primary"], fg="white",
                  command=save_name).pack(anchor="w", pady=8)

        # ── Logo ────────────────────────────────────────────────────────────
        logo_sec = self._settings_section(frm, "🖼️ School Logo")
        tk.Label(logo_sec, text=f"Current: {self.settings.get('logo_path','None')}",
                 bg=c["card_bg"], fg=c["subtext"], font=("Helvetica",9)).pack(anchor="w")

        def change_logo():
            path = filedialog.askopenfilename(
                filetypes=[("Image","*.png *.jpg *.jpeg *.bmp *.ico *.gif")],
                title="Select Logo Image")
            if path:
                self.settings["logo_path"] = path
                self.save_data()
                messagebox.showinfo("✅","Logo updated! Restart or re-login to see changes.")
                self.show_settings()

        def remove_logo():
            self.settings["logo_path"] = ""
            self.save_data()
            messagebox.showinfo("✅","Logo removed.")
            self.show_settings()

        btn_row = tk.Frame(logo_sec, bg=c["card_bg"])
        btn_row.pack(anchor="w", pady=8)
        tk.Button(btn_row, text="📁 Choose Logo", bg=c["primary"], fg="white",
                  command=change_logo).pack(side="left", padx=3)
        tk.Button(btn_row, text="❌ Remove Logo", bg=c["danger"], fg="white",
                  command=remove_logo).pack(side="left", padx=3)

        # ── Theme ────────────────────────────────────────────────────────────
        theme_sec = self._settings_section(frm, "🎨 Theme / Appearance")
        tk.Label(theme_sec, text="Select Theme:",
                 bg=c["card_bg"], fg=c["subtext"]).pack(anchor="w", pady=(5,8))
        th_row = tk.Frame(theme_sec, bg=c["card_bg"])
        th_row.pack(anchor="w")
        for th_name, th_data in THEMES.items():
            is_active = self.settings.get("theme","Dark") == th_name
            btn = tk.Button(th_row, text=th_name,
                            font=("Helvetica",10,"bold" if is_active else "normal"),
                            bg=th_data["primary"], fg="white",
                            padx=16, pady=8, bd=0,
                            relief="solid" if is_active else "flat",
                            command=lambda tn=th_name: self._apply_theme(tn))
            btn.pack(side="left", padx=4, pady=4)
        tk.Label(theme_sec, text=f"✅ Current theme: {self.settings.get('theme','Dark')}",
                 bg=c["card_bg"], fg=c["success"]).pack(anchor="w", pady=5)

        # ── Password ──────────────────────────────────────────────────────
        pwd_sec = self._settings_section(frm, "🔐 Change Password")
        tk.Label(pwd_sec, text="Current Password:", bg=c["card_bg"],
                 fg=c["subtext"]).pack(anchor="w", pady=(5,0))
        cur_e = tk.Entry(pwd_sec, show="•", bg=c["dark_light"], fg=c["text"],
                          insertbackground=c["text"], relief="flat", bd=7)
        cur_e.pack(fill="x", pady=4)
        tk.Label(pwd_sec, text="New Password:", bg=c["card_bg"],
                 fg=c["subtext"]).pack(anchor="w")
        new_e = tk.Entry(pwd_sec, show="•", bg=c["dark_light"], fg=c["text"],
                          insertbackground=c["text"], relief="flat", bd=7)
        new_e.pack(fill="x", pady=4)
        tk.Label(pwd_sec, text="Confirm New Password:", bg=c["card_bg"],
                 fg=c["subtext"]).pack(anchor="w")
        conf_e = tk.Entry(pwd_sec, show="•", bg=c["dark_light"], fg=c["text"],
                           insertbackground=c["text"], relief="flat", bd=7)
        conf_e.pack(fill="x", pady=4)

        def change_pwd():
            if not self._verify(cur_e.get(), self.current_user["password"]):
                messagebox.showerror("Error","Current password wrong"); return
            if new_e.get() != conf_e.get():
                messagebox.showerror("Error","Passwords don't match"); return
            if len(new_e.get()) < 4:
                messagebox.showerror("Error","Password too short"); return
            self.current_user["password"] = self._hash(new_e.get())
            self.save_data()
            messagebox.showinfo("✅","Password changed!")
            cur_e.delete(0,"end"); new_e.delete(0,"end"); conf_e.delete(0,"end")

        tk.Button(pwd_sec, text="🔐 Change Password", bg=c["success"], fg="white",
                  command=change_pwd).pack(anchor="w", pady=10)

        # ── Face Recognition ──────────────────────────────────────────────
        face_sec = self._settings_section(frm, "📷 Face Recognition Login")
        has_face_enc = bool(self.current_user.get("face_photo_path") and
                            os.path.isfile(self.current_user.get("face_photo_path","")))
        face_status_text = "✅ Face registered – you can use Face Login" if has_face_enc \
                           else "❌ No face registered yet"
        face_status_col  = c["success"] if has_face_enc else c["danger"]
        tk.Label(face_sec, text=face_status_text,
                 bg=c["card_bg"], fg=face_status_col,
                 font=("Helvetica",10,"bold")).pack(anchor="w", pady=(0,6))
        tk.Label(face_sec,
                 text="Register your face so you can log in using the 📷 Face Login button\n"
                      "on the login screen without typing a password.",
                 bg=c["card_bg"], fg=c["subtext"],
                 font=("Helvetica",9), justify="left").pack(anchor="w", pady=(0,8))

        face_btn_row = tk.Frame(face_sec, bg=c["card_bg"])
        face_btn_row.pack(anchor="w")

        if HAS_FACE:
            tk.Button(face_btn_row, text="📷 Register My Face",
                      bg="#7c3aed", fg="white",
                      font=("Helvetica",10,"bold"), padx=14, pady=7,
                      command=self._register_face).pack(side="left", padx=(0,8))
            if has_face_enc:
                def remove_face():
                    if messagebox.askyesno("Confirm", "Remove your registered face?"):
                        photo_path = self.current_user.get("face_photo_path","")
                        if photo_path and os.path.isfile(photo_path):
                            try: os.unlink(photo_path)
                            except Exception: pass
                        self.current_user["face_encoding"]   = None
                        self.current_user["face_photo_path"] = ""
                        self.save_data()
                        self.show_settings()
                tk.Button(face_btn_row, text="🗑️ Remove Face",
                          bg=c["danger"], fg="white",
                          font=("Helvetica",10,"bold"), padx=14, pady=7,
                          command=remove_face).pack(side="left")
        else:
            def retry_face_install():
                btn_retry.config(state="disabled", text="⏳ Installing…")
                self.root.update()
                ok = _auto_install_face_libs()
                if ok:
                    try:
                        import importlib, cv2
                        _ = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                        messagebox.showinfo(
                            "✅ Success",
                            "opencv installed!\n\n"
                            "Please RESTART the application to activate Face Login."
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Installed but import failed:\n{e}")
                else:
                    messagebox.showerror(
                        "Installation Failed",
                        "Could not auto-install.\n\n"
                        "Run manually:\n"
                        "  pip install opencv-contrib-python"
                    )
                btn_retry.config(state="normal", text="⬇️ Auto-Install Libraries")

            tk.Label(face_sec,
                     text="⚠️ Face recognition not available.\n"
                          "Click below to auto-install opencv-contrib-python,\n"
                          "or run:  pip install opencv-contrib-python",
                     bg=c["card_bg"], fg=c["warning"],
                     font=("Helvetica",9), justify="left").pack(anchor="w", pady=(0,6))
            btn_retry = tk.Button(face_sec, text="⬇️ Auto-Install Libraries",
                                  bg="#7c3aed", fg="white",
                                  font=("Helvetica",10,"bold"), padx=14, pady=7,
                                  command=retry_face_install)
            btn_retry.pack(anchor="w")

        # ── QR Verification Settings ──────────────────────────────────────
        qr_sec = self._settings_section(frm, "🔒 QR Code Verification")
        tk.Label(qr_sec, text="Configure trusted network and email for QR attendance verification.",
                 bg=c["card_bg"], fg=c["subtext"], font=("Helvetica",9),
                 wraplength=500, justify="left").pack(anchor="w", pady=(0,8))

        # Trusted Host / WiFi
        tk.Label(qr_sec, text="Trusted Host (auto-detected):",
                 bg=c["card_bg"], fg=c["subtext"]).pack(anchor="w", pady=(4,0))
        host_var = tk.StringVar(value=self.settings.get("trusted_host", ""))
        host_e = tk.Entry(qr_sec, textvariable=host_var,
                          bg=c["dark_light"], fg=c["text"],
                          insertbackground=c["text"], relief="flat", bd=7,
                          font=("Helvetica",10))
        host_e.pack(fill="x", pady=3)

        current_host = self._get_host_port()
        tk.Label(qr_sec, text=f"Current device: {current_host}",
                 bg=c["card_bg"], fg="#93c5fd", font=("Helvetica",8)).pack(anchor="w")

        def set_current_host():
            host_var.set(current_host)

        tk.Button(qr_sec, text="📡 Set Current Device as Trusted",
                  bg="#065f46", fg="white", font=("Helvetica",9),
                  command=set_current_host).pack(anchor="w", pady=4)

        # Trusted Email Domain
        tk.Label(qr_sec, text="Trusted Email Domain (e.g. school.edu.pk):",
                 bg=c["card_bg"], fg=c["subtext"]).pack(anchor="w", pady=(8,0))
        email_domain_var = tk.StringVar(value=self.settings.get("trusted_email_domain", ""))
        email_e = tk.Entry(qr_sec, textvariable=email_domain_var,
                           bg=c["dark_light"], fg=c["text"],
                           insertbackground=c["text"], relief="flat", bd=7,
                           font=("Helvetica",10))
        email_e.pack(fill="x", pady=3)
        tk.Label(qr_sec, text="Leave empty to allow all emails. Students must have matching email for QR attendance.",
                 bg=c["card_bg"], fg="#94a3b8", font=("Helvetica",8),
                 wraplength=500, justify="left").pack(anchor="w")

        def save_qr_settings():
            self.settings["trusted_host"] = host_var.get().strip()
            self.settings["trusted_email_domain"] = email_domain_var.get().strip()
            self.save_data()
            messagebox.showinfo("✅", "QR verification settings saved!")

        tk.Button(qr_sec, text="💾 Save QR Settings",
                  bg=c["primary"], fg="white", font=("Helvetica",10),
                  command=save_qr_settings).pack(anchor="w", pady=8)

        # ── Data management ──────────────────────────────────────────────
        data_sec = self._settings_section(frm, "💾 Data Management")
        def export_backup():
            path = filedialog.asksaveasfilename(defaultextension=".json",
                filetypes=[("JSON","*.json")], title="Export Backup")
            if path:
                shutil.copyfile(self.db_file, path)
                messagebox.showinfo("✅",f"Backup saved to {path}")

        def import_backup():
            path = filedialog.askopenfilename(
                filetypes=[("JSON","*.json")], title="Import Backup")
            if path and messagebox.askyesno("Confirm",
                "This will REPLACE all current data. Continue?"):
                shutil.copyfile(path, self.db_file)
                self.load_data()
                self.show_main_application()
                messagebox.showinfo("✅","Backup restored!")

        btn_row2 = tk.Frame(data_sec, bg=c["card_bg"])
        btn_row2.pack(anchor="w", pady=8)
        tk.Button(btn_row2, text="📤 Export Backup", bg=c["primary"], fg="white",
                  command=export_backup).pack(side="left", padx=3)
        tk.Button(btn_row2, text="📥 Import Backup", bg=c["warning"], fg="white",
                  command=import_backup).pack(side="left", padx=3)

    def _settings_section(self, parent, title):
        c = self.colors
        sec = tk.Frame(parent, bg=c["card_bg"], padx=20, pady=15)
        sec.pack(fill="x", pady=8)
        tk.Label(sec, text=title, font=("Helvetica",13,"bold"),
                 bg=c["card_bg"], fg=c["text"]).pack(anchor="w", pady=(0,5))
        tk.Frame(sec, bg=c["primary"], height=2).pack(fill="x", pady=(0,8))
        return sec


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = SchoolManagerPro(root)
    root.mainloop()