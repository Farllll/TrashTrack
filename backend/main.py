
import asyncio
import io
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent.resolve()
MODEL_PATH      = ROOT_DIR / "model" / "yolo_best.pt"   # model klasifikasi 4 kategori
CONF_THRESHOLD  = 60.0    # di bawah nilai ini, prediksi ditandai kurang yakin
MIN_CONFIDENCE  = 20.0    # di bawah nilai ini, gambar TIDAK diklasifikasikan (dianggap bukan sampah dikenal)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH} — jalankan 2_training.py dulu")

_model = YOLO(MODEL_PATH)
print(f"  Model dimuat: {MODEL_PATH.name} | kategori: {list(_model.names.values())}")


# ─────────────────────────────────────────────────
#  PREPROCESSING: CLAHE
# ─────────────────────────────────────────────────
def enhance_image(img: Image.Image) -> Image.Image:
    """Tingkatkan kontras pakai CLAHE (channel L, color space LAB)."""
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr[:, :, 0] = clahe.apply(arr[:, :, 0])
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))


# ─────────────────────────────────────────────────
#  AUGMENTASI SAAT INFERENSI / TTA (5 variasi)
# ─────────────────────────────────────────────────
def _run_model(pil_img: Image.Image) -> np.ndarray:
    res = _model(pil_img, imgsz=224, verbose=False)
    return res[0].probs.data.cpu().numpy()

def classify_with_tta(img: Image.Image) -> tuple[str, float, np.ndarray]:
    """Jalankan model 5x pada variasi gambar lalu rata-rata probabilitasnya (TTA)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    crop_margin = int(min(h, w) * 0.10)

    variants: list[Image.Image] = [
        img,                                                              # gambar asli
        Image.fromarray(cv2.flip(arr, 1)),                               # cermin horizontal
        Image.fromarray(np.clip(arr.astype(np.float32)*1.25,0,255).astype(np.uint8)),  # lebih terang
        Image.fromarray(np.clip(arr.astype(np.float32)*0.78,0,255).astype(np.uint8)),  # lebih gelap
        Image.fromarray(arr[crop_margin:h-crop_margin, crop_margin:w-crop_margin]).resize((224,224)),  # potong tengah
    ]

    all_probs = np.stack([_run_model(v) for v in variants])
    avg = all_probs.mean(axis=0)
    top1 = int(np.argmax(avg))
    return _model.names[top1], float(avg[top1]) * 100, avg


# ─────────────────────────────────────────────────
#  INFORMASI TIAP KATEGORI (4 kategori)
# ─────────────────────────────────────────────────
TRASH_INFO: dict[str, dict] = {
    "anorganik": {
        "label": "Anorganik", "code": "ANO", "category": "Anorganik",
        "decompose": "Puluhan hingga ratusan tahun", "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Sampah anorganik seperti plastik, kaca, logam, dan kertas umumnya bisa "
               "didaur ulang. Bersihkan dari sisa makanan lalu setor ke bank sampah.",
    },
    "organik": {
        "label": "Organik", "code": "ORG", "category": "Organik",
        "decompose": "Beberapa minggu sampai bulan", "bin": "Hijau", "binHex": "#10B981",
        "tip": "Sampah organik seperti sisa makanan, daun, dan ranting bisa dijadikan "
               "kompos. Pisahkan dari sampah lain agar mudah terurai.",
    },
    "b3": {
        "label": "B3 (Bahan Berbahaya & Beracun)", "code": "B3", "category": "B3",
        "decompose": "Sulit terurai & berbahaya", "bin": "Merah", "binHex": "#EF4444",
        "tip": "Limbah B3 seperti baterai, elektronik, lampu, obat, dan bahan kimia "
               "mengandung zat berbahaya. Jangan buang ke tempat sampah biasa — "
               "serahkan ke drop box B3 atau fasilitas pengelolaan limbah khusus.",
    },
    "residu": {
        "label": "Residu", "code": "RSD", "category": "Residu",
        "decompose": "Sulit terurai", "bin": "Abu-abu", "binHex": "#6B7280",
        "tip": "Sampah residu seperti popok, pembalut, dan puntung rokok tidak dapat "
               "didaur ulang maupun dikompos. Bungkus rapat dan buang ke tempat sampah residu.",
    },
}

_FALLBACK = {
    "label": "Sampah Tidak Dikenali", "code": "UNK", "category": "Anorganik",
    "decompose": "bervariasi", "bin": "Hitam", "binHex": "#6B7280",
    "tip": "Sampah tidak dapat dikenali dengan yakin. Coba foto ulang dengan pencahayaan "
           "lebih baik dan objek lebih jelas.",
}

# ─────────────────────────────────────────────────
#  STATUS PIPELINE
# ─────────────────────────────────────────────────
_pipeline: dict = {
    "status": "idle",   # idle | running | done | error
    "mode":   "",
    "step":   "",       # preprocess | train
    "logs":   [],
}


def _blocking_run(args: list[str], cwd: str) -> tuple[int, list[str]]:
    """Jalankan subprocess secara sinkron dan kumpulkan semua output-nya."""
    lines: list[str] = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONLEGACYWINDOWSSTDIO": "0"}
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in proc.stdout:
        lines.append(line.rstrip())
    proc.wait()
    return proc.returncode, lines


async def _run_pipeline(mode: str) -> None:
    """Background task: preprocessing lalu training. Update state _pipeline langsung."""
    global _model

    def push(msg: str) -> None:
        _pipeline["logs"].append(msg)

    py           = sys.executable
    pipeline_dir = ROOT_DIR / "pipeline"

    try:
        # ── Step 1: Preprocessing ──────────────────────────────────────────
        push("[1/2] Mulai preprocessing...")
        _pipeline["step"] = "preprocess"

        rc, logs = await asyncio.to_thread(
            _blocking_run,
            [py, str(pipeline_dir / "1_preprocessing.py")],
            str(ROOT_DIR),
        )
        _pipeline["logs"].extend(logs)
        if rc != 0:
            push(f"[ERROR] Preprocessing gagal (exit code {rc})")
            _pipeline["status"] = "error"
            return
        push("[1/2] Preprocessing selesai.")

        # ── Step 2: Training ───────────────────────────────────────────────
        push(f"[2/2] Mulai training ({mode})...")
        _pipeline["step"] = "train"

        train_args = [py, str(pipeline_dir / "2_training.py")]
        if mode == "finetune":
            train_args += ["--finetune", "--epochs", "30"]

        rc, logs = await asyncio.to_thread(_blocking_run, train_args, str(ROOT_DIR))
        _pipeline["logs"].extend(logs)
        if rc != 0:
            push(f"[ERROR] Training gagal (exit code {rc})")
            _pipeline["status"] = "error"
            return
        push("[2/2] Training selesai.")

        # ── Muat ulang model tanpa perlu restart backend ─────────────────────
        push("[RELOAD] Memuat ulang model ke memori...")
        try:
            _model = YOLO(MODEL_PATH)
            push("[DONE] Model berhasil dimuat. Backend siap digunakan.")
        except Exception as exc:
            push(f"[WARN] Gagal memuat ulang model: {exc}. Restart backend secara manual.")

        _pipeline.update({"status": "done", "step": ""})

    except Exception as exc:
        push(f"[ERROR] Terjadi kesalahan tak terduga pada pipeline: {exc}")
        _pipeline["status"] = "error"


# ─────────────────────────────────────────────────
#  APLIKASI
# ─────────────────────────────────────────────────
app = FastAPI(title="TrashTrack API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Periksa status backend dan model."""
    return {"status": "ok", "model": MODEL_PATH.name, "classes": len(TRASH_INFO),
            "kategori": list(TRASH_INFO.keys())}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Terima gambar sampah, klasifikasikan ke 4 kategori, kembalikan hasilnya sebagai JSON."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (image/*)")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Gambar tidak dapat dibaca")

    # ── Langkah 1: Peningkatan kontras dengan CLAHE ──────────────────────
    img = enhance_image(img)

    # ── Langkah 2: Klasifikasi dengan TTA (5 variasi gambar) ─────────────
    kategori, confidence, avg_probs = classify_with_tta(img)
    confidence = round(confidence, 2)

    # ── Tiga kemungkinan teratas ─────────────────────────────────────────
    top3 = sorted(
        [{"lvl2": _model.names[i], "confidence": round(float(avg_probs[i])*100, 2)}
         for i in range(len(_model.names))],
        key=lambda x: -x["confidence"]
    )[:3]

    # Bila keyakinan terlalu rendah, jangan paksa klasifikasi ke kategori apa pun.
    if confidence < MIN_CONFIDENCE:
        return {
            **_FALLBACK,
            "confidence"     : confidence,
            "low_confidence" : True,
            "unknown"        : True,
            "lvl2"           : "unknown",
            "lvl1"           : "unknown",
            "top3"           : top3,
            "method"         : "TTA",
        }

    info = TRASH_INFO.get(kategori, _FALLBACK)
    return {
        **info,
        "confidence"     : confidence,
        "low_confidence" : confidence < CONF_THRESHOLD,
        "unknown"        : False,
        "lvl2"           : kategori,
        "lvl1"           : kategori,
        "top3"           : top3,
        "method"         : "TTA",
    }


# ─────────────────────────────────────────────────
#  POST /pipeline/run — PREPROCESSING + TRAINING
# ─────────────────────────────────────────────────
@app.post("/pipeline/run")
async def pipeline_run(mode: str = "finetune"):
    """
    Jalankan pipeline: preprocessing dataset/raw lalu training model.

    mode: 'finetune' : lanjut training 30 epoch dari model yang ada (lebih cepat)
          'full'     : training dari awal (lebih akurat)
    """
    if _pipeline["status"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline sedang berjalan")
    if mode not in ("finetune", "full"):
        raise HTTPException(status_code=400, detail="mode harus 'finetune' atau 'full'")

    _pipeline.update({"status": "running", "mode": mode, "step": "", "logs": []})
    asyncio.create_task(_run_pipeline(mode))
    return {"status": "started", "mode": mode}


# ─────────────────────────────────────────────────
#  GET /pipeline/status — STATUS & LOG PIPELINE
# ─────────────────────────────────────────────────
@app.get("/pipeline/status")
def pipeline_status():
    """Kembalikan status pipeline dan semua log-nya."""
    return {
        "status": _pipeline["status"],
        "mode":   _pipeline["mode"],
        "step":   _pipeline["step"],
        "logs":   list(_pipeline["logs"]),
    }
