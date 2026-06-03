"""
====================================================
  BACKEND TRASHTRACK — FastAPI
  ────────────────────────────────────────────────
  Menerima gambar yang diunggah, menjalankan model
  klasifikasi YOLO, lalu mengembalikan informasi
  sampah dalam bentuk JSON ke frontend React.

  Daftar endpoint:
    GET  /health    : memeriksa status backend
    POST /classify  : menerima gambar (field "file")

  Cara menjalankan (dari folder TrashTrack_UI/):
    pip install -r backend/requirements.txt
    uvicorn backend.main:app --reload --port 8000
====================================================
"""

import asyncio
import io
import json
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
MODEL_PATH      = ROOT_DIR / "model" / "yolo_best.pt"     # Level 2 (kelas spesifik)
LVL1_MODEL_PATH = ROOT_DIR / "model" / "yolo_lvl1.pt"     # Level 1 (4 kategori)
MAPPING_PATH    = ROOT_DIR / "dataset" / "label_mapping.json"
CONF_THRESHOLD  = 60.0    # di bawah nilai ini, prediksi ditandai kurang yakin

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
if not MAPPING_PATH.exists():
    raise FileNotFoundError(f"Berkas pemetaan label tidak ditemukan: {MAPPING_PATH}")

_model      = YOLO(MODEL_PATH)
_lvl1_model = YOLO(LVL1_MODEL_PATH) if LVL1_MODEL_PATH.exists() else None

if _lvl1_model:
    print(f"  Model Level 1 berhasil dimuat: {LVL1_MODEL_PATH.name}")
else:
    print(f"  Model Level 1 belum ada ({LVL1_MODEL_PATH.name}) — jalankan 2_pelatihan_yolo.py")

with open(MAPPING_PATH, encoding="utf-8") as f:
    _mapping = json.load(f)
_sub_to_lvl1: dict[str, str]       = _mapping["sub_to_lvl1"]
_hierarchy:   dict[str, list[str]] = _mapping["hierarchy"]   # lvl1 → [lvl2, ...]

# Pemetaan nama kelas Level 2 ke indeksnya di dalam model
_lvl2_name_to_idx: dict[str, int] = {v: k for k, v in _model.names.items()}


# ─────────────────────────────────────────────────
#  PRA-PEMROSESAN: PENINGKATAN KONTRAS CLAHE
# ─────────────────────────────────────────────────
def enhance_image(img: Image.Image) -> Image.Image:
    """Tingkatkan kontras gambar webcam dengan CLAHE pada channel L (ruang warna LAB)."""
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr[:, :, 0] = clahe.apply(arr[:, :, 0])
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))


# ─────────────────────────────────────────────────
#  AUGMENTASI SAAT INFERENSI / TTA (5 variasi)
# ─────────────────────────────────────────────────
def _run_model(model: YOLO, pil_img: Image.Image) -> np.ndarray:
    res = model(pil_img, imgsz=224, verbose=False)
    return res[0].probs.data.cpu().numpy()

def classify_with_tta(img: Image.Image) -> tuple[str, float, np.ndarray]:
    """
    Jalankan model sebanyak 5 kali pada variasi gambar yang berbeda,
    lalu rata-ratakan probabilitasnya. Cara ini membuat prediksi lebih
    stabil untuk gambar yang ambigu atau dengan pencahayaan tidak merata.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    crop_margin = int(min(h, w) * 0.10)

    variants: list[Image.Image] = [
        img,                                                              # gambar asli
        Image.fromarray(cv2.flip(arr, 1)),                               # cermin horizontal
        Image.fromarray(np.clip(arr.astype(np.float32)*1.25,0,255).astype(np.uint8)),  # lebih terang
        Image.fromarray(np.clip(arr.astype(np.float32)*0.78,0,255).astype(np.uint8)),  # lebih gelap
        Image.fromarray(arr[crop_margin:h-crop_margin, crop_margin:w-crop_margin]).resize((224,224)),  # potong bagian tengah
    ]

    all_probs = np.stack([_run_model(_model, v) for v in variants])
    avg = all_probs.mean(axis=0)
    top1 = int(np.argmax(avg))
    return _model.names[top1], float(avg[top1]) * 100, avg


# ─────────────────────────────────────────────────
#  INFERENSI HIERARKI: Level 1 → Level 2
# ─────────────────────────────────────────────────
def classify_hierarchical(img: Image.Image) -> tuple[str, float, str, float]:
    """
    Klasifikasi dilakukan dalam dua tahap:
      1. Model Level 1 menentukan kategori besar (anorganik/organik/b3/residu).
      2. Model Level 2 dengan TTA menentukan jenis spesifik di dalam kategori itu.
    Mengembalikan: (nama_lvl2, keyakinan_lvl2, nama_lvl1, keyakinan_lvl1)
    """
    # ── Tahap 1: Level 1 ─────────────────────────────────────────────────
    l1_probs = _run_model(_lvl1_model, img)
    l1_top1  = int(np.argmax(l1_probs))
    lvl1_pred = _lvl1_model.names[l1_top1]
    lvl1_conf = float(l1_probs[l1_top1]) * 100

    # ── Tahap 2: Level 2 dengan TTA ──────────────────────────────────────
    lvl2_raw, lvl2_conf_raw, all_probs = classify_with_tta(img)

    # Daftar kandidat Level 2 yang sesuai dengan hasil Level 1
    candidates = _hierarchy.get(lvl1_pred, [])

    if lvl2_raw in candidates:
        # Prediksi Level 2 sudah konsisten dengan Level 1, langsung dipakai
        return lvl2_raw, lvl2_conf_raw, lvl1_pred, lvl1_conf

    # Jika tidak konsisten, pilih kelas Level 2 terbaik di dalam kategori Level 1
    best_cls, best_score = lvl2_raw, -1.0
    for cls in candidates:
        idx = _lvl2_name_to_idx.get(cls)
        if idx is not None and float(all_probs[idx]) > best_score:
            best_score = float(all_probs[idx])
            best_cls   = cls

    return best_cls, best_score * 100, lvl1_pred, lvl1_conf


# ─────────────────────────────────────────────────
#  INFORMASI TIAP KELAS (4 kategori · 21 jenis sampah)
# ─────────────────────────────────────────────────
TRASH_INFO: dict[str, dict] = {
    # ── ANORGANIK ──────────────────────────────────
    "kaca": {
        "label": "Kaca", "code": "ANO-01", "category": "Anorganik",
        "decompose": "1 juta tahun",  "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Kaca dapat didaur ulang tanpa batas tanpa kehilangan kualitas. "
               "Bersihkan dan setor ke bank sampah terdekat.",
    },
    "karet": {
        "label": "Karet", "code": "ANO-02", "category": "Anorganik",
        "decompose": "50–80 tahun",   "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Karet bekas (ban, sandal) bisa didaur ulang menjadi bahan lantai atau aspal. "
               "Hubungi fasilitas daur ulang karet di kota kamu.",
    },
    "logam": {
        "label": "Logam / Kaleng", "code": "ANO-03", "category": "Anorganik",
        "decompose": "50–200 tahun",  "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Kaleng aluminium hemat 95% energi dibanding produksi baru. "
               "Cuci sebelum dibuang atau jual ke pengepul besi.",
    },
    "styrofoam": {
        "label": "Styrofoam", "code": "ANO-04", "category": "Anorganik",
        "decompose": "lebih dari 500 tahun", "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Styrofoam sangat sulit terurai dan tidak bisa didaur ulang biasa. "
               "Minimalkan penggunaan dan cari titik pengumpulan styrofoam khusus.",
    },
    "kardus": {
        "label": "Kardus", "code": "ANO-05", "category": "Anorganik",
        "decompose": "2–6 bulan",     "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Lipat kardus agar hemat tempat. Lepas selotip dan lapisan plastik "
               "sebelum disetor ke bank sampah.",
    },
    "plastik": {
        "label": "Plastik", "code": "ANO-06", "category": "Anorganik",
        "decompose": "20–500 tahun",  "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Periksa kode daur ulang di bawah kemasan (segitiga berangka 1 sampai 7). "
               "Bersihkan dari sisa makanan sebelum dibuang ke bank sampah.",
    },
    "sepatu": {
        "label": "Sepatu", "code": "ANO-07", "category": "Anorganik",
        "decompose": "25–40 tahun",   "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Sepatu yang masih layak pakai bisa didonasikan. Sepatu rusak berbahan campuran "
               "sulit didaur ulang, jadi setor ke bank sampah atau titik pengumpulan tekstil.",
    },
    "pakaian": {
        "label": "Pakaian", "code": "ANO-08", "category": "Anorganik",
        "decompose": "1–5 tahun",     "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Pakaian yang masih layak pakai sebaiknya didonasikan ke panti asuhan atau bank pakaian. "
               "Pakaian tidak layak bisa dijadikan kain lap atau disetor ke daur ulang tekstil.",
    },
    "topi": {
        "label": "Topi", "code": "ANO-09", "category": "Anorganik",
        "decompose": "1–5 tahun",     "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Topi berbahan kain bisa didonasikan selama masih layak. "
               "Topi dengan rangka plastik atau logam perlu dipisahkan dulu sebelum didaur ulang.",
    },
    "tas": {
        "label": "Tas", "code": "ANO-10", "category": "Anorganik",
        "decompose": "5–20 tahun",    "bin": "Kuning", "binHex": "#F59E0B",
        "tip": "Tas yang masih layak pakai bisa didonasikan. Tas berbahan kulit sintetis sulit terurai, "
               "jadi setor ke bank sampah atau titik pengumpulan tekstil terdekat.",
    },

    # ── ORGANIK ────────────────────────────────────
    "ampas": {
        "label": "Ampas Organik", "code": "ORG-01", "category": "Organik",
        "decompose": "1–4 minggu",    "bin": "Hijau",  "binHex": "#10B981",
        "tip": "Ampas kopi, teh, dan buah sangat baik untuk kompos. "
               "Campur dengan bahan hijau lain agar pengomposan lebih cepat.",
    },
    "kayu": {
        "label": "Kayu Bekas", "code": "ORG-02", "category": "Organik",
        "decompose": "2–3 tahun",     "bin": "Hijau",  "binHex": "#10B981",
        "tip": "Kayu polos bisa dijadikan bahan bakar biomassa atau dicacah menjadi mulsa. "
               "Kayu berlapis cat atau bahan kimia sebaiknya dikirim ke fasilitas daur ulang khusus.",
    },
    "daun_ranting": {
        "label": "Daun & Ranting", "code": "ORG-03", "category": "Organik",
        "decompose": "1–3 bulan",     "bin": "Hijau",  "binHex": "#10B981",
        "tip": "Daun kering dan ranting sangat baik untuk kompos atau mulsa. "
               "Jangan dibakar karena menghasilkan polusi udara yang berbahaya.",
    },
    "kertas_tisu": {
        "label": "Kertas / Tisu", "code": "ORG-04", "category": "Organik",
        "decompose": "2–6 minggu",    "bin": "Hijau",  "binHex": "#10B981",
        "tip": "Tisu bekas tidak bisa didaur ulang, tetapi bisa dijadikan kompos. "
               "Kertas yang bersih dan kering masih bisa disetor ke bank sampah.",
    },

    # ── B3 ─────────────────────────────────────────
    "baterai": {
        "label": "Baterai", "code": "B3-01", "category": "B3",
        "decompose": "tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "tip": "Jangan dibuang ke tempat sampah biasa karena mengandung merkuri dan kadmium. "
               "Setor ke kotak pengumpulan limbah B3 di Indomaret atau Alfamart.",
    },
    "elektronik": {
        "label": "Elektronik Bekas", "code": "B3-02", "category": "B3",
        "decompose": "tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "tip": "Limbah elektronik mengandung logam berat yang berbahaya. Serahkan ke program "
               "tarik-kembali dari produsen atau gerai resmi seperti iBox dan Samsung Store.",
    },
    "lampu_merkuri": {
        "label": "Lampu Merkuri", "code": "B3-03", "category": "B3",
        "decompose": "tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "tip": "Lampu neon atau CFL mengandung merkuri. Jangan dipecahkan, "
               "bawa dalam keadaan utuh ke fasilitas pengumpulan limbah B3.",
    },
    "medis": {
        "label": "Limbah Medis", "code": "B3-04", "category": "B3",
        "decompose": "tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "tip": "Limbah medis seperti jarum, perban, dan obat kedaluwarsa sangat berbahaya. "
               "Serahkan ke fasilitas kesehatan atau apotek terdekat.",
    },
    "kimia": {
        "label": "Limbah Kimia", "code": "B3-05", "category": "B3",
        "decompose": "tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "tip": "Cat, pelarut, dan bahan kimia rumah tangga mengandung senyawa mudah menguap yang berbahaya. "
               "Jangan dituang ke saluran air. Serahkan ke fasilitas limbah B3.",
    },

    # ── RESIDU ─────────────────────────────────────
    "popok_pembalut": {
        "label": "Popok / Pembalut", "code": "RSD-01", "category": "Residu",
        "decompose": "200–500 tahun", "bin": "Hitam", "binHex": "#6B7280",
        "tip": "Bungkus rapat sebelum dibuang ke tempat sampah residu (hitam). "
               "Tidak dapat didaur ulang, jadi kurangi pemakaian barang sekali pakai.",
    },
    "puntung_rokok": {
        "label": "Puntung Rokok", "code": "RSD-02", "category": "Residu",
        "decompose": "10–15 tahun",   "bin": "Hitam", "binHex": "#6B7280",
        "tip": "Puntung rokok mengandung ribuan bahan kimia berbahaya. "
               "Jangan dibuang sembarangan karena dapat mencemari air tanah.",
    },
}

_FALLBACK = {
    "label": "Sampah Tidak Dikenali", "code": "UNK-01", "category": "Anorganik",
    "decompose": "bervariasi", "bin": "Hitam", "binHex": "#6B7280",
    "tip": "Sampah tidak dapat diklasifikasikan. Pisahkan dari sampah organik dan setor ke TPS terdekat.",
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
    """Jalankan subprocess secara sinkron, lalu kumpulkan seluruh keluarannya baris per baris."""
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
    """Tugas latar belakang: pra-pemrosesan lalu pelatihan model. Mengubah _pipeline secara langsung."""
    global _model, _lvl1_model, _lvl2_name_to_idx

    def push(msg: str) -> None:
        _pipeline["logs"].append(msg)

    py           = sys.executable
    pipeline_dir = ROOT_DIR / "pipeline"

    try:
        # ── Langkah 1: Pra-pemrosesan ──────────────────────────────────────
        push("[1/2] Memulai pra-pemrosesan...")
        _pipeline["step"] = "preprocess"

        rc, logs = await asyncio.to_thread(
            _blocking_run,
            [py, str(pipeline_dir / "1_prapemrosesan.py")],
            str(ROOT_DIR),
        )
        _pipeline["logs"].extend(logs)

        if rc != 0:
            push(f"[ERROR] Pra-pemrosesan gagal (kode keluar {rc})")
            _pipeline["status"] = "error"
            return
        push("[1/2] Pra-pemrosesan selesai.")

        # ── Langkah 2: Pelatihan ────────────────────────────────────────────
        push(f"[2/2] Memulai pelatihan ({mode})...")
        _pipeline["step"] = "train"

        train_args = [py, str(pipeline_dir / "2_pelatihan_yolo.py")]
        if mode == "finetune":
            train_args += ["--finetune", "--epochs", "30"]

        rc, logs = await asyncio.to_thread(_blocking_run, train_args, str(ROOT_DIR))
        _pipeline["logs"].extend(logs)

        if rc != 0:
            push(f"[ERROR] Pelatihan gagal (kode keluar {rc})")
            _pipeline["status"] = "error"
            return
        push("[2/2] Pelatihan selesai.")

        # ── Muat ulang model tanpa perlu restart backend ─────────────────────
        push("[RELOAD] Memuat ulang model ke memori...")
        try:
            _model            = YOLO(MODEL_PATH)
            _lvl1_model       = YOLO(LVL1_MODEL_PATH) if LVL1_MODEL_PATH.exists() else None
            _lvl2_name_to_idx = {v: k for k, v in _model.names.items()}
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
app = FastAPI(title="TrashTrack API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Periksa status backend dan model."""
    return {"status": "ok", "model": MODEL_PATH.name, "classes": len(TRASH_INFO), "lvl1": 4, "lvl2": len(TRASH_INFO)}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Mengklasifikasikan gambar sampah menggunakan model YOLO.

    Permintaan : multipart/form-data dengan field `file` berisi gambar (JPEG/PNG/WebP).
    Jawaban    : JSON berisi label, kode, kategori, tingkat keyakinan, dan tips pengelolaan.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (image/*)")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Gambar tidak dapat dibaca")

    # ── Langkah 1: Peningkatan kontras dengan CLAHE ──────────────────────
    img = enhance_image(img)

    # ── Langkah 2: Inferensi hierarki (bila model Level 1 ada) atau TTA biasa ─
    if _lvl1_model:
        lvl2, confidence, lvl1, lvl1_conf = classify_hierarchical(img)
    else:
        lvl2, confidence, _ = classify_with_tta(img)
        lvl1      = _sub_to_lvl1.get(lvl2, "anorganik")
        lvl1_conf = 0.0

    confidence = round(confidence, 2)

    # ── Tiga kemungkinan teratas (rata-rata probabilitas TTA) ────────────
    _, _, avg_probs = classify_with_tta(img)
    top3 = sorted(
        [{"lvl2": _model.names[i], "confidence": round(float(avg_probs[i])*100, 2)}
         for i in range(len(_model.names))],
        key=lambda x: -x["confidence"]
    )[:3]

    info = TRASH_INFO.get(lvl2, _FALLBACK)
    return {
        **info,
        "confidence"     : confidence,
        "low_confidence" : confidence < CONF_THRESHOLD,
        "lvl2"           : lvl2,
        "lvl1"           : lvl1,
        "lvl1_conf"      : round(lvl1_conf, 2),
        "top3"           : top3,
        "method"         : "hierarchical+TTA" if _lvl1_model else "TTA",
    }


# ─────────────────────────────────────────────────
#  POST /pipeline/run — PRA-PEMROSESAN + PELATIHAN
# ─────────────────────────────────────────────────
@app.post("/pipeline/run")
async def pipeline_run(mode: str = "finetune"):
    """
    Menjalankan pipeline: pra-pemrosesan seluruh dataset/raw lalu melatih model.
    Data lama tetap dipakai karena pra-pemrosesan membaca semua isi dataset/raw/.

    mode: 'finetune' : melatih ulang 30 epoch dari model yang ada (lebih cepat)
          'full'     : melatih dari awal sepenuhnya (lebih akurat)
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
    """Mengembalikan status pipeline saat ini beserta seluruh lognya."""
    return {
        "status": _pipeline["status"],
        "mode":   _pipeline["mode"],
        "step":   _pipeline["step"],
        "logs":   list(_pipeline["logs"]),
    }
