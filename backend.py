"""
====================================================
  TRASHTRACK BACKEND — FastAPI
  ────────────────────────────────────────────────
  Terima upload gambar → jalankan YOLO classifier
  → return JSON info sampah ke frontend React.

  Endpoint:
    GET  /health       → status backend
    POST /classify     → multipart: file=<image>

  Jalankan:
    pip install fastapi uvicorn python-multipart ultralytics pillow
    uvicorn backend:app --reload --port 8000
====================================================
"""

import io
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
MODEL_PATH   = BASE_DIR / "model"   / "yolo_best.pt"
MAPPING_PATH = BASE_DIR / "dataset" / "label_mapping.json"

for p in (MODEL_PATH, MAPPING_PATH):
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")

_model = YOLO(MODEL_PATH)

with open(MAPPING_PATH, encoding="utf-8") as f:
    _mapping = json.load(f)
_sub_to_lvl1: dict[str, str] = _mapping["sub_to_lvl1"]

# ─────────────────────────────────────────────────
#  INFO PER KELAS (12 kelas Level-2)
# ─────────────────────────────────────────────────
TRASH_INFO: dict[str, dict] = {
    "daun_ranting": {
        "label": "Daun & Ranting", "code": "ORG-01", "category": "Organik",
        "decompose": "1–3 bulan",     "bin": "Hijau",       "binHex": "#10B981",
        "icon": "🍂",
        "tip": "Daun kering dan ranting sangat baik untuk kompos atau mulsa. "
               "Jangan dibakar — menghasilkan polusi udara berbahaya.",
    },
    "kayu": {
        "label": "Kayu Bekas", "code": "ORG-02", "category": "Organik",
        "decompose": "2–3 tahun",     "bin": "Hijau",       "binHex": "#10B981",
        "icon": "🪵",
        "tip": "Kayu polos bisa dijadikan bahan bakar biomassa atau dicacah jadi mulsa. "
               "Kayu berlapis cat/kimia kirim ke fasilitas daur ulang khusus.",
    },
    "kertas_kardus": {
        "label": "Kertas / Kardus", "code": "KTS-01", "category": "Anorganik",
        "decompose": "2–6 bulan",     "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "📦",
        "tip": "Kertas dan kardus kering mudah didaur ulang. "
               "Jauhkan dari minyak dan air karena mengurangi nilai daur ulang.",
    },
    "plastik": {
        "label": "Plastik", "code": "PLT-01", "category": "Anorganik",
        "decompose": "20–500 tahun",  "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "🧴",
        "tip": "Cek kode daur ulang di bawah kemasan (segitiga angka 1–7). "
               "Bersihkan dari sisa makanan sebelum dibuang ke bank sampah.",
    },
    "logam": {
        "label": "Logam / Kaleng", "code": "LGM-01", "category": "Anorganik",
        "decompose": "50–200 tahun",  "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "🥫",
        "tip": "Kaleng aluminium hemat 95% energi dibanding produksi baru. "
               "Cuci sebelum dibuang atau jual ke pengepul besi.",
    },
    "kaca": {
        "label": "Kaca / Botol Kaca", "code": "KCA-01", "category": "Anorganik",
        "decompose": "1 juta tahun",  "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "🍾",
        "tip": "Kaca dapat didaur ulang tanpa batas tanpa kehilangan kualitas. "
               "Bersihkan dan setor ke bank sampah terdekat.",
    },
    "tekstil": {
        "label": "Tekstil / Pakaian", "code": "TKS-01", "category": "Anorganik",
        "decompose": "1–5 tahun",     "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "👕",
        "tip": "Pakaian layak pakai: donasikan. Tidak layak: jadikan kain lap "
               "atau setor ke bank tekstil untuk didaur ulang.",
    },
    "karet": {
        "label": "Karet", "code": "KRT-01", "category": "Anorganik",
        "decompose": "50–80 tahun",   "bin": "Kuning",      "binHex": "#F59E0B",
        "icon": "⚙️",
        "tip": "Karet bekas (ban, sandal) bisa didaur ulang menjadi bahan lantai karet "
               "atau aspal. Hubungi fasilitas daur ulang karet di kota kamu.",
    },
    "baterai_aki": {
        "label": "Baterai & Aki", "code": "B3-01", "category": "B3",
        "decompose": "Tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "icon": "🔋",
        "tip": "JANGAN buang ke tempat sampah biasa! Mengandung merkuri, timbal, dan asam. "
               "Setor ke drop box B3 di Indomaret/Alfamart.",
    },
    "elektronik": {
        "label": "Elektronik Bekas", "code": "B3-02", "category": "B3",
        "decompose": "Tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "icon": "📱",
        "tip": "E-waste mengandung logam berat berbahaya. Serahkan ke program take-back "
               "produsen atau gerai resmi (iBox, Samsung Store, dll).",
    },
    "cat_pelarut": {
        "label": "Cat & Pelarut", "code": "B3-03", "category": "B3",
        "decompose": "Tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "icon": "🪣",
        "tip": "Cat dan pelarut mengandung VOC berbahaya. JANGAN tuang ke saluran air. "
               "Serahkan ke fasilitas pengolahan limbah B3 setempat.",
    },
    "lampu_merkuri": {
        "label": "Lampu Merkuri", "code": "B3-04", "category": "B3",
        "decompose": "Tidak terurai", "bin": "Drop Box B3", "binHex": "#EF4444",
        "icon": "💡",
        "tip": "Lampu neon/CFL mengandung merkuri. Jangan dihancurkan! "
               "Bawa utuh ke fasilitas pengumpulan limbah B3.",
    },
}

_FALLBACK = {
    "label": "Sampah Tidak Dikenali", "code": "UNK-01", "category": "Anorganik",
    "decompose": "Bervariasi", "bin": "Hitam", "binHex": "#6B7280",
    "icon": "🗑️",
    "tip": "Sampah tidak terklasifikasi. Pisahkan dari sampah organik dan setor ke TPS terdekat.",
}

# ─────────────────────────────────────────────────
#  APP
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
    """Cek status backend dan model."""
    return {"status": "ok", "model": MODEL_PATH.name, "classes": len(TRASH_INFO)}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Klasifikasikan gambar sampah menggunakan model YOLO.

    Request : multipart/form-data, field `file` berisi gambar (JPEG/PNG/WebP)
    Response: JSON berisi label, kode, kategori, kepercayaan, tips pengelolaan
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (image/*)")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Gambar tidak dapat dibaca")

    results   = _model(img, imgsz=224, verbose=False)
    probs     = results[0].probs
    lvl2      = _model.names[probs.top1]
    confidence = round(float(probs.top1conf) * 100, 2)
    lvl1      = _sub_to_lvl1.get(lvl2, "anorganik")

    info = TRASH_INFO.get(lvl2, _FALLBACK)
    return {**info, "confidence": confidence, "lvl2": lvl2, "lvl1": lvl1}
