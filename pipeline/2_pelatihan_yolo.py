"""
====================================================
  SKRIP 2: PELATIHAN KLASIFIKASI YOLO
  ─────────────────────────────────────────────────
  Latih YOLO classifier untuk klasifikasi sampah 18 kelas.

  Alur:
    Langkah 1 – Ratakan dataset ke format YOLO cls (hapus layer lvl1)
    Langkah 2 – Pelatihan YOLO11-cls (pretrained ImageNet)
    Langkah 3 – Evaluasi akurasi pada test set

  Input  : dataset/processed/{split}/{lvl1}/{lvl2}/*.jpg
  Output : model/yolo_classifier/weights/best.pt

  Model disalin ke model/yolo_best.pt (dibaca oleh backend)
====================================================
Install:
  pip install ultralytics
"""

import sys, shutil, json
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.parent.resolve()   # → TrashTrack_UI/


# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
PROCESSED_DIR  = ROOT_DIR / "dataset" / "processed"
YOLO_DIR       = ROOT_DIR / "dataset" / "yolo_cls"       # Level 2 (18 kelas)
YOLO_LVL1_DIR  = ROOT_DIR / "dataset" / "yolo_cls_lvl1"  # Level 1 (4 kelas)
MAPPING_PATH   = ROOT_DIR / "dataset" / "label_mapping.json"
MODEL_DIR      = ROOT_DIR / "model"
YOLO_BEST      = MODEL_DIR / "yolo_best.pt"    # Level 2 — dibaca backend
YOLO_LVL1_BEST = MODEL_DIR / "yolo_lvl1.pt"   # Level 1 — dibaca backend

EPOCHS   = 100
IMG_SIZE = 224
BATCH    = 16    
PATIENCE = 15
DEVICE   = "0"   

MODEL_DIR.mkdir(exist_ok=True)

with open(MAPPING_PATH) as f:
    mapping = json.load(f)

print(f"  Level 1 : {mapping['lvl1_classes']}")
print(f"  Level 2 : {mapping['lvl2_classes']}")


# ─────────────────────────────────────────────────
#  STEP 1A: DATASET LEVEL 2 (18 kelas)
# ─────────────────────────────────────────────────
def prepare_yolo_dataset():
    """
    Rapikan susunan folder buat model Level 2.
    YOLO maunya struktur folder = nama kelas, jadi kita "ratakan":
    buang layer kategori besar (lvl1), sisakan langsung folder per jenis (lvl2).
    Contoh: processed/anorganik/plastik → yolo_cls/plastik.
    """
    print("\n[PREP-L2] Dataset Level 2 (18 kelas)...")
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    total = 0
    for split in ["train", "val", "test"]:
        split_src = PROCESSED_DIR / split
        if not split_src.exists():
            print(f"  {split_src} tidak ditemukan — jalankan 1_prapemrosesan.py dulu")
            return False
        count = 0
        for lvl1_dir in split_src.iterdir():
            if not lvl1_dir.is_dir(): continue
            for lvl2_dir in lvl1_dir.iterdir():
                if not lvl2_dir.is_dir(): continue
                dst = YOLO_DIR / split / lvl2_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for img in lvl2_dir.glob("*.jpg"):
                    shutil.copy2(img, dst / img.name); count += 1
        print(f"  {split:<6}: {count} gambar")
        total += count
    print(f"  Total: {total} | Lokasi: {YOLO_DIR}\n")
    return True


# ─────────────────────────────────────────────────
#  STEP 1B: DATASET LEVEL 1 (4 kelas)
# ─────────────────────────────────────────────────
def prepare_yolo_lvl1_dataset():
    """
    Sama seperti di atas, tapi buat model Level 1 (kategori besar).
    Di sini yang dijadikan folder kelas justru kategorinya (anorganik/organik/b3/residu),
    semua jenis sampah di bawahnya digabung jadi satu.
    """
    print("\n[PREP-L1] Dataset Level 1 (4 kelas)...")
    if YOLO_LVL1_DIR.exists():
        shutil.rmtree(YOLO_LVL1_DIR)

    total = 0
    for split in ["train", "val", "test"]:
        split_src = PROCESSED_DIR / split
        if not split_src.exists():
            return False
        count = 0
        for lvl1_dir in split_src.iterdir():
            if not lvl1_dir.is_dir(): continue
            dst = YOLO_LVL1_DIR / split / lvl1_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for lvl2_dir in lvl1_dir.iterdir():
                if not lvl2_dir.is_dir(): continue
                for img in lvl2_dir.glob("*.jpg"):
                    shutil.copy2(img, dst / img.name); count += 1
        print(f"  {split:<6}: {count} gambar")
        total += count
    print(f"  Total: {total} | Lokasi: {YOLO_LVL1_DIR}\n")
    return True


# ─────────────────────────────────────────────────
#  STEP 2A: TRAINING LEVEL 1 (4 kelas — cepat)
# ─────────────────────────────────────────────────
def train_yolo_lvl1():
    """
    Latih model "penyaring kasar" (Level 1) yang cuma nentuin kategori besar:
    anorganik / organik / b3 / residu. Karena cuma 4 pilihan, modelnya lebih ringan
    dan jarang salah. Hasilnya dipakai mempersempit tebakan model Level 2.
    """
    print("\n[TRAIN-L1] Level 1 — 4 kelas, model yolo11l (78.3% ImageNet Top-1)...")
    print(f"  Model  : yolo11l-cls.pt")

    model   = YOLO("yolo11l-cls.pt")
    results = model.train(
        data     = str(YOLO_LVL1_DIR),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        project  = str(MODEL_DIR),
        name     = "yolo_lvl1",
        patience = PATIENCE,
        save     = True,
        plots    = True,
        dropout  = 0.2,
        label_smoothing = 0.05,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_LVL1_BEST)
    print(f"  Model Level 1: {YOLO_LVL1_BEST}")
    return YOLO_LVL1_BEST


# ─────────────────────────────────────────────────
#  STEP 2B: TRAINING LEVEL 2 (18 kelas — utama)
# ─────────────────────────────────────────────────
def train_yolo():
    """
    Latih model "detail" (Level 2) yang nebak jenis spesifik sampah (18 kelas).
    Ini model utama yang dibaca backend. Pakai versi YOLO terbesar (paling akurat)
    karena tugasnya paling berat — bedain plastik vs kaca vs tekstil, dst.
    """
    print("\n[TRAIN-L2] Level 2 — 18 kelas, model yolo11x (terbesar, paling akurat)...")
    print(f"  Model  : yolo11x-cls.pt (79.9% ImageNet Top-1)")
    print(f"  Epochs : {EPOCHS}  |  Batch: {BATCH}  |  Device: {DEVICE}")

    model   = YOLO("yolo11x-cls.pt")
    results = model.train(
        data            = str(YOLO_DIR),
        epochs          = EPOCHS,
        imgsz           = IMG_SIZE,
        batch           = BATCH,
        device          = DEVICE,
        project         = str(MODEL_DIR),
        name            = "yolo_classifier",
        patience        = PATIENCE,
        save            = True,
        plots           = True,
        dropout         = 0.3,
        mixup           = 0.1,
        label_smoothing = 0.1,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  Model terbaik (hasil run) : {best}")
    print(f"  Model terbaik (final)     : {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  FINE-TUNING (untuk data baru tanpa train dari nol)
# ─────────────────────────────────────────────────
def finetune_yolo(base_model_path: Path = YOLO_BEST, epochs: int = 30):
    """
    Lanjutin training dari model yang udah ada, bukan mulai dari nol (fine-tuning).
    Cocok kalau cuma nambah sedikit data baru — jauh lebih cepat.

    Cara kerjanya:
      1. Ambil model lama yang udah pinter.
      2. Lanjut latih pakai dataset terbaru (data lama + data baru).
      3. Learning rate sengaja dibikin kecil + 10 layer awal dibekukan,
         biar model nggak "lupa" hal yang udah dia pelajari sebelumnya.

    Jalankan setelah preprocessing:
      python pipeline/1_prapemrosesan.py  (biar data baru ikut terproses)
    """
    if not base_model_path.exists():
        print(f"  Model tidak ditemukan: {base_model_path}")
        print(f"    Jalankan training penuh dulu: python pipeline/2_pelatihan_yolo.py")
        return None

    print(f"\n[FINETUNE] Fine-tuning dari model existing...")
    print(f"  Base model : {base_model_path.name}")
    print(f"  Epochs     : {epochs} (lebih sedikit dari training penuh)")
    print(f"  LR         : lebih kecil (0.001 → model tidak lupa data lama)")

    model   = YOLO(str(base_model_path))
    results = model.train(
        data            = str(YOLO_DIR),
        epochs          = epochs,
        imgsz           = IMG_SIZE,
        batch           = BATCH,
        device          = DEVICE,
        project         = str(MODEL_DIR),
        name            = "yolo_finetune",
        patience        = 10,
        save            = True,
        plots           = True,
        lr0             = 0.001,    # learning rate awal kecil
        lrf             = 0.01,     # learning rate final
        warmup_epochs   = 3,
        dropout         = 0.2,
        label_smoothing = 0.05,
        freeze          = 10,       # bekukan 10 layer awal — jaga fitur dasar
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  Fine-tuning selesai.")
    print(f"  Model diperbarui: {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  STEP 3: EVALUASI
# ─────────────────────────────────────────────────
def evaluate_yolo(model_path):
    """
    Uji model pakai data test (yang belum pernah dilihat saat training).
    Top-1 = seberapa sering tebakan pertama langsung benar.
    Top-5 = seberapa sering jawaban benar ada di 5 tebakan teratas.
    """
    print(f"\n[EVAL] Evaluasi pada test set...")
    model   = YOLO(model_path)
    metrics = model.val(
        data   = str(YOLO_DIR),
        split  = "test",
        imgsz  = IMG_SIZE,
        device = DEVICE,
        verbose= False,
    )
    print(f"  Akurasi Top-1 : {metrics.top1*100:.2f}%")
    print(f"  Akurasi Top-5 : {metrics.top5*100:.2f}%")
    return metrics


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pelatihan YOLO TrashTrack")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune dari model existing (lebih cepat dari training penuh)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Jumlah epoch untuk fine-tuning (default: 30)")
    args = parser.parse_args()

    if args.finetune:
        print("=" * 60)
        print("  FINE-TUNING YOLO — DETEKSI SAMPAH")
        print(f"  Epoch: {args.epochs}  |  Base: {YOLO_BEST.name}")
        print("=" * 60)
        # Siapkan dataset (sertakan data baru)
        if not prepare_yolo_dataset():
            sys.exit(1)
        finetune_yolo(epochs=args.epochs)
        print(f"\n  Fine-tuning selesai.")
        print(f"  Model Level 2 : {YOLO_BEST}")
    else:
        print("=" * 60)
        print("  PELATIHAN HIERARKI YOLO — DETEKSI SAMPAH")
        print("  Level 1: 4 kelas  |  Level 2: 18 kelas")
        print("=" * 60)

        # Siapkan kedua dataset
        if not prepare_yolo_dataset():
            sys.exit(1)
        if not prepare_yolo_lvl1_dataset():
            sys.exit(1)

        # Train Level 1 dulu (cepat, nano model)
        train_yolo_lvl1()

        # Train Level 2 (lebih lama, large model)
        best_model = train_yolo()
        evaluate_yolo(best_model)

        print(f"\n  Selesai.")
        print(f"  Model Level 1 : {YOLO_LVL1_BEST}")
        print(f"  Model Level 2 : {YOLO_BEST}")
