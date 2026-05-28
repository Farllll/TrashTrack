"""
====================================================
  SCRIPT 3B: YOLO CLASSIFICATION TRAINING
  ─────────────────────────────────────────────────
  Latih YOLOv8 classifier untuk klasifikasi sampah 12 kelas.

  Alur:
    Step 1 – Flatten dataset ke format YOLO cls (hapus layer lvl1)
    Step 2 – Training YOLOv8n-cls (pretrained ImageNet)
    Step 3 – Evaluasi akurasi pada test set

  Input  : dataset/processed/{split}/{lvl1}/{lvl2}/*.jpg
  Output : model/yolo_classifier/weights/best.pt

  Output model disalin ke model/yolo_best.pt (dibaca oleh backend)
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
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
YOLO_DIR      = ROOT_DIR / "dataset" / "yolo_cls"
MAPPING_PATH  = ROOT_DIR / "dataset" / "label_mapping.json"
MODEL_DIR     = ROOT_DIR / "model"
YOLO_BEST     = MODEL_DIR / "yolo_best.pt"   # dibaca oleh backend/main.py

EPOCHS   = 30
IMG_SIZE = 224
BATCH    = 16    # turunkan ke 8 jika RAM terbatas
PATIENCE = 6
DEVICE   = "0"   # GPU NVIDIA RTX 3050

MODEL_DIR.mkdir(exist_ok=True)

with open(MAPPING_PATH) as f:
    mapping = json.load(f)

print(f"  Level 1 : {mapping['lvl1_classes']}")
print(f"  Level 2 : {mapping['lvl2_classes']}")


# ─────────────────────────────────────────────────
#  STEP 1: PERSIAPAN DATASET
# ─────────────────────────────────────────────────
def prepare_yolo_dataset():
    """
    Flatten dataset/processed/{split}/{lvl1}/{lvl2} → dataset/yolo_cls/{split}/{lvl2}.
    YOLOv8 cls butuh struktur flat: {split}/{class_name}/*.jpg.
    """
    print("\n[PREP] Menyiapkan dataset format YOLO Classification...")

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        print("  ✓ Folder lama dihapus")

    total = 0
    for split in ["train", "val", "test"]:
        split_src = PROCESSED_DIR / split
        if not split_src.exists():
            print(f"  ✗ {split_src} tidak ditemukan — jalankan 2_preprocessor.py dulu")
            return False

        count = 0
        for lvl1_dir in split_src.iterdir():
            if not lvl1_dir.is_dir():
                continue
            for lvl2_dir in lvl1_dir.iterdir():
                if not lvl2_dir.is_dir():
                    continue
                dst = YOLO_DIR / split / lvl2_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for img in lvl2_dir.glob("*.jpg"):
                    shutil.copy2(img, dst / img.name)
                    count += 1

        print(f"  ✓ {split:<6}: {count} gambar")
        total += count

    print(f"  ✓ Total  : {total} gambar")
    print(f"  ✓ Lokasi : {YOLO_DIR}\n")
    return True


# ─────────────────────────────────────────────────
#  STEP 2: TRAINING
# ─────────────────────────────────────────────────
def train_yolo():
    """
    Train YOLOv8n-cls (nano classifier, ImageNet pretrained) pada 12 kelas Level-2.
    Gunakan yolov8s-cls.pt untuk akurasi lebih tinggi (lebih lambat).
    """
    print("[TRAIN] Memulai YOLOv8 Classification Training...")
    print(f"  Model   : yolov8n-cls.pt (nano, ImageNet pretrained)")
    print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH}  |  Device: {DEVICE}")
    print(f"  Kelas   : {len(mapping['lvl2_classes'])} kelas Level-2")

    model   = YOLO("yolov8n-cls.pt")
    results = model.train(
        data    = str(YOLO_DIR),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = DEVICE,
        project = str(MODEL_DIR),
        name    = "yolo_classifier",
        patience= PATIENCE,
        save    = True,
        plots   = True,
    )

    # YOLO may auto-increment the folder name; use save_dir from results
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  ✓ Training selesai!")
    print(f"  ✓ Best model (run)   : {best}")
    print(f"  ✓ Best model (fixed) : {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  STEP 3: EVALUASI
# ─────────────────────────────────────────────────
def evaluate_yolo(model_path):
    """Evaluasi model pada test split; cetak Top-1 dan Top-5 accuracy."""
    print(f"\n[EVAL] Evaluasi pada test set...")
    model   = YOLO(model_path)
    metrics = model.val(
        data   = str(YOLO_DIR),
        split  = "test",
        imgsz  = IMG_SIZE,
        device = DEVICE,
        verbose= False,
    )
    print(f"  ✓ Top-1 Accuracy : {metrics.top1*100:.2f}%")
    print(f"  ✓ Top-5 Accuracy : {metrics.top5*100:.2f}%")
    return metrics


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO CLASSIFICATION TRAINING — GARBAGE DETECTION")
    print("=" * 60)

    if not prepare_yolo_dataset():
        sys.exit(1)

    best_model = train_yolo()
    evaluate_yolo(best_model)

    print(f"\n  ✓ Selesai! Jalankan:")
    print(f"    python 4_real_time_inference.py")
