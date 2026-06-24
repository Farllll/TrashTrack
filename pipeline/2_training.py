import sys, shutil
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.parent.resolve()

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
DATA_YAML = ROOT_DIR / "dataset" / "yolo_seg" / "data.yaml"
MODEL_DIR = ROOT_DIR / "model"
YOLO_BEST = MODEL_DIR / "yolo_best.pt"

EPOCHS   = 60
IMG_SIZE = 512   # 640 butuh VRAM > 4GB — 512 muat di RTX 3050 Laptop tanpa spill ke RAM
BATCH    = 8
PATIENCE = 15
DEVICE   = "0"   # ganti "cpu" kalau nggak ada GPU

MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────
def train_yolo():
    # Train model segmentasi 4 kategori dari awal pakai YOLO11s-seg, simpan ke yolo_best.pt
    print("\n[TRAIN] Training model segmentasi 4 kategori (yolo11s-seg)...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH} | Imgsz: {IMG_SIZE} | Device: {DEVICE}")

    model   = YOLO("yolo11s-seg.pt")
    results = model.train(
        data     = str(DATA_YAML),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        project  = str(MODEL_DIR),
        name     = "yolo_seg",
        patience = PATIENCE,
        save     = True,
        plots    = True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  Model terbaik disalin ke: {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  FINE-TUNING
# ─────────────────────────────────────────────────
def finetune_yolo(base_model_path: Path = YOLO_BEST, epochs: int = 30):
    # Lanjutin training dari model yang sudah ada
    # Learning rate kecil + beberapa layer di-freeze biar model nggak lupa yang lama
    if not base_model_path.exists():
        print(f"  Model tidak ditemukan: {base_model_path}")
        print(f"    Jalankan training penuh dulu: python pipeline/2_training.py")
        return None

    print(f"\n[FINETUNE] Fine-tuning dari {base_model_path.name} ({epochs} epoch)...")
    model   = YOLO(str(base_model_path))
    results = model.train(
        data          = str(DATA_YAML),
        epochs        = epochs,
        imgsz         = IMG_SIZE,
        batch         = BATCH,
        device        = DEVICE,
        project       = str(MODEL_DIR),
        name          = "yolo_finetune",
        patience      = 10,
        save          = True,
        plots         = True,
        lr0           = 0.001,
        lrf           = 0.01,
        warmup_epochs = 3,
        freeze        = 10,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  Fine-tuning selesai. Model diperbarui: {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  EVALUASI
# ─────────────────────────────────────────────────
def evaluate_yolo(model_path):
    # Uji model di test set. mAP50 = akurasi deteksi+mask pada threshold IoU 50%
    print(f"\n[EVAL] Evaluasi pada test set...")
    model   = YOLO(model_path)
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = "test",
        imgsz   = IMG_SIZE,
        device  = DEVICE,
        verbose = False,
    )
    print(f"  mAP50 (box)  : {metrics.box.map50*100:.2f}%")
    print(f"  mAP50 (mask) : {metrics.seg.map50*100:.2f}%")

    # mAP50 per kategori (kalau tersedia)
    try:
        names = metrics.names
        for idx, ap in zip(metrics.seg.ap_class_index, metrics.seg.ap50):
            print(f"    {names[int(idx)]:<12}: {ap*100:.2f}%")
    except Exception:
        pass
    return metrics


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO-seg Training TrashTrack (4 kategori)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune dari model yang ada (lebih cepat dari training penuh)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Jumlah epoch untuk fine-tuning (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO-SEG TRAINING — 4 KATEGORI SAMPAH")
    print("  Kategori: anorganik | organik | b3 | residu")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"  {DATA_YAML} tidak ditemukan — jalankan 1_preprocessing.py dulu")
        sys.exit(1)

    if args.finetune:
        finetune_yolo(epochs=args.epochs)
    else:
        train_yolo()

    evaluate_yolo(YOLO_BEST)
    print(f"\n  Selesai. Model: {YOLO_BEST}")
