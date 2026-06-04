import sys, shutil, random
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.parent.resolve()
random.seed(42)


# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
YOLO_DIR      = ROOT_DIR / "dataset" / "yolo_cls"
MODEL_DIR     = ROOT_DIR / "model"
YOLO_BEST     = MODEL_DIR / "yolo_best.pt"

EPOCHS   = 100
IMG_SIZE = 224
BATCH    = 16
PATIENCE = 15
DEVICE   = "0"

MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────
#  STEP 1: SIAPKAN DATASET 4 KATEGORI
# ─────────────────────────────────────────────────
def prepare_dataset():
    # Gabung semua jenis per kategori jadi satu folder, lalu balance dengan
    # cap ke jumlah kategori terkecil (downsample acak) biar model nggak bias
    print("\n[PREP] Menyiapkan dataset 4 kategori...")
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    total = 0
    for split in ["train", "val", "test"]:
        split_src = PROCESSED_DIR / split
        if not split_src.exists():
            print(f"  {split_src} tidak ditemukan — jalankan 1_preprocessing.py dulu")
            return False

        per_kategori = {}
        for kategori_dir in split_src.iterdir():
            if not kategori_dir.is_dir():
                continue
            imgs = list(kategori_dir.glob("*.jpg"))
            if imgs:
                per_kategori[kategori_dir.name] = imgs
        if not per_kategori:
            return False

        batas = min(len(v) for v in per_kategori.values())
        count = 0
        for kategori, imgs in per_kategori.items():
            random.shuffle(imgs)
            dst = YOLO_DIR / split / kategori
            dst.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(imgs[:batas]):
                shutil.copy2(img, dst / f"{i:06d}.jpg")
                count += 1
        print(f"  {split:<6}: {count} foto (cap per kategori: {batas})")
        total += count

    print(f"  Total: {total} | Lokasi: {YOLO_DIR}\n")
    return True


# ─────────────────────────────────────────────────
#  STEP 2: TRAINING
# ─────────────────────────────────────────────────
def train_yolo():
    # Train model klasifikasi 4 kategori dari awal pakai YOLO11l-cls, simpan ke yolo_best.pt
    print("\n[TRAIN] Training model 4 kategori (yolo11l-cls)...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH} | Device: {DEVICE}")

    model   = YOLO("yolo11l-cls.pt")
    results = model.train(
        data            = str(YOLO_DIR),
        epochs          = EPOCHS,
        imgsz           = IMG_SIZE,
        batch           = BATCH,
        device          = DEVICE,
        project         = str(MODEL_DIR),
        name            = "yolo_kategori",
        patience        = PATIENCE,
        save            = True,
        plots           = True,
        dropout         = 0.2,
        label_smoothing = 0.1,
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
        lr0             = 0.001,
        lrf             = 0.01,
        warmup_epochs   = 3,
        dropout         = 0.2,
        label_smoothing = 0.1,
        freeze          = 10,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, YOLO_BEST)
    print(f"\n  Fine-tuning selesai. Model diperbarui: {YOLO_BEST}")
    return YOLO_BEST


# ─────────────────────────────────────────────────
#  STEP 3: EVALUASI
# ─────────────────────────────────────────────────
def evaluate_yolo(model_path):
    # Uji model di test set. Top-1 = seberapa sering tebakan pertama benar
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
    return metrics


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Training TrashTrack (4 kategori)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune dari model yang ada (lebih cepat dari training penuh)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Jumlah epoch untuk fine-tuning (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO TRAINING — 4 KATEGORI SAMPAH")
    print("  Kategori: anorganik | organik | b3 | residu")
    print("=" * 60)

    if not prepare_dataset():
        sys.exit(1)

    if args.finetune:
        finetune_yolo(epochs=args.epochs)
    else:
        train_yolo()

    evaluate_yolo(YOLO_BEST)
    print(f"\n  Selesai. Model: {YOLO_BEST}")
