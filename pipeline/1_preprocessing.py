import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os, shutil, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import albumentations as A

ROOT_DIR = Path(__file__).parent.parent.resolve()

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
RAW_DIR    = ROOT_DIR / "dataset" / "raw"
OUT_DIR    = ROOT_DIR / "dataset" / "processed"
TEMP_DIR   = ROOT_DIR / "dataset" / "temp"

IMG_SIZE   = (224, 224)
SEED       = 42
SPLIT      = {"train": 0.80, "val": 0.10, "test": 0.10}
TARGET_PER_CLASS = 1000

CATEGORIES = ["anorganik", "organik", "b3", "residu"]

COLORS = {
    "anorganik": "#2196F3",
    "organik"  : "#4CAF50",
    "b3"       : "#F44336",
    "residu"   : "#9C27B0",
}

random.seed(SEED)


# ─────────────────────────────────────────────────
#  STEP 1: CLEANING
# ─────────────────────────────────────────────────
# threshold duplicate: dua foto dianggap sama kalau jarak Hamming hash-nya <= nilai ini
PHASH_THRESHOLD = 5

def get_phash(path: Path) -> int:
    # Bikin fingerprint foto pakai dHash — bisa deteksi foto yang tampak sama walau sudah di-resize atau re-compress
    img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits

def jarak_hash(a: int, b: int) -> int:
    # Hamming distance antara dua hash
    return bin(a ^ b).count("1")

def clean_category(kategori: str) -> list:
    # Buang foto rusak dan duplikat dari satu folder kategori, kembalikan daftar foto yang valid
    folder = RAW_DIR / kategori
    if not folder.exists():
        print(f"  Folder tidak ada: {kategori}")
        return []

    valid, seen_hashes = [], []
    removed_corrupt = removed_dup = 0

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue
        try:
            ph = get_phash(f)
        except Exception:
            try: f.unlink()
            except: pass
            removed_corrupt += 1
            continue

        if any(jarak_hash(ph, prev) <= PHASH_THRESHOLD for prev in seen_hashes):
            f.unlink(); removed_dup += 1
        else:
            seen_hashes.append(ph); valid.append(f)

    print(f"  [{kategori:<12}] Valid: {len(valid):>4} | Duplikat: {removed_dup} | Rusak: {removed_corrupt}")
    return valid

def clean_all():
    # Jalanin clean_category ke semua 4 kategori
    print("\n[STEP 1] Cleaning dataset...")
    cleaned = {}
    for kategori in CATEGORIES:
        cleaned[kategori] = clean_category(kategori)
    return cleaned


# ─────────────────────────────────────────────────
#  STEP 2: AUGMENTASI
# ─────────────────────────────────────────────────
# augmentation config — tiap foto dibikin beberapa variasi (rotate, gelap, blur, dll)
# biar model nggak overfitting. p = probabilitas efek dipakai.
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.Perspective(scale=(0.05, 0.15), p=0.5),

    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
    A.RandomGamma(gamma_limit=(60, 140), p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.3),

    A.OneOf([
        A.Blur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0),
    ], p=0.4),

    A.GaussNoise(std_range=(0.05, 0.20), p=0.4),
    A.ImageCompression(quality_range=(60, 95), p=0.3),

    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(10, 40),
        hole_width_range=(10, 40),
        p=0.4
    ),

    A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.4),
])

def resize_save(src: Path, dst: Path):
    # Resize foto ke 224x224 dan simpan
    try:
        img = Image.open(src).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=90)
    except Exception as e:
        print(f"  Gagal memproses {src.name}: {e}")

def augment_image(src: Path, out_dir: Path, n: int):
    # Dari 1 foto asli, bikin n variasi baru pakai aug_pipeline
    img = cv2.imread(str(src))
    if img is None: return
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
    for i in range(n):
        aug = aug_pipeline(image=img)["image"]
        Image.fromarray(aug).save(out_dir / f"{src.stem}_aug{i}.jpg", quality=90)


# ─────────────────────────────────────────────────
#  STEP 3: RESIZE + AUGMENTASI + SPLIT
# ─────────────────────────────────────────────────
def prepare_all(cleaned: dict):
    # Resize semua foto, balance ke 800 per kategori (kurang → augmentasi, kebanyakan → downsample), lalu split 80/10/10
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    print("\n[STEP 2] Resize + Augmentasi...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for kategori, paths in cleaned.items():
        tmp = TEMP_DIR / kategori
        tmp.mkdir(parents=True, exist_ok=True)

        for p in tqdm(paths, desc=f"  Resize {kategori}", leave=False):
            resize_save(p, tmp / p.name)

        n_existing = len(list(tmp.glob("*.jpg")))

        if n_existing > TARGET_PER_CLASS:
            semua = list(tmp.glob("*.jpg"))
            random.shuffle(semua)
            for p in semua[TARGET_PER_CLASS:]:
                p.unlink()
            print(f"  [{kategori}] Downsample {n_existing} → {TARGET_PER_CLASS}")

        elif n_existing < TARGET_PER_CLASS:
            needed   = TARGET_PER_CLASS - n_existing
            sources  = list(tmp.glob("*.jpg"))
            if not sources:
                print(f"  [{kategori}] Tidak ada foto, dilewati")
                continue
            aug_each = max(1, needed // len(sources))
            extra    = needed - aug_each * len(sources)
            print(f"  [{kategori}] Augmentasi +{needed} foto (×{aug_each} per foto)")
            for i, src in enumerate(tqdm(sources, desc=f"  Augment {kategori}", leave=False)):
                augment_image(src, tmp, aug_each + (1 if i < extra else 0))

        total = len(list(tmp.glob("*.jpg")))
        print(f"  [{kategori}] Total sebelum split: {total} (target {TARGET_PER_CLASS})")

    print("\n[STEP 3] Split train/val/test...")

    stats = defaultdict(dict)

    for kategori in CATEGORIES:
        imgs = list((TEMP_DIR / kategori).glob("*.jpg")) if (TEMP_DIR / kategori).exists() else []
        random.shuffle(imgs)

        n      = len(imgs)
        n_test = int(n * SPLIT["test"])
        n_val  = int(n * SPLIT["val"])

        splits_data = {
            "test" : imgs[:n_test],
            "val"  : imgs[n_test:n_test + n_val],
            "train": imgs[n_test + n_val:],
        }

        for split, split_imgs in splits_data.items():
            dst = OUT_DIR / split / kategori
            dst.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                if p.exists():
                    shutil.copy2(p, dst / p.name)
                else:
                    print(f"  [WARN] File hilang, dilewati: {p.name}")
            stats[split][kategori] = len(split_imgs)

        print(f"  [{kategori}] train={len(splits_data['train'])} | val={len(splits_data['val'])} | test={len(splits_data['test'])}")

    shutil.rmtree(TEMP_DIR)
    print("\n  Split selesai.")
    return stats


# ─────────────────────────────────────────────────
#  STEP 4: VISUALISASI
# ─────────────────────────────────────────────────
def visualize(stats: dict):
    # Bikin grafik sebaran 4 kategori dan simpan ke distribusi_dataset.png
    print("\n[STEP 4] Visualisasi distribusi...")

    labels       = CATEGORIES
    train_counts = [stats["train"].get(k, 0) for k in CATEGORIES]
    val_counts   = [stats["val"].get(k, 0)   for k in CATEGORIES]
    test_counts  = [stats["test"].get(k, 0)  for k in CATEGORIES]
    colors       = [COLORS[k] for k in CATEGORIES]

    x = np.arange(len(CATEGORIES))
    w = 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, train_counts, w, label="Train", color=[c+"CC" for c in colors])
    ax1.bar(x, val_counts,   w, bottom=train_counts, label="Val", color=[c+"88" for c in colors])
    bottom2 = [t + v for t, v in zip(train_counts, val_counts)]
    ax1.bar(x, test_counts,  w, bottom=bottom2, label="Test", color=[c+"44" for c in colors])

    ax1.set_xticks(x)
    ax1.set_xticklabels([k.upper() for k in CATEGORIES], fontsize=11)
    ax1.set_ylabel("Jumlah Foto")
    ax1.set_title("Distribusi Dataset per Kategori")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(handles=[
        mpatches.Patch(color="gray", alpha=0.8, label="Train"),
        mpatches.Patch(color="gray", alpha=0.5, label="Val"),
        mpatches.Patch(color="gray", alpha=0.2, label="Test"),
    ])

    totals = {k: train_counts[i] + val_counts[i] + test_counts[i] for i, k in enumerate(CATEGORIES)}
    nonzero = {k: v for k, v in totals.items() if v > 0}
    if nonzero:
        ax2.pie(
            nonzero.values(),
            labels=[f"{k.upper()}\n({v} foto)" for k, v in nonzero.items()],
            colors=[COLORS[k] for k in nonzero],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
    else:
        ax2.text(0.5, 0.5, "Belum ada data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.set_title("Proporsi Kategori")

    plt.tight_layout()
    (ROOT_DIR / "dataset").mkdir(exist_ok=True)
    plt.savefig(ROOT_DIR / "dataset" / "distribusi_dataset.png", dpi=150)
    plt.close()
    print("  Grafik disimpan: dataset/distribusi_dataset.png")


# ─────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────
def print_summary(stats: dict):
    # Tampilkan total foto per split dan status siap training atau belum
    print("\n" + "=" * 50)
    print("  RINGKASAN DATASET")
    print("=" * 50)
    grand = 0
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        print(f"  {split.upper():<8}: {total} foto")
        grand += total
    print(f"  {'TOTAL':<8}: {grand} foto")
    print(f"  STATUS  : {'SIAP TRAINING' if grand >= 1000 else 'Belum cukup'}")
    print("=" * 50)


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  DATASET PREPROCESSING")
    print("=" * 50)

    cleaned = clean_all()
    stats   = prepare_all(cleaned)
    visualize(stats)
    print_summary(stats)
