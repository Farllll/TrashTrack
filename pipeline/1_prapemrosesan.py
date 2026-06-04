"""
====================================================
  SKRIP 1: PRA-PEMROSESAN DATASET (HIERARKI 4 Lvl1, 21 Kelas)
  ─────────────────────────────────────────────────
  Tahapan (dijalankan berurutan):
    Langkah 1 – Cleaning   : Hapus file rusak & duplikat (cek MD5)
    Langkah 2 – Resize     : Semua gambar diubah ke 224×224 px
    Langkah 3 – Augmentasi : Tambah gambar buatan jika jumlahnya kurang
    Langkah 4 – Split      : Bagi 80% train / 10% val / 10% test
    Langkah 5 – Mapping    : Simpan label_mapping.json (dipakai model)

  Input  : dataset/raw/{lvl1}/{lvl2}/*.jpg
  Output : dataset/processed/{split}/{lvl1}/{lvl2}/*.jpg
           dataset/label_mapping.json
           dataset/distribusi_dataset.png
====================================================
Install:
  pip install Pillow albumentations scikit-learn tqdm matplotlib opencv-python
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os, shutil, random, hashlib, json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import albumentations as A

ROOT_DIR = Path(__file__).parent.parent.resolve()   # → TrashTrack_UI/

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
RAW_DIR    = ROOT_DIR / "dataset" / "raw"
OUT_DIR    = ROOT_DIR / "dataset" / "processed"
TEMP_DIR   = ROOT_DIR / "dataset" / "temp"

IMG_SIZE   = (224, 224)
SEED       = 42
SPLIT      = {"train": 0.80, "val": 0.10, "test": 0.10}
TARGET_PER_SUBCLASS = 800

HIERARCHY = {
    "anorganik": ["kaca", "karet", "logam", "styrofoam", "kardus", "plastik", "tekstil"],
    "organik"  : ["ampas", "kayu", "daun_ranting", "kertas_tisu"],
    "b3"       : ["baterai", "elektronik", "lampu_merkuri", "medis", "kimia"],
    "residu"   : ["popok_pembalut", "puntung_rokok"],
}

COLORS = {
    "anorganik": "#2196F3",
    "organik"  : "#4CAF50",
    "b3"       : "#F44336",
    "residu"   : "#9C27B0",
}

random.seed(SEED)


# ─────────────────────────────────────────────────
#  LABEL MAPPING
# ─────────────────────────────────────────────────
def build_label_maps():
    """
    Bikin "kamus" yang menghubungkan nama kelas dengan angka indeksnya.
    Model nggak ngerti teks, dia maunya angka — jadi 'plastik' kita ubah jadi 5, dst.
    Hasilnya disimpan ke label_mapping.json biar backend tahu arti tiap angka prediksi.
    """
    lvl1_map    = {k: i for i, k in enumerate(HIERARCHY.keys())}
    lvl2_list   = [sub for subs in HIERARCHY.values() for sub in subs]
    lvl2_map    = {k: i for i, k in enumerate(lvl2_list)}
    sub_to_lvl1 = {}
    for lvl1, subs in HIERARCHY.items():
        for sub in subs:
            sub_to_lvl1[sub] = lvl1

    mapping = {
        "lvl1_classes"  : list(HIERARCHY.keys()),
        "lvl2_classes"  : lvl2_list,
        "lvl1_map"      : lvl1_map,
        "lvl2_map"      : lvl2_map,
        "sub_to_lvl1"   : sub_to_lvl1,
        "hierarchy"     : {k: list(v) for k, v in HIERARCHY.items()},
    }
    return mapping


# ─────────────────────────────────────────────────
#  STEP 1: CLEANING
# ─────────────────────────────────────────────────
def get_hash(path: Path) -> str:
    """
    Bikin "sidik jari" unik dari isi file pakai MD5.
    Dua gambar yang isinya persis sama bakal punya sidik jari sama —
    dari situ kita tahu mana yang duplikat dan bisa dibuang.
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_subclass(lvl1: str, lvl2: str) -> list:
    """
    Beresin satu folder kelas: buang gambar yang rusak (nggak bisa dibuka)
    dan yang duplikat (sidik jarinya kembar). Sisanya yang masih bagus
    dikembalikan sebagai daftar untuk diproses lebih lanjut.
    """
    folder = RAW_DIR / lvl1 / lvl2
    if not folder.exists():
        print(f"  Folder tidak ada: {lvl1}/{lvl2}")
        return []

    valid, seen = [], {}
    removed_corrupt = removed_dup = 0

    for f in folder.iterdir():
        if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue
        try:
            img = Image.open(f)
            img.verify()
            h = get_hash(f)
            if h in seen:
                f.unlink(); removed_dup += 1
            else:
                seen[h] = f; valid.append(f)
        except Exception:
            try: f.unlink()
            except: pass
            removed_corrupt += 1

    print(f"  [{lvl1}/{lvl2:<20}] Valid: {len(valid):>4} | Duplikat: {removed_dup} | Rusak: {removed_corrupt}")
    return valid

def clean_all():
    """
    Jalanin proses bersih-bersih tadi ke SEMUA kelas sekaligus.
    Hasilnya satu kumpulan berisi daftar gambar bagus per kelas.
    """
    print("\n[LANGKAH 1] Membersihkan dataset...")
    cleaned = {}
    for lvl1, subs in HIERARCHY.items():
        print(f"\n  ── {lvl1.upper()} ──")
        for lvl2 in subs:
            cleaned[(lvl1, lvl2)] = clean_subclass(lvl1, lvl2)
    return cleaned


# ─────────────────────────────────────────────────
#  STEP 2: AUGMENTASI
# ─────────────────────────────────────────────────
# "Resep" untuk memperbanyak gambar secara buatan. Tiap gambar asli
# dibikin versi-versi baru yang sedikit beda: diputar, digelapkan, diblur, dll.
# Tujuannya biar model terbiasa lihat foto yang nggak sempurna (seperti dunia nyata)
# dan nggak gampang hafalan (overfitting). Angka p = peluang efek itu dipakai.
aug_pipeline = A.Compose([
    # Flip & rotasi
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.Perspective(scale=(0.05, 0.15), p=0.5),   # simulasi sudut pengambilan gambar

    # Pencahayaan — webcam sering backlit atau overexposed
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
    A.RandomGamma(gamma_limit=(60, 140), p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.3),              # simulasi pencahayaan tidak rata

    # Blur — simulasi kamera bergerak / out of focus
    A.OneOf([
        A.Blur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),        # simulasi gerakan tangan
        A.GaussianBlur(blur_limit=5, p=1.0),
    ], p=0.4),

    # Noise & degradasi kualitas — simulasi webcam murah
    A.GaussNoise(std_range=(0.05, 0.20), p=0.4),
    A.ImageCompression(quality_range=(60, 95), p=0.3),

    # Occlusion — simulasi objek sebagian tertutup
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(10, 40),
        hole_width_range=(10, 40),
        p=0.4
    ),

    # Random crop & resize — simulasi objek tidak full frame
    A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.4),
])

def resize_save(src: Path, dst: Path):
    """
    Seragamkan ukuran gambar jadi 224x224 lalu simpan.
    Model maunya semua input seukuran, jadi gambar apa pun kita paksa ke ukuran itu.
    """
    try:
        img = Image.open(src).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=90)
    except Exception as e:
        print(f"  Gagal memproses {src.name}: {e}")

def augment_image(src: Path, out_dir: Path, n: int):
    """
    Dari 1 gambar asli, bikin 'n' gambar variasi baru pakai resep augmentasi di atas.
    Dipakai kalau jumlah gambar di suatu kelas masih kurang dari target.
    """
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
    """
    Inti dari preprocessing. Untuk tiap kelas:
      1. Semua gambar di-resize.
      2. Kalau jumlahnya kurang dari target (~800), ditambah lewat augmentasi
         biar tiap kelas porsinya seimbang.
      3. Dibagi jadi 3 bagian: train (buat belajar), val (buat ngecek saat training),
         test (buat ujian akhir model). Perbandingannya 80/10/10.
    Mengembalikan statistik jumlah gambar tiap bagian.
    """
    print("\n[LANGKAH 2] Resize + Augmentasi...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for (lvl1, lvl2), paths in cleaned.items():
        tmp = TEMP_DIR / lvl1 / lvl2
        tmp.mkdir(parents=True, exist_ok=True)

        for p in tqdm(paths, desc=f"  Resize {lvl2}", leave=False):
            resize_save(p, tmp / p.name)

        n_existing = len(list(tmp.glob("*.jpg")))
        if n_existing < TARGET_PER_SUBCLASS:
            needed    = TARGET_PER_SUBCLASS - n_existing
            sources   = list(tmp.glob("*.jpg"))
            if not sources:
                print(f"  [{lvl1}/{lvl2}] Tidak ada gambar sama sekali, dilewati")
                continue
            aug_each  = max(1, needed // len(sources))
            extra     = needed - aug_each * len(sources)
            print(f"  [{lvl1}/{lvl2}] Augmentasi +{needed} gambar (×{aug_each} per foto)")
            for i, src in enumerate(tqdm(sources, desc=f"  Augment {lvl2}", leave=False)):
                augment_image(src, tmp, aug_each + (1 if i < extra else 0))

        total = len(list(tmp.glob("*.jpg")))
        print(f"  [{lvl1}/{lvl2}] Jumlah akhir sebelum split: {total}")

    print("\n[LANGKAH 3] Membagi train/val/test...")

    for split in ["train", "val", "test"]:
        for lvl1, subs in HIERARCHY.items():
            for lvl2 in subs:
                (OUT_DIR / split / lvl1 / lvl2).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: defaultdict(dict))

    for lvl1, subs in HIERARCHY.items():
        for lvl2 in subs:
            imgs = list((TEMP_DIR / lvl1 / lvl2).glob("*.jpg"))
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
                dst = OUT_DIR / split / lvl1 / lvl2
                for p in split_imgs:
                    shutil.copy2(p, dst / p.name)
                stats[split][lvl1][lvl2] = len(split_imgs)

            print(f"  [{lvl1}/{lvl2}] train={len(splits_data['train'])} | val={len(splits_data['val'])} | test={len(splits_data['test'])}")

    shutil.rmtree(TEMP_DIR)
    print("\n  Split selesai.")
    return stats


# ─────────────────────────────────────────────────
#  STEP 4: VISUALISASI
# ─────────────────────────────────────────────────
def visualize(stats: dict):
    """
    Bikin gambar grafik biar gampang lihat sebaran data:
    diagram batang jumlah gambar per kelas + diagram lingkaran proporsi kategori besar.
    Berguna buat ngecek apakah datanya timpang. Disimpan ke distribusi_dataset.png.
    """
    print("\n[LANGKAH 4] Visualisasi distribusi...")

    all_subs = [(lvl1, lvl2) for lvl1, subs in HIERARCHY.items() for lvl2 in subs]
    labels   = [f"{lvl2}\n({lvl1})" for lvl1, lvl2 in all_subs]
    colors   = [COLORS[lvl1] for lvl1, lvl2 in all_subs]

    train_counts = [stats["train"].get(lvl1, {}).get(lvl2, 0) for lvl1, lvl2 in all_subs]
    val_counts   = [stats["val"].get(lvl1, {}).get(lvl2, 0)   for lvl1, lvl2 in all_subs]
    test_counts  = [stats["test"].get(lvl1, {}).get(lvl2, 0)  for lvl1, lvl2 in all_subs]

    x     = np.arange(len(all_subs))
    w     = 0.28
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    ax1.bar(x, train_counts, w, label="Train", color=[c+"CC" for c in colors])
    ax1.bar(x, val_counts,   w, bottom=train_counts, label="Val", color=[c+"88" for c in colors])
    bottom2 = [t + v for t, v in zip(train_counts, val_counts)]
    ax1.bar(x, test_counts,  w, bottom=bottom2, label="Test", color=[c+"44" for c in colors])

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Jumlah Gambar")
    ax1.set_title("Distribusi Dataset per Subkelas (Level 2)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    seps = [len(HIERARCHY["organik"]) - 0.5, len(HIERARCHY["organik"]) + len(HIERARCHY["anorganik"]) - 0.5]
    for sep in seps:
        ax1.axvline(sep, color="gray", linestyle="--", alpha=0.5)

    legend_patches = [mpatches.Patch(color=COLORS[c], label=c.upper()) for c in HIERARCHY]
    ax1.legend(handles=legend_patches + [
        mpatches.Patch(color="gray", alpha=0.8, label="Train"),
        mpatches.Patch(color="gray", alpha=0.5, label="Val"),
        mpatches.Patch(color="gray", alpha=0.2, label="Test"),
    ])

    lvl1_totals = {}
    for lvl1, subs in HIERARCHY.items():
        total = sum(
            stats["train"].get(lvl1, {}).get(s, 0) +
            stats["val"].get(lvl1, {}).get(s, 0) +
            stats["test"].get(lvl1, {}).get(s, 0)
            for s in subs
        )
        lvl1_totals[lvl1] = total

    nonzero = {k: v for k, v in lvl1_totals.items() if v > 0}
    if nonzero:
        ax2.pie(
            nonzero.values(),
            labels=[f"{k.upper()}\n({v} gambar)" for k, v in nonzero.items()],
            colors=[COLORS[k] for k in nonzero],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
    else:
        ax2.text(0.5, 0.5, "Belum ada data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.set_title("Proporsi Level 1")

    plt.tight_layout()
    (ROOT_DIR / "dataset").mkdir(exist_ok=True)
    plt.savefig(ROOT_DIR / "dataset" / "distribusi_dataset.png", dpi=150)
    plt.close()
    print("  Grafik disimpan: dataset/distribusi_dataset.png")


# ─────────────────────────────────────────────────
#  STEP 5: SIMPAN LABEL MAPPING
# ─────────────────────────────────────────────────
def save_mapping():
    """
    Simpan 'kamus' label tadi ke file label_mapping.json.
    File ini jadi jembatan: backend bacanya buat menerjemahkan angka prediksi
    model jadi nama kelas yang dimengerti manusia.
    """
    mapping = build_label_maps()
    out = ROOT_DIR / "dataset" / "label_mapping.json"
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\n  Label mapping disimpan: {out}")
    print(f"     Kelas Level 1 : {mapping['lvl1_classes']}")
    print(f"     Kelas Level 2 : {mapping['lvl2_classes']}")
    return mapping


# ─────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────
def print_summary(stats: dict):
    """
    Tampilkan ringkasan akhir: berapa total gambar di train/val/test
    dan apakah jumlahnya sudah cukup buat mulai training.
    """
    print("\n" + "=" * 55)
    print("  RINGKASAN DATASET AKHIR")
    print("=" * 55)
    grand = 0
    for split in ["train", "val", "test"]:
        total = sum(
            v for lvl1_data in stats[split].values()
            for v in lvl1_data.values()
        )
        print(f"  {split.upper():<8}: {total} gambar")
        grand += total
    print(f"  {'TOTAL':<8}: {grand} gambar")
    print(f"  STATUS  : {'SIAP DILATIH' if grand >= 1000 else 'Belum cukup'}")
    print("=" * 55)


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PREPROCESSOR SAMPAH – HIERARKI 2 LEVEL")
    print("=" * 55)

    cleaned = clean_all()
    stats   = prepare_all(cleaned)
    save_mapping()
    visualize(stats)
    print_summary(stats)