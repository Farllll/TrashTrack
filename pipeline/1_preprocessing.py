import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import shutil, random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
from ultralytics import FastSAM

ROOT_DIR = Path(__file__).parent.parent.resolve()

# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
RAW_DIR  = ROOT_DIR / "dataset" / "raw"
SEG_DIR  = ROOT_DIR / "dataset" / "yolo_seg"   # dataset siap-latih format YOLO-seg

SEED   = 42
SPLIT  = {"train": 0.80, "val": 0.10, "test": 0.10}
DEVICE = "0"   # ganti "cpu" kalau nggak ada GPU

CATEGORIES = ["anorganik", "organik", "b3", "residu"]

# batas area mask (proporsi terhadap frame) — di luar range ini dianggap bukan objek utama
MIN_AREA = 0.02   # terlalu kecil = noise
MAX_AREA = 0.90   # terlalu besar = kemungkinan background

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
#  STEP 2: AUTO-ANNOTATION (FastSAM)
# ─────────────────────────────────────────────────
# FastSAM dipakai SEKALI di sini untuk bikin label segmentasi otomatis.
# Kelas tiap foto sudah ketahuan dari nama foldernya, jadi FastSAM cuma perlu
# nyari DI MANA objeknya (mask), bukan APA objeknya.
_sam = None

def load_sam():
    global _sam
    if _sam is None:
        print("  Load FastSAM-s.pt (auto-download kalau belum ada)...")
        _sam = FastSAM("FastSAM-s.pt")
    return _sam

def annotate_image(path: Path, cls_id: int) -> str | None:
    # Jalankan FastSAM ke satu foto, ambil mask objek utama,
    # return satu baris label format YOLO-seg ("cls x1 y1 x2 y2 ...") atau None kalau gagal
    model = load_sam()
    try:
        res = model(str(path), device=DEVICE, retina_masks=True, imgsz=640,
                    conf=0.4, iou=0.9, verbose=False)[0]
    except Exception:
        return None

    if res.masks is None or len(res.masks) == 0:
        return None

    # Hitung proporsi area tiap mask terhadap frame, lalu filter yang masuk akal
    data = res.masks.data.cpu().numpy()            # (N, H, W)
    frame_area = data.shape[1] * data.shape[2]
    fracs = data.sum(axis=(1, 2)) / frame_area

    candidates = [i for i, fr in enumerate(fracs) if MIN_AREA <= fr <= MAX_AREA]
    if not candidates:
        return None

    # Objek utama = mask valid dengan area terbesar
    best = max(candidates, key=lambda i: fracs[i])
    polygon = res.masks.xyn[best]                  # polygon ternormalisasi (P, 2)
    if polygon is None or len(polygon) < 3:
        return None

    coords = " ".join(f"{min(max(x,0),1):.4f} {min(max(y,0),1):.4f}" for x, y in polygon)
    return f"{cls_id} {coords}"

def annotate_all(cleaned: dict) -> dict:
    # Anotasi semua foto. Return {kategori: [(path, label_line), ...]}
    print("\n[STEP 2] Auto-annotation dengan FastSAM...")
    annotated = {}
    for cls_id, kategori in enumerate(CATEGORIES):
        items, skipped = [], 0
        for p in tqdm(cleaned[kategori], desc=f"  Annotate {kategori}", leave=False):
            label = annotate_image(p, cls_id)
            if label:
                items.append((p, label))
            else:
                skipped += 1
        annotated[kategori] = items
        print(f"  [{kategori:<12}] Teranotasi: {len(items):>4} | Gagal/di-skip: {skipped}")
    return annotated


# ─────────────────────────────────────────────────
#  STEP 3: SPLIT + TULIS DATASET YOLO-SEG
# ─────────────────────────────────────────────────
def build_dataset(annotated: dict):
    # Bagi 80/10/10 lalu tulis images/ + labels/ + data.yaml
    print("\n[STEP 3] Split train/val/test...")
    if SEG_DIR.exists():
        shutil.rmtree(SEG_DIR)
    for split in SPLIT:
        (SEG_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SEG_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {split: {} for split in SPLIT}

    for kategori, items in annotated.items():
        random.shuffle(items)
        n      = len(items)
        n_test = int(n * SPLIT["test"])
        n_val  = int(n * SPLIT["val"])

        splits_data = {
            "test" : items[:n_test],
            "val"  : items[n_test:n_test + n_val],
            "train": items[n_test + n_val:],
        }

        for split, sub in splits_data.items():
            for j, (src, label) in enumerate(sub):
                stem = f"{kategori}_{split}_{j:06d}"
                shutil.copy2(src, SEG_DIR / "images" / split / f"{stem}.jpg")
                (SEG_DIR / "labels" / split / f"{stem}.txt").write_text(label + "\n")
            stats[split][kategori] = len(sub)

        print(f"  [{kategori}] train={len(splits_data['train'])} | val={len(splits_data['val'])} | test={len(splits_data['test'])}")

    # data.yaml — dibaca YOLO saat training
    names = "\n".join(f"  {i}: {k}" for i, k in enumerate(CATEGORIES))
    yaml_text = (
        f"path: {SEG_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n"
    )
    (SEG_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\n  data.yaml ditulis: {SEG_DIR / 'data.yaml'}")
    return stats


# ─────────────────────────────────────────────────
#  STEP 4: PREVIEW ANOTASI
# ─────────────────────────────────────────────────
def preview_annotations(annotated: dict, n: int = 12):
    # Simpan grid foto + overlay polygon ke cek_anotasi.png — cek mata dulu sebelum training
    print("\n[STEP 4] Bikin preview anotasi...")
    samples = []
    for kategori, items in annotated.items():
        samples += [(kategori, p, label) for p, label in items]
    if not samples:
        print("  Tidak ada sample untuk preview")
        return
    random.shuffle(samples)
    samples = samples[:n]

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (kategori, p, label) in zip(axes, samples):
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        vals = label.split()[1:]
        poly = np.array(vals, dtype=float).reshape(-1, 2) * [w, h]
        ax.imshow(img)
        ax.add_patch(plt.Polygon(poly, closed=True, fill=True,
                                 facecolor=COLORS[kategori], alpha=0.35,
                                 edgecolor=COLORS[kategori], linewidth=2))
        ax.set_title(kategori, fontsize=10, color=COLORS[kategori])
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    out = ROOT_DIR / "dataset" / "cek_anotasi.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Preview disimpan: {out} — CEK DULU sebelum training!")


# ─────────────────────────────────────────────────
#  STEP 5: VISUALISASI DISTRIBUSI
# ─────────────────────────────────────────────────
def visualize(stats: dict):
    # Bikin grafik sebaran 4 kategori dan simpan ke distribusi_dataset.png
    print("\n[STEP 5] Visualisasi distribusi...")

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
    plt.savefig(ROOT_DIR / "dataset" / "distribusi_dataset.png", dpi=150)
    plt.close()
    print("  Grafik disimpan: dataset/distribusi_dataset.png")


# ─────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────
def print_summary(stats: dict):
    # Tampilkan total foto per split dan status siap training atau belum
    print("\n" + "=" * 50)
    print("  RINGKASAN DATASET (YOLO-SEG)")
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
    print("  DATASET PREPROCESSING — AUTO-ANNOTATION SEG")
    print("=" * 50)

    cleaned   = clean_all()
    annotated = annotate_all(cleaned)
    stats     = build_dataset(annotated)
    preview_annotations(annotated)
    visualize(stats)
    print_summary(stats)
