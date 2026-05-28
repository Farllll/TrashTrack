"""
====================================================
  SCRIPT 1: DATASET COLLECTOR – HIERARKIS
  Level 1 : Anorganik | Organik | B3 | Residu
  Level 2 : Subkategori per level 1
  ─────────────────────────────────────────────────
  Dua sumber data yang didukung:
    A) Scraping Bing Image Search via icrawler
       → Gunakan keyword bahasa Inggris & Indonesia
       → ~80 gambar per keyword, 4 keyword per subkelas
    B) Download dataset TrashNet (opsional, ~500MB)
       → Dipetakan ke hierarki lokal via TRASHNET_MAP

  Output: dataset/raw/{lvl1}/{lvl2}/*.jpg
  Lanjutkan dengan: python 2_preprocessor.py
====================================================
Install:
  pip install icrawler requests tqdm Pillow
"""

from pathlib import Path
import shutil, zipfile, requests
from tqdm import tqdm
from icrawler.builtin import BingImageCrawler

ROOT_DIR = Path(__file__).parent.parent.resolve()   # → TrashTrack_UI/

# ─────────────────────────────────────────────────
#  HIERARKI KELAS
# ─────────────────────────────────────────────────
HIERARCHY = {
    "anorganik": {
        "kaca"      : ["broken glass waste", "glass bottle garbage", "sampah kaca pecahan", "glass jar trash"],
        "karet"     : ["rubber waste pile", "old tire garbage", "sandal rubber trash", "ban bekas sampah"],
        "logam"     : ["metal can waste", "aluminum scrap garbage", "iron scrap pile", "kaleng sampah logam"],
        "styrofoam" : ["styrofoam waste pile", "foam packaging garbage", "styrofoam trash", "styrofoam bekas sampah"],
        "kardus"    : ["cardboard box waste", "corrugated cardboard trash", "dus kardus sampah", "cardboard garbage pile"],
        "plastik"   : ["plastic bottle waste", "plastic bag garbage", "plastic trash pile", "sampah plastik"],
        "tekstil"   : ["textile waste pile", "old clothes garbage", "fabric scrap waste", "sampah kain baju"],
    },
    "organik": {
        "ampas"       : ["coffee grounds waste", "fruit pulp garbage", "ampas kopi sampah", "food pulp organic waste"],
        "kayu"        : ["wood waste scrap", "wooden plank garbage", "bamboo waste pile", "potongan kayu sampah"],
        "daun_ranting": ["dry leaves garbage", "fallen leaves pile", "twigs branches waste", "sampah daun kering"],
        "kertas_tisu" : ["tissue paper waste", "used tissue garbage", "tisu kertas sampah", "paper tissue trash"],
    },
    "b3": {
        "baterai"      : ["battery waste disposal", "used battery garbage", "baterai bekas sampah", "dead battery waste"],
        "elektronik"   : ["electronic waste e-waste", "broken phone garbage", "PCB circuit board waste", "sampah elektronik"],
        "lampu_merkuri": ["fluorescent lamp waste", "mercury bulb garbage", "broken CFL bulb waste", "lampu TL bekas sampah"],
        "medis"        : ["medical waste disposal", "used syringe garbage", "limbah medis sampah", "hospital waste trash"],
        "kimia"        : ["chemical waste disposal", "paint can garbage", "solvent waste pile", "limbah kimia berbahaya"],
    },
    "residu": {
        "popok_pembalut": ["used diaper waste", "sanitary pad garbage", "popok bekas sampah", "diaper trash pile"],
        "puntung_rokok" : ["cigarette butt waste", "smoking litter garbage", "puntung rokok sampah", "cigarette stub trash"],
    },
}

RAW_DIR          = ROOT_DIR / "dataset" / "raw"
SCRAPE_PER_QUERY = 80   # Gambar per keyword


# ─────────────────────────────────────────────────
#  TRASHNET MAPPING (Opsional ~500MB)
# ─────────────────────────────────────────────────
TRASHNET_MAP = {
    "paper"    : ("organik",   "kertas_tisu"),
    "cardboard": ("anorganik", "kardus"),
    "plastic"  : ("anorganik", "plastik"),
    "metal"    : ("anorganik", "logam"),
    "glass"    : ("anorganik", "kaca"),
}

def download_trashnet():
    """Download dan ekstrak dataset TrashNet (~500MB), lalu peta ke hierarki lokal."""
    url      = "https://github.com/garythung/trashnet/releases/download/v1.0/dataset-resized.zip"
    zip_path = ROOT_DIR / "dataset" / "trashnet.zip"
    extract  = ROOT_DIR / "dataset" / "trashnet_raw"

    print("\n[TrashNet] Downloading (~500MB)...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        r = requests.get(url, stream=True, timeout=120)
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract)

    for tn_cls, (lvl1, lvl2) in TRASHNET_MAP.items():
        src = extract / "dataset-resized" / tn_cls
        dst = RAW_DIR / lvl1 / lvl2
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            copied = 0
            for img in src.iterdir():
                shutil.copy2(img, dst / f"tn_{img.name}")
                copied += 1
            print(f"  ✓ {tn_cls} → {lvl1}/{lvl2}: {copied} gambar")

    print("[TrashNet] Selesai!\n")


# ─────────────────────────────────────────────────
#  SCRAPING PER SUBKELAS
# ─────────────────────────────────────────────────
def scrape_subclass(lvl1: str, lvl2: str, keywords: list):
    """Scrape gambar Bing untuk satu subkelas menggunakan semua keyword yang diberikan."""
    out_dir = RAW_DIR / lvl1 / lvl2
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    print(f"\n  [{lvl1}/{lvl2}] Ada {existing} gambar, mulai scraping...")

    for kw in keywords:
        try:
            crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
            crawler.crawl(keyword=kw, max_num=SCRAPE_PER_QUERY,
                          filters={"type": "photo", "size": "medium"})
        except Exception as e:
            print(f"    ✗ Error '{kw}': {e}")

    total = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    print(f"  ✓ [{lvl1}/{lvl2}] Total: {total} gambar")
    return total


def scrape_all():
    """Jalankan scraping untuk semua subkelas dalam HIERARCHY."""
    print("\n[SCRAPING] Mulai scraping semua subkelas...")
    for lvl1, subclasses in HIERARCHY.items():
        print(f"\n── {lvl1.upper()} ──")
        for lvl2, keywords in subclasses.items():
            scrape_subclass(lvl1, lvl2, keywords)


# ─────────────────────────────────────────────────
#  CEK DATASET
# ─────────────────────────────────────────────────
def check_dataset():
    """Cetak ringkasan jumlah gambar per subkelas; tandai subkelas yang kurang dari 80 gambar."""
    print("\n" + "=" * 55)
    print("  RINGKASAN DATASET")
    print("=" * 55)

    grand_total = 0
    for lvl1, subclasses in HIERARCHY.items():
        lvl1_total = 0
        print(f"\n  [{lvl1.upper()}]")
        for lvl2 in subclasses:
            folder = RAW_DIR / lvl1 / lvl2
            count  = len([
                f for f in folder.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]) if folder.exists() else 0
            status = "✓" if count >= 80 else "✗ "
            print(f"    {status} {lvl2:<20}: {count:>4} gambar")
            lvl1_total  += count
            grand_total += count
        print(f"    {'─'*36}")
        print(f"    Subtotal {lvl1:<18}: {lvl1_total:>4} gambar")

    print("\n" + "=" * 55)
    status = "✓ SIAP" if grand_total >= 1000 else "✗ BELUM 1000+"
    print(f"  GRAND TOTAL : {grand_total} gambar  [{status}]")
    print("=" * 55)


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  GARBAGE COLLECTOR – HIERARKI 2 LEVEL")
    print("=" * 55)

    # Option A: Download TrashNet (butuh internet stabil ~500MB)
    # download_trashnet()

    # Option B: Scraping Bing per subkelas
    scrape_all()

    # Cek hasil akhir
    check_dataset()