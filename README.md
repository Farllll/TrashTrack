# TrashTrack — Klasifikasi Sampah Otomatis

Aplikasi web untuk mengenali jenis sampah dari gambar. User tinggal unggah foto atau arahkan webcam, lalu model AI (YOLO segmentasi) mendeteksi objek sampahnya, menentukan kategorinya, harus dibuang ke tempat sampah warna apa, berapa lama terurai, plus tips pengelolaannya.

Dibuat sebagai proyek UAS Machine Learning.

---

## Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Cara Kerja Singkat](#cara-kerja-singkat)
- [Struktur Folder](#struktur-folder)
- [Cara Menjalankan](#cara-menjalankan)
- [Pipeline: Menyiapkan Data & Melatih Model](#pipeline-menyiapkan-data--melatih-model)
- [Peta Kode per Fitur](#peta-kode-per-fitur)
- [Catatan Penting](#catatan-penting)
- [Teknologi yang Dipakai](#teknologi-yang-dipakai)

---

## Fitur Utama

- **Unggah foto** → langsung dideteksi & diklasifikasi.
- **Webcam langsung** → taruh sampah di kotak panduan, sistem auto-jepret & klasifikasi.
- **Model segmentasi (YOLO11-seg)** → model belajar AREA objek sampahnya saja, jadi background otomatis diabaikan — tidak perlu hapus background manual.
- **4 kategori** → anorganik, organik, B3, residu.
- **Info lengkap per sampah** → kategori, warna tempat sampah, lama terurai, dan tips buang yang benar.
- **Penanda "tidak yakin"** → kalau tidak ada objek terdeteksi atau confidence di bawah 20%, user diberi tahu daripada dipaksakan ke kategori yang salah.
- **Auto-annotation** → label segmentasi dataset di-generate otomatis pakai FastSAM, tanpa anotasi manual.

---

## Cara Kerja Singkat

```
Foto/Webcam (Frontend React)
        │  kirim gambar lewat HTTP
        ▼
Backend FastAPI  ──►  Perbaiki kontras (CLAHE)
        │                     │
        │              Model YOLO11-seg
        │              (deteksi + segmentasi, 4 kategori)
        │              → ambil objek dengan confidence tertinggi
        ▼
Hasil JSON (kategori, keyakinan, tips) ──► tampil di panel kanan
```

Model dilatih lewat folder `pipeline/`, hasilnya disimpan sebagai `model/yolo_best.pt` dan dibaca backend.

**Kenapa segmentasi, bukan klasifikasi biasa?** Classifier melihat SELURUH gambar — termasuk background — jadi bisa "nyontek" dari meja/lantai di foto training. Model segmentasi cuma belajar dari area objeknya, jadi lebih tahan terhadap background yang berbeda-beda saat scan kamera.

---

## Struktur Folder

```
TrashTrack_UI/
├── backend/
│   ├── main.py              # Server FastAPI + logika inferensi
│   └── requirements.txt     # Dependensi Python backend
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Komponen utama + komunikasi ke backend
│   │   └── components/      # Navbar, UploadZone, WebcamCapture, ResultPanel, StatCard
│   ├── package.json
│   └── vite.config.js       # Proxy /api → localhost:8000
├── pipeline/
│   ├── 1_preprocessing.py   # Cleaning + auto-annotation (FastSAM) + split
│   └── 2_training.py        # Latih model YOLO11-seg (+ fine-tune)
├── dataset/
│   ├── raw/{kategori}/      # Foto mentah per kategori (sumber kebenaran)
│   └── yolo_seg/            # Dataset siap-latih (images + labels + data.yaml), dibuat otomatis
└── model/
    └── yolo_best.pt         # Model segmentasi 4 kategori (dibaca backend)
```

---

## Cara Menjalankan

### 1. Backend (FastAPI)
Dari folder `TrashTrack_UI/`:
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Cek jalan atau tidak: buka `http://localhost:8000/health`.

> Backend butuh `model/yolo_best.pt` ada. Kalau belum, jalankan pipeline dulu.

### 2. Frontend (React + Vite)
Dari folder `frontend/`:
```bash
npm install
npm run dev
```
Buka alamat yang ditampilkan Vite (biasanya `http://localhost:5173`). Frontend otomatis meneruskan request `/api/...` ke backend di port 8000.

### 3. Akses dari luar jaringan (Cloudflare Tunnel)

Biar website bisa dibuka dari HP atau laptop lain lewat internet, pakai Cloudflare Tunnel — gratis, tanpa perlu daftar akun.

**Install cloudflared** (sekali saja):
```bash
winget install Cloudflare.cloudflared
```

**Jalankan tunnel** di terminal terpisah (backend & frontend harus sudah jalan dulu):
```bash
cloudflared tunnel --url http://localhost:5173
```

Tunggu beberapa detik sampai muncul output seperti ini:
```
+----------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at         |
|  https://xxxx-xxxx-xxxx.trycloudflare.com                |
+----------------------------------------------------------+
```

URL `trycloudflare.com` itu langsung bisa dibuka dari mana saja. Aktif selama terminal tunnel masih jalan.

> **Catatan:** URL berubah setiap kali tunnel di-restart. Untuk URL permanen, perlu akun Cloudflare (tetap gratis).

---

## Pipeline: Menyiapkan Data & Melatih Model

Urutannya: **taruh gambar → preprocessing (auto-annotation) → training**.

### Langkah 1 — Taruh gambar mentah
Masukkan gambar langsung ke folder kategorinya:
```
dataset/raw/anorganik/000001.jpg
dataset/raw/organik/000001.jpg
dataset/raw/b3/000001.jpg
dataset/raw/residu/000001.jpg
```

### Langkah 2 — Preprocessing + Auto-annotation
```bash
python pipeline/1_preprocessing.py
```
Yang dilakukan:
1. Hapus foto rusak & duplikat (perceptual hash / dHash)
2. **Auto-annotation**: tiap foto dijalankan ke FastSAM untuk dapat mask objek utamanya, kelasnya diambil dari nama folder → label segmentasi di-generate tanpa anotasi manual
3. Split 80/10/10 ke `dataset/yolo_seg/` (format YOLO-seg: images + labels + data.yaml)
4. Simpan preview anotasi ke `dataset/cek_anotasi.png` — **cek dulu hasilnya sebelum training!**

### Langkah 3 — Training
Training penuh dari awal:
```bash
python pipeline/2_training.py
```
Fine-tune dari model yang sudah ada (lebih cepat, untuk data tambahan):
```bash
python pipeline/2_training.py --finetune --epochs 30
```

> Setelah training selesai, **restart backend** supaya model baru termuat (atau jalankan via endpoint `/pipeline/run` yang otomatis reload model).

---

## Peta Kode per Fitur

> Nomor baris bisa sedikit bergeser kalau file diedit. Nama fungsi lebih stabil sebagai acuan.

### Backend — `backend/main.py`
| Fitur | Lokasi |
|------|--------|
| Path model & threshold keyakinan | `backend/main.py:9` |
| Load model YOLO-seg saat backend start | `backend/main.py:22` |
| Perbaikan kontras CLAHE | `enhance_image()` — `backend/main.py:30` |
| Inferensi segmentasi (1× per request) | `run_segmentation()` — `backend/main.py:40` |
| Data info tiap kategori (label, warna, tips) | `TRASH_INFO` — `backend/main.py:65` |
| Fallback untuk sampah tidak dikenali | `_FALLBACK` — `backend/main.py:95` |
| Endpoint cek status backend | `GET /health` |
| Endpoint klasifikasi utama | `POST /classify` |
| Jalankan pipeline dari API | `POST /pipeline/run` |
| Cek status & log pipeline | `GET /pipeline/status` |

### Frontend — `frontend/src/`
| Fitur | Lokasi |
|------|--------|
| Cek koneksi backend saat dibuka | `frontend/src/App.jsx:21` |
| Kirim gambar ke `/classify` & terima hasil | `handleFile()` — `frontend/src/App.jsx:44` |
| Simulasi progress bar saat memproses | `startProgressSim()` — `frontend/src/App.jsx:30` |
| Ganti mode Upload ↔ Webcam | `switchMode()` — `frontend/src/App.jsx:92` |
| Tema gelap/terang | `theme` — `frontend/src/App.jsx:102` |
| Area drag & drop / pilih file | `frontend/src/components/UploadZone.jsx` |
| Panel hasil deteksi (kartu, bar, tips) | `frontend/src/components/ResultPanel.jsx` |
| Deteksi objek webcam (background subtraction) | `findLargestBlob()` — `frontend/src/components/WebcamCapture.jsx:56` |
| Auto-jepret saat objek terkunci 3 detik | `triggerScan()` — `frontend/src/components/WebcamCapture.jsx:235` |

### Pipeline — `pipeline/`
| Fitur | Lokasi |
|------|--------|
| Daftar 4 kategori | `CATEGORIES` — `pipeline/1_preprocessing.py:27` |
| Fingerprint foto (perceptual hash / dHash) | `get_phash()` — `pipeline/1_preprocessing.py:49` |
| Bersihkan file rusak & duplikat per kategori | `clean_category()` — `pipeline/1_preprocessing.py:62` |
| Auto-annotation satu foto via FastSAM | `annotate_image()` — `pipeline/1_preprocessing.py:115` |
| Split + tulis dataset YOLO-seg + data.yaml | `build_dataset()` — `pipeline/1_preprocessing.py:163` |
| Preview overlay anotasi | `preview_annotations()` — `pipeline/1_preprocessing.py:210` |
| Grafik distribusi 4 kategori | `visualize()` — `pipeline/1_preprocessing.py:250` |
| Training penuh YOLO11s-seg | `train_yolo()` — `pipeline/2_training.py:29` |
| Fine-tune dari model yang sudah ada | `finetune_yolo()` — `pipeline/2_training.py:55` |
| Evaluasi mAP50 (box + mask, per kategori) | `evaluate_yolo()` — `pipeline/2_training.py:92` |

---

## Catatan Penting

- **4 kategori:** anorganik (kuning), organik (hijau), B3 (merah), residu (abu-abu).
- **Model = YOLO11s-seg** (deteksi + segmentasi), input 512px — dipilih supaya muat di VRAM 4GB (RTX 3050 Laptop). FastSAM hanya dipakai sekali di preprocessing untuk generate label — bukan bagian dari model final.
- **Threshold keyakinan:** tidak ada deteksi atau confidence < 20% → "Tidak Dikenali", user diminta foto ulang. Di bawah 60% → hasil ditampilkan tapi diberi tanda "kurang yakin".
- **Training butuh GPU.** Default `DEVICE = "0"` (GPU NVIDIA). Ganti ke `"cpu"` di `pipeline/2_training.py:20` kalau tanpa GPU.
- **Augmentasi offline tidak diperlukan** — training YOLO-seg sudah punya augmentasi bawaan (mosaic, flip, HSV shift, scale, dll).
- **Kotak di webcam bukan output AI.** Itu hasil deteksi gerakan (background subtraction) yang dikode manual, hanya untuk memicu auto-jepret.

---

## Teknologi yang Dipakai

**Backend:** Python, FastAPI, Uvicorn, Ultralytics YOLO11, OpenCV, Pillow, NumPy.  
**Frontend:** React 19, Vite, Tailwind CSS.  
**Model:** YOLO11s-seg (segmentasi instans, 4 kategori), transfer learning dari bobot pretrained COCO.  
**Pipeline:** FastSAM (auto-annotation), Matplotlib, tqdm.
