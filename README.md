# TrashTrack — Klasifikasi Sampah Otomatis

Aplikasi web untuk mengenali jenis sampah dari gambar. User tinggal unggah foto atau arahkan webcam, lalu model AI (YOLO) menentukan sampah ini masuk kategori mana, harus dibuang ke tempat sampah warna apa, berapa lama terurai, plus tips pengelolaannya.

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

- **Unggah foto** → langsung diklasifikasi.
- **Webcam langsung** → taruh sampah di kotak panduan, sistem auto-jepret & klasifikasi.
- **Klasifikasi 4 kategori** → anorganik, organik, B3, residu.
- **Info lengkap per sampah** → kategori, warna tempat sampah, lama terurai, dan tips buang yang benar.
- **Penanda "tidak yakin"** → kalau keyakinan model di bawah 20%, user diberi tahu daripada dipaksakan ke kategori yang salah.
- **TTA (Test-Time Augmentation)** → model dijalankan 5× pada variasi gambar berbeda, hasilnya dirata-rata untuk prediksi yang lebih stabil.

---

## Cara Kerja Singkat

```
Foto/Webcam (Frontend React)
        │  kirim gambar lewat HTTP
        ▼
Backend FastAPI  ──►  Perbaiki kontras (CLAHE)
        │                     │
        │              Model YOLO11-cls
        │              (1 model, 4 kategori)
        │              + TTA (5 variasi)
        ▼
Hasil JSON (kategori, keyakinan, tips) ──► tampil di panel kanan
```

Model dilatih lewat folder `pipeline/`, hasilnya disimpan sebagai `model/yolo_best.pt` dan dibaca backend.

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
│   ├── 1_preprocessing.py   # Bersihkan, resize, augmentasi, bagi dataset
│   └── 2_training.py        # Latih model YOLO11-cls (+ fine-tune)
├── dataset/
│   ├── raw/{kategori}/      # Gambar mentah per kategori (sumber kebenaran)
│   └── processed/           # Hasil preprocessing, dibuat otomatis
└── model/
    └── yolo_best.pt         # Model klasifikasi 4 kategori (dibaca backend)
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

---

## Pipeline: Menyiapkan Data & Melatih Model

Urutannya: **taruh gambar → preprocessing → training**.

### Langkah 1 — Taruh gambar mentah
Masukkan gambar langsung ke folder kategorinya:
```
dataset/raw/anorganik/000001.jpg
dataset/raw/organik/000001.jpg
dataset/raw/b3/000001.jpg
dataset/raw/residu/000001.jpg
```

### Langkah 2 — Preprocessing
```bash
python pipeline/1_preprocessing.py
```
Membersihkan gambar rusak/duplikat (perceptual hash), resize ke 224×224, balance ke 1000 foto per kategori (augmentasi kalau kurang, downsample kalau kebanyakan), lalu bagi 80/10/10 ke `dataset/processed/`.

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
| Load model YOLO saat backend start | `backend/main.py:21` |
| Perbaikan kontras CLAHE | `enhance_image()` — `backend/main.py:29` |
| TTA — jalankan model 5 variasi, rata-rata probabilitas | `classify_with_tta()` — `backend/main.py:39` |
| Data info tiap kategori (label, warna, tips) | `TRASH_INFO` — `backend/main.py:62` |
| Fallback untuk sampah tidak dikenali | `_FALLBACK` — `backend/main.py:92` |
| Endpoint cek status backend | `GET /health` — `backend/main.py:112` |
| Endpoint klasifikasi utama | `POST /classify` — `backend/main.py:118` |
| Jalankan pipeline dari API | `POST /pipeline/run` — `backend/main.py:173` |
| Cek status & log pipeline | `GET /pipeline/status` — `backend/main.py:185` |

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
| Daftar 4 kategori | `CATEGORIES` — `pipeline/1_preprocessing.py:33` |
| Fingerprint foto (perceptual hash / dHash) | `get_phash()` — `pipeline/1_preprocessing.py:50` |
| Bersihkan file rusak & duplikat per kategori | `clean_category()` — `pipeline/1_preprocessing.py:60` |
| Config augmentasi gambar | `aug_pipeline` — `pipeline/1_preprocessing.py:88` |
| Resize + balance + split dataset | `prepare_all()` — `pipeline/1_preprocessing.py:138` |
| Grafik distribusi 4 kategori | `visualize()` — `pipeline/1_preprocessing.py:193` |
| Siapkan dataset YOLO (balance per kategori) | `prepare_dataset()` — `pipeline/2_training.py:31` |
| Training penuh YOLO11l-cls | `train_yolo()` — `pipeline/2_training.py:72` |
| Fine-tune dari model yang sudah ada | `finetune_yolo()` — `pipeline/2_training.py:98` |
| Evaluasi akurasi Top-1 | `evaluate_yolo()` — `pipeline/2_training.py:130` |

---

## Catatan Penting

- **4 kategori:** anorganik (kuning), organik (hijau), B3 (merah), residu (abu-abu). Klasifikasi langsung satu level, tidak ada hierarki.
- **Threshold keyakinan:** di bawah 20% → tidak diklasifikasikan, user diminta foto ulang. Di bawah 60% → hasil ditampilkan tapi diberi tanda "kurang yakin".
- **Training butuh GPU.** Default `DEVICE = "0"` (GPU NVIDIA). Ganti ke `"cpu"` di `pipeline/2_training.py:22` kalau tanpa GPU.
- **TTA menambah beban.** Tiap klasifikasi menjalankan model 5×. Kalau butuh lebih cepat bisa dikurangi di `classify_with_tta()`.
- **Kotak di webcam bukan output AI.** Itu hasil deteksi gerakan (background subtraction) yang dikode manual, hanya untuk memicu auto-jepret.

---

## Teknologi yang Dipakai

**Backend:** Python, FastAPI, Uvicorn, Ultralytics YOLO11, OpenCV, Pillow, NumPy.  
**Frontend:** React 19, Vite, Tailwind CSS.  
**Model:** YOLO11l-cls (klasifikasi), transfer learning dari bobot pretrained ImageNet.  
**Pipeline:** Albumentations (augmentasi), Matplotlib, tqdm.
