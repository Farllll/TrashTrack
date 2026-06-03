# TrashTrack — Klasifikasi Sampah Otomatis

Aplikasi web untuk mengenali jenis sampah dari gambar. User tinggal unggah foto atau arahkan webcam, lalu model AI (YOLO) menebak ini sampah apa, masuk kategori mana, harus dibuang ke tempat sampah warna apa, berapa lama terurai, plus tips pengelolaannya.

Dibuat sebagai proyek UAS Machine Learning.

---

## Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Cara Kerja Singkat](#cara-kerja-singkat)
- [Struktur Folder](#struktur-folder)
- [Cara Menjalankan](#cara-menjalankan)
- [Pipeline: Menyiapkan Data & Melatih Model](#pipeline-menyiapkan-data--melatih-model)
- [Peta Kode (file:baris) per Fitur](#peta-kode-filebaris-per-fitur)
- [Status & Catatan Penting](#status--catatan-penting)
- [Teknologi yang Dipakai](#teknologi-yang-dipakai)

---

## Fitur Utama

- **Unggah foto** → langsung diklasifikasi.
- **Webcam langsung** → taruh sampah di kotak panduan, sistem melacak ~3 detik lalu otomatis menjepret & klasifikasi.
- **Klasifikasi bertingkat (hierarki 2 level)** → tentukan kategori besar dulu (anorganik/organik/B3/residu), baru jenis spesifiknya.
- **Info lengkap per sampah** → kategori, warna tempat sampah, lama terurai, dan tips buang yang benar.
- **Penanda "tidak yakin"** → kalau keyakinan model rendah, user diberi tahu + ditampilkan kemungkinan lain.
- **Pipeline training built-in** → preprocessing, training penuh, dan fine-tune bisa dijalankan lewat skrip atau GUI.

---

## Cara Kerja Singkat

```
Foto/Webcam (Frontend React)
        │  kirim gambar lewat HTTP
        ▼
Backend FastAPI  ──►  Perbaiki kontras (CLAHE)
        │                     │
        │              Model Level 1 (4 kategori)
        │                     │
        │              Model Level 2 + TTA (jenis spesifik)
        ▼
Hasil JSON (label, kategori, keyakinan, tips) ──► tampil di panel kanan
```

Model dilatih terpisah lewat folder `pipeline/`, hasilnya disimpan sebagai `model/yolo_best.pt` (Level 2) dan `model/yolo_lvl1.pt` (Level 1), lalu dibaca backend.

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
│   ├── 1_prapemrosesan.py   # Bersihkan, resize, augmentasi, bagi dataset
│   ├── 2_pelatihan_yolo.py  # Latih model Level 1 & Level 2 (+ fine-tune)
│   └── manajer_dataset.py   # GUI untuk tambah data & jalankan training
├── dataset/
│   ├── raw/{lvl1}/{lvl2}/    # Gambar mentah (sumber kebenaran)
│   ├── processed/            # Hasil preprocessing (dibuat otomatis)
│   └── label_mapping.json    # Peta nama kelas ↔ indeks model
└── model/
    ├── yolo_best.pt         # Model Level 2 (dibaca backend)
    └── yolo_lvl1.pt         # Model Level 1 (dibaca backend)
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

> Backend butuh `model/yolo_best.pt` dan `dataset/label_mapping.json` ada. Kalau belum, jalankan pipeline dulu (lihat bawah).

### 2. Frontend (React + Vite)
Dari folder `frontend/`:
```bash
npm install
npm run dev
```
Buka alamat yang ditampilkan Vite (biasanya `http://localhost:5173`). Frontend otomatis meneruskan request `/api/...` ke backend di port 8000.

---

## Pipeline: Menyiapkan Data & Melatih Model

Urutannya: **kumpulkan gambar → preprocessing → training**.

### Langkah 1 — Taruh gambar mentah
Masukkan gambar ke `dataset/raw/{kategori}/{jenis}/`, contoh:
```
dataset/raw/anorganik/plastik/000001.jpg
dataset/raw/organik/kayu/000001.jpg
```

### Langkah 2 — Preprocessing
```bash
python pipeline/1_prapemrosesan.py
```
Membersihkan gambar rusak/duplikat, resize ke 224×224, augmentasi sampai ~800/kelas, bagi 80/10/10, dan menulis ulang `label_mapping.json`.

### Langkah 3 — Training
Training penuh dari awal (latih Level 1 + Level 2):
```bash
python pipeline/2_pelatihan_yolo.py
```
Fine-tune (lanjut dari model lama, lebih cepat — untuk data tambahan):
```bash
python pipeline/2_pelatihan_yolo.py --finetune --epochs 30
```

### Alternatif — GUI
```bash
python pipeline/manajer_dataset.py
```
Aplikasi jendela untuk memilih gambar, menetapkan kategori, dan menjalankan preprocessing + training dengan tombol (tanpa ketik command).

> Setelah training selesai, **restart backend** supaya model baru termuat (kecuali training dijalankan lewat endpoint `/pipeline/run`, yang otomatis memuat ulang model).

---

## Peta Kode (file:baris) per Fitur

> Nomor baris bisa sedikit bergeser kalau file diedit. Acuan paling stabil adalah nama fungsinya.

### Backend — `backend/main.py`
| Fitur | Lokasi |
|------|--------|
| Konfigurasi path model & mapping | `backend/main.py:36` |
| Muat 2 model (Level 1 & Level 2) | `backend/main.py:47` |
| Perbaikan kontras CLAHE | `enhance_image()` — `backend/main.py:68` |
| Test-Time Augmentation (5 variasi) | `classify_with_tta()` — `backend/main.py:83` |
| Klasifikasi hierarki Level 1 → Level 2 | `classify_hierarchical()` — `backend/main.py:110` |
| Database info tiap kelas (label, tips, dll) | `TRASH_INFO` — `backend/main.py:147` |
| Info cadangan untuk sampah tak dikenal | `_FALLBACK` — `backend/main.py:283` |
| Endpoint cek status backend | `GET /health` — `backend/main.py:395` |
| Endpoint klasifikasi utama | `POST /classify` — `backend/main.py:401` |
| Ambang batas keyakinan (60%) | `CONF_THRESHOLD` — `backend/main.py:40` |
| Jalankan pipeline dari API (preprocess + train) | `POST /pipeline/run` — `backend/main.py:455` |
| Cek status & log pipeline | `GET /pipeline/status` — `backend/main.py:477` |
| Eksekutor pipeline di latar belakang | `_run_pipeline()` — `backend/main.py:320` |

### Frontend — `frontend/src/`
| Fitur | Lokasi |
|------|--------|
| Cek koneksi backend saat dibuka | `frontend/src/App.jsx:22` |
| Kirim gambar ke `/classify` & terima hasil | `handleFile()` — `frontend/src/App.jsx:44` |
| Simulasi progress bar saat memproses | `startProgressSim()` — `frontend/src/App.jsx:30` |
| Ganti mode Upload ↔ Webcam | `switchMode()` — `frontend/src/App.jsx:92` |
| Tema gelap/terang | `theme` — `frontend/src/App.jsx:102` |
| Area drag & drop / pilih file | `frontend/src/components/UploadZone.jsx` |
| Panel hasil deteksi (kartu, bar, tips) | `frontend/src/components/ResultPanel.jsx` |
| Deteksi objek webcam (background subtraction) | `findLargestBlob()` — `frontend/src/components/WebcamCapture.jsx:56` |
| Loop pemrosesan frame webcam | `processFrame()` — `frontend/src/components/WebcamCapture.jsx:274` |
| Auto-jepret saat objek terkunci 3 detik | `triggerScan()` — `frontend/src/components/WebcamCapture.jsx:235` |
| Operasi morfologi (bersihkan mask) | `erode()`/`dilate()` — `frontend/src/components/WebcamCapture.jsx:35` |

### Pipeline — `pipeline/`
| Fitur | Lokasi |
|------|--------|
| Definisi hierarki kelas | `HIERARCHY` — `pipeline/1_prapemrosesan.py:51` |
| Bangun peta label (`label_mapping.json`) | `build_label_maps()` — `pipeline/1_prapemrosesan.py:72` |
| Hapus duplikat (hash MD5) | `get_hash()` — `pipeline/1_prapemrosesan.py:100` |
| Bersihkan file rusak & duplikat | `clean_subclass()` — `pipeline/1_prapemrosesan.py:109` |
| Resep augmentasi gambar | `aug_pipeline` — `pipeline/1_prapemrosesan.py:163` |
| Resize + augmentasi + split | `prepare_all()` — `pipeline/1_prapemrosesan.py:225` |
| Grafik distribusi dataset | `visualize()` — `pipeline/1_prapemrosesan.py:301` |
| Siapkan dataset Level 2 (format YOLO) | `prepare_yolo_dataset()` — `pipeline/2_pelatihan_yolo.py:60` |
| Siapkan dataset Level 1 | `prepare_yolo_lvl1_dataset()` — `pipeline/2_pelatihan_yolo.py:95` |
| Latih model Level 1 (4 kelas) | `train_yolo_lvl1()` — `pipeline/2_pelatihan_yolo.py:128` |
| Latih model Level 2 (kelas spesifik) | `train_yolo()` — `pipeline/2_pelatihan_yolo.py:161` |
| Fine-tune dari model lama | `finetune_yolo()` — `pipeline/2_pelatihan_yolo.py:197` |
| Evaluasi akurasi (Top-1 & Top-5) | `evaluate_yolo()` — `pipeline/2_pelatihan_yolo.py:251` |
| Pengaturan training (epoch, batch, device) | `pipeline/2_pelatihan_yolo.py:42` |

---

## Status & Catatan Penting

- **Hierarki kelas sedang transisi.** Kode (`TRASH_INFO` di backend dan `HIERARCHY` di pipeline) sudah disiapkan untuk **21 kelas** — kelas `tekstil` lama dipecah jadi `sepatu`, `pakaian`, `topi`, `tas`. **Tetapi** `dataset/label_mapping.json` dan model terlatih (`yolo_best.pt`) yang ada sekarang **masih versi lama (18 kelas, dengan `tekstil`)**.
  → Supaya 4 kelas baru benar-benar aktif, **dataset harus dipreprocess + model dilatih ulang** (Langkah 2 & 3 di atas). Sebelum itu, prediksi `tekstil` dari model lama akan jatuh ke info cadangan.
- **Data 4 kelas baru terbatas.** Berasal dari pemecahan ~714 gambar tekstil, ditambal augmentasi. Untuk akurasi serius perlu lebih banyak data asli.
- **Training butuh GPU.** Pengaturan default memakai `DEVICE = "0"` (GPU NVIDIA). Ubah ke `"cpu"` di `pipeline/2_pelatihan_yolo.py:46` jika tanpa GPU (akan jauh lebih lambat).
- **TTA menambah beban.** Tiap klasifikasi menjalankan model 5× (untuk akurasi). Kalau butuh lebih cepat, ini bisa dipangkas.
- **Kotak di webcam bukan output AI.** Itu hasil deteksi gerakan (background subtraction) yang dikode manual, hanya untuk memicu auto-jepret.

---

## Teknologi yang Dipakai

**Backend:** Python, FastAPI, Uvicorn, Ultralytics YOLO11, OpenCV, Pillow, NumPy.
**Frontend:** React 19, Vite, Tailwind CSS.
**Model:** YOLO11-cls (klasifikasi), transfer learning dari bobot pretrained ImageNet.
**Pipeline:** Albumentations (augmentasi), scikit-learn, Matplotlib, tqdm.
