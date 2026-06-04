"""
====================================================
  TRASHTRACK — MANAJER DATASET
  ─────────────────────────────────────────────────
  Aplikasi lokal untuk menambahkan gambar ke dataset,
  lalu menjalankan pra-pemrosesan dan fine-tune model
  secara otomatis.

  Jalankan:
    python pipeline/manajer_dataset.py
====================================================
"""

import sys, shutil, threading, subprocess
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()
RAW_DIR  = ROOT_DIR / "dataset" / "raw"

HIERARCHY = {
    "anorganik": ["kaca", "karet", "logam", "styrofoam", "kardus", "plastik", "tekstil"],
    "organik"  : ["ampas", "kayu", "daun_ranting", "kertas_tisu"],
    "b3"       : ["baterai", "elektronik", "lampu_merkuri", "medis", "kimia"],
    "residu"   : ["popok_pembalut", "puntung_rokok"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BG       = "#0f0f0f"
BG2      = "#1a1a1a"
BG3      = "#252525"
ACCENT   = "#10B981"
ACCENT2  = "#059669"
FG       = "#e5e5e5"
FG2      = "#9ca3af"
RED      = "#ef4444"
YELLOW   = "#f59e0b"
FONT     = ("Consolas", 10)
FONT_SM  = ("Consolas", 9)
FONT_LG  = ("Consolas", 13, "bold")


class DatasetManager:
    """
    Aplikasi jendela (GUI) biar nambah data & latih model nggak perlu ngetik command.
    Alurnya: pilih gambar → tentukan kategorinya → masukkan ke dataset →
    klik tombol untuk preprocessing + training. Semua progress muncul di kotak log.
    """

    def __init__(self):
        """Siapkan jendela utama dan semua komponen tampilannya saat aplikasi dibuka."""
        self.root = tk.Tk()
        self.root.title("TrashTrack — Manajer Dataset")
        self.root.geometry("920x720")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.selected_files: list[Path] = []
        self._thumb_refs: list          = []   # cegah garbage collection
        self._running                   = False

        self.lvl1_var = tk.StringVar(value="anorganik")
        self.lvl2_var = tk.StringVar(value="plastik")

        self._build_ui()
        self._refresh_lvl2()
        self._refresh_status()

    # ─── UI BUILDER ───────────────────────────────────────────────────────────
    def _build_ui(self):
        """Bangun seluruh tampilan: header, kolom kiri (pilih gambar + kategori), kolom kanan (ringkasan, tombol training, log)."""
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG2, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="TrashTrack", font=("Consolas", 18, "bold"),
                 fg=ACCENT, bg=BG2).pack()
        tk.Label(hdr, text="Manajer Dataset — Tambah Data & Fine-tune Model",
                 font=FONT_SM, fg=FG2, bg=BG2).pack()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # ── Kolom kiri ────────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Section 1: Pilih Gambar
        self._section(left, "1  PILIH GAMBAR")

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(0, 6))
        self._btn(btn_row, "Pilih File", self._select_files).pack(side="left", padx=(0, 6))
        self._btn(btn_row, "Pilih Folder", self._select_folder).pack(side="left")
        self._btn(btn_row, "Bersihkan", self._clear_selection,
                  bg=BG3, fg=FG2).pack(side="right")

        # Preview grid
        preview_frame = tk.Frame(left, bg=BG3, bd=1, relief="flat")
        preview_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.canvas = tk.Canvas(preview_frame, bg=BG3, highlightthickness=0)
        sb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)

        self.preview_inner = tk.Frame(self.canvas, bg=BG3)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.preview_inner, anchor="nw")
        self.preview_inner.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        self.file_count_label = tk.Label(left, text="Belum ada gambar dipilih",
                                         font=FONT_SM, fg=FG2, bg=BG)
        self.file_count_label.pack(pady=(0, 4))

        # Section 2: Kategori
        self._section(left, "2  PILIH KATEGORI")

        cat_frame = tk.Frame(left, bg=BG)
        cat_frame.pack(fill="x", pady=(0, 6))

        # Level 1
        tk.Label(cat_frame, text="Level 1", font=FONT_SM, fg=FG2, bg=BG).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        lvl1_cb = ttk.Combobox(cat_frame, textvariable=self.lvl1_var,
                                values=list(HIERARCHY.keys()),
                                state="readonly", width=16, font=FONT)
        lvl1_cb.grid(row=0, column=1, sticky="w", padx=(0, 16))
        lvl1_cb.bind("<<ComboboxSelected>>", self._refresh_lvl2)

        # Level 2
        tk.Label(cat_frame, text="Level 2", font=FONT_SM, fg=FG2, bg=BG).grid(
            row=0, column=2, sticky="w", padx=(0, 8))
        self.lvl2_combo = ttk.Combobox(cat_frame, textvariable=self.lvl2_var,
                                       state="readonly", width=20, font=FONT)
        self.lvl2_combo.grid(row=0, column=3, sticky="w")
        self.lvl2_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_status())

        self.status_label = tk.Label(left, text="", font=FONT_SM, fg=FG2, bg=BG)
        self.status_label.pack(pady=(4, 0))

        # Tombol tambah
        self._btn(left, "Tambahkan ke Dataset", self._add_to_dataset,
                  bg=ACCENT, fg="white", pad=12).pack(fill="x", pady=8)

        # ── Kolom kanan ───────────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG, width=310)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        # Section 3: Ringkasan Dataset
        self._section(right, "3  RINGKASAN DATASET")

        self.summary_text = tk.Text(right, font=FONT_SM, bg=BG2, fg=FG,
                                    state="disabled", relief="flat",
                                    height=12, wrap="word")
        self.summary_text.pack(fill="x", pady=(0, 8))
        self._btn(right, "Muat Ulang", self._refresh_summary,
                  bg=BG3, fg=FG2).pack(fill="x", pady=(0, 8))

        # Section 4: Pelatihan
        self._section(right, "4  PELATIHAN")

        self._btn(right, "Pra-pemrosesan + Fine-tune",
                  lambda: self._run_pipeline("finetune"),
                  bg=ACCENT, fg="white", pad=10).pack(fill="x", pady=(0, 6))
        self._btn(right, "Pra-pemrosesan + Latih Penuh",
                  lambda: self._run_pipeline("full"),
                  bg=BG3, fg=FG2, pad=10).pack(fill="x", pady=(0, 8))

        tk.Label(right,
                 text="Fine-tune: cepat, untuk data baru\nTrain penuh: lambat, dari nol",
                 font=FONT_SM, fg=FG2, bg=BG, justify="left").pack(anchor="w")

        # Section 5: Log
        self._section(right, "5  LOG", pady_top=12)

        log_frame = tk.Frame(right, bg=BG3)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, font=("Consolas", 8), bg=BG3, fg=FG,
                                state="disabled", relief="flat", wrap="word")
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        self._refresh_summary()
        self._log("Manajer Dataset siap. Pilih gambar dan kategori untuk memulai.")

    def _section(self, parent, title, pady_top=8):
        """Bikin judul section + garis pemisah, biar tampilan rapi terbagi per bagian."""
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(pady_top, 4))
        tk.Label(f, text=title, font=("Consolas", 10, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Frame(f, bg=BG3, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=6)

    def _btn(self, parent, text, cmd, bg=BG3, fg=FG, pad=6):
        """Pembuat tombol biar gayanya seragam tanpa nulis ulang konfigurasi yang sama terus."""
        return tk.Button(parent, text=text, command=cmd, font=FONT,
                         bg=bg, fg=fg, activebackground=ACCENT2, activeforeground="white",
                         relief="flat", bd=0, padx=pad, pady=pad, cursor="hand2")

    # ─── PILIH FILE ───────────────────────────────────────────────────────────
    def _select_files(self):
        """Buka dialog untuk memilih satu atau beberapa file gambar."""
        files = filedialog.askopenfilenames(
            title="Pilih Gambar",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Semua", "*.*")]
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self._update_preview()

    def _select_folder(self):
        """Pilih satu folder sekaligus — semua gambar di dalamnya langsung diambil."""
        folder = filedialog.askdirectory(title="Pilih Folder Berisi Gambar")
        if folder:
            self.selected_files = [f for f in Path(folder).iterdir()
                                   if f.suffix.lower() in IMG_EXTS]
            self._update_preview()

    def _clear_selection(self):
        """Kosongkan daftar gambar yang dipilih beserta thumbnail-nya."""
        self.selected_files = []
        self._thumb_refs.clear()
        for w in self.preview_inner.winfo_children():
            w.destroy()
        self.file_count_label.config(text="Belum ada gambar dipilih")

    # ─── PREVIEW GRID ─────────────────────────────────────────────────────────
    def _update_preview(self):
        """Tampilkan thumbnail gambar yang dipilih (maksimal 30) biar bisa dicek dulu sebelum dimasukkan."""
        for w in self.preview_inner.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        n = len(self.selected_files)
        self.file_count_label.config(text=f"{n} gambar dipilih")

        if not PIL_OK:
            tk.Label(self.preview_inner, text="Install Pillow untuk preview gambar",
                     font=FONT_SM, fg=FG2, bg=BG3).pack(padx=8, pady=8)
            return

        cols = 5
        for i, path in enumerate(self.selected_files[:30]):   # max 30 preview
            row, col = divmod(i, cols)
            cell = tk.Frame(self.preview_inner, bg=BG3, padx=2, pady=2)
            cell.grid(row=row, column=col, padx=2, pady=2)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((80, 80), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_refs.append(photo)
                tk.Label(cell, image=photo, bg=BG3).pack()
            except Exception:
                tk.Label(cell, text="?", font=FONT, fg=FG2, bg=BG3,
                         width=5, height=3).pack()
            tk.Label(cell, text=path.name[:12], font=("Consolas", 7),
                     fg=FG2, bg=BG3).pack()

        if n > 30:
            tk.Label(self.preview_inner,
                     text=f"... dan {n-30} gambar lainnya",
                     font=FONT_SM, fg=FG2, bg=BG3).grid(
                         row=(30//cols)+1, column=0, columnspan=cols, pady=4)

    # ─── KATEGORI ─────────────────────────────────────────────────────────────
    def _refresh_lvl2(self, *_):
        """Saat kategori besar (Level 1) diganti, isi ulang pilihan jenis (Level 2) yang sesuai."""
        lvl1    = self.lvl1_var.get()
        options = HIERARCHY.get(lvl1, [])
        self.lvl2_combo["values"] = options
        self.lvl2_var.set(options[0] if options else "")
        self._refresh_status()

    def _refresh_status(self, *_):
        """Tampilkan berapa gambar yang sudah ada di folder kategori yang sedang dipilih."""
        lvl1  = self.lvl1_var.get()
        lvl2  = self.lvl2_var.get()
        folder = RAW_DIR / lvl1 / lvl2
        count  = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        self.status_label.config(
            text=f"dataset/raw/{lvl1}/{lvl2}/ — {count} gambar tersedia",
            fg=ACCENT if count > 0 else YELLOW)

    # ─── RINGKASAN DATASET ────────────────────────────────────────────────────
    def _refresh_summary(self):
        """Hitung & tampilkan berapa gambar yang sudah ada di tiap kelas, lengkap dengan bar sederhana."""
        lines = []
        total = 0
        for lvl1, classes in HIERARCHY.items():
            lvl1_total = 0
            for cls in classes:
                folder = RAW_DIR / lvl1 / cls
                n = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
                lvl1_total += n
                total      += n
                bar = "#" * min(10, n // 100) + "-" * max(0, 10 - n // 100)
                lines.append(f"  {cls:<18} {n:>4}  {bar}")
            lines.insert(len(lines) - len(classes),
                         f"\n{lvl1.upper()} ({lvl1_total})")
        lines.append(f"\nTOTAL: {total} gambar")

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)
        for line in lines:
            color = ACCENT if line.startswith("\n") and not line.startswith("\nTOTAL") else FG
            tag = f"tag_{id(line)}"
            self.summary_text.tag_config(tag, foreground=color)
            self.summary_text.insert(tk.END, line + "\n", tag)
        self.summary_text.config(state="disabled")

    # ─── TAMBAH KE DATASET ────────────────────────────────────────────────────
    def _add_to_dataset(self):
        """
        Salin gambar yang dipilih ke folder dataset/raw/{kategori}/{jenis}/.
        Penamaan otomatis lanjut dari nomor terakhir (000001, 000002, ...) biar nggak nimpa data lama.
        """
        if not self.selected_files:
            messagebox.showwarning("Tidak ada file", "Pilih gambar terlebih dahulu.")
            return

        lvl1 = self.lvl1_var.get()
        lvl2 = self.lvl2_var.get()
        if not lvl1 or not lvl2:
            messagebox.showwarning("Kategori kosong", "Pilih Level 1 dan Level 2 terlebih dahulu.")
            return

        dst = RAW_DIR / lvl1 / lvl2
        dst.mkdir(parents=True, exist_ok=True)

        # Nomor file berikutnya
        existing_nums = []
        for f in dst.glob("*.jpg"):
            try: existing_nums.append(int(f.stem))
            except: pass
        next_num = max(existing_nums) + 1 if existing_nums else 1

        copied, failed = 0, 0
        for i, src in enumerate(self.selected_files):
            dst_file = dst / f"{next_num + i:06d}.jpg"
            try:
                if PIL_OK:
                    img = Image.open(src).convert("RGB")
                    img.save(dst_file, "JPEG", quality=92)
                else:
                    shutil.copy2(src, dst_file)
                copied += 1
            except Exception as e:
                self._log(f"  Gagal {src.name}: {e}")
                failed += 1

        self._log(f"{copied} gambar → dataset/raw/{lvl1}/{lvl2}/"
                  + (f"  ({failed} gagal)" if failed else ""))
        self._clear_selection()
        self._refresh_status()
        self._refresh_summary()

    # ─── PIPELINE ─────────────────────────────────────────────────────────────
    def _run_pipeline(self, mode: str):
        """
        Jalanin preprocessing lalu training di thread terpisah (biar GUI nggak nge-freeze).
        mode 'finetune' = lanjut dari model lama (cepat); mode 'full' = latih dari nol.
        """
        if self._running:
            messagebox.showinfo("Sedang berjalan", "Pipeline sudah berjalan. Tunggu selesai.")
            return

        confirm = messagebox.askyesno(
            "Konfirmasi",
            f"{'Fine-tune (cepat)' if mode == 'finetune' else 'Training penuh (lama)'} akan dimulai.\n"
            f"Proses ini membutuhkan GPU dan bisa memakan waktu lama.\n\nLanjutkan?"
        )
        if not confirm:
            return

        self._running = True

        def task():
            python = sys.executable
            self._log(f"\n{'='*45}")
            self._log(f"Memulai {'fine-tune' if mode == 'finetune' else 'training penuh'}...")

            # Langkah 1: Pra-pemrosesan
            self._log("\n[1/2] Pra-pemrosesan dataset...")
            ok = self._run_subprocess([python, str(ROOT_DIR / "pipeline" / "1_prapemrosesan.py")])

            if not ok:
                self._log("Pra-pemrosesan gagal!")
                self._running = False
                return

            # Langkah 2: Pelatihan / Fine-tune
            self._log(f"\n[2/2] {'Fine-tuning' if mode == 'finetune' else 'Training penuh'}...")
            if mode == "finetune":
                ok = self._run_subprocess([
                    python, str(ROOT_DIR / "pipeline" / "2_pelatihan_yolo.py"),
                    "--finetune", "--epochs", "30",
                ])
            else:
                ok = self._run_subprocess([python, str(ROOT_DIR / "pipeline" / "2_pelatihan_yolo.py")])

            if ok:
                self._log("\nSelesai! Model diperbarui di model/yolo_best.pt")
                self._log("  Restart backend agar model baru ter-load.")
            else:
                self._log("\nTraining gagal. Cek log di atas.")

            self._running = False

        threading.Thread(target=task, daemon=True).start()

    def _run_subprocess(self, cmd: list) -> bool:
        """Jalankan skrip Python lain (preprocessing/training) sambil menyalurkan outputnya ke kotak log."""
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(ROOT_DIR), encoding="utf-8", errors="replace"
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log(line)
            proc.wait()
            return proc.returncode == 0
        except Exception as e:
            self._log(f"Error: {e}")
            return False

    # ─── LOG ──────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        """Tulis pesan ke kotak log dengan stempel waktu. Aman dipanggil dari thread lain."""
        ts = datetime.now().strftime("%H:%M:%S")
        def update():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")
        self.root.after(0, update)

    def run(self):
        """Tampilkan jendela dan mulai aplikasi (loop event tkinter)."""
        self.root.mainloop()


if __name__ == "__main__":
    app = DatasetManager()
    app.run()
