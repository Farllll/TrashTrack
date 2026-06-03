import { useRef, useState, useEffect, useCallback } from "react";

// ─── Konstanta ────────────────────────────────────────────────────────────────
const DW = 160, DH = 120;            // resolusi deteksi 4:3
const GW = 16,  GH = 12;             // grid cells untuk blob detection
const CW = DW / GW, CH = DH / GH;

// Gaussian background model
const BG_ALPHA       = 0.006;         // laju adaptasi background (lambat)
const BG_ALPHA_FAST  = 0.05;          // laju saat tidak ada objek
const MIN_STD        = 9;             // std minimum (hindari false negative di area flat)
const K_SIGMA        = 2.8;           // sensitivitas threshold (higher = kurang sensitif)

// Blob & tracking
const CELL_THRESH    = 0.25;
const MIN_CELLS      = 4;
const FOCUS_MARGIN   = 3;             // cell margin untuk lock zone
const EMA            = 0.15;          // bbox smoothing (lebih kecil = lebih halus)
const TRACK_MS       = 3000;
const CLEAR_MS       = 1500;
const COOLDOWN       = 4000;
const BOX_PAD        = 0.04;

// Kotak panduan — deteksi awal hanya dalam area ini (normalized 0-1)
// Setelah objek terdeteksi di sini, tracking bisa mengikuti ke mana saja
const GUIDE = { x1: 0.18, y1: 0.18, x2: 0.82, y2: 0.82 };
const GUIDE_GRID = {
  gx1: Math.round(GUIDE.x1 * GW),
  gy1: Math.round(GUIDE.y1 * GH),
  gx2: Math.round(GUIDE.x2 * GW) - 1,
  gy2: Math.round(GUIDE.y2 * GH) - 1,
};

// ─── Operasi Morfologi ────────────────────────────────────────────────────────
function erode(mask, W, H) {
  const out = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (mask[i] && mask[i-1] && mask[i+1] && mask[i-W] && mask[i+W]) out[i] = 1;
    }
  return out;
}

function dilate(mask, W, H) {
  const out = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (mask[i] || mask[i-1] || mask[i+1] || mask[i-W] || mask[i+W]) out[i] = 1;
    }
  return out;
}

// ─── Cari blob terbesar dalam zona ───────────────────────────────────────────
function findLargestBlob(grid, lockZone) {
  const gx1s = lockZone ? Math.max(0,    lockZone.gx1 - FOCUS_MARGIN) : 0;
  const gy1s = lockZone ? Math.max(0,    lockZone.gy1 - FOCUS_MARGIN) : 0;
  const gx2s = lockZone ? Math.min(GW-1, lockZone.gx2 + FOCUS_MARGIN) : GW - 1;
  const gy2s = lockZone ? Math.min(GH-1, lockZone.gy2 + FOCUS_MARGIN) : GH - 1;

  const visited = new Uint8Array(GW * GH);
  let bestBlob = null, bestScore = 0;

  for (let gy = gy1s; gy <= gy2s; gy++) {
    for (let gx = gx1s; gx <= gx2s; gx++) {
      const seed = gy * GW + gx;
      if (visited[seed] || grid[seed] <= CELL_THRESH) continue;

      const cells = [];
      const q = [seed];
      visited[seed] = 1;
      let score = 0;

      while (q.length) {
        const idx = q.shift();
        const cx = idx % GW, cy = Math.floor(idx / GW);
        cells.push(idx);
        score += grid[idx];
        for (const [nx, ny] of [[cx-1,cy],[cx+1,cy],[cx,cy-1],[cx,cy+1]]) {
          if (nx < gx1s || nx > gx2s || ny < gy1s || ny > gy2s) continue;
          const ni = ny * GW + nx;
          if (!visited[ni] && grid[ni] > CELL_THRESH) { visited[ni] = 1; q.push(ni); }
        }
      }

      if (cells.length >= MIN_CELLS && score > bestScore) {
        bestScore = score;
        let bx1 = GW, by1 = GH, bx2 = 0, by2 = 0;
        for (const i of cells) {
          const cx = i % GW, cy = Math.floor(i / GW);
          if (cx < bx1) bx1 = cx; if (cx > bx2) bx2 = cx;
          if (cy < by1) by1 = cy; if (cy > by2) by2 = cy;
        }
        bestBlob = { gx1: bx1, gy1: by1, gx2: bx2, gy2: by2 };
      }
    }
  }
  return bestBlob;
}

export default function WebcamCapture({ onCapture, dark, theme }) {
  const videoRef   = useRef(null);
  const detectRef  = useRef(null);
  const overlayRef = useRef(null);
  const streamRef  = useRef(null);
  const rafRef     = useRef(null);

  // Gaussian background model
  const bgMean     = useRef(null);     // Float32Array(N * 3)
  const bgVar      = useRef(null);     // Float32Array(N * 3)
  const bboxRef    = useRef(null);
  const lockZoneRef    = useRef(null);
  const lastSeenRef    = useRef(0);
  const trackStartRef  = useRef(null);
  const lastScanRef    = useRef(0);

  const [camState, setCamState]   = useState("idle");
  const [errorMsg, setErrorMsg]   = useState("");
  const [cameras, setCameras]     = useState([]);
  const [camIdx, setCamIdx]       = useState(0);
  const [flash, setFlash]         = useState(false);
  const [autoScan, setAutoScan]   = useState(true);
  const [scanning, setScanning]   = useState(false);
  const [phase, setPhase]         = useState("idle");
  const [trackPct, setTrackPct]   = useState(0);
  const [warmup, setWarmup]       = useState(0);   // 0-100 warmup progress

  const accent      = dark ? "#A3E635" : "#10B981";
  const zoneBorder  = dark ? "border-gray-700" : "border-gray-200";
  const cardBg      = dark ? "bg-[#111]"       : "bg-white";
  const labelColor  = dark ? "text-gray-500"   : "text-gray-400";
  const headingColor = dark ? "text-gray-50"   : "text-gray-800";
  const subColor    = dark ? "text-gray-500"   : "text-gray-500";
  const btnSec      = dark
    ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
    : "bg-gray-100 border-gray-200 text-gray-600 hover:bg-gray-200";
  const errColor    = dark ? "text-red-400" : "text-red-500";
  const errBg       = dark ? "bg-red-950/30 border-red-900" : "bg-red-50 border-red-200";
  const toggleBg    = autoScan
    ? (dark ? "bg-lime-400/20 border-lime-500 text-lime-400"  : "bg-emerald-50 border-emerald-400 text-emerald-600")
    : (dark ? "bg-gray-800 border-gray-700 text-gray-500"     : "bg-gray-100 border-gray-200 text-gray-400");

  const warmupFrameRef = useRef(0);   // frame counter warmup
  const WARMUP_TOTAL   = 40;          // frames untuk inisialisasi model Gaussian

  // ── Reset model ───────────────────────────────────────────────────────────
  const resetModel = useCallback(() => {
    bgMean.current = null; bgVar.current = null;
    bboxRef.current = null; lockZoneRef.current = null;
    trackStartRef.current = null; lastSeenRef.current = 0;
    warmupFrameRef.current = 0;
    setPhase("warmup"); setTrackPct(0); setWarmup(0);
    const cv = overlayRef.current;
    if (cv) cv.getContext("2d")?.clearRect(0, 0, cv.width, cv.height);
  }, []);

  // ── Gambar overlay ────────────────────────────────────────────────────────
  const drawOverlay = useCallback((bbox, pct, curPhase) => {
    const cv = overlayRef.current;
    if (!cv) return;
    const W = cv.offsetWidth, H = cv.offsetHeight;
    if (!W || !H) return;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    if (curPhase === "warmup") {
      // Progress bar tipis di atas
      ctx.fillStyle = accent + "33";
      ctx.fillRect(0, 0, W, 2);
      ctx.fillStyle = accent;
      ctx.fillRect(0, 0, W * (pct / 100), 2);
      return;
    }

    // Guide zone — kotak panduan untuk penempatan objek
    if (!bbox) {
      const gx = W * 0.18, gy = H * 0.18;
      const gw = W * 0.64, gh = H * 0.64;
      const cs = 16;
      ctx.strokeStyle = accent + "45"; ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 5]);
      ctx.strokeRect(gx, gy, gw, gh);
      ctx.setLineDash([]);
      // Solid corners
      ctx.strokeStyle = accent + "80"; ctx.lineWidth = 2; ctx.lineCap = "round";
      [[gx,gy,cs,0,0,cs],[gx+gw,gy,-cs,0,0,cs],[gx,gy+gh,cs,0,0,-cs],[gx+gw,gy+gh,-cs,0,0,-cs]]
        .forEach(([cx,cy,hx,hy,vx,vy]) => {
          ctx.beginPath(); ctx.moveTo(cx+hx,cy+hy); ctx.lineTo(cx,cy); ctx.lineTo(cx+vx,cy+vy); ctx.stroke();
        });
      ctx.font = "bold 11px monospace"; ctx.fillStyle = accent + "70";
      ctx.textAlign = "center";
      ctx.fillText("Letakkan sampah di sini", W / 2, gy + gh + 16);
      ctx.textAlign = "left";
      return;
    }

    const x  = Math.max(0, bbox.x1 - BOX_PAD) * W;
    const y  = Math.max(0, bbox.y1 - BOX_PAD) * H;
    const x2 = Math.min(1, bbox.x2 + BOX_PAD) * W;
    const y2 = Math.min(1, bbox.y2 + BOX_PAD) * H;
    const bw = x2 - x, bh = y2 - y;
    const cs = Math.max(14, Math.min(bw, bh) * 0.2);
    const color = curPhase === "locked" ? "#10B981" : "#F59E0B";

    // Gelap luar bbox
    ctx.fillStyle = "rgba(0,0,0,0.38)";
    ctx.fillRect(0,0,W,y); ctx.fillRect(0,y2,W,H-y2);
    ctx.fillRect(0,y,x,bh); ctx.fillRect(x2,y,W-x2,bh);

    ctx.fillStyle = color + "15"; ctx.fillRect(x, y, bw, bh);

    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineCap = "round";
    [[x,y,cs,0,0,cs],[x2,y,-cs,0,0,cs],[x,y2,cs,0,0,-cs],[x2,y2,-cs,0,0,-cs]]
      .forEach(([cx,cy,hx,hy,vx,vy]) => {
        ctx.beginPath(); ctx.moveTo(cx+hx,cy+hy); ctx.lineTo(cx,cy); ctx.lineTo(cx+vx,cy+vy); ctx.stroke();
      });

    // Progress bar
    const barY = y2 - 7;
    ctx.fillStyle = "rgba(0,0,0,0.3)"; ctx.fillRect(x+4, barY, bw-8, 3);
    ctx.fillStyle = color;              ctx.fillRect(x+4, barY, (bw-8)*(pct/100), 3);

    // Label
    const secs  = ((TRACK_MS*(1-pct/100))/1000).toFixed(1);
    const label = curPhase === "locked" ? "● SCAN..." : `◎  ${secs}s`;
    ctx.font = "bold 11px monospace";
    const lw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(x+4, y+4, lw+8, 18);
    ctx.fillStyle = color;              ctx.fillText(label, x+8, y+16);
  }, [accent]);

  // ── Crop ke bbox lalu kirim ke backend ───────────────────────────────────
  const triggerScan = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    setPhase("locked"); setScanning(true);
    setFlash(true); setTimeout(() => setFlash(false), 200);

    const tmpCanvas = document.createElement("canvas");
    const tmpCtx    = tmpCanvas.getContext("2d");
    const bbox      = bboxRef.current;

    if (bbox) {
      // Crop ke area objek (dengan padding 12%) untuk kirim ke model
      const vw = video.videoWidth, vh = video.videoHeight;
      const padX = (bbox.x2 - bbox.x1) * 0.12;
      const padY = (bbox.y2 - bbox.y1) * 0.12;
      const sx = Math.max(0, (bbox.x1 - padX) * vw);
      const sy = Math.max(0, (bbox.y1 - padY) * vh);
      const sw = Math.min(vw - sx, (bbox.x2 - bbox.x1 + padX * 2) * vw);
      const sh = Math.min(vh - sy, (bbox.y2 - bbox.y1 + padY * 2) * vh);
      tmpCanvas.width = 224; tmpCanvas.height = 224;
      tmpCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);
    } else {
      tmpCanvas.width = video.videoWidth; tmpCanvas.height = video.videoHeight;
      tmpCtx.drawImage(video, 0, 0);
    }

    tmpCanvas.toBlob((blob) => {
      if (!blob) { setScanning(false); return; }
      onCapture(new File([blob], "autoscan.jpg", { type: "image/jpeg" }));
      bboxRef.current = null; lockZoneRef.current = null;
      trackStartRef.current = null; lastSeenRef.current = 0;
      setTrackPct(0); setPhase("active");
      drawOverlay(null, 0, "active");
      setTimeout(() => setScanning(false), 2500);
    }, "image/jpeg", 0.95);
  }, [onCapture, drawOverlay]);

  // ── Perulangan utama: pengurangan latar belakang Gaussian + tracking ────────
  const processFrame = useCallback(() => {
    const video = videoRef.current, cv = detectRef.current;
    if (!video || !cv || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(processFrame); return;
    }

    cv.width = DW; cv.height = DH;
    const ctx = cv.getContext("2d");
    ctx.drawImage(video, 0, 0, DW, DH);
    const pix = ctx.getImageData(0, 0, DW, DH).data;
    const N   = DW * DH;

    // ── Inisialisasi model Gaussian ───────────────────────────────────────
    if (!bgMean.current) {
      bgMean.current = new Float32Array(N * 3);
      bgVar.current  = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        bgMean.current[i*3]   = pix[i*4];
        bgMean.current[i*3+1] = pix[i*4+1];
        bgMean.current[i*3+2] = pix[i*4+2];
        bgVar.current[i*3]    = MIN_STD * MIN_STD;
        bgVar.current[i*3+1]  = MIN_STD * MIN_STD;
        bgVar.current[i*3+2]  = MIN_STD * MIN_STD;
      }
    }

    // ── Warmup: belajar background selama WARMUP_TOTAL frame ─────────────
    const frame = ++warmupFrameRef.current;
    if (frame <= WARMUP_TOTAL) {
      const wPct = Math.round((frame / WARMUP_TOTAL) * 100);
      setWarmup(wPct);
      drawOverlay(null, wPct, "warmup");
      const mu = bgMean.current, v = bgVar.current;
      for (let i = 0; i < N; i++) {
        for (let c = 0; c < 3; c++) {
          const diff = pix[i*4+c] - mu[i*3+c];
          mu[i*3+c]  += 0.2 * diff;
          v[i*3+c]    = Math.max(MIN_STD**2, v[i*3+c] + 0.2 * (diff*diff - v[i*3+c]));
        }
      }
      if (frame === WARMUP_TOTAL) { setPhase("active"); drawOverlay(null, 0, "active"); }
      rafRef.current = requestAnimationFrame(processFrame); return;
    }

    const mu = bgMean.current, v = bgVar.current;
    const bbox = bboxRef.current;

    // ── Tentukan apakah pixel dalam lock zone (bbox yang sedang di-track) ──
    const lz = lockZoneRef.current;
    const lockPx = lz ? {
      x1: lz.gx1 * CW, y1: lz.gy1 * CH, x2: (lz.gx2+1) * CW, y2: (lz.gy2+1) * CH,
    } : null;

    // ── Langkah 1: Deteksi FG dengan threshold adaptif per pixel ─────────
    const fgMask = new Uint8Array(N);
    for (let py = 0; py < DH; py++) {
      for (let px = 0; px < DW; px++) {
        const i = py * DW + px;
        const p = i * 4, b = i * 3;
        let isFG = false;
        for (let c = 0; c < 3; c++) {
          const diff = Math.abs(pix[p+c] - mu[b+c]);
          const std  = Math.sqrt(v[b+c]);
          if (diff > K_SIGMA * std) { isFG = true; break; }
        }
        fgMask[i] = isFG ? 1 : 0;

        // Update background (STOP di dalam lock zone agar objek diam tetap terdeteksi)
        const inLock = lockPx &&
          px >= lockPx.x1 && px <= lockPx.x2 &&
          py >= lockPx.y1 && py <= lockPx.y2;
        const rate = inLock ? 0 : (isFG ? BG_ALPHA * 0.3 : BG_ALPHA_FAST);

        if (rate > 0) {
          for (let c = 0; c < 3; c++) {
            const diff = pix[p+c] - mu[b+c];
            mu[b+c] += rate * diff;
            v[b+c]   = Math.max(MIN_STD**2, v[b+c] + rate * (diff*diff - v[b+c]));
          }
        }
      }
    }

    // ── Langkah 2: Penutupan morfologi (dilate → erode) untuk bersihkan mask
    const dilated = dilate(fgMask, DW, DH);
    const clean   = erode(dilated, DW, DH);

    // ── Langkah 3: Kepadatan grid dari mask bersih ───────────────────────
    const grid = new Float32Array(GW * GH);
    for (let gy = 0; gy < GH; gy++)
      for (let gx = 0; gx < GW; gx++) {
        let cnt = 0, tot = 0;
        const x0 = Math.round(gx*CW), x1 = Math.round((gx+1)*CW);
        const y0 = Math.round(gy*CH), y1 = Math.round((gy+1)*CH);
        for (let py = y0; py < y1; py++)
          for (let px = x0; px < x1; px++) { if (clean[py*DW+px]) cnt++; tot++; }
        grid[gy*GW+gx] = tot ? cnt/tot : 0;
      }

    // ── Step 4: Blob terbesar → lock zone → bbox ─────────────────────────
    // Jika belum ada lock zone → cari hanya dalam GUIDE_GRID (kotak panduan)
    // Jika sudah tracking → cari dalam lock zone (mengikuti objek)
    const searchZone = lockZoneRef.current ?? GUIDE_GRID;
    const blob = findLargestBlob(grid, searchZone);
    const now  = Date.now();

    if (blob) {
      const raw = {
        x1: (blob.gx1 * CW) / DW,
        y1: (blob.gy1 * CH) / DH,
        x2: ((blob.gx2+1) * CW) / DW,
        y2: ((blob.gy2+1) * CH) / DH,
      };
      if (!bboxRef.current) {
        bboxRef.current       = raw;
        lockZoneRef.current   = { gx1: blob.gx1, gy1: blob.gy1, gx2: blob.gx2, gy2: blob.gy2 };
        trackStartRef.current = now;
        setPhase("tracking");
      } else {
        bboxRef.current = {
          x1: bboxRef.current.x1*(1-EMA) + raw.x1*EMA,
          y1: bboxRef.current.y1*(1-EMA) + raw.y1*EMA,
          x2: bboxRef.current.x2*(1-EMA) + raw.x2*EMA,
          y2: bboxRef.current.y2*(1-EMA) + raw.y2*EMA,
        };
        lockZoneRef.current = { gx1: blob.gx1, gy1: blob.gy1, gx2: blob.gx2, gy2: blob.gy2 };
      }
      lastSeenRef.current = now;
    }

    // Objek hilang terlalu lama
    if (bboxRef.current && (now - lastSeenRef.current) > CLEAR_MS) {
      bboxRef.current = null; lockZoneRef.current = null;
      trackStartRef.current = null;
      setPhase("active"); setTrackPct(0);
      drawOverlay(null, 0, "active");
      rafRef.current = requestAnimationFrame(processFrame); return;
    }

    // ── Langkah 5: Hitung mundur → picu scan ─────────────────────────────
    if (bboxRef.current && trackStartRef.current) {
      const pct = Math.min(100, ((now - trackStartRef.current) / TRACK_MS) * 100);
      setTrackPct(pct);
      if (pct >= 100 && autoScan && !scanning && now - lastScanRef.current > COOLDOWN) {
        lastScanRef.current = now;
        drawOverlay(bboxRef.current, 100, "locked");
        setTimeout(() => triggerScan(), 150);
      } else {
        drawOverlay(bboxRef.current, pct, "tracking");
      }
    } else {
      drawOverlay(null, 0, "active");
    }

    rafRef.current = requestAnimationFrame(processFrame);
  }, [autoScan, scanning, drawOverlay, triggerScan]);

  useEffect(() => {
    if (camState === "active") {
      resetModel();
      rafRef.current = requestAnimationFrame(processFrame);
    } else {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      bgMean.current = null; bgVar.current = null;
      bboxRef.current = null; lockZoneRef.current = null;
      warmupFrameRef.current = 0;
      setPhase("idle"); setTrackPct(0); setWarmup(0);
    }
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [camState, processFrame, resetModel]);

  // ── Kamera ────────────────────────────────────────────────────────────────
  const loadCameraList = useCallback(async () => {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter(d => d.kind === "videoinput").map((d, i) => ({
        id: d.deviceId, label: d.label || `Kamera ${i + 1}`,
      }));
      setCameras(cams);
    } catch (_) {}
  }, []);

  const startCamera = useCallback(async (deviceId = null, idx = 0) => {
    setCamState("loading"); setErrorMsg("");
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId
          ? { deviceId: { exact: deviceId }, width:{ideal:1280}, height:{ideal:960} }
          : { facingMode: "environment",    width:{ideal:1280}, height:{ideal:960} },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play(); }
      setCamState("active"); setCamIdx(idx);
      await loadCameraList();
    } catch (err) {
      setErrorMsg(
        err.name === "NotAllowedError" ? "Akses kamera ditolak. Izinkan di browser."
        : err.name === "NotFoundError"  ? "Kamera tidak ditemukan."
        : "Gagal membuka kamera."
      );
      setCamState("error");
    }
  }, [loadCameraList]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamState("idle"); setScanning(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const captureManual = () => {
    const video = videoRef.current;
    if (!video) return;
    setFlash(true); setTimeout(() => setFlash(false), 200);
    const tmpCanvas = document.createElement("canvas");
    const tmpCtx    = tmpCanvas.getContext("2d");
    const bbox      = bboxRef.current;
    if (bbox) {
      const vw = video.videoWidth, vh = video.videoHeight;
      const padX = (bbox.x2 - bbox.x1) * 0.15;
      const padY = (bbox.y2 - bbox.y1) * 0.15;
      const sx = Math.max(0, (bbox.x1-padX)*vw), sy = Math.max(0, (bbox.y1-padY)*vh);
      const sw = Math.min(vw-sx, (bbox.x2-bbox.x1+padX*2)*vw);
      const sh = Math.min(vh-sy, (bbox.y2-bbox.y1+padY*2)*vh);
      tmpCanvas.width = 224; tmpCanvas.height = 224;
      tmpCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);
    } else {
      tmpCanvas.width = video.videoWidth; tmpCanvas.height = video.videoHeight;
      tmpCtx.drawImage(video, 0, 0);
    }
    tmpCanvas.toBlob((blob) => {
      stopCamera();
      onCapture(new File([blob], "webcam-capture.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.95);
  };

  const statusLabel = scanning              ? "Mengklasifikasi..."
    : phase === "locked"                    ? "Mengambil gambar..."
    : phase === "tracking"                  ? `Melacak... ${((TRACK_MS*(1-trackPct/100))/1000).toFixed(1)}s`
    : phase === "active"                    ? "Letakkan sampah di dalam kotak panduan"
    : phase === "warmup"                    ? `Mempersiapkan... ${warmup}%`
    : "";

  const statusColor = (scanning||phase==="locked") ? "#10B981"
    : phase === "tracking" ? "#F59E0B" : accent;

  return (
    <section className={`flex flex-col gap-4 px-4 md:px-8 py-6 md:py-10 border-b md:border-b-0 md:border-r ${theme.divider} transition-colors duration-300`}>

      <div className="flex flex-col gap-1">
        <p className={`text-[11px] ${theme.accent} font-bold tracking-[0.15em]`}>/ WEBCAM LANGSUNG</p>
        <h1 className={`text-[24px] md:text-[32px] font-bold ${headingColor} leading-tight m-0`}>Scan Langsung</h1>
        <p className={`text-[12px] ${subColor} m-0`}>
          Letakkan sampah di kotak panduan — sistem track 3 detik lalu klasifikasi otomatis.
        </p>
      </div>

      {/* Frame kamera — full width 4:3 */}
      <div className={`relative w-full rounded-xl overflow-hidden border ${zoneBorder} ${cardBg}`}
        style={{ aspectRatio: "4/3" }}>

        <video ref={videoRef} autoPlay playsInline muted
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${camState==="active"?"opacity-100":"opacity-0"}`}
        />
        <canvas ref={overlayRef}
          className={`absolute inset-0 w-full h-full z-10 pointer-events-none transition-opacity duration-300 ${camState==="active"?"opacity-100":"opacity-0"}`}
        />
        <canvas ref={detectRef} className="hidden" />

        {flash && <div className="absolute inset-0 bg-white/80 z-20 pointer-events-none" />}

        {camState === "active" && (
          <>
            <button onClick={resetModel} title="Reset background (tekan jika tracking tidak akurat)"
              className="absolute top-2 right-2 z-10 text-[11px] px-2 py-1 rounded-md bg-black/55 text-white border border-white/20 hover:bg-black/75 cursor-pointer">
              ↺ Reset
            </button>
            {statusLabel && (
              <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-10 text-[10px] px-3 py-1 rounded-full font-bold whitespace-nowrap"
                style={{ background: "rgba(0,0,0,0.65)", color: statusColor, border: `1px solid ${statusColor}55` }}>
                {statusLabel}
              </div>
            )}
            {scanning && (
              <div className="absolute inset-0 z-10 pointer-events-none rounded-xl"
                style={{ border: `2px solid ${accent}`, animation: "pulse 1s ease-in-out infinite", opacity: 0.6 }} />
            )}
          </>
        )}

        {camState === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
            <p className={`text-sm font-bold ${labelColor}`}>Kamera belum aktif</p>
          </div>
        )}
        {camState === "loading" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 opacity-70">
            <span className="text-[48px] animate-spin-slow select-none" style={{ color: accent }}>◌</span>
            <p className="text-sm font-bold" style={{ color: accent }}>Membuka kamera...</p>
          </div>
        )}
        {camState === "error" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
            <div className={`text-center text-xs ${errColor} border ${errBg} rounded-lg px-4 py-3`}>{errorMsg}</div>
          </div>
        )}
      </div>

      {/* Pilihan kamera */}
      {camState === "active" && cameras.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-0.5">
          {cameras.map((cam, i) => (
            <button key={cam.id} onClick={() => startCamera(cam.id, i)}
              className={`shrink-0 text-[11px] px-3 py-1.5 rounded-lg border font-bold transition-all cursor-pointer ${
                i === camIdx
                  ? (dark ? "bg-lime-400/20 border-lime-500 text-lime-400" : "bg-emerald-50 border-emerald-400 text-emerald-700")
                  : btnSec
              }`}>
              {cam.label.length > 22 ? cam.label.slice(0,22)+"…" : cam.label}
            </button>
          ))}
        </div>
      )}

      {/* Tombol aksi */}
      <div className="flex gap-2">
        {(camState === "idle" || camState === "error") && (
          <button onClick={() => startCamera()}
            className="flex-1 py-2.5 rounded-lg text-sm font-bold text-black cursor-pointer border-0 hover:opacity-90 transition-opacity"
            style={{ background: accent }}>
            Buka Kamera
          </button>
        )}
        {camState === "active" && (
          <>
            <button onClick={() => setAutoScan(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-2.5 rounded-lg text-[11px] font-bold border cursor-pointer transition-all ${toggleBg}`}>
              <span>{autoScan ? "Auto ON" : "Auto OFF"}</span>
            </button>
            <button onClick={captureManual} disabled={scanning}
              className="flex-1 py-2.5 rounded-lg text-sm font-bold text-black cursor-pointer border-0 hover:opacity-90 disabled:opacity-50 transition-opacity"
              style={{ background: accent }}>
              Ambil Foto
            </button>
            <button onClick={stopCamera}
              className={`px-3 py-2.5 rounded-lg text-sm border cursor-pointer transition-colors ${btnSec}`}>✕</button>
          </>
        )}
      </div>

      <style>{`@keyframes pulse{0%,100%{opacity:.3}50%{opacity:.8}}`}</style>
    </section>
  );
}
