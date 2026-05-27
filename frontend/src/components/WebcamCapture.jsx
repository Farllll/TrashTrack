import { useRef, useState, useEffect, useCallback } from "react";

const MOTION_THRESHOLD = 30;      // sensitivitas perubahan pixel (0-255)
const MOTION_MIN_PIXELS = 1500;   // minimum pixel berubah baru dianggap ada objek
const SCAN_COOLDOWN = 3000;       // jeda antar scan (ms) biar tidak spam
const CONFIDENCE_MIN = 75;        // minimum confidence yang ditampilkan

export default function WebcamCapture({ onCapture, dark, theme }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const prevFrameRef = useRef(null); // frame sebelumnya untuk diff
  const rafRef = useRef(null); // requestAnimationFrame id
  const lastScanRef = useRef(0);    // timestamp scan terakhir
  const streamRef = useRef(null);

  const [camState, setCamState] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [facingMode, setFacingMode] = useState("environment");
  const [flash, setFlash] = useState(false);
  const [motionLevel, setMotionLevel] = useState(0);   // 0-100 untuk UI bar
  const [autoScan, setAutoScan] = useState(true); // toggle autoscan
  const [scanning, setScanning] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Menunggu objek...");

  const accentHex = dark ? "#A3E635" : "#10B981";
  const zoneBg = dark ? "bg-[#111]" : "bg-white";
  const zoneBorder = dark ? "border-gray-700" : "border-gray-300";
  const labelColor = dark ? "text-gray-500" : "text-gray-400";
  const headingColor = dark ? "text-gray-50" : "text-gray-800";
  const subColor = dark ? "text-gray-500" : "text-gray-500";
  const btnSecondary = dark
    ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
    : "bg-gray-100 border-gray-300 text-gray-600 hover:bg-gray-200";
  const errorColor = dark ? "text-red-400" : "text-red-500";
  const errorBg = dark ? "bg-red-950/30 border-red-900" : "bg-red-50 border-red-200";
  const motionBg = dark ? "bg-gray-800" : "bg-gray-200";
  const toggleBg = autoScan
    ? (dark ? "bg-lime-400/20 border-lime-500 text-lime-400" : "bg-emerald-50 border-emerald-400 text-emerald-600")
    : (dark ? "bg-gray-800 border-gray-700 text-gray-500" : "bg-gray-100 border-gray-300 text-gray-400");

  // ─── Motion detection loop ───────────────────────────────────────────
  const detectMotion = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(detectMotion);
      return;
    }

    const W = 160, H = 90; // resolusi kecil untuk performa
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, W, H);
    const curr = ctx.getImageData(0, 0, W, H).data;

    let changedPixels = 0;

    if (prevFrameRef.current) {
      const prev = prevFrameRef.current;
      for (let i = 0; i < curr.length; i += 4) {
        const dr = Math.abs(curr[i] - prev[i]);
        const dg = Math.abs(curr[i + 1] - prev[i + 1]);
        const db = Math.abs(curr[i + 2] - prev[i + 2]);
        if ((dr + dg + db) / 3 > MOTION_THRESHOLD) changedPixels++;
      }
    }

    // Simpan frame saat ini sebagai prev
    prevFrameRef.current = new Uint8ClampedArray(curr);

    const totalPixels = W * H;
    const motionPct = Math.min(100, Math.round((changedPixels / totalPixels) * 100 * 4));
    setMotionLevel(motionPct);

    // Trigger scan kalau motion cukup dan cooldown sudah lewat
    const now = Date.now();
    if (
      autoScan &&
      !scanning &&
      changedPixels > MOTION_MIN_PIXELS &&
      now - lastScanRef.current > SCAN_COOLDOWN
    ) {
      lastScanRef.current = now;
      triggerAutoScan();
    } else if (changedPixels > MOTION_MIN_PIXELS) {
      setStatusMsg("Objek terdeteksi! Menahan posisi...");
    } else {
      setStatusMsg("Menunggu objek di depan kamera...");
    }

    rafRef.current = requestAnimationFrame(detectMotion);
  }, [autoScan, scanning]);

  // ─── Auto scan: capture frame → kirim ke onCapture ──────────────────
  const triggerAutoScan = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    setScanning(true);
    setStatusMsg("Menganalisis objek...");
    setFlash(true);
    setTimeout(() => setFlash(false), 200);

    // Render frame resolusi penuh ke canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) { setScanning(false); return; }
      const file = new File([blob], "autoscan.jpg", { type: "image/jpeg" });
      onCapture(file);

      // Reset scanning setelah animasi selesai (estimasi 2.5 detik)
      setTimeout(() => {
        setScanning(false);
        setStatusMsg("Menunggu objek di depan kamera...");
      }, 2500);
    }, "image/jpeg", 0.92);
  }, [onCapture]);

  // ─── Start / stop loop saat kamera aktif ────────────────────────────
  useEffect(() => {
    if (camState === "active") {
      rafRef.current = requestAnimationFrame(detectMotion);
    } else {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      prevFrameRef.current = null;
      setMotionLevel(0);
    }
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [camState, detectMotion]);

  // ─── Kamera ─────────────────────────────────────────────────────────
  const startCamera = useCallback(async (facing = facingMode) => {
    setCamState("loading");
    setErrorMsg("");
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCamState("active");
      setStatusMsg("Menunggu objek di depan kamera...");
    } catch (err) {
      const msg =
        err.name === "NotAllowedError" ? "Akses kamera ditolak. Izinkan akses kamera di browser kamu."
          : err.name === "NotFoundError" ? "Kamera tidak ditemukan di perangkat ini."
            : "Gagal membuka kamera. Coba refresh halaman.";
      setErrorMsg(msg);
      setCamState("error");
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamState("idle");
    setScanning(false);
    setMotionLevel(0);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const captureManual = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    setFlash(true);
    setTimeout(() => setFlash(false), 200);
    canvas.toBlob((blob) => {
      const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
      stopCamera();
      onCapture(file);
    }, "image/jpeg", 0.95);
  };

  const flipCamera = () => {
    const next = facingMode === "environment" ? "user" : "environment";
    setFacingMode(next);
    startCamera(next);
  };

  // Warna motion bar
  const motionColor = motionLevel > 60 ? "#EF4444" : motionLevel > 30 ? "#F59E0B" : accentHex;

  return (
    <section className={`flex flex-col gap-5 md:gap-6 px-4 md:px-8 py-6 md:py-10 border-b md:border-b-0 md:border-r ${theme.divider} transition-colors duration-300`}>

      {/* Heading */}
      <div className="flex flex-col gap-1.5 md:gap-2">
        <p className={`text-[11px] ${theme.accent} font-bold tracking-[0.15em]`}>/ WEBCAM LANGSUNG</p>
        <h1 className={`text-[28px] md:text-[40px] font-bold ${headingColor} leading-tight m-0`}>
          Scan<br />Langsung
        </h1>
        <p className={`text-[12px] md:text-[13px] ${subColor} leading-relaxed max-w-sm m-0`}>
          Arahkan kamera ke sampah — sistem akan mendeteksi otomatis saat ada objek di frame.
        </p>
      </div>

      {/* Camera viewport */}
      <div className={`relative rounded-xl overflow-hidden border ${zoneBorder} ${zoneBg} min-h-48 md:min-h-65 flex items-center justify-center`}>
        <video
          ref={videoRef}
          autoPlay playsInline muted
          className={`w-full h-48 md:h-70 object-cover block ${camState === "active" ? "opacity-100" : "opacity-0"} transition-opacity duration-300`}
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Flash */}
        {flash && <div className="absolute inset-0 bg-white/80 z-20 pointer-events-none" />}

        {/* Corner frame + controls saat aktif */}
        {camState === "active" && (
          <>
            {["top-3 left-3 border-t-2 border-l-2", "top-3 right-3 border-t-2 border-r-2",
              "bottom-3 left-3 border-b-2 border-l-2", "bottom-3 right-3 border-b-2 border-r-2"
            ].map((cls, i) => (
              <div key={i} className={`absolute w-6 h-6 ${cls} pointer-events-none z-10`} style={{ borderColor: accentHex }} />
            ))}

            {/* Flip button */}
            <button
              onClick={flipCamera}
              className="absolute top-3 left-1/2 -translate-x-1/2 z-10 text-[11px] px-2.5 py-1 rounded-lg bg-black/50 text-white border border-white/20 hover:bg-black/70 transition cursor-pointer"
            >
              🔄 Balik Kamera
            </button>

            {/* Status badge */}
            <div
              className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 text-[11px] px-3 py-1 rounded-full font-bold tracking-wide whitespace-nowrap"
              style={{ background: "rgba(0,0,0,0.6)", color: scanning ? "#FCD34D" : accentHex, border: `1px solid ${accentHex}44` }}
            >
              {scanning ? "⏳ " : motionLevel > 30 ? "⚡ " : "◉ "}{statusMsg}
            </div>

            {/* Scanning pulse overlay */}
            {scanning && (
              <div className="absolute inset-0 z-10 pointer-events-none" style={{ border: `2px solid ${accentHex}`, borderRadius: "inherit", animation: "pulse 1s ease-in-out infinite", opacity: 0.6 }} />
            )}
          </>
        )}

        {/* Idle */}
        {camState === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <span className={`text-[48px] ${labelColor} select-none`}>📷</span>
            <p className={`text-sm font-bold ${labelColor}`}>Kamera belum aktif</p>
            <p className={`text-[11px] ${labelColor} opacity-60`}>Tekan tombol di bawah untuk mulai</p>
          </div>
        )}

        {/* Loading */}
        {camState === "loading" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 opacity-70">
            <span className="text-[48px] animate-spin-slow select-none" style={{ color: accentHex }}>◌</span>
            <p className="text-sm font-bold" style={{ color: accentHex }}>Membuka kamera...</p>
          </div>
        )}

        {/* Error */}
        {camState === "error" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
            <span className="text-[40px] select-none">⚠️</span>
            <div className={`text-center text-xs ${errorColor} border ${errorBg} rounded-lg px-4 py-3 leading-relaxed`}>
              {errorMsg}
            </div>
          </div>
        )}
      </div>

      {/* Motion level bar — hanya tampil saat kamera aktif */}
      {camState === "active" && (
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center">
            <span className={`text-[11px] ${labelColor} tracking-widest`}>Motion Level</span>
            <span className="text-[11px] font-bold" style={{ color: motionColor }}>{motionLevel}%</span>
          </div>
          <div className={`h-1 ${motionBg} rounded-full overflow-hidden`}>
            <div
              className="h-full rounded-full transition-all duration-100"
              style={{ width: `${motionLevel}%`, background: motionColor }}
            />
          </div>
          <p className={`text-[10px] ${labelColor} opacity-60`}>
            Threshold: {Math.round(MOTION_MIN_PIXELS / (160 * 90) * 100 * 4)}% · Cooldown: {SCAN_COOLDOWN / 1000}s
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        {(camState === "idle" || camState === "error") && (
          <button
            onClick={() => startCamera()}
            className="flex-1 py-2.5 rounded-lg text-sm font-bold tracking-wide text-black cursor-pointer border-0 transition-opacity hover:opacity-90"
            style={{ background: accentHex }}
          >
            📷 &nbsp;Buka Kamera
          </button>
        )}

        {camState === "active" && (
          <>
            {/* Toggle autoscan */}
            <button
              onClick={() => setAutoScan(v => !v)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-[12px] font-bold border transition-all cursor-pointer ${toggleBg}`}
            >
              <span>{autoScan ? "⚡" : "⏸"}</span>
              <span>{autoScan ? "Auto ON" : "Auto OFF"}</span>
            </button>

            {/* Manual capture */}
            <button
              onClick={captureManual}
              className="flex-1 py-2.5 rounded-lg text-sm font-bold tracking-wide text-black cursor-pointer border-0 transition-opacity hover:opacity-90"
              style={{ background: accentHex }}
              disabled={scanning}
            >
              ⊙ &nbsp;Capture Manual
            </button>

            {/* Stop */}
            <button
              onClick={stopCamera}
              className={`px-4 py-2.5 rounded-lg text-sm border ${btnSecondary} cursor-pointer transition-colors`}
            >
              ✕
            </button>
          </>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50%       { opacity: 0.8; }
        }
      `}</style>
    </section>
  );
}