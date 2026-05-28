import { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import UploadZone from "./components/UploadZone";
import WebcamCapture from "./components/WebcamCapture";
import ResultPanel from "./components/ResultPanel";

// Backend FastAPI — di-proxy oleh Vite (lihat vite.config.js)
const BACKEND = "/api";

export default function App() {
  const [image, setImage]       = useState(null);
  const [result, setResult]     = useState(null);
  const [scanning, setScanning] = useState(false);
  const [scanPct, setScanPct]   = useState(0);
  const [dark, setDark]         = useState(false);
  const [mode, setMode]         = useState("upload");
  const [apiError, setApiError] = useState(null);
  const [backendOk, setBackendOk] = useState(null); // null=belum cek, true/false

  // Cek backend saat mount
  useEffect(() => {
    fetch(BACKEND + "/health")
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  const progressRef = useRef(null);

  const startProgressSim = (maxPct = 90) => {
    let pct = 0;
    progressRef.current = setInterval(() => {
      pct += Math.random() * 15;
      if (pct >= maxPct) { clearInterval(progressRef.current); pct = maxPct; }
      setScanPct(Math.min(Math.round(pct), maxPct));
    }, 120);
  };

  const stopProgress = () => {
    clearInterval(progressRef.current);
    progressRef.current = null;
  };

  const handleFile = async (file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImage(url);
    setResult(null);
    setApiError(null);
    setScanning(true);
    setScanPct(0);

    startProgressSim(85);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(BACKEND + "/classify", { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `Server error ${res.status}`);
      }

      const data = await res.json();
      stopProgress();
      setScanPct(100);

      setTimeout(() => {
        setResult(data);
        setScanning(false);
      }, 300);

    } catch (err) {
      stopProgress();
      setApiError(`Error: ${err.message}`);
      setScanning(false);
      setScanPct(0);
    }
  };

  const handleReset = () => {
    setImage(null);
    setResult(null);
    setScanning(false);
    setScanPct(0);
    setApiError(null);
    stopProgress();
  };

  const switchMode = (next) => { handleReset(); setMode(next); };

  // Status backend indicator
  const statusDot = backendOk === null ? "bg-blue-400 animate-pulse"
    : backendOk ? "bg-emerald-500"
      : "bg-red-400";
  const statusLabel = backendOk === null ? "MENGHUBUNGKAN..."
    : backendOk ? "BACKEND SIAP"
      : "BACKEND OFFLINE";

  const theme = dark
    ? {
      bg: "bg-[#0A0A0A]", text: "text-gray-200",
      navBg: "bg-[rgba(10,10,10,0.95)]", navBorder: "border-gray-800",
      navTitle: "text-gray-50", navBadgeBg: "bg-gray-800",
      navBadgeText: "text-lime-400", navBadgeBorder: "border-gray-700",
      navSub: "text-gray-600", footerText: "text-gray-700",
      footerBorder: "border-gray-800", divider: "border-gray-800",
      gridColor: "rgba(163,230,53,0.04)", accent: "text-lime-400",
      accentBg: "bg-lime-400", toggleBg: "bg-gray-800",
      toggleIcon: "☀️", toggleLabel: "Light Mode",
      tabActive: "bg-gray-800 text-lime-400 border-gray-700",
      tabInactive: "text-gray-600 border-transparent hover:text-gray-400",
    }
    : {
      bg: "bg-slate-50", text: "text-gray-800",
      navBg: "bg-white/95", navBorder: "border-gray-200",
      navTitle: "text-gray-900", navBadgeBg: "bg-emerald-50",
      navBadgeText: "text-emerald-700", navBadgeBorder: "border-emerald-200",
      navSub: "text-gray-400", footerText: "text-gray-400",
      footerBorder: "border-gray-200", divider: "border-gray-200",
      gridColor: "rgba(16,185,129,0.04)", accent: "text-emerald-600",
      accentBg: "bg-emerald-500", toggleBg: "bg-gray-100",
      toggleIcon: "🌙", toggleLabel: "Dark Mode",
      tabActive: "bg-white text-emerald-600 border-gray-200 shadow-sm",
      tabInactive: "text-gray-400 border-transparent hover:text-gray-600",
    };

  return (
    <div className={`relative min-h-screen ${theme.bg} ${theme.text} font-mono overflow-hidden transition-colors duration-300`}>
      <div
        aria-hidden="true"
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          backgroundImage: `linear-gradient(${theme.gridColor} 1px, transparent 1px), linear-gradient(90deg, ${theme.gridColor} 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />
      <style>{`
        @keyframes scanMove  { 0%{top:0%} 50%{top:90%} 100%{top:0%} }
        @keyframes spinSlow  { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes bboxIn    { from{opacity:0;transform:scale(0.94)} to{opacity:1;transform:scale(1)} }
        .animate-scan      { animation: scanMove 1.5s ease-in-out infinite; }
        .animate-spin-slow { animation: spinSlow 1s linear infinite; }
      `}</style>

      <Navbar theme={theme} dark={dark} onToggle={() => setDark(!dark)} />

      <div className={`relative z-10 flex items-center gap-1 px-4 md:px-8 py-2.5 md:py-3 border-b ${theme.divider} ${theme.navBg} backdrop-blur-md transition-colors duration-300`}>
        <div className={`flex gap-1 p-1 rounded-lg border ${dark ? "border-gray-800 bg-gray-900" : "border-gray-200 bg-gray-100"}`}>
          <button
            onClick={() => switchMode("upload")}
            className={`px-3 md:px-4 py-1.5 rounded-md text-[12px] font-bold tracking-wide border transition-all duration-200 cursor-pointer ${mode === "upload" ? theme.tabActive : theme.tabInactive + " border-transparent"}`}
          >
            ⬆ <span className="hidden sm:inline">Upload </span>Foto
          </button>
          <button
            onClick={() => switchMode("webcam")}
            className={`px-3 md:px-4 py-1.5 rounded-md text-[12px] font-bold tracking-wide border transition-all duration-200 cursor-pointer ${mode === "webcam" ? theme.tabActive : theme.tabInactive + " border-transparent"}`}
          >
            📷 Webcam
          </button>
        </div>
        <span className={`ml-2 text-[11px] ${dark ? "text-gray-600" : "text-gray-400"} hidden md:inline`}>
          {mode === "upload" ? "Unggah gambar dari perangkat" : "Ambil foto dari kamera"}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${statusDot}`} />
          <span className={`text-[10px] tracking-widest ${dark ? "text-gray-600" : "text-gray-400"}`}>
            {statusLabel}
          </span>
        </div>
      </div>

      {apiError && (
        <div className={`relative z-10 px-8 py-2 text-[11px] flex items-center gap-2 ${dark
            ? "bg-red-950/40 text-red-400 border-b border-red-900"
            : "bg-red-50 text-red-700 border-b border-red-200"
          }`}>
          <span>⚠️</span><span>{apiError}</span>
          {!backendOk && <span className="ml-2 opacity-70">— Pastikan backend berjalan: <code>uvicorn backend:app --reload --port 8000</code></span>}
        </div>
      )}

      <main className="relative z-10 grid grid-cols-1 md:grid-cols-2">
        {mode === "upload" ? (
          <UploadZone
            image={image} scanning={scanning} scanPct={scanPct}
            onFile={handleFile} onReset={handleReset}
            dark={dark} theme={theme} result={result}
          />
        ) : (
          <WebcamCapture onCapture={handleFile} dark={dark} theme={theme} />
        )}
        <ResultPanel result={result} scanning={scanning} dark={dark} theme={theme} />
      </main>

      <footer className={`relative z-10 flex items-center gap-3 px-4 md:px-8 py-3 border-t ${theme.footerBorder} ${theme.footerText} text-[11px] tracking-wide`}>
        <span>TrashTrack · Proyek ML Semester 4</span>
        <span className="hidden sm:inline">·</span>
        <span className="hidden sm:inline">Model: YOLOv8n-cls · 12 kelas · 98.6% akurasi</span>
      </footer>
    </div>
  );
}
