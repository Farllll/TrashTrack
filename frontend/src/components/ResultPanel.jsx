import { useEffect, useRef } from "react";
import StatCard from "./StatCard";

export default function ResultPanel({ result, scanning, dark, theme }) {
  const resultRef = useRef(null);

  // Trigger animasi setiap kali result berubah
  useEffect(() => {
    if (!result || !resultRef.current) return;
    const el = resultRef.current;
    el.style.animation = "none";
    // Force reflow
    void el.offsetHeight;
    el.style.animation = "";
  }, [result]);

  const confColor =
    result?.confidence > 95 ? (dark ? "#A3E635" : "#10B981")
      : result?.confidence > 80 ? "#F59E0B"
        : "#EF4444";

  const catColor =
    result?.category === "Organik" ? "#10B981"
      : result?.category === "B3" ? "#EF4444"
        : "#F59E0B";

  const cardBg = dark ? "bg-[#111] border-gray-800" : "bg-white border-gray-200";
  const badgeBg = dark ? "bg-[#111] border-gray-800" : "bg-gray-50 border-gray-200";
  const labelColor = dark ? "text-gray-500" : "text-gray-400";
  const nameColor = dark ? "text-gray-50" : "text-gray-900";
  const confLabel = dark ? "text-gray-500" : "text-gray-400";
  const confPctColor = dark ? "text-lime-400" : "text-emerald-600";
  const confBg = dark ? "bg-gray-800" : "bg-gray-200";
  const tipBg = dark ? "rgba(163,230,53,0.05)" : "rgba(16,185,129,0.05)";
  const tipBorder = dark ? "rgba(163,230,53,0.15)" : "rgba(16,185,129,0.2)";
  const tipText = dark ? "text-gray-400" : "text-gray-500";
  const emptyIcon = dark ? "text-gray-700" : "text-gray-300";
  const emptyText = dark ? "text-gray-500" : "text-gray-400";
  const emptySubtext = dark ? "text-gray-600" : "text-gray-300";
  const spinColor = dark ? "text-lime-400" : "text-emerald-500";

  return (
    <section className="flex flex-col px-4 md:px-8 py-6 md:py-10">

      <style>{`
        @keyframes resultFadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes bboxIn {
          from { opacity: 0; transform: scale(0.92); }
          to   { opacity: 1; transform: scale(1); }
        }
        @keyframes cardSlideIn {
          from { opacity: 0; transform: translateX(12px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes badgePop {
          0%   { opacity: 0; transform: scale(0.7); }
          70%  { transform: scale(1.08); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes barGrow {
          from { width: 0%; }
        }
        .result-root { animation: resultFadeUp 0.4s cubic-bezier(0.22,1,0.36,1) both; }
        .result-badge { animation: badgePop 0.35s cubic-bezier(0.22,1,0.36,1) 0.05s both; }
        .result-card  { animation: cardSlideIn 0.4s cubic-bezier(0.22,1,0.36,1) 0.1s both; }
        .result-bar   { animation: barGrow 0.7s cubic-bezier(0.22,1,0.36,1) 0.25s both; }
        .result-stat1 { animation: resultFadeUp 0.4s cubic-bezier(0.22,1,0.36,1) 0.3s both; opacity: 0; animation-fill-mode: forwards; }
        .result-stat2 { animation: resultFadeUp 0.4s cubic-bezier(0.22,1,0.36,1) 0.38s both; opacity: 0; animation-fill-mode: forwards; }
        .result-tip   { animation: resultFadeUp 0.4s cubic-bezier(0.22,1,0.36,1) 0.46s both; opacity: 0; animation-fill-mode: forwards; }
      `}</style>

      {/* Empty state */}
      {!result && !scanning && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 opacity-40">
          <span className={`text-[56px] ${emptyIcon} select-none`}>◎</span>
          <p className={`text-sm ${emptyText} font-bold`}>Hasil deteksi akan tampil di sini</p>
          <p className={`text-xs ${emptySubtext}`}>Upload gambar terlebih dahulu</p>
        </div>
      )}

      {/* Scanning state */}
      {scanning && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 opacity-60">
          <span className={`text-[56px] ${spinColor} select-none animate-spin-slow`}>◌</span>
          <p className={`text-sm ${spinColor} font-bold`}>Memproses gambar...</p>
        </div>
      )}

      {/* Result — semua anak pakai kelas animasi staggered */}
      {result && !scanning && (
        <div ref={resultRef} className="result-root flex flex-col gap-5">

          {/* Code badge */}
          <div className={`result-badge self-start flex items-center gap-1 text-[11px] ${badgeBg} border rounded-md px-2.5 py-1 tracking-widest`}>
            <span className={labelColor}>ID /</span>
            <span className={`${theme.accent} font-bold`}>{result.code}</span>
          </div>

          {/* Main result card */}
          <div className={`result-card flex items-center gap-4 p-5 ${cardBg} border rounded-xl transition-colors duration-300`}>
            {/* Icon dengan animasi bounce kecil */}
            <span
              className="text-[52px] select-none"
              style={{ animation: "badgePop 0.5s cubic-bezier(0.22,1,0.36,1) 0.15s both" }}
            >
              {result.icon}
            </span>
            <div>
              <p className={`text-[11px] ${labelColor} tracking-widest mb-1`}>Terdeteksi sebagai</p>
              <p className={`text-[22px] md:text-[30px] font-bold ${nameColor} leading-none`}>{result.label}</p>
              <span
                className="inline-block mt-2 text-[11px] font-bold px-2.5 py-0.5 rounded-full border tracking-widest"
                style={{
                  background: catColor + "22",
                  color: catColor,
                  borderColor: catColor + "55",
                  animation: "badgePop 0.4s cubic-bezier(0.22,1,0.36,1) 0.2s both",
                }}
              >
                {result.category}
              </span>
            </div>
          </div>

          {/* Confidence bar */}
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <span className={`text-[11px] ${confLabel} tracking-widest`}>Kepercayaan Model</span>
              <span className={`text-sm font-bold ${confPctColor}`}>{result.confidence.toFixed(1)}%</span>
            </div>
            <div className={`h-1.5 ${confBg} rounded-full overflow-hidden`}>
              <div
                className="result-bar h-full rounded-full"
                style={{ width: `${result.confidence}%`, background: confColor }}
              />
            </div>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="result-stat1">
              <StatCard
                label="Waktu Terurai"
                value={result.decompose}
                dark={dark}
              />
            </div>
            <div className="result-stat2">
              <StatCard
                label="Tempat Sampah"
                value={`● ${result.bin}`}
                valueStyle={{ color: result.binHex }}
                dark={dark}
              />
            </div>
          </div>

          {/* Tip card */}
          <div
            className="result-tip rounded-xl p-4"
            style={{ background: tipBg, border: `1px solid ${tipBorder}` }}
          >
            <p className={`text-[11px] font-bold ${tipText} tracking-widest mb-2 flex items-center gap-1.5`}>
              <span>💡</span> Tips Pengelolaan
            </p>
            <p className={`text-[13px] ${tipText} leading-relaxed m-0`}>{result.tip}</p>
          </div>

        </div>
      )}
    </section>
  );
}