import { useRef, useState } from "react";

export default function UploadZone({ image, scanning, scanPct, onFile, onReset, dark, theme, result }) {
  const inputRef = useRef();
  const [imgSize, setImgSize] = useState({ w: 0, h: 0, natW: 0, natH: 0 });
  const imgRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    onFile(e.dataTransfer.files[0]);
  };

  const handleImgLoad = () => {
    const el = imgRef.current;
    if (!el) return;
    setImgSize({
      w: el.clientWidth,
      h: el.clientHeight,
      natW: el.naturalWidth,
      natH: el.naturalHeight,
    });
  };

  // Token warna per mode
  const zoneBg = dark ? "bg-[#111]" : "bg-white";
  const zoneBorder = dark ? "border-gray-700" : "border-gray-300";
  const zoneHover = dark ? "hover:border-lime-500 hover:bg-lime-950/10" : "hover:border-emerald-400 hover:bg-emerald-50/40";
  const iconWrap = dark ? "border-gray-700 text-lime-400" : "border-gray-300 text-emerald-500";
  const uploadText = dark ? "text-gray-300" : "text-gray-600";
  const uploadSub = dark ? "text-gray-600" : "text-gray-400";
  const resetHover = dark ? "hover:text-red-400 hover:border-red-800" : "hover:text-red-500 hover:border-red-300";
  const resetBorder = dark ? "border-gray-700 text-gray-500" : "border-gray-300 text-gray-400";
  const scanBarBg = dark ? "bg-gray-800" : "bg-gray-200";
  const scanLabelColor = dark ? "text-lime-400" : "text-emerald-600";
  const scanAccent = dark ? "#A3E635" : "#10B981";
  const headingColor = dark ? "text-gray-50" : "text-gray-800";
  const subColor = dark ? "text-gray-500" : "text-gray-500";

  // Warna bounding box per kategori
  const bboxColor =
    result?.category === "Organik" ? "#10B981"
      : result?.category === "B3" ? "#EF4444"
        : "#F59E0B";

  // Bounding box mock — normalized 0–1 (akan diganti data API YOLO nanti)
  // Format: { x1, y1, x2, y2 } dalam satuan 0–1 relatif ke gambar
  const bbox = result?.bbox ?? { x1: 0.2, y1: 0.15, x2: 0.8, y2: 0.85 };

  const bboxStyle = imgSize.w > 0 && result ? {
    left: bbox.x1 * imgSize.w,
    top: bbox.y1 * imgSize.h,
    width: (bbox.x2 - bbox.x1) * imgSize.w,
    height: (bbox.y2 - bbox.y1) * imgSize.h,
  } : null;

  return (
    <section className={`flex flex-col gap-5 md:gap-7 px-4 md:px-8 py-6 md:py-10 border-b md:border-b-0 md:border-r ${theme.divider} transition-colors duration-300`}>

      {/* Heading */}
      <div className="flex flex-col gap-1.5 md:gap-2">
        <p className={`text-[11px] ${theme.accent} font-bold tracking-[0.15em]`}>/ DETEKSI GAMBAR</p>
        <h1 className={`text-[28px] md:text-[40px] font-bold ${headingColor} leading-tight m-0`}>
          Scan &amp;<br />Klasifikasi
        </h1>
        <p className={`text-[12px] md:text-[13px] ${subColor} leading-relaxed max-w-sm m-0`}>
          Unggah foto sampah. Model akan mendeteksi jenisnya dalam hitungan detik.
        </p>
      </div>

      {/* Upload zone */}
      <div
        onClick={() => !image && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{ aspectRatio: "4/3" }}
        className={[
          `relative rounded-xl w-full flex items-center justify-center overflow-hidden transition-all duration-200 ${zoneBg}`,
          image
            ? `border ${zoneBorder} cursor-default`
            : `border border-dashed ${zoneBorder} cursor-pointer ${zoneHover}`,
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => onFile(e.target.files[0])}
        />

        {!image && (
          <div className="flex flex-col items-center gap-2.5 p-8">
            <div className={`w-16 h-16 border ${iconWrap} rounded-xl flex items-center justify-center text-3xl mb-1`}>
              ⬆
            </div>
            <p className={`text-sm font-bold ${uploadText}`}>Klik atau drag &amp; drop</p>
            <p className={`text-[11px] ${uploadSub} tracking-widest`}>JPG · PNG · WEBP</p>
          </div>
        )}

        {image && (
          <div className="absolute inset-0">
            <img
              ref={imgRef}
              src={image}
              alt="Preview"
              className="w-full h-full object-cover block"
              onLoad={handleImgLoad}
            />

            {/* Scanning overlay */}
            {scanning && (
              <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center gap-3">
                <div
                  className="absolute left-0 right-0 h-0.5 animate-scan"
                  style={{ background: `linear-gradient(90deg, transparent, ${scanAccent}, transparent)` }}
                />
                <p className={`relative z-10 text-xs ${scanLabelColor} tracking-widest`}>
                  Menganalisis... {scanPct}%
                </p>
                <div className={`relative z-10 w-3/5 h-1 ${scanBarBg} rounded-full overflow-hidden`}>
                  <div
                    className="h-full rounded-full transition-all duration-100"
                    style={{ width: `${scanPct}%`, background: scanAccent }}
                  />
                </div>
              </div>
            )}

            {/* Bounding Box Overlay */}
            {result && !scanning && bboxStyle && (
              <div
                className="absolute pointer-events-none"
                style={{
                  ...bboxStyle,
                  border: `2px solid ${bboxColor}`,
                  borderRadius: "6px",
                  boxShadow: `0 0 12px ${bboxColor}55`,
                  animation: "bboxIn 0.35s cubic-bezier(0.22,1,0.36,1) both",
                }}
              >
                {/* Label tag di atas box */}
                <div
                  className="absolute -top-6 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold tracking-wide"
                  style={{
                    background: bboxColor,
                    color: "#000",
                    whiteSpace: "nowrap",
                  }}
                >
                  <span>{result.label}</span>
                  <span className="opacity-70 font-normal">· {result.confidence?.toFixed(1)}%</span>
                </div>

                {/* Corner accents */}
                {[
                  "top-0 left-0 border-t-2 border-l-2",
                  "top-0 right-0 border-t-2 border-r-2",
                  "bottom-0 left-0 border-b-2 border-l-2",
                  "bottom-0 right-0 border-b-2 border-r-2",
                ].map((cls, i) => (
                  <div
                    key={i}
                    className={`absolute w-3 h-3 ${cls}`}
                    style={{ borderColor: "#fff", margin: "-2px" }}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Reset button */}
      {image && !scanning && (
        <button
          onClick={onReset}
          className={`self-start text-xs border rounded-lg px-4 py-2 tracking-wide transition-colors cursor-pointer bg-transparent ${resetBorder} ${resetHover}`}
        >
          ✕ &nbsp;Hapus &amp; Ganti Gambar
        </button>
      )}
    </section>
  );
}