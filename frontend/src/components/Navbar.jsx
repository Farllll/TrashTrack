export default function Navbar({ theme, dark, onToggle }) {
  return (
    <nav className={`relative z-10 flex items-center gap-2 md:gap-3 px-4 md:px-8 py-3 md:py-3.5 border-b ${theme.navBorder} ${theme.navBg} backdrop-blur-md transition-colors duration-300`}>
      <div className={`w-7.5 h-7.5 md:w-8.5 md:h-8.5 ${theme.accentBg} rounded-md flex items-center justify-center text-white font-bold text-base md:text-lg shrink-0`}>
        T
      </div>
      <span className={`text-sm md:text-base font-bold ${theme.navTitle} tracking-wide`}>TrashTrack</span>
      <span className={`hidden sm:inline text-[10px] ${theme.navBadgeBg} ${theme.navBadgeText} border ${theme.navBadgeBorder} rounded px-1.5 py-0.5 tracking-widest`}>
        v2.1 · YOLO
      </span>
      <div className="flex-1" />
      <span className={`text-[11px] ${theme.navSub} tracking-wide hidden lg:block`}>
        Klasifikasi Sampah Otomatis
      </span>
      <button
        onClick={onToggle}
        className={`flex items-center gap-1.5 px-2.5 md:px-3 py-1.5 rounded-lg text-[11px] ${theme.toggleBg} ${theme.navSub} border ${theme.navBorder} hover:opacity-80 transition-all cursor-pointer shrink-0`}
        aria-label="Ganti mode gelap/terang"
      >
        <span className="tracking-wide hidden sm:inline">{theme.toggleLabel}</span>
      </button>
    </nav>
  );
}