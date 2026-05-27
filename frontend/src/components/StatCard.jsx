export default function StatCard({ label, value, valueStyle, dark }) {
  const cardBg = dark ? "bg-[#111] border-gray-800" : "bg-white border-gray-200";
  const labelColor = dark ? "text-gray-600" : "text-gray-400";
  const valueColor = dark ? "text-gray-300" : "text-gray-700";

  return (
    <div className={`${cardBg} border rounded-xl p-4 transition-colors duration-300`}>
      <p className={`text-[10px] ${labelColor} tracking-widest mb-1.5`}>{label}</p>
      <p className={`text-sm font-bold ${valueColor}`} style={valueStyle}>
        {value}
      </p>
    </div>
  );
}