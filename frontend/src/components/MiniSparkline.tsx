type MiniSparklineProps = {
  values: number[];
  positive?: boolean;
};

export function MiniSparkline({ values, positive = true }: MiniSparklineProps) {
  const points = buildPoints(values);

  return (
    <svg viewBox="0 0 120 34" className="h-8 w-full overflow-visible" aria-hidden="true">
      <polyline
        fill="none"
        stroke={positive ? "#00E676" : "#FF5252"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

function buildPoints(values: number[]): string {
  const safeValues = values.length > 1 ? values : [1, 1];
  const min = Math.min(...safeValues);
  const max = Math.max(...safeValues);
  const range = max - min || 1;

  return safeValues
    .map((value, index) => {
      const x = (index / (safeValues.length - 1)) * 120;
      const y = 30 - ((value - min) / range) * 26;
      return `${x},${y}`;
    })
    .join(" ");
}
