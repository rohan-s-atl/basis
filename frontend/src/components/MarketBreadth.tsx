import { ArrowDown, ArrowRight, ArrowUp } from "lucide-react";
import type { ReactNode } from "react";

import type { EventRecord } from "../types";

type MarketBreadthProps = {
  events: EventRecord[];
  embedded?: boolean;
};

export function MarketBreadth({ events, embedded = false }: MarketBreadthProps) {
  const positive = events.filter((event) => event.impact_direction === "positive").length;
  const negative = events.filter((event) => event.impact_direction === "negative").length;
  const neutral = events.filter((event) => event.impact_direction === "neutral").length;
  const total = Math.max(events.length, 1);

  return (
    <section className={embedded ? "" : "quant-panel p-4"}>
      <div className="mb-3">
        <p className="quant-eyebrow">Market Breadth</p>
        <h2 className="text-lg font-bold text-quant-text">Signal balance</h2>
      </div>

      <div className="grid gap-2">
        <BreadthRow label="Positive" value={positive} total={total} color="bg-quant-green" icon={<ArrowUp size={15} />} />
        <BreadthRow label="Negative" value={negative} total={total} color="bg-quant-red" icon={<ArrowDown size={15} />} />
        <BreadthRow label="Neutral" value={neutral} total={total} color="bg-quant-yellow" icon={<ArrowRight size={15} />} />
      </div>
    </section>
  );
}

function BreadthRow({
  label,
  value,
  total,
  color,
  icon
}: {
  label: string;
  value: number;
  total: number;
  color: string;
  icon: ReactNode;
}) {
  const width = `${Math.round((value / total) * 100)}%`;

  return (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="flex items-center gap-2 font-bold text-quant-text">
          {icon}
          {label}
        </span>
        <strong>{value}</strong>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-quant-bg">
        <span className={`block h-full rounded-full ${color}`} style={{ width }} />
      </div>
    </div>
  );
}
