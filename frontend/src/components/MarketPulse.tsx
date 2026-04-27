import { ArrowDown, ArrowUp, Activity } from "lucide-react";

import { MiniSparkline } from "./MiniSparkline";
import type { AssetImpactRow, PriceQuote } from "../types";

type MarketPulseProps = {
  rows: AssetImpactRow[];
  quotes: Record<string, PriceQuote>;
};

export function MarketPulse({ rows, quotes }: MarketPulseProps) {
  const leaders = rows.slice(0, 4);

  return (
    <section className="quant-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="quant-eyebrow">Live Tape</p>
          <h2 className="text-lg font-bold text-quant-text">Price pulse</h2>
        </div>
        <Activity size={17} className="text-quant-green" />
      </div>

      <div className="grid grid-cols-4 gap-2">
        {leaders.map((row) => {
          const quote = quotes[row.symbol];
          const price = quote?.price ?? row.price;
          const history = quote?.history ?? [];
          const delta = history.length > 1 ? history[history.length - 1] - history[history.length - 2] : 0;
          const positive = delta >= 0;

          return (
            <div key={row.symbol} className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
              <div className="mb-1 flex items-center justify-between">
                <strong className="text-sm text-quant-text">{row.symbol}</strong>
                <span className={positive ? "text-quant-green" : "text-quant-red"}>
                  {positive ? <ArrowUp size={15} /> : <ArrowDown size={15} />}
                </span>
              </div>
              <div className="text-lg font-black text-quant-text">
                {typeof price === "number" ? `$${price.toFixed(2)}` : "Pending"}
              </div>
              <div className={`text-xs font-bold ${positive ? "text-quant-green" : "text-quant-red"}`}>
                {delta >= 0 ? "+" : ""}
                {delta.toFixed(2)}
              </div>
              <MiniSparkline values={quote?.history ?? []} positive={positive} />
            </div>
          );
        })}
      </div>
    </section>
  );
}
