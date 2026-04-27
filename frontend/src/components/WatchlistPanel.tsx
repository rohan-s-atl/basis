import { useMemo, useState } from "react";
import { Crosshair } from "lucide-react";

import { fetchWatchlistImpact } from "../lib/api";
import { directionColor, formatLabel } from "../lib/intelligence";
import type { EventRecord, WatchlistImpact } from "../types";

type WatchlistPanelProps = {
  events: EventRecord[];
  embedded?: boolean;
};

export function WatchlistPanel({ events, embedded = false }: WatchlistPanelProps) {
  const [symbolsText, setSymbolsText] = useState("AAPL, MSFT, QQQ, XLE, DAL, GLD");
  const [serverImpact, setServerImpact] = useState<WatchlistImpact | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const localImpact = useMemo(() => buildLocalImpact(parseSymbols(symbolsText), events), [symbolsText, events]);
  const impact = serverImpact ?? localImpact;

  async function analyze() {
    const symbols = parseSymbols(symbolsText);
    if (symbols.length === 0) return;

    setIsAnalyzing(true);
    try {
      setServerImpact(await fetchWatchlistImpact(symbols));
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <section className={embedded ? "" : "quant-panel p-3"}>
      <div className="mb-2 flex items-center justify-between">
        <div>
          <p className="quant-eyebrow">Watchlist Risk</p>
          <h2 className="text-base font-bold text-quant-text">Portfolio impact</h2>
        </div>
        <div className="grid h-8 w-8 place-items-center rounded-lg border border-quant-blue/40 bg-quant-blue/10 text-quant-blue">
          <Crosshair size={16} />
        </div>
      </div>

      <div className="mb-2 grid grid-cols-[1fr_auto] gap-2">
        <input
          className="quant-input"
          value={symbolsText}
          onChange={(event) => {
            setSymbolsText(event.target.value);
            setServerImpact(null);
          }}
          placeholder="AAPL, MSFT, QQQ, XLE"
        />
        <button className="quant-button" onClick={analyze} disabled={isAnalyzing}>
          {isAnalyzing ? "Analyzing" : "Analyze"}
        </button>
      </div>

      <div className="mb-2 grid grid-cols-[92px_1fr] gap-2">
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
          <p className="text-xs font-bold uppercase text-quant-muted">Risk Score</p>
          <strong className="text-2xl font-black text-quant-text">{impact.portfolio_risk_score}</strong>
        </div>
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
          <p className="mb-1 text-xs font-bold uppercase text-quant-muted">Impacted Holdings</p>
          <div className="flex flex-wrap gap-1.5">
            {impact.impacted_assets.slice(0, 8).map((asset) => (
              <span key={asset.symbol} className={`rounded-md border border-quant-line px-2 py-1 text-xs font-bold ${directionColor(asset.net_direction)}`}>
                {asset.symbol} {directionIcon(asset.net_direction)}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="scrollbar-quant grid max-h-[112px] gap-2 overflow-auto pr-1">
        {impact.impacted_assets.length === 0 ? (
          <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-sm text-quant-muted">
            No watchlist symbols currently match active macro events.
          </div>
        ) : (
          impact.impacted_assets.map((asset) => (
            <div key={asset.symbol} className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
              <div className="mb-1 flex items-center justify-between">
                <strong className="text-quant-text">{asset.symbol}</strong>
                <span className={`text-sm font-black capitalize ${directionColor(asset.net_direction)}`}>
                  {directionIcon(asset.net_direction)} {asset.net_direction}
                </span>
              </div>
              <p className="text-xs text-quant-muted">
                {asset.matched_events} matched events | {formatLabel(asset.highest_severity)} severity
              </p>
              <p className="mt-1 text-xs text-quant-muted">{asset.reason}</p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function parseSymbols(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(",")
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean)
    )
  );
}

function directionIcon(direction: string) {
  if (direction === "positive") return "↑";
  if (direction === "negative") return "↓";
  return "→";
}

function buildLocalImpact(symbols: string[], events: EventRecord[]): WatchlistImpact {
  const impacted_assets = symbols
    .map((symbol) => {
      const matched = events.filter((event) =>
        (event.assets ?? []).some((asset) => asset.symbol === symbol) ||
        (event.mapped_assets ?? []).includes(symbol)
      );

      if (matched.length === 0) return null;

      const negative = matched.filter((event) => event.impact_direction === "negative").length;
      const positive = matched.filter((event) => event.impact_direction === "positive").length;
      const net_direction = positive > negative ? "positive" : negative > positive ? "negative" : "neutral";

      return {
        symbol,
        matched_events: matched.length,
        highest_severity: matched[0].severity ?? "low",
        net_direction,
        reason: `${symbol} appears in ${matched.length} active event-linked asset maps.`
      };
    })
    .filter(Boolean) as WatchlistImpact["impacted_assets"];

  return {
    watchlist: symbols,
    impacted_assets,
    portfolio_risk_score: Math.min(100, impacted_assets.length * 18)
  };
}
