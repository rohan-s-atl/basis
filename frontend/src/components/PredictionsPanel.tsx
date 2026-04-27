import { useEffect, useState } from "react";
import { BrainCircuit, RefreshCw, TrendingDown, TrendingUp } from "lucide-react";

import { fetchPredictions } from "../lib/api";
import { directionArrow, directionColor, formatLabel } from "../lib/intelligence";
import type { PredictionSummary } from "../types";

export function PredictionsPanel() {
  const [summary, setSummary] = useState<PredictionSummary | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    loadPredictions();
  }, []);

  async function loadPredictions() {
    setIsLoading(true);
    setError("");
    try {
      const payload = await fetchPredictions();
      setSummary(payload);
      setSelectedIndex(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load predictions");
    } finally {
      setIsLoading(false);
    }
  }

  const predictions = summary?.predictions ?? [];
  const selected = predictions[selectedIndex];

  return (
    <section>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="quant-eyebrow">Prediction Engine</p>
          <h2 className="text-base font-bold text-quant-text">Forward signal ranking</h2>
        </div>
        <button className="quant-button" onClick={loadPredictions} disabled={isLoading}>
          <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="mb-3 rounded-lg border border-quant-red/40 bg-quant-red/10 p-2 text-xs font-semibold text-quant-red">
          {error}
        </div>
      )}

      {predictions.length === 0 ? (
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-sm text-quant-muted">
          No predictions available yet. Refresh events, then reload predictions.
        </div>
      ) : (
        <div className="grid gap-3">
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Signals" value={summary?.count.toString() ?? "0"} />
            <Metric label="Top Prob." value={`${Math.round(predictions[0].probability * 100)}%`} />
            <Metric label="Model" value={summary?.model_version ?? "v1"} icon />
          </div>

          <div className="grid min-h-[360px] grid-cols-[0.82fr_1.18fr] gap-3">
            <div className="scrollbar-quant grid h-full max-h-[430px] content-start gap-2 overflow-auto pr-1">
              {predictions.slice(0, 10).map((prediction, index) => (
                <button
                  key={`${prediction.symbol}-${prediction.title}-${index}`}
                  onClick={() => setSelectedIndex(index)}
                  className={`rounded-lg border p-2 text-left transition ${
                    index === selectedIndex
                      ? "border-quant-green/60 bg-quant-green/10"
                      : "border-quant-line bg-quant-panel2 hover:border-quant-green/40"
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <strong className="text-sm text-quant-text">{prediction.symbol}</strong>
                    <span className={`text-xs font-black ${directionColor(prediction.impact_direction)}`}>
                      {directionArrow(prediction.impact_direction)} {prediction.expected_move_pct}%
                    </span>
                  </div>
                  <p className="line-clamp-2 text-xs text-quant-muted">{prediction.title}</p>
                  <div className="mt-2 flex items-center justify-between text-[0.68rem] font-bold text-quant-muted">
                    <span>{prediction.horizon}</span>
                    <span>{Math.round(prediction.probability * 100)}% probability</span>
                  </div>
                </button>
              ))}
            </div>

            {selected && (
              <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
                <div className="mb-3 flex items-start justify-between gap-2">
                  <div>
                    <p className="quant-eyebrow">Selected forecast</p>
                    <h3 className="text-lg font-black text-quant-text">{selected.symbol}</h3>
                  </div>
                  <span className={`inline-flex items-center gap-1 text-sm font-black ${directionColor(selected.impact_direction)}`}>
                    {selected.impact_direction === "positive" ? <TrendingUp size={15} /> : <TrendingDown size={15} />}
                    {selected.impact_direction}
                  </span>
                </div>

                <div className="mb-3 grid grid-cols-2 gap-2">
                  <SmallStat label="Expected move" value={`${selected.expected_move_pct}%`} />
                  <SmallStat label="Range" value={`${selected.expected_move_low_pct}% to ${selected.expected_move_high_pct}%`} />
                  <SmallStat label="Horizon" value={selected.horizon} />
                  <SmallStat label="Probability" value={`${Math.round(selected.probability * 100)}%`} />
                </div>

                <div className="mb-3 rounded-lg border border-quant-line bg-quant-bg/60 p-2">
                  <p className="quant-eyebrow mb-1">Scenario cases</p>
                  <p className="mb-1 text-xs text-quant-green">Bull: {selected.bull_case}</p>
                  <p className="mb-1 text-xs text-quant-muted">Base: {selected.base_case}</p>
                  <p className="text-xs text-quant-red">Bear: {selected.bear_case}</p>
                </div>

                <div>
                  <p className="quant-eyebrow mb-2">Drivers</p>
                  <div className="flex flex-wrap gap-1.5">
                    {selected.drivers.map((driver) => (
                      <span key={driver} className="quant-tag text-[0.66rem]">
                        {formatLabel(driver)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
  icon = false
}: {
  label: string;
  value: string;
  icon?: boolean;
}) {
  return (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
      <div className="mb-1 flex items-center justify-between">
        <p className="text-[0.65rem] font-bold uppercase text-quant-muted">{label}</p>
        {icon && <BrainCircuit size={12} className="text-quant-green" />}
      </div>
      <strong className="text-lg font-black text-quant-text">{value}</strong>
    </div>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/50 p-2">
      <p className="text-[0.62rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className="text-sm font-black text-quant-text">{value}</strong>
    </div>
  );
}
