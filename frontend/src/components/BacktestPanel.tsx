import { useEffect, useState } from "react";
import { BrainCircuit, CheckCircle, MinusCircle, Play, XCircle } from "lucide-react";

import { fetchBacktestSummary, runBacktest } from "../lib/api";
import { directionArrow, directionColor, formatLabel } from "../lib/intelligence";
import type { BacktestSummary } from "../types";

export function BacktestPanel() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchBacktestSummary()
      .then(setSummary)
      .catch(() => {});
  }, []);

  async function handleRun() {
    setIsRunning(true);
    setError("");
    try {
      setSummary(await runBacktest());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <section>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="quant-eyebrow">Backtest Lab</p>
          <h2 className="text-base font-bold text-quant-text">Signal outcome model</h2>
        </div>
        <button className="quant-button" onClick={handleRun} disabled={isRunning}>
          <Play size={14} />
          {isRunning ? "Running" : "Run"}
        </button>
      </div>

      {error && (
        <div className="mb-3 rounded-lg border border-quant-red/40 bg-quant-red/10 p-2 text-xs font-semibold text-quant-red">
          {error}
        </div>
      )}

      {!summary || summary.total_signals === 0 ? (
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-sm text-quant-muted">
          Run the backtest after events are ingested to generate per-asset signal outcomes.
        </div>
      ) : (
        <div className="grid gap-3">
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Actionable Accuracy" value={`${summary.accuracy_pct}%`} />
            <Metric label="Avg Return" value={`${summary.avg_return_pct >= 0 ? "+" : ""}${summary.avg_return_pct}%`} />
            <Metric label="ML Score" value={summary.avg_ml_score.toFixed(2)} icon />
          </div>

          <div className="grid grid-cols-3 gap-2">
            <Metric label="Total Signals" value={summary.total_signals.toString()} compact />
            <Metric label="Actionable" value={summary.actionable_signals.toString()} compact />
            <Metric label="Flat / Pending" value={summary.flat_signals.toString()} compact />
          </div>

          {summary.actionable_signals === 0 && (
            <div className="rounded-lg border border-quant-yellow/30 bg-quant-yellow/10 p-2 text-xs font-semibold text-quant-yellow">
              Prices have not moved enough from entry yet. Flat signals are stored but excluded from actionable accuracy until return magnitude is at least 0.05%.
            </div>
          )}

          <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
            <p className="quant-eyebrow mb-2">Top ranked signals</p>
            <div className="scrollbar-quant grid max-h-[11.25rem] gap-2 overflow-auto pr-1">
              {summary.top_signals.slice(0, 6).map((signal) => (
                <div key={`${signal.symbol}-${signal.event_type}-${signal.evaluated_at}`} className="rounded-md border border-quant-line bg-quant-bg/60 p-2">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <strong className="text-sm text-quant-text">{signal.symbol}</strong>
                    <span className={outcomeColor(signal.outcome_status)}>
                      {signal.outcome_status === "flat"
                        ? <MinusCircle size={14} />
                        : signal.correct
                          ? <CheckCircle size={14} />
                          : <XCircle size={14} />
                      }
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-[0.68rem] font-bold text-quant-muted">
                    <span className={`capitalize ${directionColor(signal.expected_direction)}`}>
                      {directionArrow(signal.expected_direction)} {signal.expected_direction}
                    </span>
                    <span>{formatLabel(signal.event_type)}</span>
                    <span className={signal.return_pct >= 0 ? "text-quant-green" : "text-quant-red"}>
                      {signal.return_pct >= 0 ? "+" : ""}{signal.return_pct}%
                    </span>
                    <span>score {signal.ml_score.toFixed(2)}</span>
                    <span className={outcomeColor(signal.outcome_status)}>{signal.outcome_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
            <p className="quant-eyebrow mb-2">Accuracy by event type</p>
            <div className="grid gap-2">
              {Object.entries(summary.by_event_type).slice(0, 5).map(([eventType, stats]) => (
                <div key={eventType}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="font-bold capitalize text-quant-text">{formatLabel(eventType)}</span>
                    <span className="font-black text-quant-muted">
                      {stats.accuracy_pct}% | {stats.actionable}/{stats.total} actionable
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-quant-bg">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-quant-green to-quant-blue"
                      style={{ width: `${stats.accuracy_pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
  icon = false,
  compact = false
}: {
  label: string;
  value: string;
  icon?: boolean;
  compact?: boolean;
}) {
  return (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
      <div className="mb-1 flex items-center justify-between gap-2">
        <p className="text-[0.65rem] font-bold uppercase text-quant-muted">{label}</p>
        {icon && <BrainCircuit size={12} className="text-quant-green" />}
      </div>
      <strong className={`${compact ? "text-base" : "text-lg"} font-black text-quant-text`}>{value}</strong>
    </div>
  );
}

function outcomeColor(status: string) {
  if (status === "correct") return "text-quant-green";
  if (status === "incorrect") return "text-quant-red";
  return "text-quant-yellow";
}
