import { useEffect, useState } from "react";
import { BrainCircuit, CheckCircle, MinusCircle, Play, XCircle } from "lucide-react";

import { fetchBacktestSummary, fetchPortfolioSimulation, runBacktest } from "../lib/api";
import { directionArrow, directionColor, formatLabel } from "../lib/intelligence";
import type { BacktestSummary, PortfolioSimulation } from "../types";

export function BacktestPanel() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioSimulation | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchBacktestSummary(), fetchPortfolioSimulation()])
      .then(([nextSummary, nextPortfolio]) => {
        setSummary(nextSummary);
        setPortfolio(nextPortfolio);
      })
      .catch(() => {});
  }, []);

  async function handleRun() {
    setIsRunning(true);
    setError("");
    try {
      const nextSummary = await runBacktest();
      setSummary(nextSummary);
      setPortfolio(await fetchPortfolioSimulation());
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
          {portfolio && portfolio.signals > 0 && (
            <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div>
                  <p className="quant-eyebrow">Portfolio simulation</p>
                  <h3 className="text-sm font-black text-quant-text">Signal curve vs SPY</h3>
                  <p className="mt-1 text-xs text-quant-muted">Compounded signal return compared with the same-window SPY benchmark.</p>
                </div>
                <span className={`text-sm font-black ${portfolio.excess_return_pct >= 0 ? "text-quant-green" : "text-quant-red"}`}>
                  {portfolio.excess_return_pct >= 0 ? "+" : ""}{portfolio.excess_return_pct}% excess
                </span>
              </div>
              <PortfolioCurve simulation={portfolio} />
              <div className="mt-2 grid grid-cols-4 gap-2">
                <SmallStat label="Return" value={`${portfolio.total_return_pct >= 0 ? "+" : ""}${portfolio.total_return_pct}%`} />
                <SmallStat label="SPY" value={`${portfolio.benchmark_return_pct >= 0 ? "+" : ""}${portfolio.benchmark_return_pct}%`} />
                <SmallStat label="Win rate" value={`${portfolio.win_rate_pct}%`} />
                <SmallStat label="Risk/signal" value={`${portfolio.allocation_pct}%`} />
              </div>
            </div>
          )}

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
              {summary.top_signals.slice(0, 6).map((signal, index) => (
                <button
                  type="button"
                  key={`${signal.symbol}-${signal.event_type}-${signal.evaluated_at}`}
                  onClick={() => { window.location.hash = `/signals/${index}`; }}
                  className="rounded-md border border-quant-line bg-quant-bg/60 p-2 text-left transition hover:border-quant-green/50 hover:bg-quant-green/5"
                  title="Open signal detail"
                >
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
                </button>
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
                    <span className="font-black text-quant-muted" title={`${stats.correct} correct of ${stats.actionable} actionable signals. ${stats.total - stats.actionable} flat or pending signals are excluded from accuracy.`}>
                      {stats.accuracy_pct}% accuracy | {stats.correct}/{stats.actionable} correct | {stats.actionable}/{stats.total} actionable
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

function PortfolioCurve({ simulation }: { simulation: PortfolioSimulation }) {
  const points = simulation.points;
  const W = 1000;
  const H = 260;
  const padLeft = 54;
  const padRight = 24;
  const padY = 28;
  const values = points.flatMap((p) => [p.return_pct, p.benchmark_return_pct]);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = Math.max(max - min, 0.01);

  function pathFor(key: "return_pct" | "benchmark_return_pct") {
    return points.map((point, index) => {
      const x = padLeft + (index / Math.max(points.length - 1, 1)) * (W - padLeft - padRight);
      const y = H - padY - ((point[key] - min) / span) * (H - padY * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(" ");
  }

  const zeroY = H - padY - ((0 - min) / span) * (H - padY * 2);
  const yPct = (value: number) => `${Math.max(0, Math.min(100, (value / H) * 100))}%`;
  const strategyEnd = points[points.length - 1]?.return_pct ?? 0;
  const benchmarkEnd = points[points.length - 1]?.benchmark_return_pct ?? 0;

  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/60 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs font-bold text-quant-muted">
        <span>Return by signal index</span>
        <span className="text-quant-green">Strategy {strategyEnd >= 0 ? "+" : ""}{strategyEnd}%</span>
        <span>SPY {benchmarkEnd >= 0 ? "+" : ""}{benchmarkEnd}%</span>
      </div>
      <div className="relative h-72 overflow-hidden rounded-md border border-quant-line bg-quant-bg/55">
        <span className="absolute left-3 top-4 text-[0.68rem] font-bold text-quant-muted">{max.toFixed(2)}%</span>
        <span className="absolute left-3 text-[0.68rem] font-bold text-quant-muted" style={{ top: yPct(zeroY) }}>0%</span>
        <span className="absolute bottom-8 left-3 text-[0.68rem] font-bold text-quant-muted">{min.toFixed(2)}%</span>
        <span className="absolute bottom-4 left-[3.35rem] text-[0.68rem] font-bold text-quant-muted">first signal</span>
        <span className="absolute bottom-4 right-6 text-[0.68rem] font-bold text-quant-muted">latest signal</span>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" role="img" aria-label="Portfolio simulation return curve">
          {[0.25, 0.5, 0.75].map((tick) => {
            const y = padY + tick * (H - padY * 2);
            return <line key={tick} x1={padLeft} x2={W - padRight} y1={y} y2={y} className="stroke-quant-line/70" strokeDasharray="3 6" vectorEffect="non-scaling-stroke" />;
          })}
          <line x1={padLeft} x2={W - padRight} y1={zeroY} y2={zeroY} className="stroke-quant-muted/60" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
          <path d={pathFor("benchmark_return_pct")} fill="none" className="stroke-quant-muted" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <path d={pathFor("return_pct")} fill="none" className="stroke-quant-green" strokeWidth="3" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <div className="mt-1 flex items-center justify-between text-[0.65rem] font-bold text-quant-muted">
        <span>Signals: {simulation.signals}</span>
        <span className="inline-flex items-center gap-1 text-quant-green"><span className="h-1.5 w-5 rounded-full bg-quant-green" /> Strategy</span>
        <span className="inline-flex items-center gap-1"><span className="h-1.5 w-5 rounded-full bg-quant-muted" /> SPY benchmark</span>
      </div>
    </div>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/50 p-2">
      <p className="text-[0.6rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className="text-xs font-black text-quant-text">{value}</strong>
    </div>
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
