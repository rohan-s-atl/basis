import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle, RefreshCw, TrendingDown, TrendingUp } from "lucide-react";

import { fetchMarketRegime, fetchModelEvaluation, fetchModelHealth, fetchTrainingHistory, triggerTrainModel } from "../lib/api";
import type { EvaluationMetric, MarketRegime, ModelEvaluation, ModelHealth, TrainingRun } from "../types";

export function MLIntelligencePanel() {
  const [health, setHealth] = useState<ModelHealth | null>(null);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [history, setHistory] = useState<TrainingRun[]>([]);
  const [regime, setRegime] = useState<MarketRegime | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRetraining, setIsRetraining] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setIsLoading(true);
    const [h, hist, r, evalResult] = await Promise.allSettled([
      fetchModelHealth(),
      fetchTrainingHistory(10),
      fetchMarketRegime(),
      fetchModelEvaluation(true),
    ]);
    if (h.status === "fulfilled") setHealth(h.value);
    if (hist.status === "fulfilled") setHistory(hist.value);
    if (r.status === "fulfilled") setRegime(r.value);
    if (evalResult.status === "fulfilled") setEvaluation(evalResult.value);
    setIsLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleRetrain() {
    setIsRetraining(true);
    setError("");
    try {
      await triggerTrainModel();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrain failed");
    } finally {
      setIsRetraining(false);
    }
  }

  const latestRun = history[0] ?? null;
  const activeFeatures = latestRun?.top_features.filter((f) => f.importance > 0) ?? [];
  const maxImportance = activeFeatures[0]?.importance ?? 1;

  return (
    <section>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="quant-eyebrow">ML Intelligence</p>
          <h2 className="text-base font-bold text-quant-text">Model performance &amp; regime</h2>
        </div>
        <button className="quant-button" onClick={handleRetrain} disabled={isRetraining}>
          <RefreshCw size={14} className={isRetraining ? "animate-spin" : ""} />
          {isRetraining ? "Training…" : "Retrain"}
        </button>
      </div>

      {error && (
        <div className="mb-2 rounded-lg border border-quant-red/40 bg-quant-red/10 p-2 text-xs font-semibold text-quant-red">
          {error}
        </div>
      )}

      {isLoading && !health && <SectionLoading title="Loading active model readout" rows={4} />}

      {/* Model health */}
      {health && (
        <div className={`mb-2 rounded-lg border p-3 ${
          health.status === "drift_detected"
            ? "border-quant-red/40 bg-quant-red/5"
            : health.status === "no_model"
              ? "border-quant-yellow/40 bg-quant-yellow/5"
              : "border-quant-line bg-quant-panel2"
        }`}>
          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              {health.status === "drift_detected"
                ? <AlertTriangle size={13} className="text-quant-red" />
                : <CheckCircle size={13} className="text-quant-green" />}
              <span className={`text-xs font-black uppercase tracking-wide ${
                health.status === "drift_detected" ? "text-quant-red"
                  : health.status === "no_model" ? "text-quant-yellow"
                    : "text-quant-green"
              }`}>
                {health.status === "drift_detected" ? "Drift Detected"
                  : health.status === "no_model" ? "No Model"
                    : "Healthy"}
              </span>
            </div>
            {health.dataset_size_at_training !== null && (
              <span className="text-[0.65rem] text-quant-muted">
                {health.dataset_size_at_training} samples · +{health.retraining_threshold} to retrain
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {health.rolling_accuracy && (
              <div>
                <p className="text-quant-muted">Rolling {health.rolling_accuracy.window}-pred</p>
                <p className="mt-0.5 font-black text-quant-text">
                  {(health.rolling_accuracy.accuracy * 100).toFixed(1)}%
                  <span className="ml-1 text-[0.65rem] font-semibold text-quant-muted">
                    ({health.rolling_accuracy.samples} samples)
                  </span>
                </p>
              </div>
            )}
            {(health.deployment_accuracy ?? health.peak_accuracy) !== null && (
              <div>
                <p className="text-quant-muted">Deployed accuracy</p>
                <p className="mt-0.5 font-black text-quant-text">
                  {((health.deployment_accuracy ?? health.peak_accuracy ?? 0) * 100).toFixed(1)}%
                </p>
              </div>
            )}
            {health.confidence_drift && (
              <div>
                <p className="text-quant-muted">Confidence PSI</p>
                <p className={`mt-0.5 font-black ${health.confidence_drift.drift_detected ? "text-quant-red" : "text-quant-text"}`}>
                  {health.confidence_drift.psi.toFixed(3)}
                  <span className="ml-1 text-[0.65rem] font-semibold text-quant-muted">
                    ({health.confidence_drift.samples} preds)
                  </span>
                </p>
              </div>
            )}
          </div>

          {health.confidence_drift && health.confidence_drift.samples > 0 && (
            <div className="mt-2">
              <div className="mb-1 flex items-center justify-between text-[0.62rem] font-bold uppercase text-quant-muted">
                <span>Training confidence</span>
                <span>Recent</span>
              </div>
              <div className="grid grid-cols-4 gap-1">
                {health.confidence_drift.recent.map((share, index) => (
                  <div key={index} className="h-1.5 rounded-full bg-quant-bg">
                    <div
                      className={`h-full rounded-full ${health.confidence_drift.drift_detected ? "bg-quant-red" : "bg-quant-green"}`}
                      style={{ width: `${Math.max(4, share * 100)}%` }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {health.last_trained_at && (
            <p className="mt-2 text-[0.65rem] text-quant-muted">
              Last trained {new Date(health.last_trained_at).toLocaleString([], {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
              })}
            </p>
          )}
        </div>
      )}

      {isLoading && health && !evaluation && <SectionLoading title="Loading current-model evaluation" rows={3} />}

      {evaluation && (
        <div className="mb-2 rounded-lg border border-quant-line bg-quant-panel2 p-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <div>
              <p className="quant-eyebrow">Active-model evaluation</p>
              <p className="text-[0.68rem] text-quant-muted">Current deployed model only</p>
            </div>
            <span className="text-[0.65rem] text-quant-muted">{evaluation.sample_count} labeled samples</span>
          </div>

          <div className="mb-3 grid grid-cols-3 gap-2">
            <EvalStat label="Model" metric={evaluation.overall} />
            <EvalStat label="High conf." metric={evaluation.high_confidence} />
            <EvalStat
              label="Vs SPY"
              metric={{
                samples: evaluation.benchmark_relative.samples,
                accuracy: evaluation.benchmark_relative.accuracy
              }}
            />
          </div>

          <div className="mb-3 rounded-md border border-quant-line bg-quant-bg/50 p-2">
            <div className="mb-2 flex items-center justify-between text-[0.65rem] font-bold uppercase text-quant-muted">
              <span>Directional baselines</span>
              <span>accuracy</span>
            </div>
            <div className="grid gap-1.5">
              {Object.entries(evaluation.baselines).map(([name, metric]) => (
                <BarMetric key={name} label={prettifyFeature(name)} metric={metric} />
              ))}
            </div>
          </div>

          {evaluation.raw_return_direction && (
            <div className="mb-3 rounded-md border border-quant-line bg-quant-bg/50 p-2">
              <div className="mb-2 flex items-center justify-between text-[0.65rem] font-bold uppercase text-quant-muted">
                <span>Raw return direction</span>
                <span>{evaluation.raw_return_direction.samples} samples</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <DirectionStat label="Up" value={evaluation.raw_return_direction.up} count={evaluation.raw_return_direction.up_count} tone="green" />
                <DirectionStat label="Down" value={evaluation.raw_return_direction.down} count={evaluation.raw_return_direction.down_count} tone="red" />
                <DirectionStat label="Flat" value={evaluation.raw_return_direction.flat} count={evaluation.raw_return_direction.flat_count} tone="yellow" />
              </div>
            </div>
          )}

          <div className="mb-3 grid grid-cols-2 gap-2">
            <MiniGroup title="By horizon" rows={evaluation.by_horizon} />
            <MiniGroup title="Return buckets" rows={evaluation.by_return_bucket} />
          </div>

          {evaluation.data_quality.issues.length > 0 && (
            <div className="mb-2 rounded-md border border-quant-yellow/30 bg-quant-yellow/10 p-2 text-[0.68rem] font-semibold text-quant-yellow">
              {evaluation.data_quality.issues.slice(0, 2).join(" | ")}
            </div>
          )}

          {evaluation.recommendations.length > 0 && (
            <div className="grid gap-1 text-[0.68rem] text-quant-muted">
              {evaluation.recommendations.slice(0, 3).map((item) => (
                <p key={item}>{item}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Market regime */}
      {regime && (
        <div className="mb-2 rounded-lg border border-quant-line bg-quant-panel2 p-2">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <p className="quant-eyebrow">Market Regime</p>
            <RegimeBadge encoded={regime.market_regime_encoded} />
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs">
            <div>
              <span className={`font-black ${vixColor(regime.vix_level)}`}>
                {regime.vix_level.toFixed(1)}
              </span>
              <span className="text-quant-muted"> VIX</span>
            </div>
            <div>
              <span className={`font-black ${regime.spy_trend >= 0 ? "text-quant-green" : "text-quant-red"}`}>
                {regime.spy_trend >= 0 ? "+" : ""}{(regime.spy_trend * 100).toFixed(2)}%
              </span>
              <span className="text-quant-muted"> SPY 20D</span>
            </div>
            <div>
              <span className="font-black text-quant-text">{regime.rate_level.toFixed(2)}%</span>
              <span className="text-quant-muted"> 10Y</span>
            </div>
          </div>
        </div>
      )}

      {/* Feature importance */}
      {activeFeatures.length > 0 && (
        <div className="mb-2 rounded-lg border border-quant-line bg-quant-panel2 p-2">
          <p className="quant-eyebrow mb-2">Top features</p>
          <div className="grid gap-1.5">
            {activeFeatures.slice(0, 6).map((f) => (
              <div key={f.feature}>
                <div className="mb-0.5 flex items-center justify-between text-[0.67rem]">
                  <span className="font-semibold text-quant-text">{prettifyFeature(f.feature)}</span>
                  <span className="font-black text-quant-muted">{(f.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1 rounded-full bg-quant-bg">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-quant-blue to-quant-green"
                    style={{ width: `${(f.importance / maxImportance) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training history */}
      {history.length > 0 && (
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
          <div className="mb-2 flex items-center justify-between gap-2">
            <p className="quant-eyebrow">Training history</p>
            <span className="text-[0.65rem] text-quant-muted">{history.length} run{history.length !== 1 ? "s" : ""}</span>
          </div>
          {history.length >= 2 && <AccuracySparkline runs={history} />}
          <div className="mt-2 grid gap-1">
            {history.slice(0, 4).map((run, index) => (
              <button
                type="button"
                key={run.id}
                onClick={() => { window.location.hash = `/training/${index}`; }}
                className="flex items-center justify-between gap-1 rounded-md px-1 py-1 text-left text-[0.65rem] transition hover:bg-quant-green/5"
                title="Open training run detail"
              >
                <span className="shrink-0 text-quant-muted">
                  {new Date(run.trained_at).toLocaleString([], {
                    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
                  })}
                </span>
                <span className={`shrink-0 font-bold ${run.triggered_by === "auto" ? "text-quant-blue" : "text-quant-muted"}`}>
                  {run.triggered_by}
                </span>
                <span className="shrink-0 font-black text-quant-text">
                  {((run.deployment_accuracy ?? run.calibrated_accuracy ?? run.accuracy) * 100).toFixed(1)}%
                </span>
                <span className="shrink-0 text-quant-muted">
                  AUC {run.roc_auc !== null ? run.roc_auc.toFixed(3) : "—"}
                </span>
                <span className="shrink-0 text-quant-muted">{run.dataset_size}s</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {!isLoading && !health && !regime && history.length === 0 && (
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-sm text-quant-muted">
          No model data yet. Train a model to begin tracking performance.
        </div>
      )}
    </section>
  );
}

function SectionLoading({ title, rows = 3 }: { title: string; rows?: number }) {
  return (
    <div className="mb-2 rounded-lg border border-quant-line bg-quant-panel2 p-3">
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-xs font-black uppercase text-quant-muted">{title}</p>
        <RefreshCw size={13} className="animate-spin text-quant-green" />
      </div>
      <div className="grid gap-2">
        {Array.from({ length: rows }).map((_, index) => (
          <div key={index} className="h-9 overflow-hidden rounded-md bg-quant-bg">
            <div className="loading-rail h-full w-1/2 bg-white/60" style={{ animationDelay: `${index * 100}ms` }} />
          </div>
        ))}
      </div>
    </div>
  );
}

function RegimeBadge({ encoded }: { encoded: number }) {
  if (encoded === 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-quant-green/15 px-2 py-0.5 text-[0.6rem] font-black uppercase tracking-wide text-quant-green">
        <TrendingUp size={9} /> Risk On
      </span>
    );
  }
  if (encoded === 2) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-quant-red/15 px-2 py-0.5 text-[0.6rem] font-black uppercase tracking-wide text-quant-red">
        <TrendingDown size={9} /> Risk Off
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-quant-yellow/15 px-2 py-0.5 text-[0.6rem] font-black uppercase tracking-wide text-quant-yellow">
      Neutral
    </span>
  );
}

function vixColor(vix: number): string {
  if (vix < 15) return "text-quant-green";
  if (vix < 25) return "text-quant-text";
  if (vix < 35) return "text-quant-yellow";
  return "text-quant-red";
}

function prettifyFeature(name: string): string {
  return name
    .replace(/^(event_|market_|derived_)/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, 30);
}

function EvalStat({ label, metric }: { label: string; metric: EvaluationMetric }) {
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/50 p-2">
      <p className="text-[0.62rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className={`text-sm font-black ${accuracyColor(metric.accuracy)}`}>
        {(metric.accuracy * 100).toFixed(1)}%
      </strong>
      <p className="text-[0.6rem] text-quant-muted">{metric.samples} samples</p>
    </div>
  );
}

function BarMetric({ label, metric }: { label: string; metric: EvaluationMetric }) {
  return (
    <div>
      <div className="mb-0.5 flex items-center justify-between text-[0.67rem]">
        <span className="font-semibold text-quant-text">{label}</span>
        <span className={`font-black ${accuracyColor(metric.accuracy)}`}>
          {(metric.accuracy * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-1 rounded-full bg-quant-panel">
        <div
          className="h-full rounded-full bg-gradient-to-r from-quant-blue to-quant-green"
          style={{ width: `${Math.max(3, metric.accuracy * 100)}%` }}
        />
      </div>
    </div>
  );
}

function MiniGroup({ title, rows }: { title: string; rows: Record<string, EvaluationMetric> }) {
  const entries = Object.entries(rows).slice(0, 4);
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/50 p-2">
      <p className="quant-eyebrow mb-1.5">{title}</p>
      <div className="grid gap-1">
        {entries.map(([name, metric]) => (
          <div key={name} className="flex items-center justify-between gap-2 text-[0.65rem]">
            <span className="truncate text-quant-muted">{prettifyFeature(name)}</span>
            <span className={`shrink-0 font-black ${accuracyColor(metric.accuracy)}`}>
              {(metric.accuracy * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DirectionStat({ label, value, count, tone }: { label: string; value: number; count: number; tone: "green" | "red" | "yellow" }) {
  const color = tone === "green" ? "text-quant-green" : tone === "red" ? "text-quant-red" : "text-quant-yellow";
  return (
    <div className="rounded-md border border-quant-line bg-quant-panel/60 p-2">
      <p className="text-[0.6rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className={`text-sm font-black ${color}`}>{(value * 100).toFixed(1)}%</strong>
      <p className="text-[0.6rem] text-quant-muted">{count} samples</p>
    </div>
  );
}

function accuracyColor(value: number): string {
  if (value >= 0.58) return "text-quant-green";
  if (value >= 0.48) return "text-quant-yellow";
  return "text-quant-red";
}

function AccuracySparkline({ runs }: { runs: TrainingRun[] }) {
  const sorted = [...runs].reverse();
  const accuracies = sorted.map((r) => r.accuracy);
  const aucs = sorted.map((r) => r.roc_auc ?? null);

  const W = 200;
  const H = 32;
  const padX = 4;
  const padY = 4;

  function toPoints(values: (number | null)[], minY: number, maxY: number): string {
    const range = maxY - minY || 0.05;
    return values
      .map((v, i) => {
        if (v === null) return null;
        const x = padX + (i / Math.max(1, values.length - 1)) * (W - padX * 2);
        const y = H - padY - ((v - minY) / range) * (H - padY * 2);
        return `${x},${y}`;
      })
      .filter(Boolean)
      .join(" ");
  }

  const allVals = [...accuracies, ...aucs.filter((v): v is number => v !== null)];
  const minY = Math.min(...allVals) - 0.02;
  const maxY = Math.max(...allVals) + 0.02;

  const accPoints = toPoints(accuracies, minY, maxY);
  const aucPoints = toPoints(aucs, minY, maxY);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      {aucPoints && (
        <polyline points={aucPoints} fill="none" stroke="#2979FF" strokeWidth="1.5" strokeOpacity="0.6" strokeDasharray="3,2" />
      )}
      <polyline points={accPoints} fill="none" stroke="#00E676" strokeWidth="1.5" strokeOpacity="0.9" />
      {accuracies.map((acc, i) => {
        const x = padX + (i / Math.max(1, accuracies.length - 1)) * (W - padX * 2);
        const range = maxY - minY || 0.05;
        const y = H - padY - ((acc - minY) / range) * (H - padY * 2);
        return <circle key={i} cx={x} cy={y} r="2.5" fill="#00E676" />;
      })}
    </svg>
  );
}
