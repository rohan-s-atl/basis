import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Clock } from "lucide-react";

import { fetchSignalAccuracy } from "../lib/api";
import { formatLabel } from "../lib/intelligence";
import type { SignalAccuracy as SignalAccuracyType } from "../types";

type SignalAccuracyProps = {
  embedded?: boolean;
};

export function SignalAccuracy({ embedded = false }: SignalAccuracyProps) {
  const [data, setData] = useState<SignalAccuracyType | null>(null);

  useEffect(() => {
    fetchSignalAccuracy()
      .then(setData)
      .catch(() => {});
  }, []);

  return (
    <section className={embedded ? "" : "quant-panel p-4"}>
      <div className="mb-3">
        <p className="quant-eyebrow">Signal Performance</p>
        <h2 className="text-lg font-bold text-quant-text">Prediction accuracy</h2>
      </div>

      {!data || data.total_evaluated === 0 ? (
        <div className="flex items-start gap-2 rounded-lg border border-quant-line bg-quant-panel2/60 p-3 text-xs text-quant-muted">
          <Clock size={14} className="mt-0.5 shrink-0 text-quant-yellow" />
          <span>
            Signals are evaluated 24 hours after classification. Check back tomorrow for accuracy data.
          </span>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Overall accuracy */}
          <div className="flex items-center justify-between rounded-lg border border-quant-line bg-quant-panel2 px-3 py-2">
            <span className="text-sm text-quant-muted">Overall accuracy</span>
            <div className="flex items-center gap-2">
              <span
                className={`text-xl font-black ${
                  data.accuracy_pct >= 60
                    ? "text-quant-green"
                    : data.accuracy_pct >= 40
                      ? "text-quant-yellow"
                      : "text-quant-red"
                }`}
              >
                {data.accuracy_pct}%
              </span>
              <span className="text-xs text-quant-muted">
                {data.correct}/{data.total_evaluated}
              </span>
            </div>
          </div>

          {/* By event type */}
          {Object.keys(data.by_event_type).length > 0 && (
            <div className="space-y-1.5">
              {Object.entries(data.by_event_type).map(([type, stats]) => (
                <div key={type} className="flex items-center gap-2">
                  <span className="w-36 truncate text-xs text-quant-muted">
                    {formatLabel(type)}
                  </span>
                  <div className="flex-1 overflow-hidden rounded-full bg-quant-panel2">
                    <div
                      className={`h-1.5 rounded-full transition-all ${
                        stats.accuracy_pct >= 60
                          ? "bg-quant-green"
                          : stats.accuracy_pct >= 40
                            ? "bg-quant-yellow"
                            : "bg-quant-red"
                      }`}
                      style={{ width: `${stats.accuracy_pct}%` }}
                    />
                  </div>
                  <span className="w-9 text-right text-xs font-bold text-quant-text">
                    {stats.accuracy_pct}%
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Recent evaluated signals */}
          {data.recent.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase text-quant-muted">Recent signals</p>
              {data.recent.map((s, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 rounded-md border border-quant-line bg-quant-panel2/60 px-2 py-1.5"
                >
                  {s.signal_correct ? (
                    <CheckCircle size={13} className="shrink-0 text-quant-green" />
                  ) : (
                    <XCircle size={13} className="shrink-0 text-quant-red" />
                  )}
                  <span className="min-w-0 flex-1 truncate text-xs text-quant-text">
                    {s.title}
                  </span>
                  <span
                    className={`shrink-0 text-xs font-bold capitalize ${
                      s.impact_direction === "positive"
                        ? "text-quant-green"
                        : s.impact_direction === "negative"
                          ? "text-quant-red"
                          : "text-quant-muted"
                    }`}
                  >
                    {s.impact_direction}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
