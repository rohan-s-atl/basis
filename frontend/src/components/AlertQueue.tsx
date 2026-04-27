import { useState } from "react";
import { BellRing, Maximize2 } from "lucide-react";

import { formatLabel } from "../lib/intelligence";
import type { AlertRule } from "../types";
import { ExpandModal } from "./ExpandModal";

type AlertItemProps = {
  alert: AlertRule;
  onAlertClick?: (alert: AlertRule) => void;
};

function AlertItem({ alert, onAlertClick }: AlertItemProps) {
  return (
    <button
      onClick={() => onAlertClick?.(alert)}
      className={`w-full rounded-lg border p-2 text-left transition ${
        alert.active
          ? "border-quant-yellow/40 bg-quant-yellow/10 hover:border-quant-yellow/60"
          : "border-quant-line bg-quant-panel2 hover:border-quant-line/80"
      }`}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <strong className="text-sm text-quant-text">{alert.label}</strong>
        <span className="text-[0.68rem] font-bold uppercase text-quant-muted">
          {formatLabel(alert.severity)}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-quant-muted">{alert.description}</p>
      <div className="mt-2 flex items-center justify-between gap-2 border-t border-quant-line pt-2">
        <span className="truncate text-[0.65rem] font-semibold text-quant-muted">{alert.rule}</span>
        <span className="shrink-0 text-[0.65rem] font-black text-quant-text">
          {alert.matchedEvents} matched
        </span>
      </div>
      <p className="mt-1 text-[0.65rem] font-bold text-quant-green">Click to filter matching events</p>
    </button>
  );
}

type AlertQueueProps = {
  alerts: AlertRule[];
  embedded?: boolean;
  onAlertClick?: (alert: AlertRule) => void;
};

export function AlertQueue({ alerts, embedded = false, onAlertClick }: AlertQueueProps) {
  const [expanded, setExpanded] = useState(false);
  const activeAlerts = alerts.filter((a) => a.active);
  const visibleAlerts = activeAlerts.length ? activeAlerts : alerts.slice(0, 2);

  return (
    <>
      <section className={embedded ? "" : "quant-panel p-3"}>
        <div className="mb-2 flex items-center justify-between">
          <div>
            <p className="quant-eyebrow">Alert Queue</p>
            <h2 className="text-base font-bold text-quant-text">Rules triggered</h2>
          </div>
          <div className="flex items-center gap-2">
            <BellRing size={16} className={activeAlerts.length ? "text-quant-yellow" : "text-quant-muted"} />
            <button
              onClick={() => setExpanded(true)}
              title="Expand"
              className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </div>
        <div className="grid gap-2">
          {visibleAlerts.map((alert) => (
            <AlertItem key={alert.id} alert={alert} onAlertClick={onAlertClick} />
          ))}
        </div>
      </section>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <section className="quant-panel p-5">
            <div className="mb-3 pr-8">
              <p className="quant-eyebrow">Alert Queue</p>
              <h2 className="text-lg font-bold text-quant-text">All triggered rules</h2>
            </div>
            <div className="grid gap-2">
              {alerts.map((alert) => (
                <AlertItem key={alert.id} alert={alert} onAlertClick={onAlertClick} />
              ))}
            </div>
          </section>
        </ExpandModal>
      )}
    </>
  );
}
