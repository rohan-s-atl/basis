import { useState } from "react";
import { Check, Copy, ExternalLink, Maximize2 } from "lucide-react";

import {
  buildSuggestedActions,
  confidenceValue,
  directionArrow,
  directionColor,
  formatLabel,
  getLinkedAssets,
  impactSummary
} from "../lib/intelligence";
import type { EventRecord } from "../types";
import { ExpandModal } from "./ExpandModal";

type PanelBodyProps = {
  event: EventRecord;
  clampDescription: boolean;
  copiedAction: string | null;
  onCopyAction: (action: string) => void;
  onAssetClick?: (symbol: string) => void;
};

function PanelBody({ event, clampDescription, copiedAction, onCopyAction, onAssetClick }: PanelBodyProps) {
  const confidence = confidenceValue(event);
  const assets = getLinkedAssets(event);

  return (
    <>
      <article className="border-b border-quant-line pb-4">
        <h3 className="mb-2 text-base font-bold leading-tight text-quant-text">{event.title}</h3>
        <p className={`text-sm leading-relaxed text-quant-muted ${clampDescription ? "line-clamp-3" : ""}`}>
          {event.description}
        </p>
      </article>

      <section className="border-b border-quant-line py-4">
        <p className="quant-eyebrow mb-2">Impact Summary</p>
        <p className="text-base font-bold text-quant-text">{impactSummary(event)}</p>
        <div className={`mt-3 text-3xl font-black ${directionColor(event.impact_direction)}`}>
          {directionArrow(event.impact_direction)} {event.impact_direction}
        </div>
      </section>

      <section className="border-b border-quant-line py-4">
        <div className="mb-2 grid grid-cols-[1fr_auto] items-center gap-3">
          <p className="quant-eyebrow">Confidence</p>
          <span className="min-w-[42px] text-right text-sm font-bold text-quant-text">
            {Math.round(confidence * 100)}%
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
          <span
            className="block h-full rounded-full bg-gradient-to-r from-quant-green to-quant-blue transition-all duration-300"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </section>

      <section className="border-b border-quant-line py-4">
        <p className="quant-eyebrow mb-2">Reasoning</p>
        <p className="text-sm leading-relaxed text-quant-muted">{event.reasoning}</p>
      </section>

      <section className="border-b border-quant-line py-4">
        <p className="quant-eyebrow mb-3">Affected Sectors</p>
        <div className="flex flex-wrap gap-2">
          {event.affected_sectors.map((sector) => (
            <span key={sector} className="quant-tag">{formatLabel(sector)}</span>
          ))}
        </div>
      </section>

      <section className="border-b border-quant-line py-4">
        <p className="quant-eyebrow mb-1">Linked Assets</p>
        <p className="mb-3 text-[0.68rem] text-quant-muted/70">Click to filter event feed by asset</p>
        <div className="grid grid-cols-2 gap-2">
          {assets.map((asset) => (
            <button
              key={asset.symbol}
              onClick={() => onAssetClick?.(asset.symbol)}
              title={`Filter events and assets by ${asset.symbol}`}
              className="group rounded-lg border border-quant-line bg-quant-panel2 px-3 py-2 text-left transition hover:border-quant-green/60 hover:bg-quant-green/5"
            >
              <div className="flex items-center justify-between gap-1">
                <span className="block text-base font-black text-quant-text">{asset.symbol}</span>
                <ExternalLink size={11} className="shrink-0 text-quant-muted opacity-0 transition group-hover:opacity-100" />
              </div>
              <span className="text-xs text-quant-muted">
                {typeof asset.price === "number" ? `$${asset.price.toFixed(2)}` : "Price pending"}
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className="pt-4">
        <p className="quant-eyebrow mb-1">Suggested Actions</p>
        <p className="mb-3 text-[0.68rem] text-quant-muted/70">Click to copy to clipboard</p>
        <div className="grid gap-2">
          {buildSuggestedActions(event).map((action) => {
            const isCopied = copiedAction === action;
            return (
              <button
                key={action}
                onClick={() => onCopyAction(action)}
                title="Click to copy this action to clipboard"
                className="group flex items-center justify-between gap-2 rounded-lg border border-quant-green/25 bg-quant-green/5 px-3 py-2 text-left text-sm font-semibold text-quant-text transition hover:border-quant-green/50 hover:bg-quant-green/10"
              >
                <span>{action}</span>
                {isCopied
                  ? <Check size={13} className="shrink-0 text-quant-green" />
                  : <Copy size={13} className="shrink-0 text-quant-muted opacity-0 transition group-hover:opacity-100" />
                }
              </button>
            );
          })}
        </div>
      </section>
    </>
  );
}

type IntelligencePanelProps = {
  event: EventRecord | null;
  onAssetClick?: (symbol: string) => void;
};

export function IntelligencePanel({ event, onAssetClick }: IntelligencePanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [copiedAction, setCopiedAction] = useState<string | null>(null);

  function copyAction(action: string) {
    navigator.clipboard.writeText(action).catch(() => {});
    setCopiedAction(action);
    setTimeout(() => setCopiedAction(null), 1500);
  }

  if (!event) {
    return (
      <aside className="quant-panel flex min-h-0 items-center justify-center p-6 text-sm text-quant-muted">
        Select an event to inspect intelligence.
      </aside>
    );
  }

  return (
    <>
      <aside className="scrollbar-quant quant-panel min-h-0 overflow-auto p-4 transition duration-200">
        <div className="mb-4 flex items-start justify-between gap-2">
          <div>
            <p className="quant-eyebrow">Intelligence Detail</p>
            <h2 className="mt-1 text-lg font-bold capitalize text-quant-text">
              {formatLabel(event.event_type)}
            </h2>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span className="rounded-md border border-quant-line bg-quant-panel2 px-2 py-1 text-xs font-bold capitalize text-quant-text">
              {event.severity ?? "low"}
            </span>
            <button
              onClick={() => setExpanded(true)}
              title="Expand intelligence panel"
              className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </div>
        <PanelBody
          event={event}
          clampDescription={true}
          copiedAction={copiedAction}
          onCopyAction={copyAction}
          onAssetClick={onAssetClick}
        />
      </aside>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <aside className="quant-panel p-6">
            <div className="mb-4 flex items-start justify-between gap-2 pr-8">
              <div>
                <p className="quant-eyebrow">Intelligence Detail</p>
                <h2 className="mt-1 text-xl font-bold capitalize text-quant-text">
                  {formatLabel(event.event_type)}
                </h2>
              </div>
              <span className="rounded-md border border-quant-line bg-quant-panel2 px-2 py-1 text-xs font-bold capitalize text-quant-text">
                {event.severity ?? "low"}
              </span>
            </div>
            <PanelBody
              event={event}
              clampDescription={false}
              copiedAction={copiedAction}
              onCopyAction={copyAction}
              onAssetClick={onAssetClick}
            />
          </aside>
        </ExpandModal>
      )}
    </>
  );
}
