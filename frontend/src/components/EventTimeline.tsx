import { useState } from "react";
import { Maximize2 } from "lucide-react";

import {
  confidenceLevel,
  confidenceValue,
  directionArrow,
  directionColor,
  formatLabel,
  getLinkedAssets,
  severityValue
} from "../lib/intelligence";
import type { EventRecord } from "../types";
import { ExpandModal } from "./ExpandModal";

type RowProps = {
  event: EventRecord;
  index: number;
  onSelectEvent?: (event: EventRecord) => void;
};

function Row({ event, index, onSelectEvent }: RowProps) {
  const assets = getLinkedAssets(event).slice(0, 4);
  const confidence = Math.round(confidenceValue(event) * 100);
  return (
    <button
      onClick={() => onSelectEvent?.(event)}
      title="Click to view full event intelligence in the detail panel"
      className="grid w-full grid-cols-[32px_minmax(0,1fr)_minmax(80px,110px)_minmax(110px,132px)] items-center gap-3 rounded-lg border border-quant-line bg-quant-panel2 px-3 py-2 text-left transition hover:border-quant-green/40 hover:bg-quant-panel2/80"
    >
      <div className="grid h-7 w-7 place-items-center rounded-md border border-quant-line bg-quant-bg text-xs font-black text-quant-muted">
        {index + 1}
      </div>
      <div className="min-w-0">
        <div className="mb-1 flex min-w-0 items-center gap-2">
          <span className={`text-sm font-black ${directionColor(event.impact_direction)}`}>
            {directionArrow(event.impact_direction)}
          </span>
          <strong className="truncate text-sm text-quant-text">{event.title}</strong>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[0.68rem] font-semibold text-quant-muted">
          <span className="capitalize">{formatLabel(event.event_type)}</span>
          <span>|</span>
          <span className="capitalize">{event.affected_sectors.slice(0, 2).map(formatLabel).join(", ")}</span>
          <span>|</span>
          <span>{assets.length ? assets.map((a) => a.symbol).join(", ") : "No linked assets"}</span>
        </div>
      </div>
      <div>
        <p className="quant-eyebrow">Severity</p>
        <strong
          className={`text-xs uppercase ${
            ["high", "critical"].includes(event.severity ?? "low")
              ? "text-quant-red"
              : event.severity === "medium"
                ? "text-quant-yellow"
                : "text-quant-muted"
          }`}
        >
          {formatLabel(event.severity ?? "low")}
        </strong>
      </div>
      <div>
        <div className="mb-1 grid grid-cols-[1fr_auto] items-center gap-2">
          <p className="quant-eyebrow">Confidence</p>
          <span className="text-[0.68rem] font-black text-quant-text">{confidence}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-quant-bg">
          <div
            className="h-full rounded-full bg-gradient-to-r from-quant-green to-quant-blue"
            style={{ width: `${confidence}%` }}
          />
        </div>
        <p className="mt-1 text-[0.65rem] font-bold uppercase text-quant-muted">
          {confidenceLevel(event)}
        </p>
      </div>
    </button>
  );
}

type EventTimelineProps = {
  events: EventRecord[];
  onSelectEvent?: (event: EventRecord) => void;
};

export function EventTimeline({ events, onSelectEvent }: EventTimelineProps) {
  const [expanded, setExpanded] = useState(false);

  const rankedEvents = events
    .slice(0, 8)
    .sort((a, b) => {
      const severityDelta = severityValue(b.severity) - severityValue(a.severity);
      if (severityDelta !== 0) return severityDelta;
      return confidenceValue(b) - confidenceValue(a);
    });

  const empty = (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-sm text-quant-muted">
      No classified events available.
    </div>
  );

  return (
    <>
      <section className="quant-panel min-h-0 p-3">
        <div className="mb-2 flex items-end justify-between gap-3">
          <div>
            <p className="quant-eyebrow">Event Sequence</p>
            <h2 className="text-base font-bold text-quant-text">Ranked macro catalysts</h2>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-quant-muted">Click any row to view intelligence</span>
            <button
              onClick={() => setExpanded(true)}
              title="Expand event sequence"
              className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </div>
        <div className="scrollbar-quant grid max-h-[9.375rem] gap-1.5 overflow-auto pr-1">
          {rankedEvents.length === 0 ? empty : rankedEvents.map((event, index) => (
            <Row key={`${event.title}-${index}`} event={event} index={index} onSelectEvent={onSelectEvent} />
          ))}
        </div>
      </section>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <section className="quant-panel p-4">
            <div className="mb-3 pr-8">
              <p className="quant-eyebrow">Event Sequence</p>
              <h2 className="text-lg font-bold text-quant-text">Ranked macro catalysts</h2>
              <p className="mt-1 text-sm text-quant-muted">Click any event to load it in the intelligence detail panel</p>
            </div>
            <div className="scrollbar-quant grid gap-1.5 overflow-auto pr-1">
              {rankedEvents.length === 0 ? empty : rankedEvents.map((event, index) => (
                <Row
                  key={`${event.title}-${index}`}
                  event={event}
                  index={index}
                  onSelectEvent={(e) => { onSelectEvent?.(e); setExpanded(false); }}
                />
              ))}
            </div>
          </section>
        </ExpandModal>
      )}
    </>
  );
}
