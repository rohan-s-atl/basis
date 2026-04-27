import { useState } from "react";
import { Filter, Maximize2, Search, X } from "lucide-react";

import {
  confidenceLevel,
  directionArrow,
  directionColor,
  eventKey,
  formatLabel
} from "../lib/intelligence";
import type { EventRecord } from "../types";
import { ExpandModal } from "./ExpandModal";

type EventFeedProps = {
  events: EventRecord[];
  selectedEvent: EventRecord | null;
  query: string;
  filterLabel?: string;
  isLoading: boolean;
  onQueryChange: (query: string) => void;
  onClearFilter?: () => void;
  onSelectEvent: (event: EventRecord) => void;
};

function EventCards({
  events,
  selectedEvent,
  isLoading,
  onSelectEvent,
  scrollClass
}: Pick<EventFeedProps, "events" | "selectedEvent" | "isLoading" | "onSelectEvent"> & { scrollClass: string }) {
  return (
    <div className={`scrollbar-quant grid gap-2 overflow-auto pr-1 ${scrollClass}`}>
      {isLoading ? (
        <EventFeedSkeleton />
      ) : events.length === 0 ? (
        <div className="rounded-lg border border-quant-line bg-quant-panel2 p-6 text-sm text-quant-muted">
          No macro events match this filter.
        </div>
      ) : (
        events.map((event) => {
          const selected = selectedEvent && eventKey(selectedEvent) === eventKey(event);
          const confidence = confidenceLevel(event);
          return (
            <button
              key={eventKey(event)}
              onClick={() => onSelectEvent(event)}
              title="Click to view event intelligence in detail panel"
              className={`group rounded-lg border p-2.5 text-left transition ${
                selected
                  ? "border-quant-green/60 bg-quant-green/10"
                  : "border-quant-line bg-quant-panel2/80 hover:border-quant-green/40 hover:bg-quant-panel2"
              }`}
            >
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <span className="rounded-md border border-quant-line bg-quant-bg/70 px-2 py-1 text-[0.68rem] font-semibold capitalize text-quant-muted">
                  {formatLabel(event.event_type)}
                </span>
                <span className={`text-sm font-bold ${directionColor(event.impact_direction)}`}>
                  {directionArrow(event.impact_direction)} {event.impact_direction}
                </span>
                <span className="ml-auto rounded-md border border-quant-line px-2 py-1 text-[0.68rem] font-semibold uppercase text-quant-text">
                  {confidence}
                </span>
              </div>
              <h3 className="mb-1 text-[0.84rem] font-bold leading-snug text-quant-text">
                {event.title}
              </h3>
              <p className="mb-2 line-clamp-2 text-xs leading-relaxed text-quant-muted">
                {event.description}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {event.affected_sectors.map((sector) => (
                  <span key={sector} className="quant-tag text-[0.68rem]">
                    {formatLabel(sector)}
                  </span>
                ))}
                <span className="ml-auto text-[0.68rem] font-bold text-quant-green opacity-0 transition group-hover:opacity-100">
                  View details
                </span>
              </div>
            </button>
          );
        })
      )}
    </div>
  );
}

export function EventFeed({
  events,
  selectedEvent,
  query,
  filterLabel,
  isLoading,
  onQueryChange,
  onClearFilter,
  onSelectEvent
}: EventFeedProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <aside className="quant-panel flex min-h-0 flex-col p-3">
        <div className="mb-3 flex shrink-0 items-start justify-between">
          <div>
            <p className="quant-eyebrow">Event Feed</p>
            <h2 className="text-base font-semibold text-quant-text">Macro signals</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-md border border-quant-green/30 bg-quant-green/10 px-2 py-1 text-xs font-semibold text-quant-green">
              {events.length}
            </span>
            <button
              onClick={() => setExpanded(true)}
              title="Expand event feed"
              className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </div>

        {filterLabel && (
          <div className="mb-2 flex shrink-0 items-center gap-1.5 rounded-md border border-quant-green/30 bg-quant-green/8 px-2 py-1.5">
            <Filter size={11} className="shrink-0 text-quant-green" />
            <span className="flex-1 truncate text-xs font-semibold text-quant-green">{filterLabel}</span>
            <button
              onClick={onClearFilter}
              title="Clear filter"
              className="shrink-0 rounded p-0.5 text-quant-muted transition hover:text-quant-text"
            >
              <X size={11} />
            </button>
          </div>
        )}

        <label className="relative mb-3 block shrink-0">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-quant-muted" size={16} />
          <input
            className="quant-input pl-9"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Search events, sectors, assets"
          />
        </label>
        <EventCards
          events={events}
          selectedEvent={selectedEvent}
          isLoading={isLoading}
          onSelectEvent={onSelectEvent}
          scrollClass="min-h-0"
        />
      </aside>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <aside className="quant-panel flex max-h-[85vh] flex-col p-4">
            <div className="mb-3 flex shrink-0 items-start justify-between pr-8">
              <div>
                <p className="quant-eyebrow">Event Feed</p>
                <h2 className="text-lg font-bold text-quant-text">Macro signals</h2>
              </div>
              <span className="rounded-md border border-quant-green/30 bg-quant-green/10 px-2 py-1 text-xs font-semibold text-quant-green">
                {events.length}
              </span>
            </div>
            {filterLabel && (
              <div className="mb-2 flex shrink-0 items-center gap-1.5 rounded-md border border-quant-green/30 bg-quant-green/8 px-2 py-1.5">
                <Filter size={11} className="shrink-0 text-quant-green" />
                <span className="flex-1 truncate text-xs font-semibold text-quant-green">{filterLabel}</span>
                <button
                  onClick={onClearFilter}
                  title="Clear filter"
                  className="shrink-0 rounded p-0.5 text-quant-muted transition hover:text-quant-text"
                >
                  <X size={11} />
                </button>
              </div>
            )}
            <label className="relative mb-3 block shrink-0">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-quant-muted" size={16} />
              <input
                className="quant-input pl-9"
                value={query}
                onChange={(e) => onQueryChange(e.target.value)}
                placeholder="Search events, sectors, assets"
              />
            </label>
            <EventCards
              events={events}
              selectedEvent={selectedEvent}
              isLoading={isLoading}
              onSelectEvent={(event) => { onSelectEvent(event); setExpanded(false); }}
              scrollClass="min-h-0 flex-1"
            />
          </aside>
        </ExpandModal>
      )}
    </>
  );
}

function EventFeedSkeleton() {
  return (
    <>
      {[0, 1, 2].map((item) => (
        <div key={item} className="animate-pulse rounded-lg border border-quant-line bg-quant-panel2 p-3">
          <div className="mb-3 h-4 w-1/2 rounded bg-quant-muted/20" />
          <div className="mb-2 h-4 w-5/6 rounded bg-quant-muted/20" />
          <div className="h-3 w-2/3 rounded bg-quant-muted/20" />
        </div>
      ))}
    </>
  );
}
