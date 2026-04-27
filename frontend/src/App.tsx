import { Activity, BrainCircuit, RefreshCw, RadioTower } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { AssetTable } from "./components/AssetTable";
import { AlertQueue } from "./components/AlertQueue";
import { BacktestPanel } from "./components/BacktestPanel";
import { EventFeed } from "./components/EventFeed";
import { EventTimeline } from "./components/EventTimeline";
import { IntelligencePanel } from "./components/IntelligencePanel";
import { MacroSituations } from "./components/MacroSituations";
import { MarketBreadth } from "./components/MarketBreadth";
import { PredictionsPanel } from "./components/PredictionsPanel";
import { SectorHeatmap } from "./components/SectorHeatmap";
import { SignalAccuracy } from "./components/SignalAccuracy";
import { WatchlistPanel } from "./components/WatchlistPanel";
import { fetchCombinedEvents, fetchMarketPrice } from "./lib/api";
import { buildAlertRules, buildAssetRows, clusterEvents, eventKey, getLinkedAssets, severityValue } from "./lib/intelligence";
import type { AlertRule, EventRecord, PriceQuote } from "./types";

type ActiveFilter =
  | { kind: "sector"; label: string; value: string }
  | { kind: "asset"; label: string; value: string }
  | { kind: "alert"; label: string; value: string }
  | null;

function App() {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [selectedEventId, setSelectedEventId] = useState("");
  const [query, setQuery] = useState("");
  const [activeFilter, setActiveFilter] = useState<ActiveFilter>(null);
  const [rightTab, setRightTab] = useState<"breadth" | "alerts" | "predict" | "backtest" | "accuracy">("breadth");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [apiStatus, setApiStatus] = useState<"connecting" | "live" | "error">("connecting");
  const [apiBaseUrl, setApiBaseUrl] = useState("auto");
  const [lastUpdated, setLastUpdated] = useState("--:--");
  const [errorMessage, setErrorMessage] = useState("");
  const [quotes, setQuotes] = useState<Record<string, PriceQuote>>({});

  useEffect(() => {
    loadEvents();
  }, []);

  const queryMatchedEvents = useMemo(() => {
    const tokens = query
      .trim()
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);

    return events
      .filter((event) => {
        if (tokens.length === 0) return true;

        const searchable = [
          event.title,
          event.description,
          event.event_type,
          event.impact_direction,
          event.severity ?? "",
          ...event.affected_sectors,
          ...getLinkedAssets(event).map((asset) => asset.symbol)
        ]
          .join(" ")
          .toLowerCase();

        return tokens.every((token) => searchable.includes(token));
      })
      .sort((a, b) => severityValue(b.severity) - severityValue(a.severity));
  }, [events, query]);

  const filteredEvents = useMemo(() => {
    if (!activeFilter) return queryMatchedEvents;

    return queryMatchedEvents.filter((event) => matchesActiveFilter(event, activeFilter));
  }, [activeFilter, queryMatchedEvents]);

  const selectedEvent = useMemo(() => {
    return (
      filteredEvents.find((event) => eventKey(event) === selectedEventId) ??
      filteredEvents[0] ??
      null
    );
  }, [filteredEvents, selectedEventId]);

  const assetRows = useMemo(() => buildAssetRows(filteredEvents), [filteredEvents]);
  const clusters = useMemo(() => clusterEvents(filteredEvents), [filteredEvents]);
  const alerts = useMemo(() => buildAlertRules(filteredEvents), [filteredEvents]);

  useEffect(() => {
    if (assetRows.length === 0) return;

    let isMounted = true;
    const symbols = assetRows.slice(0, 12).map((row) => row.symbol);

    async function updateQuotes() {
      const results = await Promise.allSettled(symbols.map((symbol) => fetchMarketPrice(symbol)));
      if (!isMounted) return;

      setQuotes((current) => {
        const next = { ...current };
        results.forEach((result) => {
          if (result.status === "fulfilled") {
            const previous = next[result.value.symbol]?.price ?? result.value.previousPrice;
            next[result.value.symbol] = {
              ...result.value,
              previousPrice: previous
            };
          }
        });
        return next;
      });
    }

    updateQuotes();
    const intervalId = window.setInterval(updateQuotes, 30000);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [assetRows]);

  async function loadEvents() {
    setIsLoading(true);
    setApiStatus("connecting");
    try {
      const result = await fetchCombinedEvents();
      setEvents(result.events);
      setSelectedEventId(result.events[0] ? eventKey(result.events[0]) : "");
      setApiBaseUrl(result.baseUrl.replace("http://", ""));
      setApiStatus("live");
      setErrorMessage("");
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    } catch (error) {
      setApiStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unable to reach /combined");
      setEvents([]);
      setSelectedEventId("");
    } finally {
      setIsLoading(false);
    }
  }

  async function refresh() {
    setIsRefreshing(true);
    await loadEvents();
    setIsRefreshing(false);
  }

  function applyFilter(filter: NonNullable<ActiveFilter>) {
    setActiveFilter((current) => {
      if (current?.kind === filter.kind && current.value === filter.value) return null;
      return filter;
    });
  }

  function clearFilter() {
    setActiveFilter(null);
  }

  function clearAllFilters() {
    setQuery("");
    setActiveFilter(null);
  }

  return (
    <main className="h-screen overflow-hidden bg-quant-bg text-quant-text">
      <div className="pointer-events-none fixed inset-0 bg-[linear-gradient(rgba(139,148,158,0.045)_1px,transparent_1px),linear-gradient(90deg,rgba(139,148,158,0.045)_1px,transparent_1px)] bg-[size:48px_48px]" />

      <section className="relative z-10 mx-auto flex h-screen w-[calc(100%-24px)] flex-col py-3">
        <header className="quant-panel mb-2 flex h-[64px] shrink-0 items-center justify-between gap-4 px-4 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg border border-quant-green/40 bg-quant-green/10 text-quant-green">
              <BrainCircuit size={20} />
            </div>
            <div>
              <p className="quant-eyebrow">Macro Event Intelligence Engine</p>
              <h1 className="text-xl font-black leading-none text-quant-text">
                Market Impact Cockpit
              </h1>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2">
            <div
              className={`inline-flex h-9 items-center gap-2 rounded-md border border-quant-line bg-quant-panel2 px-3 text-sm font-semibold ${
                apiStatus === "live"
                  ? "text-quant-green"
                  : apiStatus === "connecting"
                    ? "text-quant-yellow"
                    : "text-quant-red"
              }`}
            >
              <RadioTower size={15} />
              {apiStatus === "live" ? "Live API" : apiStatus === "connecting" ? "Connecting" : "API Issue"}
            </div>
            <div className="inline-flex h-9 items-center rounded-md border border-quant-line bg-quant-panel2 px-3 text-sm text-quant-muted">
              {apiBaseUrl}
            </div>
            <div className="inline-flex h-9 items-center rounded-md border border-quant-line bg-quant-panel2 px-3 text-sm text-quant-muted">
              Updated {lastUpdated}
            </div>
            <button className="quant-button" onClick={refresh} disabled={isRefreshing}>
              <RefreshCw size={16} className={isRefreshing ? "animate-spin" : ""} />
              Refresh
            </button>
          </div>
        </header>

        <section className="mb-3 grid shrink-0 grid-cols-4 gap-3">
          <Metric label="Tracked Events" value={filteredEvents.length.toString()} />
          <Metric
            label="High Impact"
            value={filteredEvents
              .filter((event) => ["high", "critical"].includes(event.severity ?? "low"))
              .length.toString()}
          />
          <Metric label="Linked Assets" value={assetRows.length.toString()} />
          <Metric
            label="Negative Signals"
            value={filteredEvents
              .filter((event) => event.impact_direction === "negative")
              .length.toString()}
          />
        </section>

        {apiStatus === "error" && (
          <div className="mb-2 shrink-0 rounded-lg border border-quant-red/40 bg-quant-red/10 px-4 py-2 text-sm text-quant-text">
            Unable to load macro events from <span className="font-bold">/combined</span>. Confirm the backend is running on
            {" "}127.0.0.1:8000 or 127.0.0.1:8001, then refresh. {errorMessage && (
              <span className="text-quant-muted">({errorMessage})</span>
            )}
          </div>
        )}

        <section className="grid min-h-0 flex-1 grid-cols-[minmax(300px,0.82fr)_minmax(760px,1.9fr)_minmax(360px,0.96fr)] gap-3">
          <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto_auto] gap-3">
            <EventFeed
              events={filteredEvents}
              selectedEvent={selectedEvent}
              query={query}
              filterLabel={activeFilter?.label}
              isLoading={isLoading}
              onQueryChange={(value) => {
                setQuery(value);
              }}
              onClearFilter={clearFilter}
              onSelectEvent={(event) => setSelectedEventId(eventKey(event))}
            />
            <MacroSituations
              clusters={clusters}
              onClusterClick={(cluster) => {
                if (cluster.events[0]) setSelectedEventId(eventKey(cluster.events[0]));
              }}
            />
          </div>

          <div className="grid min-h-0 grid-rows-[minmax(0,1.15fr)_auto_minmax(0,0.82fr)] gap-3">
            <AssetTable rows={assetRows} quotes={quotes} />
            <EventTimeline
              events={filteredEvents}
              onSelectEvent={(event) => setSelectedEventId(eventKey(event))}
            />
            <SectorHeatmap
              events={queryMatchedEvents}
              activeSector={activeFilter?.kind === "sector" ? activeFilter.value : ""}
              onSectorClick={(sector) =>
                applyFilter({
                  kind: "sector",
                  label: `Sector: ${sector.replaceAll("_", " ")}`,
                  value: sector
                })
              }
            />
          </div>

          <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-3">
            <IntelligencePanel
              event={selectedEvent}
              onAssetClick={(symbol) =>
                applyFilter({
                  kind: "asset",
                  label: `Asset: ${symbol}`,
                  value: symbol
                })
              }
            />
            <section className="quant-panel p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <p className="quant-eyebrow">Workbench</p>
                  <h2 className="text-base font-bold text-quant-text">Portfolio and controls</h2>
                </div>
                <div className="flex rounded-md border border-quant-line bg-quant-bg p-1">
                  {[
                    ["breadth", "Breadth"],
                    ["alerts", "Alerts"],
                    ["predict", "Predict"],
                    ["backtest", "Backtest"],
                    ["accuracy", "Accuracy"]
                  ].map(([id, label]) => (
                    <button
                      key={id}
                      onClick={() => setRightTab(id as typeof rightTab)}
                      className={`h-7 rounded px-2 text-xs font-bold transition ${
                        rightTab === id
                          ? "bg-quant-green/15 text-quant-green"
                          : "text-quant-muted hover:text-quant-text"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {rightTab === "breadth" && (
                <div className="grid gap-3">
                  <MarketBreadth events={filteredEvents} embedded />
                  <WatchlistPanel events={filteredEvents} embedded />
                </div>
              )}
              {rightTab === "alerts" && (
                <AlertQueue
                  alerts={alerts}
                  embedded
                  onAlertClick={(alert) =>
                    applyFilter({
                      kind: "alert",
                      label: `Alert: ${alert.label}`,
                      value: alert.id
                    })
                  }
                />
              )}
              {rightTab === "predict" && <PredictionsPanel />}
              {rightTab === "backtest" && <BacktestPanel />}
              {rightTab === "accuracy" && <SignalAccuracy embedded />}
            </section>
          </div>
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="quant-panel flex h-[64px] items-center gap-3 px-4">
      <div className="grid h-8 w-8 place-items-center rounded-lg border border-quant-green/35 bg-quant-green/10 text-quant-green">
        <Activity size={16} />
      </div>
      <div>
        <p className="text-xs font-semibold text-quant-muted">{label}</p>
        <strong className="text-xl font-black text-quant-text">{value}</strong>
      </div>
    </div>
  );
}

function matchesActiveFilter(event: EventRecord, filter: NonNullable<ActiveFilter>): boolean {
  if (filter.kind === "sector") {
    return event.affected_sectors.includes(filter.value);
  }

  if (filter.kind === "asset") {
    return getLinkedAssets(event).some((asset) => asset.symbol === filter.value);
  }

  if (filter.value === "energy-geopolitical-high") {
    return (
      event.event_type === "geopolitical_conflict" &&
      event.affected_sectors.includes("energy") &&
      ["high", "critical"].includes(event.severity ?? "low")
    );
  }

  if (filter.value === "broad-market-negative") {
    return event.affected_sectors.includes("broad_market") && event.impact_direction === "negative";
  }

  if (filter.value === "technology-negative") {
    return event.affected_sectors.includes("technology") && event.impact_direction === "negative";
  }

  return false;
}

export default App;
