import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  BriefcaseBusiness,
  ChevronRight,
  Database,
  FileText,
  Gauge,
  LineChart,
  RefreshCw,
  Search,
  Sparkles,
  Target,
  Waves
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { AlertQueue } from "./components/AlertQueue";
import { BacktestPanel } from "./components/BacktestPanel";
import { MLIntelligencePanel } from "./components/MLIntelligencePanel";
import { MarketBreadth } from "./components/MarketBreadth";
import { PredictionsPanel } from "./components/PredictionsPanel";
import { SectorHeatmap } from "./components/SectorHeatmap";
import { SignalAccuracy } from "./components/SignalAccuracy";
import { WatchlistPanel } from "./components/WatchlistPanel";
import {
  fetchBacktestSummary,
  fetchCombinedEvents,
  fetchMarketRegime,
  fetchMarketPrice,
  fetchModelHealth,
  fetchPortfolioSimulation,
  fetchPredictions,
  refreshPredictions,
  fetchSignalAccuracy,
  fetchWatchlistImpact,
  fetchTrainingDataStats,
  fetchTrainingDataValidation,
  fetchTrainingHistory
} from "./lib/api";
import {
  buildAlertRules,
  buildAssetRows,
  clusterEvents,
  confidenceValue,
  directionArrow,
  directionColor,
  eventKey,
  formatLabel,
  getLinkedAssets,
  severityValue
} from "./lib/intelligence";
import type {
  AssetImpactRow,
  BacktestSummary,
  EventRecord,
  MarketRegime,
  ModelHealth,
  PortfolioSimulation,
  PredictionSummary,
  PriceQuote,
  SignalAccuracy as SignalAccuracySummary,
  TrainingDataStats,
  TrainingDataValidation,
  TrainingRun,
  WatchlistImpact
} from "./types";

type Route =
  | { name: "overview" }
  | { name: "events" }
  | { name: "event"; id: string }
  | { name: "assets" }
  | { name: "asset"; symbol: string }
  | { name: "predictions" }
  | { name: "prediction"; index: number }
  | { name: "portfolio" }
  | { name: "ml" }
  | { name: "regime" }
  | { name: "training"; index: number }
  | { name: "signal"; index: number }
  | { name: "breadth" }
  | { name: "alerts" }
  | { name: "accuracy" }
  | { name: "data" };

type AppData = {
  predictions: PredictionSummary | null;
  portfolio: PortfolioSimulation | null;
  backtest: BacktestSummary | null;
  modelHealth: ModelHealth | null;
  marketRegime: MarketRegime | null;
  signalAccuracy: SignalAccuracySummary | null;
  trainingHistory: TrainingRun[];
  trainingStats: TrainingDataStats | null;
  validation: TrainingDataValidation | null;
};

type SearchResult = {
  kind: "Event" | "Asset" | "Forecast";
  title: string;
  detail: string;
  route: string;
};

const navItems = [
  { route: "#/", label: "Command", icon: Activity },
  { route: "#/predictions", label: "Signals", icon: Target },
  { route: "#/portfolio", label: "Holdings", icon: BriefcaseBusiness },
  { route: "#/events", label: "Market Feed", icon: FileText },
  { route: "#/assets", label: "Assets", icon: LineChart },
  { route: "#/ml", label: "Diagnostics", icon: BrainCircuit },
  { route: "#/data", label: "Data Health", icon: Database },
];

function parseRoute(): Route {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  if (parts[0] === "events" && parts[1]) return { name: "event", id: decodeURIComponent(parts[1]) };
  if (parts[0] === "events") return { name: "events" };
  if (parts[0] === "assets" && parts[1]) return { name: "asset", symbol: decodeURIComponent(parts[1]).toUpperCase() };
  if (parts[0] === "assets") return { name: "assets" };
  if (parts[0] === "predictions" && parts[1]) return { name: "prediction", index: Number(parts[1]) || 0 };
  if (parts[0] === "predictions") return { name: "predictions" };
  if (parts[0] === "portfolio") return { name: "portfolio" };
  if (parts[0] === "ml") return { name: "ml" };
  if (parts[0] === "regime") return { name: "regime" };
  if (parts[0] === "training" && parts[1]) return { name: "training", index: Number(parts[1]) || 0 };
  if (parts[0] === "signals" && parts[1]) return { name: "signal", index: Number(parts[1]) || 0 };
  if (parts[0] === "breadth") return { name: "breadth" };
  if (parts[0] === "alerts") return { name: "alerts" };
  if (parts[0] === "accuracy") return { name: "accuracy" };
  if (parts[0] === "data") return { name: "data" };
  return { name: "overview" };
}

function go(path: string) {
  window.location.hash = path.replace(/^#/, "");
}

function sortEvents(a: EventRecord, b: EventRecord, sort: string): number {
  if (sort === "confidence") return confidenceValue(b) - confidenceValue(a);
  if (sort === "direction") return a.impact_direction.localeCompare(b.impact_direction);
  if (sort === "type") return a.event_type.localeCompare(b.event_type);
  return severityValue(b.severity) - severityValue(a.severity);
}

function BasisLogo({ compact = false }: { compact?: boolean }) {
  const size = compact ? "h-8 w-8" : "h-10 w-10";
  return (
    <div className={`${size} basis-logo grid place-items-center rounded-lg border border-quant-line bg-white/75 shadow-sm backdrop-blur-xl`}>
      <svg viewBox="0 0 40 40" className="h-7 w-7" role="img" aria-label="Basis">
        <path d="M12 30V10h10.3c4.1 0 6.8 2.2 6.8 5.6 0 2-.9 3.5-2.7 4.4 2.3.8 3.6 2.4 3.6 4.6 0 3.5-2.8 5.4-7.2 5.4H12Zm5.3-12.1h4.3c1.4 0 2.2-.7 2.2-1.9s-.8-1.9-2.2-1.9h-4.3v3.8Zm0 7.9h5.1c1.5 0 2.3-.8 2.3-2s-.8-2-2.3-2h-5.1v4Z" fill="#17211D" />
      </svg>
    </div>
  );
}

function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute());
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [quotes, setQuotes] = useState<Record<string, PriceQuote>>({});
  const [query, setQuery] = useState("");
  const [searchFocused, setSearchFocused] = useState(false);
  const [holdingsText, setHoldingsText] = useState(() => window.localStorage.getItem("basis_holdings") ?? "AAPL, MSFT, QQQ, GLD");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [directionFilter, setDirectionFilter] = useState("all");
  const [eventTypeFilter, setEventTypeFilter] = useState("all");
  const [sectorFilter, setSectorFilter] = useState("all");
  const [eventSort, setEventSort] = useState("severity");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [apiStatus, setApiStatus] = useState<"connecting" | "live" | "error">("connecting");
  const [apiBaseUrl, setApiBaseUrl] = useState("auto");
  const [lastUpdated, setLastUpdated] = useState("--:--");
  const [errorMessage, setErrorMessage] = useState("");
  const [appData, setAppData] = useState<AppData>({
    predictions: null,
    portfolio: null,
    backtest: null,
    modelHealth: null,
    marketRegime: null,
    signalAccuracy: null,
    trainingHistory: [],
    trainingStats: null,
    validation: null,
  });

  useEffect(() => {
    const onHashChange = () => setRoute(parseRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    loadAll();
  }, []);

  const filteredEvents = useMemo(() => {
    const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return events
      .filter((event) => {
        const searchable = [
          event.title,
          event.description,
          event.event_type,
          event.impact_direction,
          event.severity ?? "",
          ...event.affected_sectors,
          ...getLinkedAssets(event).map((asset) => asset.symbol),
        ].join(" ").toLowerCase();
        const matchesSearch = tokens.every((token) => searchable.includes(token));
        const matchesSeverity = severityFilter === "all" || event.severity === severityFilter;
        const matchesDirection = directionFilter === "all" || event.impact_direction === directionFilter;
        const matchesType = eventTypeFilter === "all" || event.event_type === eventTypeFilter;
        const matchesSector = sectorFilter === "all" || event.affected_sectors.includes(sectorFilter);
        return matchesSearch && matchesSeverity && matchesDirection && matchesType && matchesSector;
      })
      .sort((a, b) => sortEvents(a, b, eventSort));
  }, [directionFilter, eventSort, eventTypeFilter, events, query, sectorFilter, severityFilter]);

  const assetRows = useMemo(() => buildAssetRows(filteredEvents), [filteredEvents]);
  const alerts = useMemo(() => buildAlertRules(filteredEvents), [filteredEvents]);
  const clusters = useMemo(() => clusterEvents(filteredEvents), [filteredEvents]);
  const nextAction = useMemo(() => getNextAction(events, appData), [appData, events]);
  const holdingSymbols = useMemo(() => parseSymbolList(holdingsText), [holdingsText]);
  const searchResults = useMemo(() => buildSearchResults(query, events, assetRows, appData.predictions), [appData.predictions, assetRows, events, query]);

  useEffect(() => {
    window.localStorage.setItem("basis_holdings", holdingsText);
  }, [holdingsText]);

  useEffect(() => {
    if (assetRows.length === 0 && holdingSymbols.length === 0) return;
    let isMounted = true;
    const symbols = Array.from(new Set([...holdingSymbols, ...assetRows.slice(0, 16).map((row) => row.symbol)])).slice(0, 28);

    async function updateQuotes() {
      const results = await Promise.allSettled(symbols.map((symbol) => fetchMarketPrice(symbol)));
      if (!isMounted) return;
      setQuotes((current) => {
        const next = { ...current };
        results.forEach((result) => {
          if (result.status === "fulfilled") {
            const previous = next[result.value.symbol]?.price ?? result.value.previousPrice;
            next[result.value.symbol] = { ...result.value, previousPrice: previous };
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
  }, [assetRows, holdingSymbols]);

  async function loadAll() {
    setIsLoading(true);
    setApiStatus("connecting");
    try {
      const combined = await fetchCombinedEvents();
      const loadedEvents = combined.events;
      setEvents(loadedEvents);
      setApiBaseUrl(combined.baseUrl.replace("http://", ""));
      setApiStatus("live");
      setErrorMessage("");

      let predictionSummary = await fetchPredictions();
      if ((predictionSummary.predictions ?? []).length === 0 && loadedEvents.length > 0) {
        try {
          await refreshPredictions();
          predictionSummary = await fetchPredictions();
        } catch {
          // Keep the rest of the workspace usable even if signal refresh is still warming up.
        }
      }

      const [portfolio, backtest, modelHealth, marketRegime, signalAccuracy, trainingHistory, trainingStats, validation] = await Promise.allSettled([
        fetchPortfolioSimulation(),
        fetchBacktestSummary(),
        fetchModelHealth(),
        fetchMarketRegime(),
        fetchSignalAccuracy(),
        fetchTrainingHistory(20),
        fetchTrainingDataStats(),
        fetchTrainingDataValidation(),
      ]);
      setAppData({
        predictions: predictionSummary,
        portfolio: portfolio.status === "fulfilled" ? portfolio.value : null,
        backtest: backtest.status === "fulfilled" ? backtest.value : null,
        modelHealth: modelHealth.status === "fulfilled" ? modelHealth.value : null,
        marketRegime: marketRegime.status === "fulfilled" ? marketRegime.value : null,
        signalAccuracy: signalAccuracy.status === "fulfilled" ? signalAccuracy.value : null,
        trainingHistory: trainingHistory.status === "fulfilled" ? trainingHistory.value : [],
        trainingStats: trainingStats.status === "fulfilled" ? trainingStats.value : null,
        validation: validation.status === "fulfilled" ? validation.value : null,
      });
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    } catch (error) {
      setApiStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unable to reach /combined");
      setEvents([]);
    } finally {
      setIsLoading(false);
    }
  }

  async function refresh() {
    setIsRefreshing(true);
    await loadAll();
    setIsRefreshing(false);
  }

  const activeNav = route.name === "event" ? "events" : route.name === "asset" ? "assets" : route.name === "prediction" ? "predictions" : route.name === "regime" || route.name === "training" ? "ml" : route.name === "signal" ? "portfolio" : route.name;
  const highImpactCount = events.filter((event) => event.severity === "high").length;
  const linkedAssetCount = new Set(assetRows.map((row) => row.symbol)).size;
  const regimeLabel = appData.marketRegime ? regimeName(appData.marketRegime.market_regime_encoded) : "Regime pending";
  const sidebarAction = sidebarNextAction(appData, apiStatus);
  const topSignal = appData.predictions?.predictions[0] ?? null;

  return (
    <main className="min-h-screen bg-quant-bg text-quant-text">
      {isLoading && events.length === 0 && <AppLoadingScreen />}
      <div className="relative z-10 flex min-h-screen">
        <aside className="scrollbar-quant sticky top-0 hidden h-screen w-72 shrink-0 overflow-y-auto border-r border-white/45 bg-white/38 px-4 py-5 shadow-panel backdrop-blur-2xl lg:block">
          <button onClick={() => go("#/")} className="mb-7 flex w-full items-center gap-3 text-left">
            <BasisLogo />
            <div>
              <h1 className="text-2xl font-black leading-tight text-quant-text">Basis</h1>
            </div>
          </button>

          <nav className="grid gap-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeNav === item.route.replace("#/", "") || (item.route === "#/" && activeNav === "overview");
              return (
                <button
                  key={item.route}
                  onClick={() => go(item.route)}
                  className={`flex h-10 items-center gap-3 rounded-md px-3 text-sm font-bold transition ${
                    isActive
                      ? "border border-quant-line bg-white/65 text-quant-text shadow-sm"
                      : "text-quant-muted hover:bg-white/40 hover:text-quant-text"
                  }`}
                >
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="mt-6 rounded-lg border border-quant-line bg-white/50 p-3 backdrop-blur-xl">
            <p className="quant-eyebrow mb-2">Status</p>
            <div className={`mb-2 flex items-center gap-2 text-sm font-black ${apiStatus === "live" ? "text-quant-green" : apiStatus === "error" ? "text-quant-red" : "text-quant-yellow"}`}>
              <Waves size={15} />
              {apiStatus === "live" ? "Live market feed" : apiStatus === "error" ? "Connection issue" : "Connecting"}
            </div>
            <p className="truncate text-xs text-quant-muted">{apiBaseUrl}</p>
            <p className="mt-1 text-xs text-quant-muted">Updated {lastUpdated}</p>
          </div>

          <div className="mt-4 grid gap-3">
            <div className="rounded-lg border border-quant-line bg-white/50 p-3 backdrop-blur-xl">
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="quant-eyebrow">Today</p>
                <span className={`text-[0.65rem] font-black uppercase ${apiStatus === "live" ? "text-quant-green" : "text-quant-yellow"}`}>
                  {apiStatus === "live" ? "Online" : "Standby"}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <SidebarMetric label="Signals" value={(appData.predictions?.count ?? 0).toString()} />
                <SidebarMetric label="Watch" value={holdingSymbols.length.toString()} />
                <SidebarMetric label="Events" value={events.length.toString()} />
                <SidebarMetric label="Risk" value={highImpactCount.toString()} tone={highImpactCount > 0 ? "red" : "muted"} />
              </div>
            </div>

            {topSignal && (
              <button
                type="button"
                onClick={() => go("#/predictions/0")}
                className="rounded-lg border border-quant-line bg-white/55 p-3 text-left shadow-sm backdrop-blur-xl transition hover:border-quant-blue/30"
              >
                <p className="quant-eyebrow mb-2">Top signal</p>
                <div className="flex items-center justify-between gap-2">
                  <strong className="text-lg font-black text-quant-text">{topSignal.symbol}</strong>
                  <span className={directionColor(topSignal.impact_direction)}>{topSignal.expected_move_pct}%</span>
                </div>
                <p className="mt-1 line-clamp-2 text-xs leading-5 text-quant-muted">{topSignal.why_this_matters || topSignal.title}</p>
              </button>
            )}

            <button
              type="button"
              onClick={() => go("#/ml")}
              className="rounded-lg border border-quant-line bg-white/50 p-3 text-left backdrop-blur-xl transition hover:border-quant-blue/30"
            >
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="quant-eyebrow">Reliability</p>
                <span className={appData.modelHealth?.drift_detected ? "text-[0.65rem] font-black uppercase text-quant-red" : "text-[0.65rem] font-black uppercase text-quant-green"}>
                  {formatLabel(appData.modelHealth?.status ?? "pending")}
                </span>
              </div>
              <strong className="text-lg font-black text-quant-text">
                {appData.modelHealth?.deployment_accuracy ? `${Math.round(appData.modelHealth.deployment_accuracy * 100)}%` : "--"}
              </strong>
              <p className="mt-2 text-xs leading-5 text-quant-muted">
                {appData.modelHealth?.status === "warming_up" ? "Collecting fresh live outcomes" : "Live model reliability monitor"}
              </p>
            </button>

            <button
              type="button"
              onClick={() => go("#/regime")}
              className="rounded-lg border border-quant-line bg-white/50 p-3 text-left backdrop-blur-xl transition hover:border-quant-blue/30"
            >
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="quant-eyebrow">Market State</p>
                <span className={appData.marketRegime?.market_regime_encoded === 0 ? "text-[0.65rem] font-black uppercase text-quant-green" : "text-[0.65rem] font-black uppercase text-quant-yellow"}>
                  {regimeLabel}
                </span>
              </div>
              <div className="grid gap-2 text-xs font-bold text-quant-muted">
                <div className="flex justify-between"><span>VIX</span><span className="text-quant-text">{appData.marketRegime?.vix_level?.toFixed(1) ?? "--"}</span></div>
                <div className="flex justify-between"><span>SPY 20D</span><span className={(appData.marketRegime?.spy_trend ?? 0) >= 0 ? "text-quant-green" : "text-quant-red"}>{formatSignedPercent(appData.marketRegime?.spy_trend)}</span></div>
                <div className="flex justify-between"><span>10Y</span><span className="text-quant-text">{appData.marketRegime?.rate_level?.toFixed(2) ?? "--"}%</span></div>
              </div>
            </button>

            <button type="button" onClick={() => go(sidebarAction.route)} className="rounded-lg border border-quant-blue/20 bg-white/55 p-3 text-left shadow-sm backdrop-blur-xl transition hover:border-quant-blue/40">
              <p className="quant-eyebrow mb-2">Next Action</p>
              <p className="text-sm font-black text-quant-blue">{sidebarAction.title}</p>
              <p className="mt-1 text-xs leading-5 text-quant-muted">{sidebarAction.detail}</p>
            </button>
          </div>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 border-b border-white/45 bg-white/35 px-4 py-3 shadow-sm backdrop-blur-2xl lg:px-7">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <Breadcrumbs route={route} />
                <p className="quant-eyebrow">{pageEyebrow(route)}</p>
                <h2 className="truncate text-2xl font-black text-quant-text">{pageTitle(route)}</h2>
              </div>
              <div className="flex items-center gap-2">
                <label className="relative hidden min-w-[260px] sm:block">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-quant-muted" size={15} />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => setSearchFocused(true)}
                    onBlur={() => window.setTimeout(() => setSearchFocused(false), 120)}
                    placeholder="Search events, assets, sectors"
                    className="quant-input h-9 pl-9"
                  />
                  {searchFocused && query.trim() && (
                    <SearchResults results={searchResults} query={query} />
                  )}
                </label>
                <button className="quant-button" onClick={refresh} disabled={isRefreshing}>
                  <RefreshCw size={15} className={isRefreshing ? "animate-spin" : ""} />
                  Refresh
                </button>
              </div>
            </div>
            <div className="mt-3 flex gap-2 overflow-x-auto lg:hidden">
              {navItems.map((item) => (
                <button
                  key={item.route}
                  onClick={() => go(item.route)}
                  className="shrink-0 rounded-md border border-quant-line bg-quant-panel2 px-3 py-2 text-xs font-black text-quant-text"
                >
                  {item.label}
                </button>
              ))}
            </div>
            <label className="relative mt-3 block sm:hidden">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-quant-muted" size={15} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => window.setTimeout(() => setSearchFocused(false), 120)}
                placeholder="Search events, assets, sectors"
                className="quant-input h-9 pl-9"
              />
              {searchFocused && query.trim() && (
                <SearchResults results={searchResults} query={query} />
              )}
            </label>
          </header>

          {apiStatus === "error" && (
            <div className="mx-4 mt-4 rounded-lg border border-quant-red/40 bg-quant-red/10 px-4 py-3 text-sm text-quant-text lg:mx-7">
              Unable to load macro events. Confirm the API URL is reachable. {errorMessage && <span className="text-quant-muted">({errorMessage})</span>}
            </div>
          )}

          <div className="flex-1 px-4 py-5 lg:px-7">
            <ActionStrip
              apiStatus={apiStatus}
              lastUpdated={lastUpdated}
              nextAction={nextAction}
              onAction={() => go(nextAction.route)}
            />
            <PageRenderer
              route={route}
              events={filteredEvents}
              allEvents={events}
              assetRows={assetRows}
              quotes={quotes}
              alerts={alerts}
              clusters={clusters}
              appData={appData}
              holdingsText={holdingsText}
              holdingSymbols={holdingSymbols}
              onHoldingsChange={setHoldingsText}
              isLoading={isLoading}
              filters={{
                severityFilter,
                directionFilter,
                eventTypeFilter,
                sectorFilter,
                eventSort,
                setSeverityFilter,
                setDirectionFilter,
                setEventTypeFilter,
                setSectorFilter,
                setEventSort,
              }}
            />
          </div>
        </section>
      </div>
    </main>
  );
}

function PageRenderer({
  route,
  events,
  allEvents,
  assetRows,
  quotes,
  alerts,
  clusters,
  appData,
  holdingsText,
  holdingSymbols,
  onHoldingsChange,
  isLoading,
  filters,
}: {
  route: Route;
  events: EventRecord[];
  allEvents: EventRecord[];
  assetRows: AssetImpactRow[];
  quotes: Record<string, PriceQuote>;
  alerts: ReturnType<typeof buildAlertRules>;
  clusters: ReturnType<typeof clusterEvents>;
  appData: AppData;
  holdingsText: string;
  holdingSymbols: string[];
  onHoldingsChange: (value: string) => void;
  isLoading: boolean;
  filters: {
    severityFilter: string;
    directionFilter: string;
    eventTypeFilter: string;
    sectorFilter: string;
    eventSort: string;
    setSeverityFilter: (value: string) => void;
    setDirectionFilter: (value: string) => void;
    setEventTypeFilter: (value: string) => void;
    setSectorFilter: (value: string) => void;
    setEventSort: (value: string) => void;
  };
}) {
  if (route.name === "events") return <EventsPage events={events} allEvents={allEvents} isLoading={isLoading} filters={filters} />;
  if (route.name === "event") return <EventDetailPage event={allEvents.find((event) => eventKey(event) === route.id) ?? events[0] ?? null} allEvents={allEvents} predictions={appData.predictions} quotes={quotes} />;
  if (route.name === "assets") return <AssetsPage rows={assetRows} quotes={quotes} events={events} predictions={appData.predictions} />;
  if (route.name === "asset") return <AssetDetailPage symbol={route.symbol} rows={assetRows} quotes={quotes} events={allEvents} predictions={appData.predictions} backtest={appData.backtest} holdingSymbols={holdingSymbols} />;
  if (route.name === "predictions") return <PredictionsPage summary={appData.predictions} />;
  if (route.name === "prediction") return <PredictionDetailPage summary={appData.predictions} index={route.index} marketRegime={appData.marketRegime} allPredictions={appData.predictions} />;
  if (route.name === "portfolio") return <PortfolioPage appData={appData} events={allEvents} quotes={quotes} holdingsText={holdingsText} holdingSymbols={holdingSymbols} onHoldingsChange={onHoldingsChange} />;
  if (route.name === "ml") return <MLPage appData={appData} />;
  if (route.name === "regime") return <RegimePage regime={appData.marketRegime} />;
  if (route.name === "training") return <TrainingRunPage history={appData.trainingHistory} index={route.index} />;
  if (route.name === "signal") return <SignalDetailPage backtest={appData.backtest} index={route.index} />;
  if (route.name === "breadth") return <BreadthPage events={events} />;
  if (route.name === "alerts") return <AlertsPage alerts={alerts} />;
  if (route.name === "accuracy") return <AccuracyPage />;
  if (route.name === "data") return <DataHealthPage appData={appData} events={allEvents} />;
  return <OverviewPage events={events} assetRows={assetRows} clusters={clusters} appData={appData} />;
}

function OverviewPage({
  events,
  assetRows,
  clusters,
  appData,
}: {
  events: EventRecord[];
  assetRows: AssetImpactRow[];
  clusters: ReturnType<typeof clusterEvents>;
  appData: AppData;
}) {
  const highImpact = events.filter((event) => ["high", "critical"].includes(event.severity ?? "low")).length;
  const signals = appData.predictions?.predictions ?? [];
  const actionable = signals.filter((signal) => signal.actionability === "actionable" || signal.is_actionable);
  const watched = signals.filter((signal) => signal.actionability === "watch");
  const blocked = signals.filter((signal) => signal.actionability === "blocked" || (!signal.is_actionable && signal.actionability !== "watch"));
  const attentionSignals = [...actionable, ...watched];
  const leadSignal = attentionSignals[0] ?? signals[0] ?? null;
  return (
    <div className="grid gap-5">
      <section className="quant-panel-strong overflow-hidden p-5 lg:p-7">
        <div className="grid items-center gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.55fr)]">
          <div>
            <p className="quant-eyebrow mb-3">Market decision layer</p>
            <h2 className="max-w-4xl text-4xl font-black leading-tight text-quant-text md:text-5xl">
              {leadSignal ? `${leadSignal.symbol}: ${formatInvestorDirection(leadSignal.impact_direction)} signal` : "Your macro watch floor is ready"}
            </h2>
            <p className="mt-4 max-w-2xl text-base leading-7 text-quant-muted">
              {leadSignal
                ? leadSignal.why_this_matters || leadSignal.title
                : "Add holdings, monitor macro events, and review investor-ready signals without digging through model internals."}
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <button onClick={() => go("#/predictions")} className="quant-button bg-quant-text text-white hover:bg-quant-text/90">Review signals</button>
              <button onClick={() => go("#/portfolio")} className="quant-button">Watch my holdings</button>
              <button onClick={() => go("#/events")} className="quant-button">Open market feed</button>
            </div>
          </div>
          <div className="flex justify-center rounded-lg border border-quant-line bg-white/45 p-4 backdrop-blur-2xl">
            {leadSignal ? (
              <div className="w-full max-w-md">
                <SignalBriefCard prediction={leadSignal} index={signals.indexOf(leadSignal)} featured />
              </div>
            ) : (
              <InlineEmpty title="No signals loaded" description="Refresh the feed or wait for predictions to populate the command view." />
            )}
          </div>
        </div>
      </section>

      <div className="grid gap-3 md:grid-cols-4">
        <MetricCard label="Actionable" value={actionable.length.toString()} icon={Target} onClick={() => go("#/predictions")} />
        <MetricCard label="Watch" value={watched.length.toString()} icon={Gauge} onClick={() => go("#/predictions")} />
        <MetricCard label="Blocked" value={blocked.length.toString()} icon={Database} onClick={() => go("#/predictions")} />
        <MetricCard label="High impact" value={highImpact.toString()} icon={FileText} onClick={() => go("#/events")} />
      </div>

      <section className="quant-panel p-4">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="quant-eyebrow">Signal queue</p>
            <h3 className="text-xl font-black text-quant-text">What deserves attention</h3>
          </div>
          <button onClick={() => go("#/predictions")} className="quant-button">All signals</button>
        </div>
        <div className="grid gap-3 xl:grid-cols-3">
          {signals.length === 0 ? (
            <InlineEmpty title="No predictions yet" description="Signals appear here once Basis has classified events and mapped assets." />
          ) : (attentionSignals.length ? attentionSignals : signals).slice(0, 6).map((prediction) => (
            <SignalBriefCard key={`${prediction.symbol}-${signals.indexOf(prediction)}`} prediction={prediction} index={signals.indexOf(prediction)} />
          ))}
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr_0.8fr]">
        <section className="quant-panel p-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Market feed</p>
              <h3 className="text-lg font-black text-quant-text">Latest catalysts</h3>
            </div>
            <button onClick={() => go("#/events")} className="text-xs font-black text-quant-blue">Open feed</button>
          </div>
          <div className="grid gap-2">
            {events.length === 0 ? (
              <InlineEmpty title="No events loaded" description="Start or refresh the backend feed to populate the priority queue." />
            ) : events.slice(0, 5).map((event) => <EventRow key={eventKey(event)} event={event} />)}
          </div>
        </section>

        <section className="quant-panel p-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Holdings</p>
              <h3 className="text-lg font-black text-quant-text">Strategy curve</h3>
            </div>
            <button onClick={() => go("#/portfolio")} className="text-xs font-black text-quant-blue">Open holdings</button>
          </div>
          <MiniReturnPanel portfolio={appData.portfolio} />
        </section>
        <section className="quant-panel p-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Market regime</p>
              <h3 className="text-lg font-black text-quant-text">Current context</h3>
            </div>
            <button onClick={() => go("#/regime")} className="text-xs font-black text-quant-blue">Open regime</button>
          </div>
          <RegimeSummary regime={appData.marketRegime} />
          <p className="mt-3 text-sm leading-6 text-quant-muted">Regime data gives each forecast context for whether the market is risk-on, neutral, or risk-off.</p>
        </section>
      </div>

      <div className="grid gap-5 xl:grid-cols-3">
        <section className="quant-panel p-4 xl:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Themes</p>
              <h3 className="text-lg font-black text-quant-text">Macro clusters</h3>
            </div>
            <button onClick={() => go("#/events")} className="text-xs font-black text-quant-blue">Investigate</button>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {clusters.length === 0 ? (
              <InlineEmpty title="No clusters yet" description="Clusters appear once multiple events share type, sector, or linked assets." />
            ) : clusters.slice(0, 4).map((cluster) => (
              <button key={cluster.id} onClick={() => cluster.events[0] && go(`#/events/${encodeURIComponent(eventKey(cluster.events[0]))}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <strong className="text-sm text-quant-text">{cluster.title}</strong>
                  <span className={directionColor(cluster.direction)}>{directionArrow(cluster.direction)}</span>
                </div>
                <p className="text-xs text-quant-muted">{cluster.events.length} event(s) | {cluster.assets.slice(0, 4).join(", ")}</p>
              </button>
            ))}
          </div>
        </section>
        <section className="quant-panel p-4">
          <div className="mb-3">
            <p className="quant-eyebrow">Model</p>
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-lg font-black text-quant-text">Health snapshot</h3>
            </div>
          </div>
          <HealthSummary health={appData.modelHealth} />
        </section>
      </div>
    </div>
  );
}

function EventsPage({
  events,
  allEvents,
  isLoading,
  filters,
}: {
  events: EventRecord[];
  allEvents: EventRecord[];
  isLoading: boolean;
  filters: {
    severityFilter: string;
    directionFilter: string;
    eventTypeFilter: string;
    sectorFilter: string;
    eventSort: string;
    setSeverityFilter: (value: string) => void;
    setDirectionFilter: (value: string) => void;
    setEventTypeFilter: (value: string) => void;
    setSectorFilter: (value: string) => void;
    setEventSort: (value: string) => void;
  };
}) {
  const eventTypes = uniqueValues(allEvents.map((event) => event.event_type));
  const sectors = uniqueValues(allEvents.flatMap((event) => event.affected_sectors));
  return (
    <section className="quant-panel p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="quant-eyebrow">Events</p>
          <h3 className="text-xl font-black text-quant-text">Article intelligence queue</h3>
        </div>
        <span className="rounded-md border border-quant-green/30 bg-quant-green/10 px-2 py-1 text-xs font-black text-quant-green">{events.length}</span>
      </div>
      <div className="mb-4 grid gap-2 md:grid-cols-5">
        <SelectControl label="Severity" value={filters.severityFilter} onChange={filters.setSeverityFilter} options={["all", "low", "medium", "high", "critical"]} />
        <SelectControl label="Direction" value={filters.directionFilter} onChange={filters.setDirectionFilter} options={["all", "positive", "negative", "neutral"]} />
        <SelectControl label="Type" value={filters.eventTypeFilter} onChange={filters.setEventTypeFilter} options={["all", ...eventTypes]} />
        <SelectControl label="Sector" value={filters.sectorFilter} onChange={filters.setSectorFilter} options={["all", ...sectors]} />
        <SelectControl label="Sort" value={filters.eventSort} onChange={filters.setEventSort} options={["severity", "confidence", "type", "direction"]} />
      </div>
      <div className="grid gap-2">
        {isLoading ? (
          <PageSkeleton rows={6} />
        ) : events.length === 0 ? (
          <InlineEmpty title="No events match this view" description="Try clearing search or refreshing the API feed." />
        ) : (
          events.map((event) => <EventRow key={eventKey(event)} event={event} expanded />)
        )}
      </div>
    </section>
  );
}

function EventDetailPage({
  event,
  allEvents,
  predictions,
  quotes,
}: {
  event: EventRecord | null;
  allEvents: EventRecord[];
  predictions: PredictionSummary | null;
  quotes: Record<string, PriceQuote>;
}) {
  if (!event) return <EmptyState title="No event selected" action="Back to events" onClick={() => go("#/events")} />;
  const assets = getLinkedAssets(event);
  const assetSymbols = assets.map((asset) => asset.symbol);
  const relatedEvents = allEvents
    .filter((candidate) => eventKey(candidate) !== eventKey(event))
    .filter((candidate) =>
      candidate.event_type === event.event_type ||
      candidate.affected_sectors.some((sector) => event.affected_sectors.includes(sector)) ||
      getLinkedAssets(candidate).some((asset) => assetSymbols.includes(asset.symbol))
    )
    .slice(0, 5);
  const linkedPredictions = predictionsForSymbols(predictions, assetSymbols).slice(0, 4);
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
      <article className="quant-panel p-5">
        <button onClick={() => go("#/events")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> Events</button>
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="quant-tag">{formatLabel(event.event_type)}</span>
          <span className={`text-sm font-black capitalize ${directionColor(event.impact_direction)}`}>{directionArrow(event.impact_direction)} {event.impact_direction}</span>
          <span className="quant-tag">{event.severity ?? "low"}</span>
        </div>
        <h2 className="mb-3 text-3xl font-black leading-tight text-quant-text">{event.title}</h2>
        <p className="mb-3 text-xs font-bold uppercase text-quant-muted">
          Source event {event.article_hash ? `| ${event.article_hash.slice(0, 10)}` : ""} {event.id ? `| ID ${event.id}` : ""}
        </p>
        <p className="mb-6 text-base leading-relaxed text-quant-muted">{event.description}</p>

        <section className="mb-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">Model reasoning</p>
          <p className="text-sm leading-relaxed text-quant-text">{event.reasoning}</p>
        </section>

        <div className="grid gap-3 md:grid-cols-3">
          <DetailStat label="Confidence" value={`${Math.round(confidenceValue(event) * 100)}%`} />
          <DetailStat label="Sectors" value={event.affected_sectors.length.toString()} />
          <DetailStat label="Linked assets" value={assets.length.toString()} />
        </div>

        <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-3">Price reaction since event</p>
          <div className="grid gap-2 md:grid-cols-2">
            {assets.length === 0 ? (
              <InlineEmpty title="No asset reaction available" description="This event has no linked asset price snapshots." />
            ) : assets.map((asset) => {
              const latest = quotes[asset.symbol]?.price;
              const entry = asset.price;
              const change = typeof latest === "number" && typeof entry === "number" && entry !== 0
                ? ((latest - entry) / entry) * 100
                : null;
              return (
                <button key={asset.symbol} onClick={() => go(`#/assets/${asset.symbol}`)} className="rounded-lg border border-quant-line bg-quant-bg/55 p-3 text-left transition hover:border-quant-green/50">
                  <div className="mb-1 flex items-center justify-between">
                    <strong className="text-quant-text">{asset.symbol}</strong>
                    <span className={change === null ? "text-quant-muted" : change >= 0 ? "text-quant-green" : "text-quant-red"}>
                      {change === null ? "Pending" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`}
                    </span>
                  </div>
                  <p className="text-xs text-quant-muted">Entry {typeof entry === "number" ? `$${entry.toFixed(2)}` : "n/a"} | Latest {typeof latest === "number" ? `$${latest.toFixed(2)}` : "pending"}</p>
                </button>
              );
            })}
          </div>
        </section>

        <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-3">Event trail</p>
          <div className="grid gap-2 md:grid-cols-3">
            <ProcessStep label="1. Ingested" detail="News article normalized into a macro event." />
            <ProcessStep label="2. Classified" detail={`${formatLabel(event.event_type)} with ${event.severity ?? "low"} severity.`} />
            <ProcessStep label="3. Linked" detail={`${assets.length} asset signal${assets.length === 1 ? "" : "s"} attached.`} />
          </div>
        </section>
      </article>

      <aside className="grid content-start gap-4">
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Linked assets</p>
          <div className="grid gap-2">
            {assets.map((asset) => (
              <button key={asset.symbol} onClick={() => go(`#/assets/${asset.symbol}`)} className="flex items-center justify-between rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <span className="font-black text-quant-text">{asset.symbol}</span>
                <ChevronRight size={15} className="text-quant-muted" />
              </button>
            ))}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Affected sectors</p>
          <div className="flex flex-wrap gap-2">
            {event.affected_sectors.map((sector) => <span key={sector} className="quant-tag">{formatLabel(sector)}</span>)}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Linked predictions</p>
          <div className="grid gap-2">
            {linkedPredictions.length === 0 ? (
              <InlineEmpty title="No linked forecasts" description="Prediction records for these assets are not available yet." />
            ) : linkedPredictions.map(({ prediction, index }) => (
              <button key={`${prediction.symbol}-${index}`} onClick={() => go(`#/predictions/${index}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <div className="mb-1 flex items-center justify-between">
                  <strong className="text-quant-text">{prediction.symbol}</strong>
                  <span className={`text-xs font-black ${directionColor(prediction.impact_direction)}`}>{prediction.expected_move_pct}%</span>
                </div>
                <p className="text-xs text-quant-muted">{Math.round(prediction.probability * 100)}% | {prediction.horizon}</p>
              </button>
            ))}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Related articles</p>
          <div className="grid gap-2">
            {relatedEvents.length === 0 ? (
              <InlineEmpty title="No related events" description="No nearby duplicates or sector matches in the current feed." />
            ) : relatedEvents.map((candidate) => <EventRow key={eventKey(candidate)} event={candidate} compact />)}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-2">Next actions</p>
          <div className="grid gap-2">
            <button onClick={() => assets[0] && go(`#/assets/${assets[0].symbol}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left text-sm font-bold text-quant-text transition hover:border-quant-green/50">
              Inspect lead asset
            </button>
            <button onClick={() => go("#/predictions")} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left text-sm font-bold text-quant-text transition hover:border-quant-green/50">
              Compare model forecasts
            </button>
          </div>
        </section>
      </aside>
    </div>
  );
}

function AssetsPage({ rows, quotes, events, predictions }: { rows: AssetImpactRow[]; quotes: Record<string, PriceQuote>; events: EventRecord[]; predictions: PredictionSummary | null }) {
  return (
    <section className="quant-panel p-4">
      <div className="mb-4">
        <p className="quant-eyebrow">Assets</p>
        <h3 className="text-xl font-black text-quant-text">Impact map</h3>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {rows.length === 0 ? (
          <InlineEmpty title="No linked assets yet" description="Refresh events or wait for ingestion to map articles to tradable symbols." />
        ) : rows.map((row) => {
          const quote = quotes[row.symbol];
          const linked = events.filter((event) => getLinkedAssets(event).some((asset) => asset.symbol === row.symbol)).length;
          const predictionCount = predictionsForSymbols(predictions, [row.symbol]).length;
          return (
            <button key={row.symbol} onClick={() => go(`#/assets/${row.symbol}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-4 text-left transition hover:border-quant-green/50">
              <div className="mb-3 flex items-center justify-between">
                <strong className="text-2xl font-black text-quant-text">{row.symbol}</strong>
                <span className={`font-black capitalize ${directionColor(row.direction)}`}>{directionArrow(row.direction)} {row.direction}</span>
              </div>
              <p className="mb-3 text-sm text-quant-muted">{formatLabel(row.eventType)} | {linked} linked event(s) | {predictionCount} forecast(s)</p>
              <div className="flex items-end justify-between">
                <span className="text-lg font-black text-quant-text">{typeof quote?.price === "number" ? `$${quote.price.toFixed(2)}` : "Price pending"}</span>
                <span className="text-xs font-bold uppercase text-quant-muted">{row.confidence}</span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function AssetDetailPage({
  symbol,
  rows,
  quotes,
  events,
  predictions,
  backtest,
  holdingSymbols,
}: {
  symbol: string;
  rows: AssetImpactRow[];
  quotes: Record<string, PriceQuote>;
  events: EventRecord[];
  predictions: PredictionSummary | null;
  backtest: BacktestSummary | null;
  holdingSymbols: string[];
}) {
  const row = rows.find((item) => item.symbol === symbol);
  const isHolding = holdingSymbols.includes(symbol);
  const fallbackRow: AssetImpactRow | null = isHolding
    ? {
        symbol,
        direction: "neutral",
        eventType: "portfolio_holding",
        confidence: "low",
        severity: "low",
      }
    : null;
  const displayRow = row ?? fallbackRow;
  const quote = quotes[symbol];
  const linkedEvents = events.filter((event) => getLinkedAssets(event).some((asset) => asset.symbol === symbol));
  const linkedPredictions = predictionsForSymbols(predictions, [symbol]);
  const signalHistory = (backtest?.recent ?? backtest?.top_signals ?? []).filter((signal) => signal.symbol === symbol);
  const accuracy = backtest?.by_symbol?.[symbol]?.accuracy_pct;
  if (!displayRow) return <EmptyState title={`No asset data for ${symbol}`} action="Back to assets" onClick={() => go("#/assets")} />;
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
      <section className="quant-panel p-5">
        <button onClick={() => go(isHolding ? "#/portfolio" : "#/assets")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> {isHolding ? "Holdings" : "Assets"}</button>
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <p className="quant-eyebrow">Asset Detail</p>
            <h2 className="text-4xl font-black text-quant-text">{symbol}</h2>
            <p className="mt-1 text-sm text-quant-muted">{isHolding ? holdingDescription(symbol) : `${formatLabel(displayRow.eventType)} | ${displayRow.confidence} confidence`}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-quant-muted">Latest</p>
            <strong className="text-3xl font-black text-quant-text">{typeof quote?.price === "number" ? `$${quote.price.toFixed(2)}` : "Pending"}</strong>
          </div>
        </div>
        <PriceLineChart values={quote?.history ?? []} />
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <DetailStat label="Impact" value={displayRow.direction} />
          <DetailStat label="Signal source" value={row ? formatLabel(row.eventType) : "Holding watch"} />
          <DetailStat label="Linked events" value={linkedEvents.length.toString()} />
          <DetailStat label="Backtest accuracy" value={typeof accuracy === "number" ? `${accuracy}%` : "Pending"} />
          <DetailStat label="Forecasts" value={linkedPredictions.length.toString()} />
          <DetailStat label="Signal history" value={signalHistory.length.toString()} />
        </div>
      </section>
      <aside className="grid content-start gap-4">
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Linked events</p>
          <div className="grid gap-2">
            {linkedEvents.length === 0 ? (
              <InlineEmpty title="No linked events" description="This symbol is not present in the current filtered event set." />
            ) : linkedEvents.map((event) => <EventRow key={eventKey(event)} event={event} compact />)}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Asset forecasts</p>
          <div className="grid gap-2">
            {linkedPredictions.length === 0 ? (
              <InlineEmpty title="No forecasts" description="No prediction rows currently reference this symbol." />
            ) : linkedPredictions.map(({ prediction, index }) => (
              <button key={`${prediction.symbol}-${index}`} onClick={() => go(`#/predictions/${index}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <div className="mb-1 flex items-center justify-between">
                  <strong className="text-quant-text">{prediction.horizon}</strong>
                  <span className={directionColor(prediction.impact_direction)}>{prediction.expected_move_pct}%</span>
                </div>
                <p className="text-xs text-quant-muted">{Math.round(prediction.probability * 100)}% probability</p>
              </button>
            ))}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Signal history</p>
          <div className="grid gap-2">
            {signalHistory.length === 0 ? (
              <InlineEmpty title="No backtest history" description="Run the backtest to populate signal rows for this asset." />
            ) : signalHistory.slice(0, 5).map((signal, index) => {
              const signalIndex = (backtest ? allBacktestSignals(backtest) : []).findIndex((candidate) =>
                candidate.symbol === signal.symbol &&
                candidate.evaluated_at === signal.evaluated_at &&
                candidate.event_type === signal.event_type &&
                candidate.return_pct === signal.return_pct
              ) ?? -1;
              return (
              <button key={`${signal.symbol}-${signal.evaluated_at}-${index}`} onClick={() => signalIndex >= 0 && go(`#/signals/${signalIndex}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <div className="mb-1 flex items-center justify-between">
                  <strong className="text-quant-text">{signal.outcome_status}</strong>
                  <span className={signal.return_pct >= 0 ? "text-quant-green" : "text-quant-red"}>{signal.return_pct}%</span>
                </div>
                <p className="text-xs text-quant-muted">{formatLabel(signal.event_type)} | score {signal.ml_score.toFixed(2)}</p>
              </button>
              );
            })}
          </div>
        </section>
      </aside>
    </div>
  );
}

function PredictionsPage({ summary }: { summary: PredictionSummary | null }) {
  const [horizonFilter, setHorizonFilter] = useState("all");
  const [assetFilter, setAssetFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [actionFilter, setActionFilter] = useState("actionable");
  if (!summary) return <PredictionsPanel />;
  const assets = uniqueValues(summary.predictions.map((prediction) => prediction.symbol));
  const horizons = uniqueValues(summary.predictions.map((prediction) => prediction.horizon));
  const eventTypes = uniqueValues(summary.predictions.map((prediction) => prediction.event_type));
  const visible = summary.predictions.filter((prediction) =>
    (horizonFilter === "all" || prediction.horizon === horizonFilter) &&
    (assetFilter === "all" || prediction.symbol === assetFilter) &&
    (typeFilter === "all" || prediction.event_type === typeFilter) &&
    (actionFilter === "all" || (prediction.actionability ?? (prediction.is_actionable ? "actionable" : "blocked")) === actionFilter)
  );
  const actionable = summary.predictions.filter((prediction) => prediction.actionability === "actionable" || prediction.is_actionable).length;
  const watch = summary.predictions.filter((prediction) => prediction.actionability === "watch").length;
  const blocked = summary.predictions.filter((prediction) => !prediction.is_actionable && prediction.actionability !== "watch").length;
  return (
    <section className="grid gap-5">
      <div className="quant-panel-strong p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="quant-eyebrow">Signals</p>
            <h3 className="text-3xl font-black text-quant-text">Investor signal board</h3>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-quant-muted">Signals are grouped by actionability, expected move, and portfolio relevance. Model internals stay available in diagnostics, but the main view focuses on decisions.</p>
          </div>
          <span className="quant-tag">{summary.model_version}</span>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <MetricCard label="Actionable" value={actionable.toString()} icon={Target} onClick={() => setActionFilter("actionable")} />
          <MetricCard label="Watch" value={watch.toString()} icon={Gauge} onClick={() => setActionFilter("watch")} />
          <MetricCard label="Blocked" value={blocked.toString()} icon={Database} onClick={() => setActionFilter("blocked")} />
          <MetricCard label="Total" value={summary.total_considered.toString()} icon={FileText} onClick={() => setActionFilter("all")} />
        </div>
      </div>
      <section className="quant-panel p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="quant-eyebrow">Queue</p>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-xl font-black text-quant-text">Ranked opportunities and risks</h3>
          </div>
          <p className="mt-1 text-sm text-quant-muted">Use filters to narrow by holdings, horizon, catalyst type, or actionability.</p>
        </div>
      </div>
      <div className="mb-4 grid gap-2 md:grid-cols-4">
        <SelectControl label="Action" value={actionFilter} onChange={setActionFilter} options={["all", "actionable", "watch", "blocked"]} />
        <SelectControl label="Horizon" value={horizonFilter} onChange={setHorizonFilter} options={["all", ...horizons]} />
        <SelectControl label="Asset" value={assetFilter} onChange={setAssetFilter} options={["all", ...assets]} />
        <SelectControl label="Event type" value={typeFilter} onChange={setTypeFilter} options={["all", ...eventTypes]} />
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {visible.length === 0 ? (
          <InlineEmpty title="No predictions yet" description="Generate events and predictions, then return to this page." />
        ) : visible.map((prediction) => {
          const index = summary.predictions.indexOf(prediction);
          return <SignalBriefCard key={`${prediction.symbol}-${index}`} prediction={prediction} index={index} />;
        })}
      </div>
      </section>
    </section>
  );
}

function AppLoadingScreen() {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-quant-bg/88 px-6 backdrop-blur-xl">
      <div className="loading-card w-full max-w-sm rounded-lg border border-quant-line bg-white/70 p-6 text-center shadow-panel">
        <BasisLogo compact />
        <h2 className="mt-4 text-xl font-black text-quant-text">Basis</h2>
        <p className="mt-2 text-sm leading-6 text-quant-muted">Loading market feed, holdings context, and forward signals.</p>
        <div className="mt-5 grid gap-2">
          <LoadingRail />
          <LoadingRail delay="120ms" />
          <LoadingRail delay="240ms" />
        </div>
      </div>
    </div>
  );
}

function LoadingRail({ delay = "0ms" }: { delay?: string }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
      <div className="loading-rail h-full w-1/2 rounded-full bg-quant-green" style={{ animationDelay: delay }} />
    </div>
  );
}

function PredictionDetailPage({ summary, index, marketRegime, allPredictions }: { summary: PredictionSummary | null; index: number; marketRegime: MarketRegime | null; allPredictions: PredictionSummary | null }) {
  const prediction = summary?.predictions[index];
  if (!prediction) return <EmptyState title="Prediction not found" action="Back to predictions" onClick={() => go("#/predictions")} />;
  const similar = (allPredictions?.predictions ?? [])
    .map((candidate, candidateIndex) => ({ candidate, candidateIndex }))
    .filter(({ candidate, candidateIndex }) => candidateIndex !== index && (candidate.symbol === prediction.symbol || candidate.event_type === prediction.event_type))
    .slice(0, 4);
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
      <section className="quant-panel p-5">
        <button onClick={() => go("#/predictions")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> Predictions</button>
        <p className="quant-eyebrow">Signal Detail</p>
        <h2 className="mb-2 text-4xl font-black text-quant-text">{prediction.symbol}</h2>
        <p className={`mb-5 text-xl font-black capitalize ${directionColor(prediction.impact_direction)}`}>{directionArrow(prediction.impact_direction)} {formatInvestorDirection(prediction.impact_direction)} | {Math.round(prediction.probability * 100)}%</p>
        <div className="grid gap-3 md:grid-cols-4">
          <DetailStat label="Expected" value={`${prediction.expected_move_pct}%`} />
          <DetailStat label="Excess" value={`${prediction.expected_excess_return_pct > 0 ? "+" : ""}${prediction.expected_excess_return_pct ?? 0}%`} />
          <DetailStat label="Range" value={`${prediction.expected_move_low_pct}% to ${prediction.expected_move_high_pct}%`} />
          <DetailStat label="Horizon" value={prediction.horizon} />
        </div>
        <section className="mt-5 rounded-lg border border-quant-line bg-white/45 p-4 backdrop-blur-xl">
          <p className="quant-eyebrow mb-2">Why this matters</p>
          <p className="text-base leading-7 text-quant-text">{prediction.why_this_matters || prediction.base_case}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="quant-tag">{formatLabel(prediction.actionability ?? "watch")}</span>
            <span className="quant-tag">{formatLabel(prediction.confidence_tier ?? "medium")} confidence</span>
            {prediction.risk_factors?.slice(0, 3).map((risk) => <span key={risk} className="quant-tag">Risk: {risk}</span>)}
          </div>
        </section>
        <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">Source event</p>
          <h3 className="font-black text-quant-text">{prediction.title}</h3>
          <p className="mt-2 text-sm text-quant-muted">{prediction.base_case}</p>
        </section>

        <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-3">Market regime at review time</p>
          <RegimeSummary regime={marketRegime} />
        </section>

        <section className="mt-5 grid gap-3 md:grid-cols-3">
          <ScenarioCard label="Bull" text={prediction.bull_case} tone="green" />
          <ScenarioCard label="Base" text={prediction.base_case} tone="blue" />
          <ScenarioCard label="Bear" text={prediction.bear_case} tone="red" />
        </section>
      </section>
      <section className="quant-panel p-4">
        <p className="quant-eyebrow mb-3">Model diagnostics</p>
        <div className="grid gap-2">
          {(prediction.shap_contributions ?? []).length === 0 ? <p className="text-sm text-quant-muted">No feature attribution available yet.</p> : prediction.shap_contributions.map((item) => <FeatureBar key={item.feature} label={item.feature} value={item.shap_value} />)}
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-3">
          <p className="quant-eyebrow mb-2">Drivers</p>
          <div className="flex flex-wrap gap-2">
            {prediction.drivers.map((driver) => <span key={driver} className="quant-tag">{driver}</span>)}
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-3">
          <p className="quant-eyebrow mb-2">Similar forecasts</p>
          <div className="grid gap-2">
            {similar.length === 0 ? (
              <p className="text-sm text-quant-muted">No similar forecasts available.</p>
            ) : similar.map(({ candidate, candidateIndex }) => (
              <button key={`${candidate.symbol}-${candidateIndex}`} onClick={() => go(`#/predictions/${candidateIndex}`)} className="rounded-lg border border-quant-line bg-quant-bg/55 p-3 text-left transition hover:border-quant-green/50">
                <div className="flex items-center justify-between">
                  <strong className="text-quant-text">{candidate.symbol}</strong>
                  <span className={directionColor(candidate.impact_direction)}>{candidate.expected_move_pct}%</span>
                </div>
                <p className="mt-1 text-xs text-quant-muted">{formatLabel(candidate.event_type)} | {candidate.horizon}</p>
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function PortfolioPage({
  appData,
  events,
  quotes,
  holdingsText,
  holdingSymbols,
  onHoldingsChange,
}: {
  appData: AppData;
  events: EventRecord[];
  quotes: Record<string, PriceQuote>;
  holdingsText: string;
  holdingSymbols: string[];
  onHoldingsChange: (value: string) => void;
}) {
  const summary = appData.backtest;
  const portfolio = appData.portfolio;
  const actionableSignals = summary?.actionable_signals ?? portfolio?.actionable_signals ?? 0;
  const hasActionableSignals = actionableSignals > 0;
  return (
    <div className="grid gap-5">
      <div className="grid gap-3 md:grid-cols-4">
        <MetricCard label="Strategy return" value={portfolio ? hasActionableSignals ? `${portfolio.total_return_pct >= 0 ? "+" : ""}${portfolio.total_return_pct}%` : "Pending" : "--"} icon={LineChart} onClick={() => go("#/portfolio")} />
        <MetricCard label="SPY benchmark" value={portfolio ? hasActionableSignals ? `${portfolio.benchmark_return_pct >= 0 ? "+" : ""}${portfolio.benchmark_return_pct}%` : "Pending" : "--"} icon={Activity} onClick={() => go("#/portfolio")} />
        <MetricCard label="Win rate" value={portfolio ? hasActionableSignals ? `${portfolio.win_rate_pct}%` : "N/A" : "--"} icon={Target} onClick={() => go("#/portfolio")} />
        <MetricCard label="Signals" value={summary?.total_signals.toString() ?? "--"} icon={FileText} onClick={() => go("#/portfolio")} />
      </div>
      <HoldingsWorkspace
        events={events}
        predictions={appData.predictions}
        quotes={quotes}
        holdingsText={holdingsText}
        holdingSymbols={holdingSymbols}
        onHoldingsChange={onHoldingsChange}
        regime={appData.marketRegime}
      />
      <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.4fr)_minmax(420px,0.8fr)]">
        <BacktestPanel />
        <aside className="grid content-start gap-4">
          <PortfolioInsightPanel portfolio={portfolio} summary={summary} />
          <section className="quant-panel p-4">
            <div className="mb-4">
              <p className="quant-eyebrow">Signal inventory</p>
              <h3 className="text-lg font-black text-quant-text">Outcome state</h3>
            </div>
            <div className="grid gap-3">
              <ReadinessRow label="Actionable signals" ready={(summary?.actionable_signals ?? 0) > 0} detail={`${summary?.actionable_signals ?? 0} priced moves`} />
              <ReadinessRow label="Flat / pending signals" ready={(summary?.flat_signals ?? 0) === 0} detail={`${summary?.flat_signals ?? 0} still below threshold`} />
              <ReadinessRow label="Benchmark available" ready={(portfolio?.benchmark_return_pct ?? 0) !== 0} detail="SPY comparison line loaded" />
            </div>
          </section>
        </aside>
      </div>
      <section className="quant-panel p-4">
        <div className="mb-4">
          <p className="quant-eyebrow">Breakdowns</p>
          <h3 className="text-xl font-black text-quant-text">Performance by segment</h3>
          <p className="mt-1 text-sm text-quant-muted">Use these tables to see whether one event type, asset, or severity bucket is carrying the result.</p>
        </div>
        <div className="grid gap-4 xl:grid-cols-3">
          <BreakdownTable title="Event type" groups={summary?.by_event_type} />
          <BreakdownTable title="Asset" groups={summary?.by_symbol} />
          <BreakdownTable title="Severity" groups={summary?.by_severity} />
        </div>
      </section>
    </div>
  );
}

function MLPage({ appData }: { appData: AppData }) {
  const validationIssues = appData.validation?.issues ?? [];
  const stats = appData.trainingStats;
  return (
    <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.1fr)_minmax(440px,0.9fr)]">
      <div>
        <section className="mb-4 rounded-lg border border-quant-yellow/35 bg-quant-yellow/8 p-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase text-quant-yellow">Live model status</p>
              <p className="mt-1 text-sm text-quant-muted">{formatLabel(appData.modelHealth?.status ?? "pending")} across the active prediction stream.</p>
            </div>
            <strong className="text-lg font-black text-quant-yellow">{stats?.num_samples ?? "--"} labels</strong>
          </div>
        </section>
        <MLIntelligencePanel />
      </div>
      <aside className="grid content-start gap-4">
        <section className="quant-panel p-4">
          <p className="quant-eyebrow">Model readout</p>
          <h3 className="mb-3 text-lg font-black text-quant-text">What the warnings mean</h3>
          <div className="grid gap-3">
            <ProcessStep label="Drift detected" detail="The recent prediction stream no longer looks like the model's training-time confidence profile. With only 78 labels this is a watch item, not a production incident." />
            <ProcessStep label="Confidence PSI" detail="PSI compares the distribution of recent confidence scores against the distribution stored during training. Higher means the model is operating in a different confidence regime." />
            <ProcessStep label="Fresh outcomes" detail="The model becomes more reliable as live predictions mature into labeled outcomes for the active model version." />
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow mb-3">Training runs</p>
          <div className="grid gap-2">
            {appData.trainingHistory.length === 0 ? (
              <InlineEmpty title="No runs logged" description="Train a model to populate experiment history." />
            ) : appData.trainingHistory.slice(0, 8).map((run, index) => (
              <button key={run.id} onClick={() => go(`#/training/${index}`)} className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50">
                <div className="mb-1 flex items-center justify-between">
                  <strong className="text-quant-text">{((run.deployment_accuracy ?? run.calibrated_accuracy ?? run.accuracy) * 100).toFixed(1)}%</strong>
                  <span className="text-xs font-bold uppercase text-quant-muted">{run.triggered_by}</span>
                </div>
                <p className="text-xs text-quant-muted">{run.dataset_size}s | AUC {run.roc_auc?.toFixed(3) ?? "n/a"}</p>
              </button>
            ))}
          </div>
        </section>
        <section className="quant-panel p-4">
          <p className="quant-eyebrow">Dataset validation</p>
          <h3 className="mb-2 text-lg font-black text-quant-text">Is this an issue?</h3>
          <p className="mb-3 text-sm leading-6 text-quant-muted">
            It is not blocking training. It means some feature columns currently have the same value for every sample, so they add no learning signal yet. That is expected while the dataset is tiny or market context has not varied much.
          </p>
          <div className="mb-3 grid gap-2 md:grid-cols-3">
            <DetailStat label="Samples" value={stats?.num_samples.toString() ?? "--"} />
            <DetailStat label="Features" value={stats?.feature_count.toString() ?? "--"} />
            <DetailStat label="Issues" value={validationIssues.length.toString()} />
          </div>
          <div className="grid gap-2">
            {validationIssues.length === 0 ? (
              <div className="rounded-lg border border-quant-green/35 bg-quant-green/8 p-3 text-sm font-bold text-quant-green">No validation issues reported.</div>
            ) : validationIssues.map((issue) => <ValidationIssueCard key={issue} issue={issue} compact />)}
          </div>
        </section>
      </aside>
    </div>
  );
}

function BreadthPage({ events }: { events: EventRecord[] }) {
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <MarketBreadth events={events} />
      <WatchlistPanel events={events} />
      <SectorHeatmap events={events} activeSector="" onSectorClick={() => {}} />
    </div>
  );
}

function AlertsPage({ alerts }: { alerts: ReturnType<typeof buildAlertRules> }) {
  return (
    <div className="max-w-4xl">
      <AlertQueue alerts={alerts} />
    </div>
  );
}

function AccuracyPage() {
  return (
    <div className="max-w-4xl">
      <SignalAccuracy />
    </div>
  );
}

function DataHealthPage({ appData, events }: { appData: AppData; events: EventRecord[] }) {
  const stats = appData.trainingStats;
  const validation = appData.validation;
  const issues = validation?.issues ?? [];
  const usable = stats?.num_samples ?? 0;
  return (
    <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_minmax(460px,0.9fr)]">
      <section className="quant-panel p-5">
        <p className="quant-eyebrow">System Health</p>
        <h3 className="mb-4 text-xl font-black text-quant-text">Pipeline readiness</h3>
        <div className="grid gap-3 md:grid-cols-3">
          <DetailStat label="Events" value={events.length.toString()} />
          <DetailStat label="Samples" value={stats?.num_samples.toString() ?? "--"} />
          <DetailStat label="Features" value={stats?.feature_count.toString() ?? "--"} />
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">Signal label balance</p>
          <div className="grid gap-2 md:grid-cols-2">
            <FeatureBar label="Correct" value={stats?.class_distribution.positive ?? 0} />
            <FeatureBar label="Incorrect" value={stats?.class_distribution.negative ?? 0} />
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">Readiness checklist</p>
          <div className="grid gap-2">
            <ReadinessRow label="Backend API reachable" ready={true} />
            <ReadinessRow label="Events ingested" ready={events.length > 0} />
            <ReadinessRow label="Training samples available" ready={(stats?.num_samples ?? 0) > 0} />
            <ReadinessRow label="Live outcomes accumulating" ready={Boolean(appData.modelHealth?.rolling_accuracy?.samples)} detail={`${appData.modelHealth?.rolling_accuracy?.samples ?? 0} active-model labels`} />
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-3">Pipeline flow</p>
          <div className="grid gap-3 md:grid-cols-4">
            <ProcessStep label="Ingest" detail={`${events.length} events currently visible from the API.`} />
            <ProcessStep label="Predict" detail="Each event is expanded into linked asset forecasts." />
            <ProcessStep label="Label" detail={`${stats?.num_samples ?? 0} samples are usable for training right now.`} />
            <ProcessStep label="Validate" detail={`${issues.length} warning${issues.length === 1 ? "" : "s"} found before training.`} />
          </div>
        </div>
      </section>
      <section className="quant-panel p-5">
        <p className="quant-eyebrow">Validation issues</p>
        <h3 className="mb-2 text-xl font-black text-quant-text">Warnings, not crashes</h3>
        <p className="mb-4 text-sm leading-6 text-quant-muted">
          Data validation checks whether the training table is large enough, balanced enough, and whether features actually vary. Yellow cards mean the model can still run, but the resulting accuracy should be treated as early-stage.
        </p>
        <div className="mb-4 grid gap-3 md:grid-cols-3">
          <DetailStat label="Status" value={issues.length ? "Warning" : "Clean"} />
          <DetailStat label="Correct labels" value={`${stats ? Math.round(stats.class_distribution.positive * 100) : 0}%`} />
          <DetailStat label="Incorrect labels" value={`${stats ? Math.round(stats.class_distribution.negative * 100) : 0}%`} />
        </div>
        <div className="grid gap-3">
          {issues.length ? issues.map((issue) => <ValidationIssueCard key={issue} issue={issue} />) : <div className="rounded-lg border border-quant-green/35 bg-quant-green/8 p-3 text-sm font-bold text-quant-green">No validation issues reported.</div>}
        </div>
        <div className="mt-4 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">How this gets better</p>
          <div className="grid gap-2">
            <ReadinessRow label="More labels" ready={(stats?.num_samples ?? 0) > 0} detail={`${stats?.num_samples ?? 0} training labels available`} />
            <ReadinessRow label="More market variety" ready={issues.length === 0} detail="VIX, rates, trend, and return features need variation." />
            <ReadinessRow label="Outcome freshness" ready={(stats?.num_samples ?? 0) > 0} detail="Keep computing outcomes as new prices arrive." />
          </div>
        </div>
      </section>
    </div>
  );
}

function RegimePage({ regime }: { regime: MarketRegime | null }) {
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
      <section className="quant-panel p-5">
        <p className="quant-eyebrow">Market Regime</p>
        <h3 className="mb-4 text-2xl font-black text-quant-text">Context layer for every forecast</h3>
        <RegimeSummary regime={regime} large />
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <ProcessStep label="VIX" detail="Volatility state helps separate calm, normal, stressed, and crisis regimes." />
          <ProcessStep label="SPY trend" detail="Trend direction controls whether risk appetite is supportive or deteriorating." />
          <ProcessStep label="10Y yield" detail="Rate environment is injected as a model feature for macro sensitivity." />
        </div>
      </section>
      <section className="quant-panel p-5">
        <p className="quant-eyebrow mb-3">Regime encoding</p>
        <div className="grid gap-2">
          <ReadinessRow label="Risk-on" ready={regime?.market_regime_encoded === 0} />
          <ReadinessRow label="Neutral" ready={regime?.market_regime_encoded === 1} />
          <ReadinessRow label="Risk-off" ready={regime?.market_regime_encoded === 2} />
        </div>
      </section>
    </div>
  );
}

function TrainingRunPage({ history, index }: { history: TrainingRun[]; index: number }) {
  const run = history[index];
  if (!run) return <EmptyState title="Training run not found" action="Back to model" onClick={() => go("#/ml")} />;
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
      <section className="quant-panel p-5">
        <button onClick={() => go("#/ml")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> Model</button>
        <p className="quant-eyebrow">Training Run</p>
        <h3 className="mb-4 text-2xl font-black text-quant-text">{new Date(run.trained_at).toLocaleString()}</h3>
        <div className="grid gap-3 md:grid-cols-4">
          <DetailStat label="Deployed accuracy" value={`${((run.deployment_accuracy ?? run.calibrated_accuracy ?? run.accuracy) * 100).toFixed(1)}%`} />
          <DetailStat label="ROC AUC" value={run.roc_auc?.toFixed(3) ?? "n/a"} />
          <DetailStat label="Dataset" value={run.dataset_size.toString()} />
          <DetailStat label="Comparison winner" value={run.comparison_winner ?? run.winner_model} />
        </div>
        <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-3">Top features</p>
          <div className="grid gap-2">
            {run.top_features.slice(0, 10).map((feature) => <FeatureBar key={feature.feature} label={feature.feature} value={feature.importance} />)}
          </div>
        </section>
      </section>
      <section className="quant-panel p-5">
        <p className="quant-eyebrow mb-3">Run diagnostics</p>
        <div className="grid gap-3">
          <DetailStat label="Train / test" value={`${run.train_size} / ${run.test_size}`} />
          <DetailStat label="Walk-forward" value={`${(run.walk_forward_mean * 100).toFixed(1)}% ± ${(run.walk_forward_std * 100).toFixed(1)}%`} />
          <DetailStat label="Brier improvement" value={run.brier_improvement.toFixed(4)} />
          <DetailStat label="Label balance" value={`${run.label_balance.positive_count} correct / ${run.label_balance.negative_count} incorrect`} />
        </div>
      </section>
    </div>
  );
}

function SignalDetailPage({ backtest, index }: { backtest: BacktestSummary | null; index: number }) {
  const signals = backtest ? allBacktestSignals(backtest) : [];
  const signal = signals[index];
  if (!signal) return <EmptyState title="Signal not found" action="Back to portfolio" onClick={() => go("#/portfolio")} />;
  return (
    <section className="quant-panel max-w-4xl p-5">
      <button onClick={() => go("#/portfolio")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> Portfolio</button>
      <p className="quant-eyebrow">Signal Detail</p>
      <h3 className="mb-4 text-3xl font-black text-quant-text">{signal.symbol}</h3>
      <div className="grid gap-3 md:grid-cols-4">
        <DetailStat label="Outcome" value={signal.outcome_status} />
        <DetailStat label="Return" value={`${signal.return_pct}%`} />
        <DetailStat label="ML score" value={signal.ml_score.toFixed(2)} />
        <DetailStat label="Direction" value={signal.expected_direction} />
      </div>
      <section className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
        <p className="quant-eyebrow mb-2">Signal context</p>
        <p className="text-sm text-quant-muted">{formatLabel(signal.event_type)} | {signal.severity} severity | {signal.horizon}</p>
      </section>
    </section>
  );
}

function allBacktestSignals(backtest: BacktestSummary) {
  const seen = new Set<string>();
  return [...backtest.top_signals, ...backtest.recent].filter((signal) => {
    const key = `${signal.symbol}:${signal.event_type}:${signal.evaluated_at}:${signal.return_pct}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function EventRow({ event, expanded = false, compact = false }: { event: EventRecord; expanded?: boolean; compact?: boolean }) {
  return (
    <button onClick={() => go(`#/events/${encodeURIComponent(eventKey(event))}`)} className={`rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50 ${compact ? "" : "hover:bg-quant-green/5"}`}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="quant-tag text-[0.65rem]">{formatLabel(event.event_type)}</span>
        <span className={`text-sm font-black capitalize ${directionColor(event.impact_direction)}`}>{directionArrow(event.impact_direction)} {event.impact_direction}</span>
        <span className="ml-auto text-xs font-bold uppercase text-quant-muted">{event.severity ?? "low"}</span>
      </div>
      <h4 className="line-clamp-2 font-black text-quant-text">{event.title}</h4>
      {expanded && <p className="mt-2 line-clamp-2 text-sm text-quant-muted">{event.description}</p>}
    </button>
  );
}

function SignalBriefCard({ prediction, index, featured = false }: { prediction: PredictionSummary["predictions"][number]; index: number; featured?: boolean }) {
  const actionability = prediction.actionability ?? (prediction.is_actionable ? "actionable" : "blocked");
  const actionTone = actionability === "actionable" ? "text-quant-green" : actionability === "watch" ? "text-quant-yellow" : "text-quant-muted";
  const moveTone = directionColor(prediction.impact_direction);
  return (
    <button
      type="button"
      onClick={() => go(`#/predictions/${index}`)}
      className={`rounded-lg border border-quant-line bg-white/48 p-4 text-left shadow-sm backdrop-blur-xl transition hover:border-quant-blue/30 hover:bg-white/65 ${featured ? "h-full" : ""}`}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="quant-eyebrow mb-1">{formatLabel(actionability)}</p>
          <strong className={`${featured ? "text-4xl" : "text-2xl"} font-black text-quant-text`}>{prediction.symbol}</strong>
        </div>
        <span className={`rounded-md border border-quant-line bg-white/50 px-2 py-1 text-xs font-black uppercase ${actionTone}`}>
          {prediction.confidence_tier ?? "medium"}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <SignalMiniStat label="Signal" value={formatInvestorDirection(prediction.impact_direction)} tone={moveTone} />
        <SignalMiniStat label="Move" value={`${prediction.expected_move_pct > 0 ? "+" : ""}${prediction.expected_move_pct}%`} tone={moveTone} />
        <SignalMiniStat label="Excess" value={`${prediction.expected_excess_return_pct > 0 ? "+" : ""}${prediction.expected_excess_return_pct ?? 0}%`} tone={moveTone} />
      </div>
      <p className="mt-4 line-clamp-2 text-sm leading-6 text-quant-muted">
        {prediction.why_this_matters || prediction.title}
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="quant-tag">{prediction.horizon}</span>
        <span className="quant-tag">{formatLabel(prediction.event_type)}</span>
        <span className={`text-xs font-black uppercase ${actionTone}`}>{formatLabel(actionability)}</span>
      </div>
      {(prediction.risk_factors?.length ?? 0) > 0 && (
        <p className="mt-3 text-xs text-quant-muted">Risk: {prediction.risk_factors.slice(0, 2).join(", ")}</p>
      )}
    </button>
  );
}

function SignalMiniStat({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-md border border-quant-line bg-white/42 p-2">
      <p className="text-[0.6rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className={`text-sm font-black capitalize ${tone}`}>{value}</strong>
    </div>
  );
}

function formatInvestorDirection(direction: string) {
  if (direction === "positive") return "Bullish";
  if (direction === "negative") return "Bearish";
  return "Neutral";
}

function ActionStrip({
  apiStatus,
  lastUpdated,
  nextAction,
  onAction,
}: {
  apiStatus: "connecting" | "live" | "error";
  lastUpdated: string;
  nextAction: ReturnType<typeof getNextAction>;
  onAction: () => void;
}) {
  const tone = apiStatus === "live" ? "text-quant-green" : apiStatus === "error" ? "text-quant-red" : "text-quant-yellow";
  return (
    <section className="mb-5 rounded-lg border border-quant-line bg-quant-panel/75 p-3 backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2">
          <span className={`inline-flex items-center gap-2 text-xs font-black uppercase ${tone}`}>
            <span className="h-2 w-2 rounded-full bg-current" />
            {apiStatus === "live" ? "Live data" : apiStatus === "error" ? "API offline" : "Connecting"}
          </span>
          <span className="text-xs font-bold text-quant-muted">Updated {lastUpdated}</span>
          <span className="text-xs font-bold text-quant-muted">{nextAction.detail}</span>
        </div>
        <button onClick={onAction} className="rounded-md border border-quant-green/35 bg-quant-green/10 px-3 py-2 text-xs font-black text-quant-green transition hover:border-quant-green/70 hover:bg-quant-green/15">
          {nextAction.label}
        </button>
      </div>
    </section>
  );
}

function getNextAction(events: EventRecord[], appData: AppData) {
  const issues = appData.validation?.issues.length ?? 0;
  if (events.length === 0) return { label: "Open events", route: "#/events", detail: "No event feed loaded yet." };
  if (issues > 0) return { label: "Review validation", route: "#/data", detail: `${issues} validation warning${issues === 1 ? "" : "s"} to review.` };
  if (appData.modelHealth?.drift_detected) return { label: "Open model", route: "#/ml", detail: "Model drift is currently flagged." };
  return { label: "Open portfolio", route: "#/portfolio", detail: "Pipeline is ready for performance review." };
}

function Breadcrumbs({ route }: { route: Route }) {
  const items = breadcrumbItems(route);
  if (items.length === 0) return null;
  return (
    <div className="mb-1 flex min-w-0 items-center gap-1 text-[0.68rem] font-bold uppercase text-quant-muted">
      {items.map((item, index) => (
        <span key={`${item.label}-${index}`} className="flex min-w-0 items-center gap-1">
          {item.href ? (
            <button onClick={() => go(item.href ?? "#/")} className="truncate transition hover:text-quant-green">
              {item.label}
            </button>
          ) : (
            <span className="truncate text-quant-text">{item.label}</span>
          )}
          {index < items.length - 1 && <ChevronRight size={11} className="shrink-0" />}
        </span>
      ))}
    </div>
  );
}

function breadcrumbItems(route: Route): Array<{ label: string; href?: string }> {
  if (route.name === "overview") return [];
  if (route.name === "event") return [{ label: "Events", href: "#/events" }, { label: "Detail" }];
  if (route.name === "asset") return [{ label: "Assets", href: "#/assets" }, { label: route.symbol }];
  if (route.name === "prediction") return [{ label: "Predictions", href: "#/predictions" }, { label: "Detail" }];
  if (route.name === "regime") return [{ label: "Model", href: "#/ml" }, { label: "Regime" }];
  if (route.name === "training") return [{ label: "Model", href: "#/ml" }, { label: "Training run" }];
  if (route.name === "signal") return [{ label: "Portfolio", href: "#/portfolio" }, { label: "Signal" }];
  if (route.name === "ml") return [{ label: "Model" }];
  if (route.name === "data") return [{ label: "System Health" }];
  return [{ label: pageTitle(route) }];
}

function PageSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="grid gap-2">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="animate-pulse rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <div className="mb-3 h-4 w-1/3 rounded bg-quant-muted/20" />
          <div className="mb-2 h-4 w-3/4 rounded bg-quant-muted/20" />
          <div className="h-3 w-1/2 rounded bg-quant-muted/20" />
        </div>
      ))}
    </div>
  );
}

function InlineEmpty({ title, description }: { title: string; description: string }) {
  return (
    <div className="col-span-full rounded-lg border border-dashed border-quant-line bg-quant-bg/45 p-5">
      <p className="mb-1 font-black text-quant-text">{title}</p>
      <p className="text-sm text-quant-muted">{description}</p>
    </div>
  );
}

function SearchResults({ results, query }: { results: SearchResult[]; query: string }) {
  return (
    <div className="absolute right-0 top-11 z-40 w-[min(34rem,calc(100vw-2rem))] rounded-lg border border-quant-line bg-quant-panel shadow-panel">
      <div className="border-b border-quant-line px-3 py-2 text-xs font-semibold text-quant-muted">
        Search results for "{query.trim()}"
      </div>
      <div className="scrollbar-quant max-h-96 overflow-auto p-2">
        {results.length === 0 ? (
          <p className="px-2 py-4 text-sm text-quant-muted">No matching events, assets, or forecasts.</p>
        ) : results.map((result) => (
          <button
            key={`${result.route}-${result.title}`}
            type="button"
            onClick={() => go(result.route)}
            className="block w-full rounded-md px-3 py-2 text-left transition hover:bg-quant-panel2"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-bold text-quant-text">{result.title}</span>
              <span className="shrink-0 text-[0.62rem] font-bold uppercase text-quant-muted">{result.kind}</span>
            </div>
            <p className="mt-0.5 line-clamp-1 text-xs text-quant-muted">{result.detail}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

function HoldingsWorkspace({
  events,
  predictions,
  quotes,
  holdingsText,
  holdingSymbols,
  onHoldingsChange,
  regime,
}: {
  events: EventRecord[];
  predictions: PredictionSummary | null;
  quotes: Record<string, PriceQuote>;
  holdingsText: string;
  holdingSymbols: string[];
  onHoldingsChange: (value: string) => void;
  regime: MarketRegime | null;
}) {
  const [serverImpact, setServerImpact] = useState<WatchlistImpact | null>(null);
  const [isImpactLoading, setIsImpactLoading] = useState(false);

  useEffect(() => {
    let active = true;
    if (holdingSymbols.length === 0) {
      setServerImpact(null);
      setIsImpactLoading(false);
      return;
    }
    setIsImpactLoading(true);
    fetchWatchlistImpact(holdingSymbols)
      .then((impact) => {
        if (active) setServerImpact(impact);
      })
      .catch(() => {
        if (active) setServerImpact(null);
      })
      .finally(() => {
        if (active) setIsImpactLoading(false);
      });
    return () => {
      active = false;
    };
  }, [holdingSymbols.join(",")]);

  const rows = holdingSymbols.map((symbol) => {
    const symbolEvents = events.filter((event) => getLinkedAssets(event).some((asset) => asset.symbol === symbol));
    const symbolPredictions = predictionsForSymbols(predictions, [symbol]);
    const serverAsset = serverImpact?.impacted_assets.find((asset) => asset.symbol === symbol);
    const negativeEvents = symbolEvents.filter((event) => event.impact_direction === "negative").length;
    const positiveEvents = symbolEvents.filter((event) => event.impact_direction === "positive").length;
    const strongestPrediction = symbolPredictions
      .map(({ prediction, index }) => ({ prediction, index }))
      .sort((a, b) => b.prediction.ranking_score - a.prediction.ranking_score)[0];
    const serverMatchedEvents = serverAsset?.matched_events ?? 0;
    const risk = Math.min(100, negativeEvents * 24 + (symbolEvents.length + serverMatchedEvents) * 8 + (strongestPrediction ? strongestPrediction.prediction.probability * 18 : 0));
    return {
      symbol,
      events: symbolEvents,
      predictions: symbolPredictions,
      serverImpact: serverAsset,
      serverMatchedEvents,
      quote: quotes[symbol],
      net: serverAsset?.net_direction ?? (positiveEvents > negativeEvents ? "positive" : negativeEvents > positiveEvents ? "negative" : "neutral"),
      risk,
      strongestPrediction,
      description: holdingDescription(symbol),
      fiveDayChange: quoteChangePct(quotes[symbol]),
    };
  });
  const portfolioRisk = rows.length ? Math.round(rows.reduce((sum, row) => sum + row.risk, 0) / rows.length) : 0;
  const covered = rows.filter((row) => row.events.length > 0 || row.predictions.length > 0 || row.serverMatchedEvents > 0).length;
  const signaledRows = rows.filter((row) => row.strongestPrediction);
  const avgExpectedMove = signaledRows.length
    ? signaledRows.reduce((sum, row) => sum + row.strongestPrediction!.prediction.expected_move_pct, 0) / signaledRows.length
    : 0;
  const avgExpectedExcess = signaledRows.length
    ? signaledRows.reduce((sum, row) => sum + (row.strongestPrediction!.prediction.expected_excess_return_pct ?? 0), 0) / signaledRows.length
    : 0;
  const bullish = signaledRows.filter((row) => row.strongestPrediction?.prediction.impact_direction === "positive").length;
  const bearish = signaledRows.filter((row) => row.strongestPrediction?.prediction.impact_direction === "negative").length;
  const uncovered = rows.filter((row) => row.events.length === 0 && row.predictions.length === 0 && row.serverMatchedEvents === 0).map((row) => row.symbol);
  const mostAtRisk = [...rows].sort((a, b) => b.risk - a.risk)[0];
  const strongestSignal = [...signaledRows].sort((a, b) => b.strongestPrediction!.prediction.ranking_score - a.strongestPrediction!.prediction.ranking_score)[0];
  const portfolioTone = avgExpectedMove > 0.15 ? "positive" : avgExpectedMove < -0.15 ? "negative" : "neutral";

  return (
    <section className="quant-panel p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="quant-eyebrow">Your holdings</p>
          <h3 className="text-xl font-black text-quant-text">Portfolio briefing</h3>
          <p className="mt-1 text-sm text-quant-muted">Enter symbols you own to see related events, forecasts, and current market context.</p>
        </div>
        <div className="grid grid-cols-3 gap-2 text-right">
          <DetailStat label="Coverage" value={isImpactLoading ? "..." : `${covered}/${holdingSymbols.length || 0}`} />
          <DetailStat label="Risk score" value={isImpactLoading ? "..." : holdingSymbols.length ? portfolioRisk.toString() : "--"} />
          <DetailStat label="Regime" value={regime ? regimeName(regime.market_regime_encoded) : "--"} />
        </div>
      </div>
      <div className="mb-4 grid gap-2 md:grid-cols-[1fr_auto]">
        <input
          value={holdingsText}
          onChange={(event) => onHoldingsChange(event.target.value)}
          className="quant-input"
          placeholder="AAPL, MSFT, QQQ, GLD"
        />
        <button className="quant-button" onClick={() => onHoldingsChange(holdingsText)}>
          Update watchlist
        </button>
      </div>
      {isImpactLoading && (
        <div className="mb-4 rounded-lg border border-quant-line bg-white/45 p-3">
          <div className="mb-2 flex items-center justify-between gap-3">
            <p className="text-xs font-black uppercase text-quant-muted">Updating holdings impact</p>
            <RefreshCw size={13} className="animate-spin text-quant-green" />
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
            <div className="loading-rail h-full w-1/2 rounded-full bg-quant-green" />
          </div>
        </div>
      )}
      <div className="mb-4 grid gap-3 xl:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-lg border border-quant-line bg-white/45 p-4 backdrop-blur-xl">
          <p className="quant-eyebrow mb-2">Portfolio forecast</p>
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h4 className={`text-3xl font-black ${directionColor(portfolioTone)}`}>{formatInvestorDirection(portfolioTone)}</h4>
              <p className="mt-1 text-sm leading-6 text-quant-muted">
                Equal-weighted across holdings with available model signals. Basis sees {bullish} bullish and {bearish} bearish signal{bullish + bearish === 1 ? "" : "s"}.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 text-right">
              <DetailStat label="Expected move" value={signaledRows.length ? `${avgExpectedMove >= 0 ? "+" : ""}${avgExpectedMove.toFixed(2)}%` : "--"} />
              <DetailStat label="Expected excess" value={signaledRows.length ? `${avgExpectedExcess >= 0 ? "+" : ""}${avgExpectedExcess.toFixed(2)}%` : "--"} />
            </div>
          </div>
          <div className="mt-4 grid gap-2 md:grid-cols-3">
            <InvestorInsight label="Highest risk" value={mostAtRisk?.symbol ?? "--"} detail={mostAtRisk ? `${Math.round(mostAtRisk.risk)} risk score` : "Add holdings to calculate"} tone={mostAtRisk?.net} />
            <InvestorInsight label="Strongest signal" value={strongestSignal?.symbol ?? "--"} detail={strongestSignal ? `${Math.round(strongestSignal.strongestPrediction!.prediction.probability * 100)}% confidence` : "No active holding signal"} tone={strongestSignal?.strongestPrediction?.prediction.impact_direction} />
            <InvestorInsight label="Uncovered" value={uncovered.length.toString()} detail={uncovered.length ? uncovered.slice(0, 3).join(", ") : "All holdings have context"} />
          </div>
        </section>
        <section className="rounded-lg border border-quant-line bg-white/45 p-4 backdrop-blur-xl">
          <p className="quant-eyebrow mb-2">What you would normally calculate</p>
          <div className="grid gap-2">
            <PortfolioCalculation label="Macro concentration" value={isImpactLoading ? "..." : `${holdingSymbols.length ? Math.round((covered / holdingSymbols.length) * 100) : 0}%`} detail="Share of holdings currently linked to events or model signals." />
            <PortfolioCalculation label="Signal balance" value={`${bullish}-${bearish}`} detail="Bullish vs bearish holdings in the current model queue." />
            <PortfolioCalculation label="Market sensitivity" value={regime ? regimeName(regime.market_regime_encoded) : "--"} detail="Current backdrop used to interpret the holding signals." />
          </div>
        </section>
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {rows.length === 0 ? (
          <InlineEmpty title="Add holdings" description="Type a few ticker symbols above to build a personalized portfolio view." />
        ) : rows.map((row) => (
          <button
            key={row.symbol}
            type="button"
            onClick={() => go(`#/assets/${row.symbol}`)}
            className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-blue/50"
          >
            <div className="mb-2 flex items-start justify-between gap-3">
              <div>
                <strong className="text-lg font-black text-quant-text">{row.symbol}</strong>
                <p className="mt-1 line-clamp-2 text-xs leading-5 text-quant-muted">{row.description}</p>
                <p className="text-xs text-quant-muted">
                  {row.events.length + row.serverMatchedEvents} catalyst{row.events.length + row.serverMatchedEvents === 1 ? "" : "s"} | {row.predictions.length} signal{row.predictions.length === 1 ? "" : "s"}
                </p>
                {row.serverImpact && <p className="mt-1 line-clamp-1 text-xs text-quant-muted">{row.serverImpact.reason}</p>}
              </div>
              <div className="text-right">
                <p className="text-xs font-bold uppercase text-quant-muted">Risk</p>
                <strong className={row.risk >= 55 ? "text-quant-red" : row.risk >= 30 ? "text-quant-yellow" : "text-quant-green"}>{Math.round(row.risk)}</strong>
              </div>
            </div>
            <div className="grid gap-2 sm:grid-cols-3">
              <MiniHoldingStat label="Price" value={typeof row.quote?.price === "number" ? `$${row.quote.price.toFixed(2)}` : "--"} />
              <MiniHoldingStat label="5D" value={row.fiveDayChange === null ? "--" : `${row.fiveDayChange >= 0 ? "+" : ""}${row.fiveDayChange.toFixed(2)}%`} tone={row.fiveDayChange === null ? undefined : row.fiveDayChange >= 0 ? "positive" : "negative"} />
              <MiniHoldingStat label="Macro view" value={formatInvestorDirection(row.net)} tone={row.net} />
            </div>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              <MiniHoldingStat
                label="Top signal"
                value={row.strongestPrediction ? `${Math.round(row.strongestPrediction.prediction.probability * 100)}% ${formatInvestorDirection(row.strongestPrediction.prediction.impact_direction)}` : "--"}
                tone={row.strongestPrediction?.prediction.impact_direction}
              />
              <MiniHoldingStat
                label="Expected"
                value={row.strongestPrediction ? `${row.strongestPrediction.prediction.expected_move_pct >= 0 ? "+" : ""}${row.strongestPrediction.prediction.expected_move_pct}%` : "--"}
                tone={row.strongestPrediction?.prediction.impact_direction}
              />
              <MiniHoldingStat
                label="Action"
                value={row.strongestPrediction ? formatLabel(row.strongestPrediction.prediction.actionability ?? "watch") : "Monitor"}
                tone={row.strongestPrediction?.prediction.actionability === "actionable" ? "positive" : row.strongestPrediction?.prediction.actionability === "blocked" ? "negative" : undefined}
              />
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}

function InvestorInsight({ label, value, detail, tone }: { label: string; value: string; detail: string; tone?: string }) {
  const toneClass = tone === "positive" ? "text-quant-green" : tone === "negative" ? "text-quant-red" : "text-quant-text";
  return (
    <div className="rounded-md border border-quant-line bg-white/42 p-3">
      <p className="text-[0.6rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className={`block text-lg font-black ${toneClass}`}>{value}</strong>
      <p className="mt-1 text-xs leading-5 text-quant-muted">{detail}</p>
    </div>
  );
}

function PortfolioCalculation({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-md border border-quant-line bg-white/42 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-black text-quant-text">{label}</span>
        <strong className="text-quant-blue">{value}</strong>
      </div>
      <p className="mt-1 text-xs leading-5 text-quant-muted">{detail}</p>
    </div>
  );
}

function MiniHoldingStat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  const toneClass = tone === "positive" ? "text-quant-green" : tone === "negative" ? "text-quant-red" : "text-quant-text";
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/50 p-2">
      <p className="text-[0.6rem] font-bold uppercase text-quant-muted">{label}</p>
      <strong className={`text-xs font-black capitalize ${toneClass}`}>{value}</strong>
    </div>
  );
}

function quoteChangePct(quote: PriceQuote | undefined): number | null {
  const history = quote?.history ?? [];
  if (history.length < 2) return null;
  const first = history[0];
  const last = history[history.length - 1];
  if (!first) return null;
  return ((last - first) / first) * 100;
}

function holdingDescription(symbol: string): string {
  const descriptions: Record<string, string> = {
    AAPL: "Apple gives the portfolio large-cap consumer technology exposure with sensitivity to hardware demand, services growth, dollar moves, and broad risk appetite.",
    MSFT: "Microsoft is a mega-cap software and cloud holding, often driven by enterprise spending, AI infrastructure expectations, rates, and broad technology sentiment.",
    NVDA: "Nvidia is a high-beta AI and semiconductor holding, usually sensitive to chip demand, capex cycles, export controls, and growth-stock conditions.",
    AMZN: "Amazon blends consumer spending, cloud infrastructure, logistics costs, and risk appetite, making it sensitive to both growth data and margin expectations.",
    GOOGL: "Alphabet is a large-cap digital advertising and AI infrastructure holding tied to ad demand, regulation, cloud growth, and broad tech multiples.",
    META: "Meta is a communication-services growth holding with exposure to digital ads, AI capex, consumer demand, and regulatory headlines.",
    TSLA: "Tesla is a high-volatility growth and EV holding tied to rates, consumer credit, delivery expectations, margins, and risk appetite.",
    SPY: "SPY tracks broad U.S. large-cap equities and is the core benchmark for overall market exposure.",
    QQQ: "QQQ concentrates large-cap growth and technology exposure, usually more sensitive to rates and risk appetite than broad-market ETFs.",
    IWM: "IWM tracks U.S. small caps, making it more sensitive to domestic growth, credit conditions, and regional-bank stress.",
    TLT: "TLT tracks long-duration Treasuries and is highly sensitive to inflation, rates, Fed expectations, and recession risk.",
    GLD: "GLD tracks gold exposure, often used as an inflation, dollar, geopolitical, or risk-hedge proxy.",
    XLE: "XLE tracks energy equities and tends to react to oil prices, supply shocks, geopolitical risk, and global demand.",
    XLK: "XLK gives sector-level technology exposure, with sensitivity to rates, earnings revisions, AI spending, and broad growth sentiment.",
    XLF: "XLF tracks financials and tends to move with rates, credit risk, loan growth, and bank profitability expectations.",
    USO: "USO tracks oil exposure and is sensitive to crude supply, demand, inventories, OPEC policy, and geopolitical disruption.",
    JETS: "JETS tracks airlines and is sensitive to fuel costs, travel demand, labor costs, and consumer conditions.",
  };
  return descriptions[symbol.toUpperCase()] ?? `${symbol.toUpperCase()} is monitored for macro-linked events, price changes, active model signals, and portfolio risk contribution.`;
}

function ProcessStep({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="rounded-lg border border-quant-line bg-quant-bg/55 p-3">
      <p className="mb-1 text-xs font-black uppercase text-quant-green">{label}</p>
      <p className="text-sm leading-relaxed text-quant-muted">{detail}</p>
    </div>
  );
}

function ScenarioCard({ label, text, tone }: { label: string; text: string; tone: "green" | "blue" | "red" }) {
  const color = tone === "green" ? "border-quant-green/35 text-quant-green" : tone === "red" ? "border-quant-red/35 text-quant-red" : "border-quant-blue/35 text-quant-blue";
  return (
    <div className={`rounded-lg border bg-quant-panel2 p-3 ${color}`}>
      <p className="mb-2 text-xs font-black uppercase">{label}</p>
      <p className="text-sm leading-relaxed text-quant-muted">{text}</p>
    </div>
  );
}

function ReadinessRow({ label, ready, detail }: { label: string; ready: boolean; detail?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-quant-line bg-quant-bg/55 p-3">
      <span>
        <span className="block text-sm font-bold text-quant-text">{label}</span>
        {detail && <span className="mt-1 block text-xs text-quant-muted">{detail}</span>}
      </span>
      <span className={`rounded-md px-2 py-1 text-[0.65rem] font-black uppercase ${ready ? "bg-quant-green/12 text-quant-green" : "bg-quant-yellow/12 text-quant-yellow"}`}>
        {ready ? "Ready" : "Waiting"}
      </span>
    </div>
  );
}

function SelectControl({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: string[] }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[0.62rem] font-black uppercase text-quant-muted">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 w-full rounded-md border border-quant-line bg-quant-bg/70 px-2 text-xs font-bold capitalize text-quant-text outline-none focus:border-quant-green/60"
      >
        {options.map((option) => (
          <option key={option} value={option}>{formatLabel(option)}</option>
        ))}
      </select>
    </label>
  );
}

function RegimeSummary({ regime, large = false }: { regime: MarketRegime | null; large?: boolean }) {
  if (!regime) return <p className="text-sm text-quant-muted">Market regime unavailable.</p>;
  const label = regime.market_regime_encoded === 0 ? "Risk on" : regime.market_regime_encoded === 2 ? "Risk off" : "Neutral";
  return (
    <div className={`grid gap-3 ${large ? "md:grid-cols-4" : ""}`}>
      <DetailStat label="Regime" value={label} />
      <DetailStat label="VIX" value={regime.vix_level.toFixed(1)} />
      <DetailStat label="SPY 20D" value={`${regime.spy_trend >= 0 ? "+" : ""}${(regime.spy_trend * 100).toFixed(2)}%`} />
      <DetailStat label="10Y" value={`${regime.rate_level.toFixed(2)}%`} />
    </div>
  );
}

function BreakdownTable({ title, groups }: { title: string; groups?: Record<string, { total: number; actionable: number; flat?: number; correct: number; accuracy_pct: number; avg_return_pct: number; avg_ml_score: number }> }) {
  const rows = Object.entries(groups ?? {}).slice(0, 6);
  return (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
      <p className="quant-eyebrow mb-3">{title}</p>
      <div className="grid gap-2">
        {rows.length === 0 ? (
          <p className="text-sm text-quant-muted">No rows yet.</p>
        ) : rows.map(([key, stats]) => (
          <div key={key} className="rounded-md border border-quant-line bg-quant-bg/55 p-2">
            <div className="mb-1 flex items-center justify-between gap-2 text-xs">
              <strong className="truncate capitalize text-quant-text">{formatLabel(key)}</strong>
              <span className="font-black text-quant-muted" title={`${stats.correct} correct of ${stats.actionable} actionable signals. ${stats.total - stats.actionable} flat or pending signals are excluded.`}>
                {stats.accuracy_pct}% accuracy
              </span>
            </div>
            <p className="text-[0.65rem] text-quant-muted">{stats.correct}/{stats.actionable} correct | {stats.actionable}/{stats.total} actionable | avg {stats.avg_return_pct}% | score {stats.avg_ml_score.toFixed(2)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PortfolioInsightPanel({ portfolio, summary }: { portfolio: PortfolioSimulation | null; summary: BacktestSummary | null }) {
  const total = summary?.total_signals ?? portfolio?.signals ?? 0;
  const actionable = summary?.actionable_signals ?? 0;
  const pending = summary?.flat_signals ?? 0;
  const actionableShare = total > 0 ? Math.round((actionable / total) * 100) : 0;
  const pendingShare = total > 0 ? Math.round((pending / total) * 100) : 0;
  const verdict = !portfolio
    ? "Run the backtest to build a benchmarked strategy curve."
    : actionable === 0
      ? "Basis is tracking signals, but none have moved enough to count as actionable strategy outcomes yet."
    : portfolio.excess_return_pct >= 0
      ? "The signal strategy is ahead of SPY on the current labeled window."
      : "The signal strategy is trailing SPY on the current labeled window.";

  return (
    <section className="quant-panel p-4">
      <p className="quant-eyebrow">Interpretation</p>
      <h3 className="mb-2 text-lg font-black text-quant-text">How to read this page</h3>
      <p className="mb-4 text-sm leading-6 text-quant-muted">{verdict}</p>
      <div className="grid gap-3">
        <ProcessStep label="Strategy line" detail="Compounds equal-sized trades from model signals. It answers: what if every eligible signal was followed?" />
        <ProcessStep label="SPY line" detail="Normalizes SPY over the same signal sequence, giving a simple market benchmark instead of an isolated return number." />
        <ProcessStep label="Flat / pending" detail="Signals that have not moved enough yet are tracked, but they are excluded from actionable accuracy until price movement clears the threshold." />
      </div>
      <div className="mt-4 rounded-lg border border-quant-line bg-quant-panel2 p-3">
        <div className="mb-2 flex items-center justify-between text-xs font-bold text-quant-muted">
          <span>Actionable</span>
          <span>{actionableShare}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
          <div className="h-full rounded-full bg-quant-green" style={{ width: `${actionableShare}%` }} />
        </div>
        <div className="mt-3 mb-2 flex items-center justify-between text-xs font-bold text-quant-muted">
          <span>Flat / pending</span>
          <span>{pendingShare}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
          <div className="h-full rounded-full bg-quant-yellow" style={{ width: `${pendingShare}%` }} />
        </div>
      </div>
    </section>
  );
}

function ValidationIssueCard({ issue, compact = false }: { issue: string; compact?: boolean }) {
  const { title, detail, action } = explainValidationIssue(issue);
  return (
    <div className={`rounded-lg border border-quant-yellow/35 bg-quant-yellow/8 ${compact ? "p-3" : "p-4"}`}>
      <p className="mb-1 text-sm font-black text-quant-yellow">{title}</p>
      <p className="text-sm leading-6 text-quant-muted">{detail}</p>
      {!compact && <p className="mt-2 text-xs font-bold uppercase text-quant-yellow">{action}</p>}
    </div>
  );
}

function explainValidationIssue(issue: string) {
  if (issue.includes("constant feature columns")) {
    return {
      title: "Constant feature columns",
      detail: "Some model inputs currently have no variation across the labeled samples. XGBoost can ignore them, so this is not fatal, but those features are not contributing signal yet.",
      action: "Usually solved by more samples across different market regimes."
    };
  }
  if (issue.includes("dataset is empty")) {
    return {
      title: "Dataset is empty",
      detail: "No labeled outcomes are available for training. Compute outcomes and make sure the noise threshold is not filtering every label.",
      action: "Compute outcomes, then re-check stats."
    };
  }
  if (issue.includes("class")) {
    return {
      title: "Class balance warning",
      detail: "One outcome side is underrepresented. The model may overfit to the dominant class until more opposite-side labels arrive.",
      action: "Wait for more diverse outcomes before trusting accuracy."
    };
  }
  return {
    title: issue,
    detail: "This warning comes from the training-data validation endpoint. The model may still run, but treat results as early-stage until the warning clears.",
    action: "Review the data health checks and labeled sample count."
  };
}

function predictionsForSymbols(summary: PredictionSummary | null, symbols: string[]) {
  const symbolSet = new Set(symbols);
  return (summary?.predictions ?? [])
    .map((prediction, index) => ({ prediction, index }))
    .filter(({ prediction }) => symbolSet.has(prediction.symbol));
}

function parseSymbolList(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(/[,\s]+/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter((symbol) => /^[A-Z][A-Z0-9.-]{0,9}$/.test(symbol))
    )
  );
}

function buildSearchResults(
  query: string,
  events: EventRecord[],
  assets: AssetImpactRow[],
  predictions: PredictionSummary | null
): SearchResult[] {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return [];

  function matches(value: string) {
    const normalized = value.toLowerCase();
    return tokens.every((token) => normalized.includes(token));
  }

  const eventResults = events
    .filter((event) => matches([
      event.title,
      event.description,
      event.event_type,
      event.severity ?? "",
      ...event.affected_sectors,
      ...getLinkedAssets(event).map((asset) => asset.symbol),
    ].join(" ")))
    .slice(0, 4)
    .map((event) => ({
      kind: "Event" as const,
      title: event.title,
      detail: `${formatLabel(event.event_type)} | ${formatLabel(event.impact_direction)} | ${(event.mapped_assets ?? []).join(", ") || "No mapped assets"}`,
      route: `#/events/${encodeURIComponent(eventKey(event))}`,
    }));

  const assetResults = assets
    .filter((asset) => matches(`${asset.symbol} ${asset.eventType} ${asset.direction}`))
    .slice(0, 4)
    .map((asset) => ({
      kind: "Asset" as const,
      title: asset.symbol,
      detail: `${formatLabel(asset.eventType)} | ${formatLabel(asset.direction)} macro impact`,
      route: `#/assets/${asset.symbol}`,
    }));

  const forecastResults = (predictions?.predictions ?? [])
    .map((prediction, index) => ({ prediction, index }))
    .filter(({ prediction }) => matches(`${prediction.symbol} ${prediction.title} ${prediction.event_type} ${prediction.impact_direction}`))
    .slice(0, 4)
    .map(({ prediction, index }) => ({
      kind: "Forecast" as const,
      title: `${prediction.symbol} ${formatLabel(prediction.impact_direction)}`,
      detail: `${Math.round(prediction.probability * 100)}% probability | ${formatLabel(prediction.event_type)} | ${prediction.horizon}`,
      route: `#/predictions/${index}`,
    }));

  return [...assetResults, ...forecastResults, ...eventResults].slice(0, 10);
}

function uniqueValues(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function MetricCard({ label, value, icon: Icon, onClick }: { label: string; value: string; icon: typeof Activity; onClick: () => void }) {
  return (
    <button onClick={onClick} className="quant-panel flex items-center gap-3 p-4 text-left transition hover:border-quant-green/50">
      <div className="grid h-10 w-10 place-items-center rounded-lg border border-quant-green/35 bg-quant-green/10 text-quant-green">
        <Icon size={18} />
      </div>
      <div>
        <p className="text-xs font-bold uppercase text-quant-muted">{label}</p>
        <strong className="text-2xl font-black text-quant-text">{value}</strong>
      </div>
    </button>
  );
}

function SidebarMetric({ label, value, tone = "green" }: { label: string; value: string; tone?: "green" | "yellow" | "red" | "muted" }) {
  const toneClass = tone === "red" ? "text-quant-red" : tone === "yellow" ? "text-quant-yellow" : tone === "muted" ? "text-quant-muted" : "text-quant-green";
  return (
    <div className="rounded-md border border-quant-line bg-quant-bg/55 p-2">
      <p className="text-[0.58rem] font-black uppercase text-quant-muted">{label}</p>
      <strong className={`mt-0.5 block text-base font-black ${toneClass}`}>{value}</strong>
    </div>
  );
}

function sidebarNextAction(appData: AppData, apiStatus: "connecting" | "live" | "error") {
  if (apiStatus === "error") {
    return {
      title: "Reconnect backend",
      detail: "Basis cannot reach the API. Restart FastAPI or check the configured base URL.",
      route: "#/data"
    };
  }
  if (appData.modelHealth?.drift_detected) {
    return {
      title: "Review drift",
      detail: "Accuracy or confidence distribution moved away from the trained baseline.",
      route: "#/ml"
    };
  }
  if ((appData.backtest?.flat_signals ?? 0) > (appData.backtest?.actionable_signals ?? 0)) {
    return {
      title: "Wait for outcomes",
      detail: "Many signals are still flat or pending, so portfolio accuracy is early.",
      route: "#/portfolio"
    };
  }
  return {
    title: "Monitor live feed",
    detail: "Pipeline is online. Watch new events, predictions, and market regime shifts.",
    route: "#/events"
  };
}

function regimeName(encoded: number) {
  if (encoded === 0) return "Risk on";
  if (encoded === 2) return "Risk off";
  return "Neutral";
}

function formatSignedPercent(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "--";
  const pct = value * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function DetailStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
      <p className="text-xs font-bold uppercase text-quant-muted">{label}</p>
      <strong className="mt-1 block text-lg font-black text-quant-text">{value}</strong>
    </div>
  );
}

function MiniReturnPanel({ portfolio }: { portfolio: PortfolioSimulation | null }) {
  if (!portfolio || portfolio.signals === 0) {
    return <p className="text-sm text-quant-muted">Run the backtest to generate a portfolio curve.</p>;
  }
  if ((portfolio.actionable_signals ?? portfolio.signals) === 0) {
    return <p className="text-sm text-quant-muted">Signals are tracking, but the portfolio curve starts after at least one signal clears the flat band.</p>;
  }
  return (
    <div>
      <div className="mb-3 grid grid-cols-3 gap-2">
        <DetailStat label="Return" value={`${portfolio.total_return_pct >= 0 ? "+" : ""}${portfolio.total_return_pct}%`} />
        <DetailStat label="SPY" value={`${portfolio.benchmark_return_pct >= 0 ? "+" : ""}${portfolio.benchmark_return_pct}%`} />
        <DetailStat label="Excess" value={`${portfolio.excess_return_pct >= 0 ? "+" : ""}${portfolio.excess_return_pct}%`} />
      </div>
      <SimpleLineChart values={portfolio.points.map((point) => point.return_pct)} benchmark={portfolio.points.map((point) => point.benchmark_return_pct)} />
      <p className="mt-3 text-sm leading-6 text-quant-muted">
        Green is the signal strategy. Gray is SPY over the same signal sequence. This is early until more actionable labels arrive.
      </p>
    </div>
  );
}

function HealthSummary({ health }: { health: ModelHealth | null }) {
  if (!health) return <p className="text-sm text-quant-muted">Model health unavailable.</p>;
  return (
    <div className="grid gap-3">
      <div className={`rounded-lg border p-3 ${health.drift_detected ? "border-quant-red/40 bg-quant-red/10" : "border-quant-green/35 bg-quant-green/8"}`}>
        <p className={`font-black uppercase ${health.drift_detected ? "text-quant-red" : "text-quant-green"}`}>{health.status.replace("_", " ")}</p>
        <p className="mt-1 text-sm text-quant-muted">{health.dataset_size_at_training ?? 0} samples at training</p>
      </div>
      <FeatureBar label="Rolling accuracy" value={health.rolling_accuracy?.accuracy ?? 0} />
      <FeatureBar label="Confidence PSI" value={health.confidence_drift?.psi ?? 0} />
    </div>
  );
}

function FeatureBar({ label, value }: { label: string; value: number }) {
  const width = `${Math.min(100, Math.abs(value) * 100)}%`;
  const explanation = featureExplanation(label);
  return (
    <div title={explanation}>
      <div className="mb-1 flex items-center justify-between gap-2 text-xs">
        <span className="truncate font-bold capitalize text-quant-text">{formatLabel(label.replace(/^(event_|market_|derived_)/, ""))}</span>
        <span className={`shrink-0 font-black ${value >= 0 ? "text-quant-green" : "text-quant-red"}`}>{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-quant-bg">
        <div className={`h-full rounded-full ${value >= 0 ? "bg-quant-green" : "bg-quant-red"}`} style={{ width }} />
      </div>
    </div>
  );
}

function featureExplanation(label: string): string {
  const normalized = label.toLowerCase();
  if (normalized.includes("psi")) return "Population Stability Index: compares recent prediction confidence distribution with training-time confidence.";
  if (normalized.includes("accuracy")) return "Recent realized correctness over labeled signals.";
  if (normalized.includes("severity")) return "How strongly the event was classified as market-moving.";
  if (normalized.includes("sentiment")) return "Direction and tone extracted from the source event.";
  if (normalized.includes("volatility") || normalized.includes("vix")) return "Market volatility context at prediction time.";
  if (normalized.includes("asset")) return "Encoded asset or asset-class identity used by the model.";
  if (normalized.includes("price")) return "Market price context when the signal was generated.";
  if (normalized.includes("correct") || normalized.includes("incorrect") || normalized.includes("positive") || normalized.includes("negative")) return "Share of labeled samples in this outcome class.";
  return "Model feature contribution or dataset metric. Higher magnitude means this item mattered more in this view.";
}

function PriceLineChart({ values }: { values: number[] }) {
  const clean = values.length > 1 ? values : [0, 0];
  const W = 520;
  const H = 180;
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const span = Math.max(max - min, 0.01);
  const padX = 42;
  const padY = 20;
  const path = clean.map((value, index) => {
    const x = padX + (index / Math.max(clean.length - 1, 1)) * (W - padX - 16);
    const y = H - padY - ((value - min) / span) * (H - padY * 2);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
  const end = clean[clean.length - 1] ?? 0;
  return (
    <div className="rounded-lg border border-quant-line bg-quant-bg/70 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs font-bold text-quant-muted">
        <span>Recent price history</span>
        <span className="text-quant-green">Latest ${end.toFixed(2)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-44 w-full" role="img" aria-label="Recent asset price history">
        {[0.25, 0.5, 0.75].map((tick) => {
          const y = padY + tick * (H - padY * 2);
          return <line key={tick} x1={padX} x2={W - 16} y1={y} y2={y} className="stroke-quant-line/70" strokeDasharray="3 5" />;
        })}
        <text x="4" y={padY + 4} className="fill-quant-muted text-[10px]">${max.toFixed(2)}</text>
        <text x="4" y={H - padY + 4} className="fill-quant-muted text-[10px]">${min.toFixed(2)}</text>
        <path d={path} fill="none" className="stroke-quant-green" strokeWidth="3" />
      </svg>
    </div>
  );
}

function SimpleLineChart({ values, benchmark }: { values: number[]; benchmark?: number[] }) {
  const clean = values.length > 1 ? values : [0, 0];
  const bench = benchmark && benchmark.length > 1 ? benchmark : null;
  const W = 1000;
  const H = 240;
  const all = bench ? [...clean, ...bench] : clean;
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 0);
  const span = Math.max(max - min, 0.01);
  const padX = 34;
  const padY = 20;
  function pathFor(series: number[]) {
    return series.map((value, index) => {
      const x = padX + (index / Math.max(series.length - 1, 1)) * (W - padX - 16);
      const y = H - padY - ((value - min) / span) * (H - padY * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(" ");
  }
  const zeroY = H - padY - ((0 - min) / span) * (H - padY * 2);
  const yPct = (value: number) => `${Math.max(0, Math.min(100, (value / H) * 100))}%`;
  const end = clean[clean.length - 1] ?? 0;
  const benchEnd = bench ? bench[bench.length - 1] : null;
  return (
    <div className="rounded-lg border border-quant-line bg-quant-bg/70 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs font-bold text-quant-muted">
        <span>Return over signal sequence</span>
        <span className="text-quant-green">Strategy {end >= 0 ? "+" : ""}{end.toFixed(3)}%</span>
        {benchEnd !== null && <span>SPY {benchEnd >= 0 ? "+" : ""}{benchEnd.toFixed(3)}%</span>}
      </div>
      <div className="relative h-56 overflow-hidden rounded-md border border-quant-line bg-quant-bg/55">
        <span className="absolute left-3 top-4 text-[0.65rem] font-bold text-quant-muted">{max.toFixed(2)}%</span>
        <span className="absolute left-3 text-[0.65rem] font-bold text-quant-muted" style={{ top: yPct(zeroY) }}>0%</span>
        <span className="absolute bottom-4 left-3 text-[0.65rem] font-bold text-quant-muted">{min.toFixed(2)}%</span>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" role="img" aria-label="Strategy return compared with SPY benchmark">
          {[0.25, 0.5, 0.75].map((tick) => {
            const y = padY + tick * (H - padY * 2);
            return <line key={tick} x1={padX} x2={W - 16} y1={y} y2={y} className="stroke-quant-line/70" strokeDasharray="3 5" vectorEffect="non-scaling-stroke" />;
          })}
          <line x1={padX} x2={W - 16} y1={zeroY} y2={zeroY} className="stroke-quant-muted/60" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
          {bench && <path d={pathFor(bench)} fill="none" className="stroke-quant-muted" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
          <path d={pathFor(clean)} fill="none" className="stroke-quant-green" strokeWidth="3" vectorEffect="non-scaling-stroke" />
          <circle cx={W - 16} cy={H - padY - ((end - min) / span) * (H - padY * 2)} r="3" className="fill-quant-green" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <div className="flex items-center gap-4 text-[0.65rem] font-bold text-quant-muted">
        <span className="inline-flex items-center gap-1"><span className="h-1.5 w-5 rounded-full bg-quant-green" /> Strategy</span>
        <span className="inline-flex items-center gap-1"><span className="h-1.5 w-5 rounded-full bg-quant-muted" /> SPY benchmark</span>
      </div>
    </div>
  );
}

function EmptyState({ title, action, onClick }: { title: string; action: string; onClick: () => void }) {
  return (
    <section className="quant-panel grid min-h-[280px] place-items-center p-6 text-center">
      <div>
        <Sparkles className="mx-auto mb-3 text-quant-muted" size={28} />
        <h3 className="mb-3 text-xl font-black text-quant-text">{title}</h3>
        <button onClick={onClick} className="quant-button">{action}</button>
      </div>
    </section>
  );
}

function pageTitle(route: Route): string {
  if (route.name === "event") return "Event Detail";
  if (route.name === "asset") return route.symbol;
  if (route.name === "prediction") return "Signal Detail";
  if (route.name === "events") return "Market Feed";
  if (route.name === "assets") return "Assets";
  if (route.name === "predictions") return "Signals";
  if (route.name === "portfolio") return "Holdings";
  if (route.name === "ml") return "Diagnostics";
  if (route.name === "regime") return "Market Regime";
  if (route.name === "training") return "Training Run";
  if (route.name === "signal") return "Signal Detail";
  if (route.name === "breadth") return "Breadth";
  if (route.name === "alerts") return "Alerts";
  if (route.name === "accuracy") return "Accuracy";
  if (route.name === "data") return "System Health";
  return "Command";
}

function pageEyebrow(route: Route): string {
  if (route.name === "overview") return "Basis";
  if (route.name === "event" || route.name === "asset" || route.name === "prediction") return "Drilldown";
  if (route.name === "data") return "Operations";
  return "Workspace";
}

export default App;
