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
  fetchSignalAccuracy,
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
  TrainingRun
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

const navItems = [
  { route: "#/", label: "Overview", icon: Activity },
  { route: "#/events", label: "Events", icon: FileText },
  { route: "#/assets", label: "Assets", icon: BriefcaseBusiness },
  { route: "#/predictions", label: "Predictions", icon: Target },
  { route: "#/portfolio", label: "Portfolio", icon: LineChart },
  { route: "#/ml", label: "ML Lab", icon: BrainCircuit },
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

function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute());
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [quotes, setQuotes] = useState<Record<string, PriceQuote>>({});
  const [query, setQuery] = useState("");
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

  useEffect(() => {
    if (assetRows.length === 0) return;
    let isMounted = true;
    const symbols = assetRows.slice(0, 16).map((row) => row.symbol);

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
  }, [assetRows]);

  async function loadAll() {
    setIsLoading(true);
    setApiStatus("connecting");
    try {
      const combined = await fetchCombinedEvents();
      setEvents(combined.events);
      setApiBaseUrl(combined.baseUrl.replace("http://", ""));
      setApiStatus("live");
      setErrorMessage("");
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));

      const [predictions, portfolio, backtest, modelHealth, marketRegime, signalAccuracy, trainingHistory, trainingStats, validation] = await Promise.allSettled([
        fetchPredictions(),
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
        predictions: predictions.status === "fulfilled" ? predictions.value : null,
        portfolio: portfolio.status === "fulfilled" ? portfolio.value : null,
        backtest: backtest.status === "fulfilled" ? backtest.value : null,
        modelHealth: modelHealth.status === "fulfilled" ? modelHealth.value : null,
        marketRegime: marketRegime.status === "fulfilled" ? marketRegime.value : null,
        signalAccuracy: signalAccuracy.status === "fulfilled" ? signalAccuracy.value : null,
        trainingHistory: trainingHistory.status === "fulfilled" ? trainingHistory.value : [],
        trainingStats: trainingStats.status === "fulfilled" ? trainingStats.value : null,
        validation: validation.status === "fulfilled" ? validation.value : null,
      });
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
  const sampleCount = appData.trainingStats?.num_samples ?? 0;
  const sampleProgress = Math.min(100, (sampleCount / 300) * 100);
  const regimeLabel = appData.marketRegime ? regimeName(appData.marketRegime.market_regime_encoded) : "Regime pending";
  const sidebarAction = sidebarNextAction(appData, apiStatus);

  return (
    <main className="min-h-screen bg-quant-bg text-quant-text">
      <div className="pointer-events-none fixed inset-0 bg-[linear-gradient(rgba(139,148,158,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(139,148,158,0.04)_1px,transparent_1px)] bg-[size:48px_48px]" />
      <div className="relative z-10 flex min-h-screen">
        <aside className="scrollbar-quant sticky top-0 hidden h-screen w-64 shrink-0 overflow-y-auto border-r border-quant-line bg-quant-bg/92 px-4 py-5 backdrop-blur-xl lg:block">
          <button onClick={() => go("#/")} className="mb-7 flex w-full items-center gap-3 text-left">
            <div className="grid h-10 w-10 place-items-center rounded-lg border border-quant-green/40 bg-quant-green/10 text-quant-green">
              <BrainCircuit size={21} />
            </div>
            <div>
              <h1 className="text-2xl font-black leading-tight text-quant-text">Basis</h1>
              <p className="mt-0.5 text-[0.68rem] font-bold uppercase tracking-[0.18em] text-quant-muted">Market intelligence</p>
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
                      ? "bg-quant-green/12 text-quant-green"
                      : "text-quant-muted hover:bg-quant-panel2 hover:text-quant-text"
                  }`}
                >
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="mt-6 rounded-lg border border-quant-line bg-quant-panel2 p-3">
            <p className="quant-eyebrow mb-2">System</p>
            <div className={`mb-2 flex items-center gap-2 text-sm font-black ${apiStatus === "live" ? "text-quant-green" : apiStatus === "error" ? "text-quant-red" : "text-quant-yellow"}`}>
              <Waves size={15} />
              {apiStatus === "live" ? "Live API" : apiStatus === "error" ? "API issue" : "Connecting"}
            </div>
            <p className="truncate text-xs text-quant-muted">{apiBaseUrl}</p>
            <p className="mt-1 text-xs text-quant-muted">Updated {lastUpdated}</p>
          </div>

          <div className="mt-4 grid gap-3">
            <div className="rounded-lg border border-quant-line bg-quant-panel2 p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="quant-eyebrow">Pulse</p>
                <span className={`text-[0.65rem] font-black uppercase ${apiStatus === "live" ? "text-quant-green" : "text-quant-yellow"}`}>
                  {apiStatus === "live" ? "Online" : "Standby"}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <SidebarMetric label="Events" value={events.length.toString()} />
                <SidebarMetric label="High" value={highImpactCount.toString()} tone={highImpactCount > 0 ? "red" : "muted"} />
                <SidebarMetric label="Assets" value={linkedAssetCount.toString()} />
                <SidebarMetric label="Alerts" value={alerts.length.toString()} tone={alerts.length > 0 ? "yellow" : "muted"} />
              </div>
            </div>

            <button
              type="button"
              onClick={() => go("#/ml")}
              className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50"
            >
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="quant-eyebrow">Model Core</p>
                <span className={appData.modelHealth?.drift_detected ? "text-[0.65rem] font-black uppercase text-quant-red" : "text-[0.65rem] font-black uppercase text-quant-green"}>
                  {appData.modelHealth?.drift_detected ? "Drift" : "Stable"}
                </span>
              </div>
              <div className="mb-2 flex items-end justify-between">
                <strong className="text-lg font-black text-quant-text">{sampleCount}</strong>
                <span className="text-xs font-bold text-quant-muted">/ 300 samples</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-quant-bg">
                <div className="h-full rounded-full bg-gradient-to-r from-quant-green to-quant-blue" style={{ width: `${sampleProgress}%` }} />
              </div>
              <p className="mt-2 text-xs leading-5 text-quant-muted">
                {sampleCount < 300 ? "Learning window active" : "Credibility target reached"}
              </p>
            </button>

            <button
              type="button"
              onClick={() => go("#/regime")}
              className="rounded-lg border border-quant-line bg-quant-panel2 p-3 text-left transition hover:border-quant-green/50"
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

            <button
              type="button"
              onClick={() => go(sidebarAction.route)}
              className="rounded-lg border border-quant-green/30 bg-quant-green/8 p-3 text-left transition hover:border-quant-green/60"
            >
              <p className="quant-eyebrow mb-2">Next Action</p>
              <p className="text-sm font-black text-quant-green">{sidebarAction.title}</p>
              <p className="mt-1 text-xs leading-5 text-quant-muted">{sidebarAction.detail}</p>
            </button>
          </div>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 border-b border-quant-line bg-quant-bg/88 px-4 py-3 backdrop-blur-xl lg:px-7">
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
                    placeholder="Search events, assets, sectors"
                    className="quant-input h-9 pl-9"
                  />
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
                placeholder="Search events, assets, sectors"
                className="quant-input h-9 pl-9"
              />
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
  if (route.name === "asset") return <AssetDetailPage symbol={route.symbol} rows={assetRows} quotes={quotes} events={allEvents} predictions={appData.predictions} backtest={appData.backtest} />;
  if (route.name === "predictions") return <PredictionsPage summary={appData.predictions} />;
  if (route.name === "prediction") return <PredictionDetailPage summary={appData.predictions} index={route.index} marketRegime={appData.marketRegime} allPredictions={appData.predictions} />;
  if (route.name === "portfolio") return <PortfolioPage appData={appData} />;
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
  const earlyModel = (appData.trainingStats?.num_samples ?? 0) < 300;
  return (
    <div className="grid gap-5">
      <div className="grid gap-3 md:grid-cols-4">
        <MetricCard label="Events" value={events.length.toString()} icon={FileText} onClick={() => go("#/events")} />
        <MetricCard label="High impact" value={highImpact.toString()} icon={Gauge} onClick={() => go("#/events")} />
        <MetricCard label="Assets" value={assetRows.length.toString()} icon={BriefcaseBusiness} onClick={() => go("#/assets")} />
        <MetricCard label="Labeled samples" value={appData.trainingStats?.num_samples.toString() ?? "--"} icon={Database} onClick={() => go("#/data")} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr_0.8fr]">
        <section className="quant-panel p-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Priority Feed</p>
              <h3 className="text-lg font-black text-quant-text">Latest macro risks</h3>
            </div>
            <button onClick={() => go("#/events")} className="text-xs font-black text-quant-green">Open events</button>
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
              <p className="quant-eyebrow">Portfolio</p>
              <h3 className="text-lg font-black text-quant-text">Signal curve</h3>
            </div>
            <button onClick={() => go("#/portfolio")} className="text-xs font-black text-quant-green">Open portfolio</button>
          </div>
          <MiniReturnPanel portfolio={appData.portfolio} />
        </section>
        <section className="quant-panel p-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Market regime</p>
              <h3 className="text-lg font-black text-quant-text">Current context</h3>
            </div>
            <button onClick={() => go("#/regime")} className="text-xs font-black text-quant-green">Open regime</button>
          </div>
          <RegimeSummary regime={appData.marketRegime} />
          <p className="mt-3 text-sm leading-6 text-quant-muted">Regime data gives each forecast context for whether the market is risk-on, neutral, or risk-off.</p>
        </section>
      </div>

      <div className="grid gap-5 xl:grid-cols-3">
        <section className="quant-panel p-4 xl:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="quant-eyebrow">Situations</p>
              <h3 className="text-lg font-black text-quant-text">Event clusters</h3>
            </div>
            <button onClick={() => go("#/events")} className="text-xs font-black text-quant-green">Investigate</button>
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
              {earlyModel && <EarlyBadge />}
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
}: {
  symbol: string;
  rows: AssetImpactRow[];
  quotes: Record<string, PriceQuote>;
  events: EventRecord[];
  predictions: PredictionSummary | null;
  backtest: BacktestSummary | null;
}) {
  const row = rows.find((item) => item.symbol === symbol);
  const quote = quotes[symbol];
  const linkedEvents = events.filter((event) => getLinkedAssets(event).some((asset) => asset.symbol === symbol));
  const linkedPredictions = predictionsForSymbols(predictions, [symbol]);
  const signalHistory = (backtest?.recent ?? backtest?.top_signals ?? []).filter((signal) => signal.symbol === symbol);
  const accuracy = backtest?.by_symbol?.[symbol]?.accuracy_pct;
  if (!row) return <EmptyState title={`No asset data for ${symbol}`} action="Back to assets" onClick={() => go("#/assets")} />;
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
      <section className="quant-panel p-5">
        <button onClick={() => go("#/assets")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> Assets</button>
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <p className="quant-eyebrow">Asset Detail</p>
            <h2 className="text-4xl font-black text-quant-text">{symbol}</h2>
            <p className="mt-1 text-sm text-quant-muted">{formatLabel(row.eventType)} | {row.confidence} confidence</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-quant-muted">Latest</p>
            <strong className="text-3xl font-black text-quant-text">{typeof quote?.price === "number" ? `$${quote.price.toFixed(2)}` : "Pending"}</strong>
          </div>
        </div>
        <PriceLineChart values={quote?.history ?? []} />
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <DetailStat label="Impact" value={row.direction} />
          <DetailStat label="Signal source" value={formatLabel(row.eventType)} />
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
  if (!summary) return <PredictionsPanel />;
  const assets = uniqueValues(summary.predictions.map((prediction) => prediction.symbol));
  const horizons = uniqueValues(summary.predictions.map((prediction) => prediction.horizon));
  const eventTypes = uniqueValues(summary.predictions.map((prediction) => prediction.event_type));
  const visible = summary.predictions.filter((prediction) =>
    (horizonFilter === "all" || prediction.horizon === horizonFilter) &&
    (assetFilter === "all" || prediction.symbol === assetFilter) &&
    (typeFilter === "all" || prediction.event_type === typeFilter)
  );
  return (
    <section className="quant-panel p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="quant-eyebrow">Predictions</p>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-xl font-black text-quant-text">Ranked model forecasts</h3>
            <EarlyBadge />
          </div>
          <p className="mt-1 text-sm text-quant-muted">Inspect these as model case files while the labeled dataset matures.</p>
        </div>
        <span className="quant-tag">{summary.model_version}</span>
      </div>
      <div className="mb-4 grid gap-2 md:grid-cols-3">
        <SelectControl label="Horizon" value={horizonFilter} onChange={setHorizonFilter} options={["all", ...horizons]} />
        <SelectControl label="Asset" value={assetFilter} onChange={setAssetFilter} options={["all", ...assets]} />
        <SelectControl label="Event type" value={typeFilter} onChange={setTypeFilter} options={["all", ...eventTypes]} />
      </div>
      <div className="grid gap-3">
        {visible.length === 0 ? (
          <InlineEmpty title="No predictions yet" description="Generate events and predictions, then return to this page." />
        ) : visible.map((prediction) => {
          const index = summary.predictions.indexOf(prediction);
          return (
          <button key={`${prediction.symbol}-${index}`} onClick={() => go(`#/predictions/${index}`)} className="grid gap-3 rounded-lg border border-quant-line bg-quant-panel2 p-4 text-left transition hover:border-quant-green/50 md:grid-cols-[120px_1fr_160px]">
            <div>
              <strong className="text-2xl font-black text-quant-text">{prediction.symbol}</strong>
              <p className={`text-sm font-black capitalize ${directionColor(prediction.impact_direction)}`}>{directionArrow(prediction.impact_direction)} {prediction.impact_direction}</p>
            </div>
            <div>
              <h4 className="mb-1 line-clamp-1 font-black text-quant-text">{prediction.title}</h4>
              <p className="text-sm text-quant-muted">{formatLabel(prediction.event_type)} | {prediction.horizon}</p>
            </div>
            <div className="text-right">
              <strong className="text-xl font-black text-quant-text">{Math.round(prediction.probability * 100)}%</strong>
              <p className={directionColor(prediction.impact_direction)}>{prediction.expected_move_pct}% expected</p>
            </div>
          </button>
          );
        })}
      </div>
    </section>
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
        <p className="quant-eyebrow">Prediction Detail</p>
        <h2 className="mb-2 text-4xl font-black text-quant-text">{prediction.symbol}</h2>
        <p className={`mb-5 text-xl font-black capitalize ${directionColor(prediction.impact_direction)}`}>{directionArrow(prediction.impact_direction)} {prediction.impact_direction} | {Math.round(prediction.probability * 100)}%</p>
        <div className="grid gap-3 md:grid-cols-4">
          <DetailStat label="Expected" value={`${prediction.expected_move_pct}%`} />
          <DetailStat label="Range" value={`${prediction.expected_move_low_pct}% to ${prediction.expected_move_high_pct}%`} />
          <DetailStat label="Horizon" value={prediction.horizon} />
          <DetailStat label="Model" value={prediction.model_version} />
        </div>
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
        <p className="quant-eyebrow mb-3">SHAP contributors</p>
        <div className="grid gap-2">
          {(prediction.shap_contributions ?? []).length === 0 ? <p className="text-sm text-quant-muted">No SHAP contributors available yet.</p> : prediction.shap_contributions.map((item) => <FeatureBar key={item.feature} label={item.feature} value={item.shap_value} />)}
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

function PortfolioPage({ appData }: { appData: AppData }) {
  const summary = appData.backtest;
  const portfolio = appData.portfolio;
  const earlyModel = (appData.trainingStats?.num_samples ?? 0) < 300;
  return (
    <div className="grid gap-5">
      {earlyModel && (
        <section className="rounded-lg border border-quant-yellow/35 bg-quant-yellow/8 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase text-quant-yellow">Early model window</p>
              <p className="mt-1 text-sm leading-6 text-quant-muted">Portfolio results are useful for UI validation and directionality, but should not be treated as strategy evidence until the labeled sample set is larger.</p>
            </div>
            <button onClick={() => go("#/data")} className="quant-button">Open data health</button>
          </div>
        </section>
      )}
      <div className="grid gap-3 md:grid-cols-4">
        <MetricCard label="Strategy return" value={portfolio ? `${portfolio.total_return_pct >= 0 ? "+" : ""}${portfolio.total_return_pct}%` : "--"} icon={LineChart} onClick={() => go("#/portfolio")} />
        <MetricCard label="SPY benchmark" value={portfolio ? `${portfolio.benchmark_return_pct >= 0 ? "+" : ""}${portfolio.benchmark_return_pct}%` : "--"} icon={Activity} onClick={() => go("#/portfolio")} />
        <MetricCard label="Win rate" value={portfolio ? `${portfolio.win_rate_pct}%` : "--"} icon={Target} onClick={() => go("#/portfolio")} />
        <MetricCard label="Signals" value={summary?.total_signals.toString() ?? "--"} icon={FileText} onClick={() => go("#/portfolio")} />
      </div>
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
  const sampleProgress = Math.min(100, ((stats?.num_samples ?? 0) / 300) * 100);
  return (
    <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.1fr)_minmax(440px,0.9fr)]">
      <div>
        <section className="mb-4 rounded-lg border border-quant-yellow/35 bg-quant-yellow/8 p-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase text-quant-yellow">Model maturity</p>
              <p className="mt-1 text-sm text-quant-muted">{stats?.num_samples ?? 0}/300 labeled samples toward the credibility target.</p>
            </div>
            <strong className="text-lg font-black text-quant-yellow">{Math.round(sampleProgress)}%</strong>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-quant-bg">
            <div className="h-full rounded-full bg-gradient-to-r from-quant-yellow to-quant-green" style={{ width: `${sampleProgress}%` }} />
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
            <ProcessStep label="Sample target" detail="The model is usable for demo workflows now, but credibility improves materially once the labeled set reaches 300+ samples." />
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
  const sampleProgress = Math.min(100, (usable / 300) * 100);
  return (
    <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_minmax(460px,0.9fr)]">
      <section className="quant-panel p-5">
        <p className="quant-eyebrow">Data Health</p>
        <h3 className="mb-4 text-xl font-black text-quant-text">Pipeline readiness</h3>
        <div className="grid gap-3 md:grid-cols-3">
          <DetailStat label="Events" value={events.length.toString()} />
          <DetailStat label="Samples" value={stats?.num_samples.toString() ?? "--"} />
          <DetailStat label="Features" value={stats?.feature_count.toString() ?? "--"} />
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">Class balance</p>
          <div className="grid gap-2 md:grid-cols-2">
            <FeatureBar label="Positive" value={stats?.class_distribution.positive ?? 0} />
            <FeatureBar label="Negative" value={stats?.class_distribution.negative ?? 0} />
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <div className="mb-2 flex items-center justify-between text-xs font-bold text-quant-muted">
            <span>Sample maturity</span>
            <span>{usable}/300</span>
          </div>
          <div className="mb-4 h-2 overflow-hidden rounded-full bg-quant-bg">
            <div className="h-full rounded-full bg-gradient-to-r from-quant-yellow to-quant-green" style={{ width: `${sampleProgress}%` }} />
          </div>
          <p className="quant-eyebrow mb-2">Readiness checklist</p>
          <div className="grid gap-2">
            <ReadinessRow label="Backend API reachable" ready={true} />
            <ReadinessRow label="Events ingested" ready={events.length > 0} />
            <ReadinessRow label="Training samples available" ready={(stats?.num_samples ?? 0) > 0} />
            <ReadinessRow label="Credibility target 300+ samples" ready={(stats?.num_samples ?? 0) >= 300} />
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
          <DetailStat label="Positive" value={`${stats ? Math.round(stats.class_distribution.positive * 100) : 0}%`} />
          <DetailStat label="Negative" value={`${stats ? Math.round(stats.class_distribution.negative * 100) : 0}%`} />
        </div>
        <div className="grid gap-3">
          {issues.length ? issues.map((issue) => <ValidationIssueCard key={issue} issue={issue} />) : <div className="rounded-lg border border-quant-green/35 bg-quant-green/8 p-3 text-sm font-bold text-quant-green">No validation issues reported.</div>}
        </div>
        <div className="mt-4 rounded-lg border border-quant-line bg-quant-panel2 p-4">
          <p className="quant-eyebrow mb-2">How this gets better</p>
          <div className="grid gap-2">
            <ReadinessRow label="More labels" ready={(stats?.num_samples ?? 0) >= 300} detail={`${stats?.num_samples ?? 0}/300 credibility target`} />
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
  if (!run) return <EmptyState title="Training run not found" action="Back to ML Lab" onClick={() => go("#/ml")} />;
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
      <section className="quant-panel p-5">
        <button onClick={() => go("#/ml")} className="mb-4 inline-flex items-center gap-2 text-sm font-bold text-quant-muted hover:text-quant-text"><ArrowLeft size={14} /> ML Lab</button>
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
          <DetailStat label="Label balance" value={`${run.label_balance.positive_count}+ / ${run.label_balance.negative_count}-`} />
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
  const samples = appData.trainingStats?.num_samples ?? 0;
  const issues = appData.validation?.issues.length ?? 0;
  if (events.length === 0) return { label: "Open events", route: "#/events", detail: "No event feed loaded yet." };
  if (samples < 300) return { label: "Open data health", route: "#/data", detail: `${samples}/300 labeled samples collected.` };
  if (issues > 0) return { label: "Review validation", route: "#/data", detail: `${issues} validation warning${issues === 1 ? "" : "s"} to review.` };
  if (appData.modelHealth?.drift_detected) return { label: "Open ML Lab", route: "#/ml", detail: "Model drift is currently flagged." };
  return { label: "Open portfolio", route: "#/portfolio", detail: "Pipeline is ready for performance review." };
}

function EarlyBadge() {
  return (
    <span className="rounded-md border border-quant-yellow/35 bg-quant-yellow/10 px-2 py-1 text-[0.62rem] font-black uppercase text-quant-yellow">
      Early model
    </span>
  );
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
  if (route.name === "regime") return [{ label: "ML Lab", href: "#/ml" }, { label: "Regime" }];
  if (route.name === "training") return [{ label: "ML Lab", href: "#/ml" }, { label: "Training run" }];
  if (route.name === "signal") return [{ label: "Portfolio", href: "#/portfolio" }, { label: "Signal" }];
  if (route.name === "ml") return [{ label: "ML Lab" }];
  if (route.name === "data") return [{ label: "Data Health" }];
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
  const sampleCount = appData.trainingStats?.num_samples ?? 0;
  if (sampleCount < 300) {
    return {
      title: "Grow label set",
      detail: `${sampleCount}/300 samples. Let market data mature, then compute outcomes and retrain.`,
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
  if (normalized.includes("positive") || normalized.includes("negative")) return "Share of labeled samples in this outcome class.";
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
  if (route.name === "prediction") return "Prediction Detail";
  if (route.name === "events") return "Events";
  if (route.name === "assets") return "Assets";
  if (route.name === "predictions") return "Predictions";
  if (route.name === "portfolio") return "Portfolio";
  if (route.name === "ml") return "ML Lab";
  if (route.name === "regime") return "Market Regime";
  if (route.name === "training") return "Training Run";
  if (route.name === "signal") return "Signal Detail";
  if (route.name === "breadth") return "Breadth";
  if (route.name === "alerts") return "Alerts";
  if (route.name === "accuracy") return "Accuracy";
  if (route.name === "data") return "Data Health";
  return "Overview";
}

function pageEyebrow(route: Route): string {
  if (route.name === "overview") return "Command Center";
  if (route.name === "event" || route.name === "asset" || route.name === "prediction") return "Drilldown";
  if (route.name === "data") return "Operations";
  return "Workspace";
}

export default App;
