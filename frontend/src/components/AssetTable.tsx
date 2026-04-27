import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, ExternalLink, Maximize2 } from "lucide-react";

import { MiniSparkline } from "./MiniSparkline";
import { directionArrow, directionBorder, directionColor, formatLabel } from "../lib/intelligence";
import { fetchMarketHistory } from "../lib/api";
import type { AssetImpactRow, MarketHistory, PriceQuote } from "../types";
import { ExpandModal } from "./ExpandModal";

type AssetTableProps = {
  rows: AssetImpactRow[];
  quotes: Record<string, PriceQuote>;
};

type AssetSnapshot = {
  row: AssetImpactRow;
  quote?: PriceQuote;
};

function AssetTableContent({
  rows,
  quotes,
  scrollClass,
  onOpenAsset
}: AssetTableProps & {
  scrollClass: string;
  onOpenAsset: (snapshot: AssetSnapshot) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-quant-line">
      <div className="grid shrink-0 grid-cols-[0.75fr_0.85fr_1fr_1.05fr_1fr] bg-quant-bg/80 px-4 py-2.5 text-xs font-bold uppercase text-quant-muted">
        <span>Symbol</span>
        <span>Price</span>
        <span>Impact</span>
        <span>Source Event</span>
        <span className="text-left">Confidence / Chart</span>
      </div>
      {rows.length === 0 ? (
        <div className="p-8 text-sm text-quant-muted">No linked assets returned.</div>
      ) : (
        <div className={`scrollbar-quant overflow-auto ${scrollClass}`}>
          {rows.map((row) => {
            const quote = quotes[row.symbol];
            const price = quote?.price ?? row.price;
            const history = quote?.history ?? [];
            const delta = history.length > 1 ? history[history.length - 1] - history[history.length - 2] : 0;
            const isUp = delta >= 0;
            return (
              <div
                key={row.symbol}
                className={`grid min-h-[62px] grid-cols-[0.75fr_0.85fr_1fr_1.05fr_1fr] items-center border-l-2 border-t border-t-quant-line bg-quant-panel2/80 px-4 text-sm transition hover:bg-quant-panel2 ${directionBorder(row.direction)}`}
              >
                <div>
                  <strong className="block text-sm text-quant-text">{row.symbol}</strong>
                  <span className={`inline-flex items-center gap-1 text-[0.68rem] font-bold ${isUp ? "text-quant-green" : "text-quant-red"}`}>
                    {isUp ? <ArrowUp size={11} /> : <ArrowDown size={11} />}
                    {delta >= 0 ? "+" : ""}{delta.toFixed(2)}
                  </span>
                </div>
                <span className="font-medium text-quant-text">
                  {typeof price === "number" ? `$${price.toFixed(2)}` : "Pending"}
                </span>
                <span className={`font-bold capitalize ${directionColor(row.direction)}`}>
                  {directionArrow(row.direction)} {row.direction}
                </span>
                <span className="capitalize text-quant-muted">{formatLabel(row.eventType)}</span>
                <div className="grid min-w-0 grid-cols-[auto_minmax(92px,1fr)] items-center gap-2">
                  <span className="w-fit rounded-md border border-quant-line px-2 py-1 text-xs font-bold uppercase text-quant-text">
                    {row.confidence}
                  </span>
                  <button
                    type="button"
                    onClick={() => onOpenAsset({ row, quote })}
                    title={`Open ${row.symbol} market chart`}
                    className="group min-w-0 rounded-md border border-transparent px-1 py-0.5 transition hover:border-quant-green/45 hover:bg-quant-green/5"
                  >
                    {quote?.history ? (
                      <MiniSparkline values={quote.history} positive={isUp} />
                    ) : (
                      <span className="flex h-8 items-center justify-center text-[0.68rem] font-bold text-quant-muted">
                        Open chart
                      </span>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function AssetTable({ rows, quotes }: AssetTableProps) {
  const [expanded, setExpanded] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<AssetSnapshot | null>(null);

  return (
    <>
      <section className="quant-panel flex min-h-0 flex-col p-4">
        <div className="mb-3 flex shrink-0 items-start justify-between">
          <div>
            <p className="quant-eyebrow">Asset Impact</p>
            <h2 className="text-xl font-bold text-quant-text">Decision table</h2>
          </div>
          <button
            onClick={() => setExpanded(true)}
            title="Expand"
            className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
          >
            <Maximize2 size={13} />
          </button>
        </div>
        <AssetTableContent
          rows={rows}
          quotes={quotes}
          scrollClass="flex-1"
          onOpenAsset={setSelectedAsset}
        />
      </section>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <section className="quant-panel flex max-h-[85vh] flex-col p-5">
            <div className="mb-3 pr-8">
              <p className="quant-eyebrow">Asset Impact</p>
              <h2 className="text-xl font-bold text-quant-text">Decision table</h2>
            </div>
            <AssetTableContent
              rows={rows}
              quotes={quotes}
              scrollClass=""
              onOpenAsset={setSelectedAsset}
            />
          </section>
        </ExpandModal>
      )}

      {selectedAsset && (
        <ExpandModal onClose={() => setSelectedAsset(null)}>
          <AssetDetail snapshot={selectedAsset} />
        </ExpandModal>
      )}
    </>
  );
}

function AssetDetail({ snapshot }: { snapshot: AssetSnapshot }) {
  const { row, quote } = snapshot;
  const [range, setRange] = useState("1D");
  const [marketHistory, setMarketHistory] = useState<MarketHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const fallbackHistory = quote?.history?.length ? quote.history : typeof row.price === "number" ? [row.price, row.price] : [];

  useEffect(() => {
    let isMounted = true;
    setHistoryLoading(true);
    setHistoryError("");

    fetchMarketHistory(row.symbol, range)
      .then((payload) => {
        if (isMounted) setMarketHistory(payload);
      })
      .catch((error) => {
        if (isMounted) {
          setMarketHistory(null);
          setHistoryError(error instanceof Error ? error.message : "Unable to load market history");
        }
      })
      .finally(() => {
        if (isMounted) setHistoryLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [range, row.symbol]);

  const chartValues = marketHistory?.points.length
    ? marketHistory.points.map((point) => point.close)
    : fallbackHistory;
  const chartVolumes = marketHistory?.points.length
    ? marketHistory.points.map((point) => point.volume)
    : [];
  const chartTimestamps = marketHistory?.points.length
    ? marketHistory.points.map((point) => point.timestamp)
    : [];
  const latestPrice = marketHistory?.close ?? quote?.price ?? row.price;
  const firstPrice = marketHistory?.previous_close ?? chartValues[0];
  const lastPrice = marketHistory?.close ?? chartValues[chartValues.length - 1] ?? latestPrice;
  const change = typeof firstPrice === "number" && typeof lastPrice === "number" ? lastPrice - firstPrice : 0;
  const changePct = firstPrice ? (change / firstPrice) * 100 : 0;
  const high = marketHistory?.high ?? (chartValues.length ? Math.max(...chartValues) : latestPrice);
  const low = marketHistory?.low ?? (chartValues.length ? Math.min(...chartValues) : latestPrice);
  const previousClose = marketHistory?.previous_close ?? firstPrice ?? latestPrice;
  const open = marketHistory?.open ?? chartValues[Math.min(1, chartValues.length - 1)] ?? firstPrice;
  const volume = marketHistory?.volume ?? 0;
  const averageVolume = marketHistory?.average_volume ?? null;
  const positive = change >= 0;

  return (
    <section className="quant-panel p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4 pr-8">
        <div>
          <p className="quant-eyebrow">Asset Detail</p>
          <div className="mt-1 flex items-center gap-3">
            <h2 className="text-3xl font-black text-quant-text">{row.symbol}</h2>
            <span className={`rounded-md border border-quant-line px-2 py-1 text-sm font-black capitalize ${directionColor(row.direction)}`}>
              {directionArrow(row.direction)} {row.direction}
            </span>
          </div>
          <p className="mt-1 text-sm capitalize text-quant-muted">
            Macro source: {formatLabel(row.eventType)} | Confidence: {row.confidence}
          </p>
          <p className="mt-2 text-xs font-semibold text-quant-muted">
            Chart uses backend yfinance history. Open Yahoo Finance for full fundamentals, filings, options, and analyst data.
          </p>
        </div>

        <div className="text-right">
          <p className="text-sm font-semibold text-quant-muted">Latest price</p>
          <strong className="block text-3xl font-black text-quant-text">
            {typeof latestPrice === "number" ? `$${latestPrice.toFixed(2)}` : "Pending"}
          </strong>
          <span className={`inline-flex items-center justify-end gap-1 text-sm font-black ${positive ? "text-quant-green" : "text-quant-red"}`}>
            {positive ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
            {change >= 0 ? "+" : ""}{change.toFixed(2)} ({changePct >= 0 ? "+" : ""}{changePct.toFixed(2)}%)
          </span>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {["1D", "5D", "1M", "6M", "YTD", "1Y"].map((rangeOption) => (
          <button
            key={rangeOption}
            onClick={() => setRange(rangeOption)}
            className={`h-8 rounded-md border px-3 text-xs font-black ${
              range === rangeOption
                ? "border-quant-green/50 bg-quant-green/10 text-quant-green"
                : "border-quant-line bg-quant-panel2 text-quant-muted"
            }`}
            type="button"
          >
            {rangeOption}
          </button>
        ))}
        <a
          href={`https://finance.yahoo.com/quote/${encodeURIComponent(row.symbol)}`}
          target="_blank"
          rel="noreferrer"
          className="ml-auto inline-flex h-8 items-center gap-2 rounded-md border border-quant-line bg-quant-panel2 px-3 text-xs font-black text-quant-text transition hover:border-quant-blue/60"
        >
          Yahoo Finance <ExternalLink size={12} />
        </a>
      </div>

      <div className="mb-4 rounded-lg border border-quant-line bg-quant-bg/70 p-4">
        <LargePriceChart
          values={chartValues}
          volumes={chartVolumes}
          timestamps={chartTimestamps}
          positive={positive}
          symbol={row.symbol}
          previousClose={previousClose}
          range={range}
          loading={historyLoading}
        />
        {historyError && (
          <p className="mt-2 text-xs font-semibold text-quant-yellow">
            Historical endpoint unavailable, showing available quote samples. {historyError}
          </p>
        )}
      </div>

      <div className="grid grid-cols-4 gap-x-6 gap-y-2 rounded-lg border border-quant-line bg-quant-panel2 p-4">
        <QuoteMetric label="Previous Close" value={formatMoney(previousClose)} />
        <QuoteMetric label="Open" value={formatMoney(open)} />
        <QuoteMetric label="Day Range" value={`${formatMoney(low)} - ${formatMoney(high)}`} />
        <QuoteMetric label="Volume" value={volume ? formatCompact(volume) : "Pending"} />
        <QuoteMetric label="Avg. Volume" value={averageVolume ? formatCompact(averageVolume) : "Pending"} />
        <QuoteMetric label="Change" value={`${change >= 0 ? "+" : ""}${change.toFixed(2)}`} tone={positive ? "positive" : "negative"} />
        <QuoteMetric label="Change %" value={`${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%`} tone={positive ? "positive" : "negative"} />
        <QuoteMetric label="Interval" value={marketHistory?.interval ?? "Quote samples"} />
      </div>

      <div className="mt-4 rounded-lg border border-quant-line bg-quant-panel2 p-3">
        <p className="quant-eyebrow mb-1">Macro Readthrough</p>
        <p className="text-sm leading-relaxed text-quant-muted">
          {row.symbol} is currently linked to a {formatLabel(row.eventType)} signal with {row.confidence} model confidence.
          The table impact is classified as {row.direction}, so use this chart to compare price behavior against the macro signal.
        </p>
      </div>
    </section>
  );
}

function LargePriceChart({
  values,
  volumes,
  timestamps,
  positive,
  symbol,
  previousClose,
  range,
  loading
}: {
  values: number[];
  volumes: number[];
  timestamps: string[];
  positive: boolean;
  symbol: string;
  previousClose?: number;
  range: string;
  loading: boolean;
}) {
  const safeValues = values.length > 1 ? values : [1, 1];
  const min = Math.min(...safeValues);
  const max = Math.max(...safeValues);
  const priceRange = max - min || 1;
  const width = 920;
  const height = 360;
  const chartTop = 26;
  const chartBottom = 286;
  const chartLeft = 42;
  const chartRight = width - 58;
  const baseline = typeof previousClose === "number" ? previousClose : safeValues[0];
  const yFor = (value: number) => chartTop + (1 - (value - min) / priceRange) * (chartBottom - chartTop);
  const points = safeValues
    .map((value, index) => {
      const x = chartLeft + (index / (safeValues.length - 1)) * (chartRight - chartLeft);
      const y = yFor(value);
      return `${x},${y}`;
    })
    .join(" ");
  const areaPoints = `${chartLeft},${chartBottom} ${points} ${chartRight},${chartBottom}`;
  const baselineY = yFor(baseline);
  const lastValue = safeValues[safeValues.length - 1];
  const lastY = yFor(lastValue);
  const eventIndex = Math.max(1, Math.floor(safeValues.length * 0.62));
  const eventX = chartLeft + (eventIndex / (safeValues.length - 1)) * (chartRight - chartLeft);
  const eventY = yFor(safeValues[eventIndex]);
  const volumeSeed = volumes.length === safeValues.length
    ? volumes
    : safeValues.map((value, index) => Math.abs(value - safeValues[Math.max(0, index - 1)]) + 0.04);
  const maxVolumeSeed = Math.max(...volumeSeed, 1);
  const labels = buildTimeLabels(timestamps, range);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[360px] w-full" role="img" aria-label={`${symbol} ${range} price chart`}>
      <defs>
        <linearGradient id={`area-${symbol}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={positive ? "#00E676" : "#FF5252"} stopOpacity="0.22" />
          <stop offset="100%" stopColor={positive ? "#00E676" : "#FF5252"} stopOpacity="0.02" />
        </linearGradient>
        <pattern id={`dots-${symbol}`} x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="0.7" fill="rgba(139,148,158,0.22)" />
        </pattern>
      </defs>

      <rect x={chartLeft} y={chartTop} width={chartRight - chartLeft} height={chartBottom - chartTop} fill={`url(#dots-${symbol})`} />

      {[0, 1, 2, 3, 4].map((line) => (
        <line
          key={line}
          x1={chartLeft}
          x2={chartRight}
          y1={chartTop + line * ((chartBottom - chartTop) / 4)}
          y2={chartTop + line * ((chartBottom - chartTop) / 4)}
          stroke="rgba(139,148,158,0.18)"
          strokeWidth="1"
        />
      ))}
      {[0, 1, 2, 3].map((line) => (
        <line
          key={`v-${line}`}
          x1={chartLeft + line * ((chartRight - chartLeft) / 3)}
          x2={chartLeft + line * ((chartRight - chartLeft) / 3)}
          y1={chartTop}
          y2={chartBottom}
          stroke="rgba(139,148,158,0.12)"
          strokeWidth="1"
        />
      ))}

      <polygon points={areaPoints} fill={`url(#area-${symbol})`} />
      <line
        x1={chartLeft}
        x2={chartRight}
        y1={baselineY}
        y2={baselineY}
        stroke="rgba(139,148,158,0.58)"
        strokeDasharray="6 6"
        strokeWidth="1"
      />
      <polyline
        fill="none"
        stroke={positive ? "#00E676" : "#FF5252"}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.2"
        points={points}
      />

      <line x1={eventX} x2={eventX} y1={chartTop} y2={chartBottom} stroke="#2979FF" strokeDasharray="3 5" strokeOpacity="0.7" />
      <circle cx={eventX} cy={eventY} r="5" fill="#2979FF" stroke="#0B0F14" strokeWidth="2" />
      <text x={eventX + 9} y={eventY - 8} fill="#E6EDF3" fontSize="11" fontWeight="800">
        Macro signal
      </text>

      <circle cx={chartRight} cy={lastY} r="4" fill={positive ? "#00E676" : "#FF5252"} />
      <rect x={chartRight + 8} y={lastY - 13} width="64" height="24" rx="4" fill={positive ? "#00E676" : "#FF5252"} />
      <text x={chartRight + 40} y={lastY + 4} textAnchor="middle" fill="#0B0F14" fontSize="12" fontWeight="900">
        {lastValue.toFixed(2)}
      </text>

      {volumeSeed.map((value, index) => {
        const x = chartLeft + (index / Math.max(1, volumeSeed.length - 1)) * (chartRight - chartLeft);
        const barHeight = (value / maxVolumeSeed) * 42;
        return (
          <rect
            key={`${value}-${index}`}
            x={x - 2}
            y={chartBottom + 48 - barHeight}
            width="4"
            height={barHeight}
            fill="rgba(139,148,158,0.35)"
          />
        );
      })}

      <text x={chartLeft} y="18" fill="#8B949E" fontSize="12" fontWeight="700">
        {formatMoney(max)}
      </text>
      <text x={chartLeft} y={chartBottom + 16} fill="#8B949E" fontSize="12" fontWeight="700">
        {formatMoney(min)}
      </text>
      <text x={chartLeft} y={height - 10} fill="#8B949E" fontSize="12" fontWeight="700">Open</text>
      <text x={(chartLeft + chartRight) / 2} y={height - 10} textAnchor="middle" fill="#8B949E" fontSize="12" fontWeight="700">Mid-session</text>
      <text x={chartRight} y={height - 10} textAnchor="end" fill="#8B949E" fontSize="12" fontWeight="700">Latest</text>
      {labels.map((label) => (
        <text
          key={`${label.text}-${label.x}`}
          x={chartLeft + label.x * (chartRight - chartLeft)}
          y={chartBottom + 30}
          textAnchor="middle"
          fill="#8B949E"
          fontSize="11"
          fontWeight="700"
        >
          {label.text}
        </text>
      ))}
      {loading && (
        <rect x={chartLeft} y={chartTop} width={chartRight - chartLeft} height={chartBottom - chartTop} fill="rgba(11,15,20,0.45)" />
      )}
    </svg>
  );
}

function QuoteMetric({ label, value, tone }: { label: string; value: string; tone?: "positive" | "negative" }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-dashed border-quant-line py-1.5">
      <span className="text-xs font-bold text-quant-muted">{label}</span>
      <strong
        className={`text-sm font-black ${
          tone === "positive" ? "text-quant-green" : tone === "negative" ? "text-quant-red" : "text-quant-text"
        }`}
      >
        {value}
      </strong>
    </div>
  );
}

function formatMoney(value?: number) {
  return typeof value === "number" ? `$${value.toFixed(2)}` : "Pending";
}

function formatCompact(value: number) {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

function buildTimeLabels(timestamps: string[], range: string): Array<{ x: number; text: string }> {
  if (timestamps.length < 3) {
    return [
      { x: 0, text: "Open" },
      { x: 0.5, text: "Mid" },
      { x: 1, text: "Latest" }
    ];
  }

  const indexes = [0, Math.floor(timestamps.length / 2), timestamps.length - 1];
  return indexes.map((index) => {
    const date = new Date(timestamps[index]);
    const text = range === "1D" || range === "5D"
      ? date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : date.toLocaleDateString([], { month: "short", day: "numeric" });

    return {
      x: index / (timestamps.length - 1),
      text
    };
  });
}
