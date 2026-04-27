import type { BacktestSummary, EventRecord, MarketHistory, PredictionSummary, PriceQuote, SignalAccuracy, WatchlistImpact } from "../types";

const configuredApiUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;

const API_BASE_URLS = configuredApiUrl
  ? Array.from(
      new Set([
        configuredApiUrl,
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001"
      ])
    )
  : ["http://127.0.0.1:8000", "http://127.0.0.1:8001"];

export async function fetchCombinedEvents(): Promise<{
  events: EventRecord[];
  baseUrl: string;
}> {
  let lastError: unknown;

  for (const baseUrl of API_BASE_URLS) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(`${baseUrl}/combined`, {
        cache: "no-store",
        signal: controller.signal
      });
      if (!response.ok) {
        throw new Error(`/combined returned ${response.status}`);
      }

      return {
        events: (await response.json()) as EventRecord[],
        baseUrl
      };
    } catch (error) {
      lastError = error;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  throw lastError;
}

export async function fetchMarketPrice(symbol: string): Promise<PriceQuote> {
  const { payload } = await requestJson<{
    symbol: string;
    price: number;
    history: number[];
  }>(`/price/${symbol}`);

  return {
    symbol: payload.symbol,
    price: payload.price,
    history: payload.history,
    previousPrice: payload.history.length > 1 ? payload.history[payload.history.length - 2] : undefined
  };
}

export async function fetchSignalAccuracy(): Promise<SignalAccuracy> {
  const { payload } = await requestJson<SignalAccuracy>("/signals/accuracy");
  return payload;
}

export async function fetchMarketHistory(symbol: string, range: string): Promise<MarketHistory> {
  const { payload } = await requestJson<MarketHistory>(
    `/market/history/${encodeURIComponent(symbol)}?range=${encodeURIComponent(range.toLowerCase())}`
  );
  return payload;
}

export async function fetchWatchlistImpact(symbols: string[]): Promise<WatchlistImpact> {
  const { payload } = await requestJson<WatchlistImpact>("/watchlist/impact", {
    method: "POST",
    body: JSON.stringify({ symbols }),
    headers: { "Content-Type": "application/json" }
  });

  return payload;
}

export async function fetchBacktestSummary(): Promise<BacktestSummary> {
  const { payload } = await requestJson<BacktestSummary>("/backtest/summary");
  return payload;
}

export async function runBacktest(limit = 100): Promise<BacktestSummary> {
  const { payload } = await requestJson<BacktestSummary>(`/backtest/run?limit=${limit}`, {
    method: "POST"
  });
  return payload;
}

export async function fetchPredictions(limit = 50): Promise<PredictionSummary> {
  const { payload } = await requestJson<PredictionSummary>(`/predictions?limit=${limit}`);
  return payload;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit
): Promise<{ payload: T; baseUrl: string }> {
  let lastError: unknown;

  for (const baseUrl of API_BASE_URLS) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(`${baseUrl}${path}`, {
        cache: "no-store",
        signal: controller.signal,
        ...init
      });
      if (!response.ok) {
        throw new Error(`${path} returned ${response.status}`);
      }

      return {
        payload: (await response.json()) as T,
        baseUrl
      };
    } catch (error) {
      lastError = error;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  throw lastError;
}
