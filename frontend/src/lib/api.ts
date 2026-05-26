import type { BacktestSummary, EventRecord, MarketHistory, MarketRegime, ModelEvaluation, ModelHealth, PortfolioSimulation, PredictionSummary, PriceQuote, SignalAccuracy, TrainingDataStats, TrainingDataValidation, TrainingRun, WatchlistImpact } from "../types";

const configuredApiUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;

const normalizedConfiguredApiUrl = configuredApiUrl?.replace(/\/$/, "");
const isProduction = import.meta.env.PROD;

const API_BASE_URLS = normalizedConfiguredApiUrl
  ? isProduction
    ? [normalizedConfiguredApiUrl]
    : Array.from(
        new Set([
          normalizedConfiguredApiUrl,
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

export async function fetchPortfolioSimulation(): Promise<PortfolioSimulation> {
  const { payload } = await requestJson<PortfolioSimulation>("/backtest/portfolio");
  return payload;
}

export async function fetchPredictions(limit = 50): Promise<PredictionSummary> {
  const { payload } = await requestJson<PredictionSummary>(`/predictions?limit=${limit}`);
  return payload;
}

export async function fetchModelHealth(): Promise<ModelHealth> {
  const { payload } = await requestJson<ModelHealth>("/model-health");
  return payload;
}

export async function fetchModelEvaluation(): Promise<ModelEvaluation> {
  const { payload } = await requestJson<ModelEvaluation>("/model-evaluation");
  return payload;
}

export async function fetchTrainingHistory(limit = 20): Promise<TrainingRun[]> {
  const { payload } = await requestJson<TrainingRun[]>(`/training-history?limit=${limit}`);
  return payload;
}

export async function fetchTrainingDataStats(): Promise<TrainingDataStats> {
  const { payload } = await requestJson<TrainingDataStats>("/training-data/stats");
  return payload;
}

export async function fetchTrainingDataValidation(): Promise<TrainingDataValidation> {
  const { payload } = await requestJson<TrainingDataValidation>("/training-data/validation");
  return payload;
}

export async function fetchMarketRegime(): Promise<MarketRegime> {
  const { payload } = await requestJson<MarketRegime>("/market-regime");
  return payload;
}

export async function triggerTrainModel(): Promise<void> {
  await requestJson("/train-model", { method: "POST" }, 120_000);
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 8000,
): Promise<{ payload: T; baseUrl: string }> {
  let lastError: unknown;

  for (const baseUrl of API_BASE_URLS) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

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
