import type {
  AssetImpactRow,
  AssetPrice,
  AlertRule,
  ConfidenceLevel,
  EventCluster,
  EventRecord,
  ImpactDirection,
  Severity
} from "../types";

export const sectors = [
  "energy",
  "technology",
  "financials",
  "airlines",
  "commodities",
  "broad_market"
];

export function eventKey(event: EventRecord): string {
  return event.article_hash ?? `${event.id ?? event.title}-${event.event_type}`;
}

export function formatLabel(value: string): string {
  return value.replaceAll("_", " ");
}

export function directionArrow(direction: ImpactDirection): string {
  if (direction === "positive") return "↑";
  if (direction === "negative") return "↓";
  return "→";
}

export function directionColor(direction: ImpactDirection): string {
  if (direction === "positive") return "text-quant-green";
  if (direction === "negative") return "text-quant-red";
  return "text-quant-yellow";
}

export function directionBorder(direction: ImpactDirection): string {
  if (direction === "positive") return "border-l-quant-green/80";
  if (direction === "negative") return "border-l-quant-red/80";
  return "border-l-quant-yellow/80";
}

export function confidenceLevel(event: EventRecord): ConfidenceLevel {
  if (typeof event.confidence === "number") {
    if (event.confidence >= 0.75) return "high";
    if (event.confidence >= 0.45) return "medium";
    return "low";
  }

  const reasoningLength = event.reasoning?.trim().length ?? 0;
  if (reasoningLength > 140) return "high";
  if (reasoningLength > 60) return "medium";
  return "low";
}

export function confidenceValue(event: EventRecord): number {
  if (typeof event.confidence === "number") {
    return Math.max(0, Math.min(1, event.confidence));
  }

  const level = confidenceLevel(event);
  if (level === "high") return 0.82;
  if (level === "medium") return 0.58;
  return 0.32;
}

export function severityValue(severity?: Severity): number {
  if (severity === "critical") return 4;
  if (severity === "high") return 3;
  if (severity === "medium") return 2;
  return 1;
}

export function getLinkedAssets(event: EventRecord): AssetPrice[] {
  if (event.recommended_assets?.length) return event.recommended_assets;
  if (event.assets?.length) return event.assets;
  return (event.mapped_assets ?? []).map((symbol) => ({ symbol }));
}

export function buildAssetRows(events: EventRecord[]): AssetImpactRow[] {
  const rows = new Map<string, AssetImpactRow>();

  events.forEach((event) => {
    getLinkedAssets(event).forEach((asset) => {
      const existing = rows.get(asset.symbol);
      const nextSeverity = event.severity ?? "low";

      if (existing) {
        if (severityValue(nextSeverity) > severityValue(existing.severity)) {
          existing.direction = event.impact_direction;
          existing.eventType = event.event_type;
          existing.confidence = confidenceLevel(event);
          existing.severity = nextSeverity;
        }
        existing.price = existing.price ?? asset.price;
        return;
      }

      rows.set(asset.symbol, {
        symbol: asset.symbol,
        price: asset.price,
        direction: event.impact_direction,
        eventType: event.event_type,
        confidence: confidenceLevel(event),
        severity: nextSeverity
      });
    });
  });

  return Array.from(rows.values()).sort(
    (a, b) => severityValue(b.severity) - severityValue(a.severity)
  );
}

export function buildSuggestedActions(event: EventRecord): string[] {
  const actions: string[] = [];
  const sectorSet = new Set(event.affected_sectors);

  if (sectorSet.has("energy") && event.impact_direction === "positive") {
    actions.push("Increase energy exposure");
  }

  if (sectorSet.has("airlines") && event.impact_direction === "negative") {
    actions.push("Reduce airline exposure");
  }

  if (sectorSet.has("technology") && event.impact_direction === "negative") {
    actions.push("Review growth and technology risk");
  }

  if (sectorSet.has("financials") && event.event_type === "interest_rate_change") {
    actions.push("Monitor bank and rate-sensitive exposure");
  }

  if (sectorSet.has("broad_market") && event.impact_direction === "negative") {
    actions.push("Consider broad market hedge");
  }

  if (actions.length === 0) {
    actions.push("Monitor linked assets before changing exposure");
  }

  return actions;
}

export function impactSummary(event: EventRecord): string {
  return event.affected_sectors
    .slice(0, 3)
    .map((sector) => `${formatLabel(sector)} ${directionArrow(event.impact_direction)}`)
    .join(" | ");
}

export function clusterEvents(events: EventRecord[]): EventCluster[] {
  const clusters = new Map<string, EventRecord[]>();

  events.forEach((event) => {
    const key = `${event.event_type}:${dominantSector(event)}:${dominantKeyword(event)}`;
    clusters.set(key, [...(clusters.get(key) ?? []), event]);
  });

  return Array.from(clusters.entries())
    .map(([id, groupedEvents]) => {
      const primary = groupedEvents[0];
      const direction = netDirection(groupedEvents.map((event) => event.impact_direction));
      const severity = groupedEvents.reduce<Severity>(
        (highest, event) =>
          severityValue(event.severity) > severityValue(highest) ? event.severity ?? "low" : highest,
        "low"
      );
      const sectors = Array.from(new Set(groupedEvents.flatMap((event) => event.affected_sectors)));
      const assets = Array.from(
        new Set(groupedEvents.flatMap((event) => getLinkedAssets(event).map((asset) => asset.symbol)))
      );
      const confidence =
        groupedEvents.reduce((total, event) => total + confidenceValue(event), 0) / groupedEvents.length;

      return {
        id,
        title: clusterTitle(primary, sectors),
        eventType: primary.event_type,
        events: groupedEvents,
        sectors,
        assets,
        direction,
        severity,
        confidence
      };
    })
    .sort((a, b) => {
      const severityDelta = severityValue(b.severity) - severityValue(a.severity);
      if (severityDelta !== 0) return severityDelta;
      return b.events.length - a.events.length;
    });
}

export function buildAlertRules(events: EventRecord[]): AlertRule[] {
  const highEnergyConflictMatches = events.filter(
    (event) =>
      event.event_type === "geopolitical_conflict" &&
      event.affected_sectors.includes("energy") &&
      ["high", "critical"].includes(event.severity ?? "low")
  ).length;
  const broadMarketNegativeMatches = events.filter(
    (event) =>
      event.affected_sectors.includes("broad_market") &&
      event.impact_direction === "negative"
  ).length;
  const technologyNegativeMatches = events.filter(
    (event) =>
      event.affected_sectors.includes("technology") &&
      event.impact_direction === "negative"
  ).length;

  return [
    {
      id: "energy-geopolitical-high",
      label: "Energy conflict risk",
      active: highEnergyConflictMatches > 0,
      severity: "high",
      description: `${highEnergyConflictMatches} current event(s) classify as high-severity geopolitical conflict affecting energy.`,
      matchedEvents: highEnergyConflictMatches,
      rule: "event_type = geopolitical_conflict, sector includes energy, severity >= high"
    },
    {
      id: "broad-market-negative",
      label: "Broad market pressure",
      active: broadMarketNegativeMatches > 0,
      severity: "medium",
      description: `${broadMarketNegativeMatches} current event(s) have negative direction and broad market exposure.`,
      matchedEvents: broadMarketNegativeMatches,
      rule: "sector includes broad_market, impact_direction = negative"
    },
    {
      id: "technology-negative",
      label: "Technology drawdown watch",
      active: technologyNegativeMatches > 0,
      severity: "medium",
      description: `${technologyNegativeMatches} current event(s) have negative direction and technology exposure.`,
      matchedEvents: technologyNegativeMatches,
      rule: "sector includes technology, impact_direction = negative"
    }
  ];
}

export function riskRegime(events: EventRecord[]): string {
  const highRiskCount = events.filter((event) => ["high", "critical"].includes(event.severity ?? "low")).length;
  const negativeCount = events.filter((event) => event.impact_direction === "negative").length;

  if (highRiskCount >= 4 || negativeCount >= 6) return "Risk-Off";
  if (negativeCount >= 3) return "Defensive";
  return "Balanced";
}

function dominantSector(event: EventRecord): string {
  return event.affected_sectors[0] ?? "general";
}

function dominantKeyword(event: EventRecord): string {
  const text = `${event.title} ${event.description}`.toLowerCase();
  const keywords = ["hormuz", "inflation", "fed", "oil", "earnings", "ai", "battery", "market", "refinery", "tariff"];
  return keywords.find((keyword) => text.includes(keyword)) ?? event.title.toLowerCase().split(/\s+/)[0] ?? "event";
}

function clusterTitle(event: EventRecord, sectors: string[]): string {
  const keyword = dominantKeyword(event);
  if (keyword !== "event" && keyword.length > 2) {
    return `${formatLabel(keyword)} ${formatLabel(event.event_type)}`;
  }

  return `${formatLabel(sectors[0] ?? "macro")} ${formatLabel(event.event_type)}`;
}

function netDirection(directions: ImpactDirection[]): ImpactDirection {
  const score =
    directions.filter((direction) => direction === "positive").length -
    directions.filter((direction) => direction === "negative").length;
  if (score > 0) return "positive";
  if (score < 0) return "negative";
  return "neutral";
}
