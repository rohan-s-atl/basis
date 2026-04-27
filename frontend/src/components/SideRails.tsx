import { buildAlertRules, directionArrow, directionColor, riskRegime } from "../lib/intelligence";
import type { EventCluster, EventRecord } from "../types";

type SideRailsProps = {
  events: EventRecord[];
  clusters: EventCluster[];
};

export function SideRails({ events, clusters }: SideRailsProps) {
  const alerts = buildAlertRules(events).filter((alert) => alert.active);
  const regime = riskRegime(events);

  return (
    <>
      <aside className="pointer-events-none fixed left-3 top-1/2 hidden w-28 -translate-y-1/2 2xl:block">
        <div className="quant-panel p-2">
          <p className="quant-eyebrow mb-2">Regime</p>
          <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2 text-center">
            <strong className="text-sm text-quant-green">{regime}</strong>
            <p className="mt-1 text-[0.65rem] text-quant-muted">{events.length} signals</p>
          </div>
        </div>
      </aside>

      <aside className="pointer-events-none fixed right-3 top-1/2 hidden w-32 -translate-y-1/2 2xl:block">
        <div className="quant-panel grid gap-2 p-2">
          <div>
            <p className="quant-eyebrow mb-2">Top Cluster</p>
            {clusters[0] && (
              <div className="rounded-lg border border-quant-line bg-quant-panel2 p-2">
                <strong className="block truncate text-xs capitalize text-quant-text">{clusters[0].title}</strong>
                <span className={`text-xs font-black ${directionColor(clusters[0].direction)}`}>
                  {directionArrow(clusters[0].direction)} {clusters[0].events.length}
                </span>
              </div>
            )}
          </div>
          <div>
            <p className="quant-eyebrow mb-2">Alerts</p>
            <strong className="text-xl font-black text-quant-yellow">{alerts.length}</strong>
          </div>
        </div>
      </aside>
    </>
  );
}
