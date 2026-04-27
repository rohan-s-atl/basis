import { useState } from "react";
import { Filter, Maximize2 } from "lucide-react";

import { formatLabel, sectors } from "../lib/intelligence";
import type { EventRecord } from "../types";
import { ExpandModal } from "./ExpandModal";

type SectorScore = {
  sector: string;
  count: number;
  intensity: number;
};

type SectorGridProps = {
  scores: SectorScore[];
  cols: string;
  activeSector: string;
  onSectorClick?: (sector: string) => void;
  onClickClose?: () => void;
};

function SectorGrid({ scores, cols, activeSector, onSectorClick, onClickClose }: SectorGridProps) {
  return (
    <div className={`grid gap-2 ${cols}`}>
      {scores.map(({ sector, count, intensity }) => {
        const isActive = activeSector === sector;
        const isDisabled = count === 0 && !isActive;

        return (
          <button
            key={sector}
            disabled={isDisabled}
            onClick={() => {
              onSectorClick?.(sector);
              onClickClose?.();
            }}
            title={
              isActive
                ? `Clear ${formatLabel(sector)} sector filter`
                : `Filter events by ${formatLabel(sector)} - ${count} event${count !== 1 ? "s" : ""}`
            }
            className={`min-h-[86px] rounded-lg border p-3 text-left transition hover:brightness-110 ${
              isActive
                ? "border-quant-green/60 ring-1 ring-quant-green/20"
                : isDisabled
                  ? "cursor-not-allowed border-quant-line opacity-45"
                  : "border-quant-line hover:border-quant-green/50"
            }`}
            style={{
              background: `linear-gradient(135deg, rgba(18,24,33,0.96), rgba(0,230,118,${0.08 + intensity / 260}))`
            }}
          >
            <div className="mb-5 flex items-start justify-between gap-3">
              <span className={`text-sm font-semibold capitalize ${isActive ? "text-quant-green" : "text-quant-text"}`}>
                {formatLabel(sector)}
              </span>
              <strong className="text-xl text-quant-text">{count}</strong>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-quant-bg">
              <span
                className="block h-full rounded-full bg-gradient-to-r from-quant-green to-quant-blue"
                style={{ width: `${intensity}%` }}
              />
            </div>
            {isActive && (
              <span className="mt-1.5 flex items-center gap-1 text-[0.65rem] font-bold text-quant-green">
                <Filter size={9} /> Filtering
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

type SectorHeatmapProps = {
  events: EventRecord[];
  activeSector?: string;
  onSectorClick?: (sector: string) => void;
};

export function SectorHeatmap({ events, activeSector = "", onSectorClick }: SectorHeatmapProps) {
  const [expanded, setExpanded] = useState(false);

  const scores = sectors.map((sector) => {
    const count = events.filter((event) => event.affected_sectors.includes(sector)).length;
    const intensity = events.length === 0 ? 0 : Math.round((count / events.length) * 100);
    return { sector, count, intensity };
  });

  return (
    <>
      <section className="quant-panel min-h-0 p-4">
        <div className="mb-3 flex items-start justify-between">
          <div>
            <p className="quant-eyebrow">Sector Pressure</p>
            <h2 className="text-lg font-bold text-quant-text">Macro heatmap</h2>
          </div>
          <button
            onClick={() => setExpanded(true)}
            title="Expand sector heatmap"
            className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
          >
            <Maximize2 size={13} />
          </button>
        </div>
        <p className="mb-2 text-xs text-quant-muted/70">Click a sector to filter events. Click the active sector again to clear.</p>
        {activeSector && (
          <p className="mb-2 text-xs font-semibold text-quant-green">
            Filtering by {formatLabel(activeSector)}
          </p>
        )}
        <SectorGrid scores={scores} cols="grid-cols-3" activeSector={activeSector} onSectorClick={onSectorClick} />
      </section>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <section className="quant-panel p-6">
            <div className="mb-4 pr-8">
              <p className="quant-eyebrow">Sector Pressure</p>
              <h2 className="text-xl font-bold text-quant-text">Macro heatmap</h2>
              <p className="mt-1 text-sm text-quant-muted">
                Click a sector to filter all events. Click the active sector again to clear.
              </p>
            </div>
            <SectorGrid
              scores={scores}
              cols="grid-cols-3 sm:grid-cols-6"
              activeSector={activeSector}
              onSectorClick={onSectorClick}
              onClickClose={() => setExpanded(false)}
            />
          </section>
        </ExpandModal>
      )}
    </>
  );
}
