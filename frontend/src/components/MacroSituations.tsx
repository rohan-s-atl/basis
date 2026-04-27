import { useState } from "react";
import { Maximize2 } from "lucide-react";

import { directionArrow, directionColor, formatLabel } from "../lib/intelligence";
import type { EventCluster } from "../types";
import { ExpandModal } from "./ExpandModal";

type ClusterItemProps = {
  cluster: EventCluster;
  onClusterClick?: (cluster: EventCluster) => void;
};

function ClusterItem({ cluster, onClusterClick }: ClusterItemProps) {
  return (
    <button
      onClick={() => onClusterClick?.(cluster)}
      className="w-full rounded-lg border border-quant-line bg-quant-panel2 p-2 text-left transition hover:border-quant-green/40 hover:bg-quant-panel2/80"
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <strong className="truncate text-sm capitalize text-quant-text">{cluster.title}</strong>
        <span className={`shrink-0 text-xs font-black ${directionColor(cluster.direction)}`}>
          {directionArrow(cluster.direction)}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-[0.68rem] text-quant-muted">
        <span>{cluster.events.length} articles</span>
        <span>|</span>
        <span className="capitalize">{formatLabel(cluster.severity)}</span>
        <span>|</span>
        <span>{Math.round(cluster.confidence * 100)}%</span>
        <span className="ml-auto font-bold text-quant-green">View lead event</span>
      </div>
    </button>
  );
}

type MacroSituationsProps = {
  clusters: EventCluster[];
  onClusterClick?: (cluster: EventCluster) => void;
};

export function MacroSituations({ clusters, onClusterClick }: MacroSituationsProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <section className="quant-panel p-3">
        <div className="mb-2 flex items-start justify-between">
          <div>
            <p className="quant-eyebrow">Macro Situations</p>
            <h2 className="text-base font-bold text-quant-text">Clustered events</h2>
          </div>
          <button
            onClick={() => setExpanded(true)}
            title="Expand"
            className="rounded-md border border-quant-line bg-quant-panel2 p-1.5 text-quant-muted transition hover:text-quant-text"
          >
            <Maximize2 size={13} />
          </button>
        </div>
        <div className="scrollbar-quant grid max-h-[190px] gap-2 overflow-auto pr-1">
          {clusters.slice(0, 5).map((cluster) => (
            <ClusterItem key={cluster.id} cluster={cluster} onClusterClick={onClusterClick} />
          ))}
        </div>
      </section>

      {expanded && (
        <ExpandModal onClose={() => setExpanded(false)}>
          <section className="quant-panel p-5">
            <div className="mb-3 pr-8">
              <p className="quant-eyebrow">Macro Situations</p>
              <h2 className="text-lg font-bold text-quant-text">Clustered events</h2>
            </div>
            <div className="grid gap-2">
              {clusters.map((cluster) => (
                <ClusterItem key={cluster.id} cluster={cluster} onClusterClick={onClusterClick} />
              ))}
            </div>
          </section>
        </ExpandModal>
      )}
    </>
  );
}
