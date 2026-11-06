import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { AvatarViewer, type ViewerState } from "./components/AvatarViewer";
import { ControlPanel } from "./components/ControlPanel";
import { UploadPanel } from "./components/UploadPanel";
import { ensureDemoGlb } from "./lib/api";
import type { JobInfo, MorphName } from "./lib/contracts";

export default function App() {
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [morphs, setMorphs] = useState<Partial<Record<MorphName, number>>>({});
  const [handCurl, setHandCurl] = useState(0.15);
  const [handSpread, setHandSpread] = useState(0.2);
  const [hiddenParts, setHiddenParts] = useState<Set<number>>(new Set());
  const [selectedBone, setSelectedBone] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const skeletonRef = useRef<THREE.Skeleton | null>(null);
  const sceneRef = useRef<THREE.Group | null>(null);

  const state: ViewerState = useMemo(
    () => ({
      morphs,
      handCurl,
      handSpread,
      hiddenParts,
      selectedBone,
      skeletonRef,
      sceneRef,
    }),
    [morphs, handCurl, handSpread, hiddenParts, selectedBone]
  );

  const loadDemo = () => {
    setLoadError(null);
    void ensureDemoGlb()
      .then((url) => {
        setGlbUrl(`${url}?t=${Date.now()}`);
        setJob(null);
      })
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
  };

  useEffect(() => {
    if (window.location.hash === "#demo") loadDemo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">H3D</span>
          <span className="tag">素体工作台 · MVP</span>
        </div>
        {job && (
          <div className="top-meta">
            job <code>{job.id}</code> · {job.status}
          </div>
        )}
      </header>

      <div className="layout">
        <aside className="sidebar">
          <UploadPanel
            onReady={(url, j) => {
              setGlbUrl(url);
              setJob(j);
              setLoadError(null);
            }}
            onDemo={loadDemo}
          />
          {loadError && <div className="error">{loadError}</div>}
          <ControlPanel
            state={state}
            setMorph={(name, value) =>
              setMorphs((m) => ({ ...m, [name]: value }))
            }
            setHandCurl={setHandCurl}
            setHandSpread={setHandSpread}
            togglePart={(id) =>
              setHiddenParts((prev) => {
                const next = new Set(prev);
                if (next.has(id)) next.delete(id);
                else next.add(id);
                return next;
              })
            }
            setSelectedBone={setSelectedBone}
            bump={() => setTick((t) => t + 1)}
          />
        </aside>
        <main className="main">
          <AvatarViewer url={glbUrl} state={state} />
          {job?.result?.stages && (
            <details className="pipeline-log">
              <summary>管线阶段（mock）</summary>
              <pre>{JSON.stringify(job.result.stages, null, 2)}</pre>
            </details>
          )}
        </main>
      </div>
    </div>
  );
}
