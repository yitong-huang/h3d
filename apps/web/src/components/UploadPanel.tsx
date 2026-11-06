import { useRef, useState } from "react";
import { createJob, pollJob } from "../lib/api";
import type { JobInfo } from "../lib/contracts";

const PART_OPTIONS = [
  "",
  "Head",
  "Torso",
  "Arms_Lower",
  "Arms_Upper",
  "Hands",
  "Thighs",
  "Calves",
  "Feet",
];

interface Props {
  onReady: (glbUrl: string, job: JobInfo) => void;
  onDemo: () => void;
}

export function UploadPanel({ onReady, onDemo }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [hints, setHints] = useState<Record<string, string>>({});
  const [job, setJob] = useState<JobInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onPick = (list: FileList | null) => {
    if (!list) return;
    const next = Array.from(list).slice(0, 10);
    setFiles(next);
    setHints({});
    setError(null);
  };

  const submit = async () => {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createJob(files, hints);
      setJob(created);
      pollJob(created.id, (j) => {
        setJob(j);
        if (j.status === "done") {
          const url = j.result?.glbUrl ?? `/api/jobs/${j.id}/avatar.glb`;
          onReady(url, j);
          setBusy(false);
        } else if (j.status === "error") {
          setError(j.error || j.message || "Job failed");
          setBusy(false);
        }
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>输入</h2>
        <p>上传 1–10 张人像 / 局部特写</p>
      </header>

      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          onPick(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          hidden
          onChange={(e) => onPick(e.target.files)}
        />
        {files.length === 0 ? (
          <span>拖拽或点击选择 JPG / PNG / WEBP</span>
        ) : (
          <ul className="file-list">
            {files.map((f) => (
              <li key={f.name}>
                <span className="file-name">{f.name}</span>
                <select
                  value={hints[f.name] ?? ""}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) =>
                    setHints((h) => ({ ...h, [f.name]: e.target.value }))
                  }
                >
                  {PART_OPTIONS.map((p) => (
                    <option key={p || "auto"} value={p}>
                      {p ? `部位: ${p}` : "自动分流"}
                    </option>
                  ))}
                </select>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="row gap">
        <button
          className="btn primary"
          disabled={!files.length || busy}
          onClick={() => void submit()}
        >
          {busy ? "生成中…" : "开始生成素体"}
        </button>
        <button className="btn" onClick={onDemo} disabled={busy}>
          加载演示模型
        </button>
      </div>

      {job && (
        <div className="job-status">
          <div className="progress-bar">
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="muted">
            [{job.stage}] {job.message} ({Math.round(job.progress * 100)}%)
          </div>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </section>
  );
}
