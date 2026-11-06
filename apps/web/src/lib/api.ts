import type { JobInfo } from "./contracts";

export async function createJob(
  files: File[],
  partHints: Record<string, string | undefined>
): Promise<JobInfo> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const hints: Record<string, string | null> = {};
  for (const [k, v] of Object.entries(partHints)) {
    hints[k] = v ?? null;
  }
  form.append("part_hints", JSON.stringify(hints));
  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId: string): Promise<JobInfo> {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function ensureDemoGlb(): Promise<string> {
  const res = await fetch("/api/demo/generate-glb", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.glbUrl as string;
}

export function pollJob(
  jobId: string,
  onUpdate: (job: JobInfo) => void,
  intervalMs = 400
): () => void {
  let stopped = false;
  let timer: number | undefined;

  const tick = async () => {
    if (stopped) return;
    try {
      const job = await getJob(jobId);
      onUpdate(job);
      if (job.status === "done" || job.status === "error") return;
    } catch {
      /* retry */
    }
    timer = window.setTimeout(tick, intervalMs);
  };

  void tick();
  return () => {
    stopped = true;
    if (timer) window.clearTimeout(timer);
  };
}
