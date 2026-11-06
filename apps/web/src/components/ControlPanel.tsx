import {
  BODY_PARTS,
  HAND_PRESETS,
  MORPH_TARGETS,
  POSE_PRESETS,
  type MorphName,
} from "../lib/contracts";
import { applyEulerPose, downloadBlob, exportPoseJson } from "../lib/pose";
import type { ViewerState } from "./AvatarViewer";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

const BODY_MORPHS: MorphName[] = [
  "breast_scale",
  "waist_scale",
  "glute_scale",
  "thigh_scale",
];
const FACE_MORPHS: MorphName[] = [
  "mouthSmile",
  "jawOpen",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "browInnerUp",
];

const JOINT_BONES = [
  "Spine1",
  "Spine2",
  "Spine3",
  "LeftShoulder",
  "LeftElbow",
  "LeftWrist",
  "RightShoulder",
  "RightElbow",
  "RightWrist",
  "LeftHip",
  "LeftKnee",
  "LeftAnkle",
  "RightHip",
  "RightKnee",
  "RightAnkle",
  "Neck",
  "Head",
  "Jaw",
  "LeftEye",
  "RightEye",
];

interface Props {
  state: ViewerState;
  setMorph: (name: MorphName, value: number) => void;
  setHandCurl: (v: number) => void;
  setHandSpread: (v: number) => void;
  togglePart: (id: number) => void;
  setSelectedBone: (name: string | null) => void;
  bump: () => void;
}

export function ControlPanel({
  state,
  setMorph,
  setHandCurl,
  setHandSpread,
  togglePart,
  setSelectedBone,
  bump,
}: Props) {
  const applyPreset = (name: string) => {
    const skel = state.skeletonRef.current;
    if (!skel) return;
    applyEulerPose(skel, POSE_PRESETS[name] ?? {});
    bump();
  };

  const applyHandPreset = (name: string) => {
    const p = HAND_PRESETS[name];
    if (!p) return;
    setHandCurl(p.curl);
    setHandSpread(p.spread);
  };

  const exportPose = () => {
    const skel = state.skeletonRef.current;
    if (!skel) return;
    const pose = exportPoseJson(skel, state.morphs);
    downloadBlob(
      "h3d-pose.json",
      new Blob([JSON.stringify(pose, null, 2)], { type: "application/json" })
    );
  };

  const exportGlb = () => {
    const scene = state.sceneRef.current;
    if (!scene) return;
    const exporter = new GLTFExporter();
    exporter.parse(
      scene,
      (result) => {
        if (result instanceof ArrayBuffer) {
          downloadBlob(
            "h3d-avatar.glb",
            new Blob([result], { type: "model/gltf-binary" })
          );
        }
      },
      (err) => console.error(err),
      { binary: true, animations: [] }
    );
  };

  return (
    <section className="panel controls">
      <header className="panel-header">
        <h2>工作台</h2>
        <p>姿态 · 手势 · 表情 · 身材 · 分区遮罩</p>
      </header>

      <div className="control-block">
        <h3>姿态预设</h3>
        <div className="chip-row">
          {Object.keys(POSE_PRESETS).map((name) => (
            <button key={name} className="chip" onClick={() => applyPreset(name)}>
              {name}
            </button>
          ))}
        </div>
        <label className="field">
          <span>选中关节 (Gizmo)</span>
          <select
            value={state.selectedBone ?? ""}
            onChange={(e) => setSelectedBone(e.target.value || null)}
          >
            <option value="">无</option>
            {JOINT_BONES.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="control-block">
        <h3>手势</h3>
        <div className="chip-row">
          {Object.keys(HAND_PRESETS).map((name) => (
            <button
              key={name}
              className="chip"
              onClick={() => applyHandPreset(name)}
            >
              {name}
            </button>
          ))}
        </div>
        <Slider
          label="Curl"
          min={0}
          max={1}
          step={0.01}
          value={state.handCurl}
          onChange={setHandCurl}
        />
        <Slider
          label="Spread"
          min={0}
          max={1}
          step={0.01}
          value={state.handSpread}
          onChange={setHandSpread}
        />
      </div>

      <div className="control-block">
        <h3>面部表情</h3>
        {FACE_MORPHS.map((m) => (
          <Slider
            key={m}
            label={m}
            min={0}
            max={1}
            step={0.01}
            value={state.morphs[m] ?? 0}
            onChange={(v) => setMorph(m, v)}
          />
        ))}
      </div>

      <div className="control-block">
        <h3>身材微调</h3>
        {BODY_MORPHS.map((m) => (
          <Slider
            key={m}
            label={m}
            min={-0.5}
            max={1.5}
            step={0.01}
            value={state.morphs[m] ?? 0}
            onChange={(v) => setMorph(m, v)}
          />
        ))}
      </div>

      <div className="control-block">
        <h3>分区遮罩 (_PART_ID)</h3>
        <div className="chip-row wrap">
          {BODY_PARTS.map((p) => (
            <button
              key={p.id}
              className={`chip ${state.hiddenParts.has(p.id) ? "active" : ""}`}
              onClick={() => togglePart(p.id)}
              title="隐藏该分区网格"
            >
              {state.hiddenParts.has(p.id) ? "隐藏 " : ""}
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="row gap export-row">
        <button className="btn primary" onClick={exportGlb}>
          导出 GLB
        </button>
        <button className="btn" onClick={exportPose}>
          导出 Pose JSON
        </button>
      </div>

      <p className="muted tiny">
        Morph 通道：{MORPH_TARGETS.length} · 遮罩分区：{BODY_PARTS.length}
      </p>
    </section>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="field slider">
      <span>
        {label}
        <em>{value.toFixed(2)}</em>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
