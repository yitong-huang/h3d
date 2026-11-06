export const MORPH_TARGETS = [
  "breast_scale",
  "waist_scale",
  "glute_scale",
  "thigh_scale",
  "mouthSmile",
  "jawOpen",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "browInnerUp",
] as const;

export type MorphName = (typeof MORPH_TARGETS)[number];

export const BODY_PARTS = [
  { id: 0, label: "Head & Neck" },
  { id: 1, label: "Torso" },
  { id: 2, label: "Arms Upper" },
  { id: 3, label: "Arms Lower" },
  { id: 4, label: "Hands" },
  { id: 5, label: "Thighs" },
  { id: 6, label: "Calves" },
  { id: 7, label: "Feet" },
] as const;

/** Euler XYZ radians relative to rest pose. */
export const POSE_PRESETS: Record<
  string,
  Record<string, [number, number, number]>
> = {
  "T-Pose": {},
  "A-Pose": {
    LeftShoulder: [0, 0, -0.45],
    RightShoulder: [0, 0, 0.45],
  },
  Stand: {
    LeftShoulder: [0, 0, -0.25],
    RightShoulder: [0, 0, 0.25],
    LeftElbow: [0, 0, -0.15],
    RightElbow: [0, 0, 0.15],
  },
  Sit: {
    LeftHip: [1.2, 0, 0],
    RightHip: [1.2, 0, 0],
    LeftKnee: [-1.4, 0, 0],
    RightKnee: [-1.4, 0, 0],
  },
  "Hands on hips": {
    LeftShoulder: [0.2, 0.3, -0.8],
    RightShoulder: [0.2, -0.3, 0.8],
    LeftElbow: [0, 0, -1.4],
    RightElbow: [0, 0, 1.4],
  },
};

export const HAND_PRESETS: Record<string, { curl: number; spread: number }> = {
  Open: { curl: 0, spread: 0.4 },
  Fist: { curl: 1, spread: 0 },
  Victory: { curl: 0.85, spread: 0.5 },
  ThumbsUp: { curl: 0.9, spread: 0.1 },
  OK: { curl: 0.55, spread: 0.35 },
  Heart: { curl: 0.4, spread: 0.2 },
};

export interface H3DPose {
  version: "1.0";
  bones: Record<string, { rotation: [number, number, number, number] }>;
  morphs: Partial<Record<MorphName, number>>;
}

export interface JobInfo {
  id: string;
  status: string;
  stage: string;
  progress: number;
  message: string;
  files?: string[];
  result?: {
    glbUrl?: string;
    meta?: Record<string, unknown>;
    stages?: Record<string, unknown>;
  };
  error?: string | null;
}
