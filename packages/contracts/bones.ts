/** SMPL-X-aligned bone names used by H3D assets (55 base + 4 helper). */

export const BASE_BONES = [
  "Pelvis",
  "Spine1",
  "Spine2",
  "Spine3",
  "LeftCollar",
  "LeftShoulder",
  "LeftElbow",
  "LeftWrist",
  "LeftThumb1",
  "LeftThumb2",
  "LeftThumb3",
  "LeftIndex1",
  "LeftIndex2",
  "LeftIndex3",
  "LeftMiddle1",
  "LeftMiddle2",
  "LeftMiddle3",
  "LeftRing1",
  "LeftRing2",
  "LeftRing3",
  "LeftPinky1",
  "LeftPinky2",
  "LeftPinky3",
  "RightCollar",
  "RightShoulder",
  "RightElbow",
  "RightWrist",
  "RightThumb1",
  "RightThumb2",
  "RightThumb3",
  "RightIndex1",
  "RightIndex2",
  "RightIndex3",
  "RightMiddle1",
  "RightMiddle2",
  "RightMiddle3",
  "RightRing1",
  "RightRing2",
  "RightRing3",
  "RightPinky1",
  "RightPinky2",
  "RightPinky3",
  "Neck",
  "Head",
  "Jaw",
  "LeftEye",
  "RightEye",
  "LeftHip",
  "LeftKnee",
  "LeftAnkle",
  "LeftFoot",
  "RightHip",
  "RightKnee",
  "RightAnkle",
  "RightFoot",
] as const;

export const HELPER_BONES = [
  "Spine3_Breast_L",
  "Spine3_Breast_R",
  "Pelvis_Glute_L",
  "Pelvis_Glute_R",
] as const;

export const ALL_BONES = [...BASE_BONES, ...HELPER_BONES] as const;

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

export const PART_IDS = {
  HEAD_NECK: 0,
  TORSO: 1,
  ARMS_UPPER: 2,
  ARMS_LOWER: 3,
  HANDS: 4,
  THIGHS: 5,
  CALVES: 6,
  FEET: 7,
} as const;

export type BoneName = (typeof ALL_BONES)[number];
export type MorphName = (typeof MORPH_TARGETS)[number];

export interface H3DPose {
  version: "1.0";
  bones: Record<string, { rotation: [number, number, number, number] }>;
  morphs: Partial<Record<MorphName, number>>;
}
