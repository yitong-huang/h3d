import * as THREE from "three";
import type { H3DPose, MorphName } from "./contracts";

const FINGER_BONES = [
  "Thumb",
  "Index",
  "Middle",
  "Ring",
  "Pinky",
] as const;

/** Apply curl / spread to finger bones on both hands. */
export function applyHandPose(
  skeleton: THREE.Skeleton,
  curl: number,
  spread: number
) {
  for (const side of ["Left", "Right"] as const) {
    const sign = side === "Left" ? 1 : -1;
    FINGER_BONES.forEach((finger, fi) => {
      for (let joint = 1; joint <= 3; joint++) {
        const name = `${side}${finger}${joint}`;
        const bone = skeleton.getBoneByName(name);
        if (!bone) continue;
        const curlAmt = curl * (finger === "Thumb" ? 0.6 : 1.1);
        const axis = finger === "Thumb" ? "y" : "z";
        bone.rotation.set(0, 0, 0);
        if (axis === "z") {
          bone.rotation.z = -sign * curlAmt * (0.5 + joint * 0.25);
        } else {
          bone.rotation.y = sign * curlAmt * 0.7;
          bone.rotation.x = -curlAmt * 0.4;
        }
        // spread between fingers at first joint
        if (joint === 1 && finger !== "Thumb") {
          const spreadOff = (fi - 2) * spread * 0.25 * sign;
          bone.rotation.y += spreadOff;
        }
      }
    });
  }
}

export function applyEulerPose(
  skeleton: THREE.Skeleton,
  pose: Record<string, [number, number, number]>
) {
  for (const bone of skeleton.bones) {
    // Reset non-finger controllable bones when applying presets
    if (
      bone.name.includes("Thumb") ||
      bone.name.includes("Index") ||
      bone.name.includes("Middle") ||
      bone.name.includes("Ring") ||
      bone.name.includes("Pinky")
    ) {
      continue;
    }
    bone.rotation.set(0, 0, 0);
  }
  for (const [name, euler] of Object.entries(pose)) {
    const bone = skeleton.getBoneByName(name);
    if (bone) bone.rotation.set(euler[0], euler[1], euler[2]);
  }
}

export function exportPoseJson(
  skeleton: THREE.Skeleton,
  morphs: Partial<Record<MorphName, number>>
): H3DPose {
  const bones: H3DPose["bones"] = {};
  const q = new THREE.Quaternion();
  for (const bone of skeleton.bones) {
    bone.getWorldQuaternion(q);
    // Prefer local rotation for pose JSON (lighter / more portable)
    q.copy(bone.quaternion);
    bones[bone.name] = {
      rotation: [q.x, q.y, q.z, q.w],
    };
  }
  return { version: "1.0", bones, morphs };
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
