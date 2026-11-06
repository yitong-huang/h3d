import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  OrbitControls,
  TransformControls,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";
import type { MorphName } from "../lib/contracts";
import { applyHandPose } from "../lib/pose";

export interface ViewerState {
  morphs: Partial<Record<MorphName, number>>;
  handCurl: number;
  handSpread: number;
  hiddenParts: Set<number>;
  selectedBone: string | null;
  skeletonRef: React.MutableRefObject<THREE.Skeleton | null>;
  sceneRef: React.MutableRefObject<THREE.Group | null>;
}

function findPartAttr(geom: THREE.BufferGeometry): THREE.BufferAttribute | null {
  return (
    (geom.getAttribute("_PART_ID") as THREE.BufferAttribute | undefined) ||
    (geom.getAttribute("_part_id") as THREE.BufferAttribute | undefined) ||
    null
  );
}

function AvatarModel({ url, state }: { url: string; state: ViewerState }) {
  const gltf = useGLTF(url);
  const root = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  const skinned = useRef<THREE.SkinnedMesh[]>([]);
  const skeleton = useRef<THREE.Skeleton | null>(null);
  const baseColors = useRef<Map<THREE.SkinnedMesh, Float32Array>>(new Map());

  useEffect(() => {
    const meshes: THREE.SkinnedMesh[] = [];
    root.traverse((obj) => {
      if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
        const mesh = obj as THREE.SkinnedMesh;
        meshes.push(mesh);
        skeleton.current = mesh.skeleton;

        const geom = mesh.geometry as THREE.BufferGeometry;
        let colors = geom.getAttribute("color") as THREE.BufferAttribute | null;
        if (!colors) {
          const n = geom.getAttribute("position").count;
          const arr = new Float32Array(n * 4);
          for (let i = 0; i < n; i++) {
            arr[i * 4] = 0.78;
            arr[i * 4 + 1] = 0.62;
            arr[i * 4 + 2] = 0.52;
            arr[i * 4 + 3] = 1;
          }
          colors = new THREE.BufferAttribute(arr, 4);
          geom.setAttribute("color", colors);
        }
        baseColors.current.set(mesh, new Float32Array(colors.array as Float32Array));

        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.vertexColors = true;
        mat.transparent = true;
        mat.alphaTest = 0.5;
        mat.needsUpdate = true;

        // Ensure morph dictionary from mesh extras targetNames if missing
        const targetNames =
          (mesh.userData?.targetNames as string[] | undefined) ||
          ((gltf as unknown as { parser?: { json?: { meshes?: { extras?: { targetNames?: string[] } }[] } } })
            .parser?.json?.meshes?.[0]?.extras?.targetNames);
        if (
          targetNames &&
          mesh.morphTargetInfluences &&
          (!mesh.morphTargetDictionary ||
            Object.keys(mesh.morphTargetDictionary).length === 0)
        ) {
          mesh.morphTargetDictionary = Object.fromEntries(
            targetNames.map((n, i) => [n, i])
          );
        }
      }
    });
    skinned.current = meshes;
    state.skeletonRef.current = skeleton.current;
    state.sceneRef.current = root;
  }, [root, state, gltf]);

  useFrame(() => {
    const skel = skeleton.current;
    if (skel) applyHandPose(skel, state.handCurl, state.handSpread);

    for (const mesh of skinned.current) {
      const dict = mesh.morphTargetDictionary;
      const infl = mesh.morphTargetInfluences;
      if (dict && infl) {
        for (const [name, value] of Object.entries(state.morphs)) {
          const idx = dict[name];
          if (idx != null && typeof value === "number") infl[idx] = value;
        }
      }

      const geom = mesh.geometry as THREE.BufferGeometry;
      const partAttr = findPartAttr(geom);
      const colorAttr = geom.getAttribute("color") as THREE.BufferAttribute | null;
      const base = baseColors.current.get(mesh);
      if (!partAttr || !colorAttr || !base) continue;

      const hidden = state.hiddenParts;
      for (let i = 0; i < partAttr.count; i++) {
        const part = Math.round(partAttr.getX(i));
        const visible = !hidden.has(part);
        colorAttr.setXYZW(
          i,
          base[i * 4],
          base[i * 4 + 1],
          base[i * 4 + 2],
          visible ? 1 : 0
        );
      }
      colorAttr.needsUpdate = true;
    }
  });

  const selected = state.selectedBone
    ? skeleton.current?.getBoneByName(state.selectedBone) ?? null
    : null;

  return (
    <group>
      <primitive object={root} />
      {selected && (
        <TransformControls object={selected} mode="rotate" size={0.6} />
      )}
    </group>
  );
}

export function AvatarViewer({
  url,
  state,
}: {
  url: string | null;
  state: ViewerState;
}) {
  return (
    <div className="viewport">
      <Canvas
        camera={{ position: [1.4, 1.2, 2.2], fov: 40 }}
        gl={{ preserveDrawingBuffer: true }}
      >
        <color attach="background" args={["#1a1d23"]} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[3, 5, 2]} intensity={1.1} />
        <directionalLight position={[-2, 2, -3]} intensity={0.35} />
        <Suspense fallback={null}>
          {url ? <AvatarModel key={url} url={url} state={state} /> : null}
          <Environment preset="city" />
          <ContactShadows
            position={[0, 0.01, 0]}
            opacity={0.45}
            scale={6}
            blur={2.5}
          />
        </Suspense>
        <OrbitControls
          makeDefault
          target={[0, 1.0, 0]}
          maxPolarAngle={Math.PI * 0.9}
        />
        <gridHelper
          args={[4, 20, "#3a4050", "#2a2f3a"]}
          position={[0, 0.001, 0]}
        />
      </Canvas>
      {!url && <div className="viewport-empty">上传照片或加载演示模型</div>}
    </div>
  );
}
