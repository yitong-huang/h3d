"""Export a posed SMPL-X mesh to H3D contract GLB (real topology)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bones import MORPH_NAMES
from .smplx_runtime import (
    HELPER_BONES,
    SMPLX_JOINT_NAMES,
    load_smplx,
    resolve_device,
    smplx_faces,
    smplx_parents,
    smplx_weights,
)

PART_HEAD = 0
PART_TORSO = 1
PART_ARMS_UPPER = 2
PART_ARMS_LOWER = 3
PART_HANDS = 4
PART_THIGHS = 5
PART_CALVES = 6
PART_FEET = 7

# Dominant skinning bone → part id
_BONE_TO_PART = {
    0: PART_TORSO,
    1: PART_THIGHS,
    2: PART_THIGHS,
    3: PART_TORSO,
    4: PART_CALVES,
    5: PART_CALVES,
    6: PART_TORSO,
    7: PART_FEET,
    8: PART_FEET,
    9: PART_TORSO,
    10: PART_FEET,
    11: PART_FEET,
    12: PART_HEAD,
    13: PART_TORSO,
    14: PART_TORSO,
    15: PART_HEAD,
    16: PART_ARMS_UPPER,
    17: PART_ARMS_UPPER,
    18: PART_ARMS_LOWER,
    19: PART_ARMS_LOWER,
    20: PART_HANDS,
    21: PART_HANDS,
    22: PART_HEAD,
    23: PART_HEAD,
    24: PART_HEAD,
}
for i in range(25, 55):
    _BONE_TO_PART[i] = PART_HANDS


def _pad4(b: bytes, pad_byte: bytes = b"\x00") -> bytes:
    return b + (pad_byte * ((4 - (len(b) % 4)) % 4))


def _part_ids(weights: np.ndarray) -> np.ndarray:
    dominant = weights.argmax(axis=1)
    return np.array([_BONE_TO_PART.get(int(b), PART_TORSO) for b in dominant], dtype=np.float32)


def _load_vt_ft() -> tuple[np.ndarray | None, np.ndarray | None]:
    from pathlib import Path
    import pickle

    from ..config import SMPLX_MODEL_DIR

    pkl = SMPLX_MODEL_DIR / "smplx" / "SMPLX_NEUTRAL.pkl"
    if not pkl.exists():
        pkl = SMPLX_MODEL_DIR / "SMPLX_NEUTRAL.pkl"
    if not pkl.exists():
        return None, None
    with open(pkl, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    vt = np.asarray(data.get("vt"), dtype=np.float32)
    ft = np.asarray(data.get("ft"), dtype=np.int64)
    return vt, ft


def _vertex_uvs(n_verts: int, faces: np.ndarray, model) -> np.ndarray:
    """Assign a UV per mesh vertex from SMPL-X vt/ft when available."""
    uvs = np.zeros((n_verts, 2), dtype=np.float32)
    filled = np.zeros(n_verts, dtype=bool)
    vt, ft = _load_vt_ft()
    if vt is None or ft is None:
        return uvs
    faces = np.asarray(faces, dtype=np.int64)
    for face, fuv in zip(faces, ft):
        for vi, ui in zip(face, fuv):
            if 0 <= vi < n_verts and not filled[vi]:
                uvs[vi] = vt[int(ui) % len(vt)]
                filled[vi] = True
    if not filled.all():
        for face, fuv in zip(faces, ft):
            for vi, ui in zip(face, fuv):
                if 0 <= vi < n_verts and not filled[vi]:
                    uvs[vi] = vt[int(ui) % len(vt)]
                    filled[vi] = True
    return uvs


def _morph_deltas(vertices: np.ndarray, part_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Named morph targets as local geometric offsets on the real mesh."""
    v = vertices
    morphs: dict[str, np.ndarray] = {}
    for name in MORPH_NAMES:
        d = np.zeros_like(v)
        if name == "breast_scale":
            m = (part_ids == PART_TORSO) & (v[:, 1] > 0.05) & (v[:, 2] > 0)
            d[m, 2] += 0.04
            d[m, 0] += np.sign(v[m, 0] + 1e-6) * 0.01
        elif name == "waist_scale":
            m = (part_ids == PART_TORSO) & (v[:, 1] > -0.05) & (v[:, 1] < 0.12)
            d[m, 0] = v[m, 0] * 0.15
            d[m, 2] = v[m, 2] * 0.15
        elif name == "glute_scale":
            m = (part_ids == PART_TORSO) & (v[:, 2] < 0) & (v[:, 1] < 0.05)
            d[m, 2] -= 0.035
        elif name == "thigh_scale":
            m = part_ids == PART_THIGHS
            d[m, 0] = v[m, 0] * 0.12
            d[m, 2] = v[m, 2] * 0.12
        elif name == "mouthSmile":
            m = (part_ids == PART_HEAD) & (v[:, 1] < 0.68) & (v[:, 2] > 0.02)
            d[m, 0] = np.sign(v[m, 0] + 1e-6) * 0.008
            d[m, 1] += 0.004
        elif name == "jawOpen":
            m = (part_ids == PART_HEAD) & (v[:, 1] < 0.65)
            d[m, 1] -= 0.02
        elif name == "eyeBlinkLeft":
            m = (part_ids == PART_HEAD) & (v[:, 0] > 0.01) & (v[:, 1] > 0.7)
            d[m, 1] -= 0.006
        elif name == "eyeBlinkRight":
            m = (part_ids == PART_HEAD) & (v[:, 0] < -0.01) & (v[:, 1] > 0.7)
            d[m, 1] -= 0.006
        elif name == "browInnerUp":
            m = (part_ids == PART_HEAD) & (v[:, 1] > 0.75) & (np.abs(v[:, 0]) < 0.03)
            d[m, 1] += 0.008
        morphs[name] = d.astype(np.float32)
    return morphs


def write_smplx_glb(
    out_path: Path,
    *,
    betas: list[float] | np.ndarray,
    body_pose: np.ndarray | None = None,
    global_orient: np.ndarray | None = None,
    left_hand_pose: np.ndarray | None = None,
    right_hand_pose: np.ndarray | None = None,
    jaw_pose: np.ndarray | None = None,
    expression: np.ndarray | None = None,
    transl: np.ndarray | None = None,
    skin_tone: tuple[float, float, float] = (0.78, 0.62, 0.52),
    basecolor: np.ndarray | None = None,
    normals_map: np.ndarray | None = None,
    detail_regions: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Forward SMPL-X and write a skinned GLB with morphs + `_PART_ID`."""
    model = load_smplx()
    device = resolve_device()

    def t(x, shape):
        if x is None:
            return torch.zeros(1, *shape, device=device)
        arr = np.asarray(x, dtype=np.float32).reshape(1, *shape)
        return torch.from_numpy(arr).to(device)

    betas_t = t(betas, (10,))
    kwargs: dict[str, Any] = {
        "betas": betas_t,
        "body_pose": t(body_pose, (63,)),
        "global_orient": t(global_orient, (3,)),
        "transl": t(transl, (3,)),
        "expression": t(expression, (10,)),
        "jaw_pose": t(jaw_pose, (3,)),
        "left_hand_pose": t(left_hand_pose, (45,)),
        "right_hand_pose": t(right_hand_pose, (45,)),
        "leye_pose": torch.zeros(1, 3, device=device),
        "reye_pose": torch.zeros(1, 3, device=device),
    }
    with torch.no_grad():
        out = model(**kwargs)
        vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        joints = out.joints[0, :55].detach().cpu().numpy().astype(np.float32)

    faces = smplx_faces().astype(np.uint32)
    weights = smplx_weights()
    parents = smplx_parents()
    part_ids = _part_ids(weights)

    # Rest-pose local translations for skeleton
    with torch.no_grad():
        rest = model(
            betas=betas_t,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
        rest_joints = rest.joints[0, :55].detach().cpu().numpy().astype(np.float32)

    local_t = np.zeros((55, 3), dtype=np.float32)
    for j, p in enumerate(parents):
        if p < 0:
            local_t[j] = rest_joints[j]
        else:
            local_t[j] = rest_joints[j] - rest_joints[p]

    # Normals
    normals = np.zeros_like(vertices)
    tris = vertices[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn_n = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    fn = fn / fn_n
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    uvs = _vertex_uvs(len(vertices), faces, model)
    colors = np.tile(np.array([*skin_tone, 1.0], dtype=np.float32), (len(vertices), 1))
    if basecolor is not None and basecolor.shape[0] == len(vertices):
        colors[:, :3] = basecolor[:, :3]

    # Underwear tint on pelvis-adjacent torso
    und = (part_ids == PART_TORSO) & (vertices[:, 1] < 0.0) & (vertices[:, 1] > -0.15)
    colors[und, :3] = np.array([0.18, 0.18, 0.22], dtype=np.float32)

    morphs = _morph_deltas(vertices, part_ids)

    # Skin joints/weights VEC4
    top4 = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, top4, axis=1)
    w4 = w4 / (w4.sum(axis=1, keepdims=True) + 1e-8)
    joints_attr = top4.astype(np.uint16)
    weights_attr = w4.astype(np.float32)

    # Inverse bind matrices from rest joints
    ibm = np.zeros((55, 16), dtype=np.float32)
    for i in range(55):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -rest_joints[i]
        ibm[i] = m.T.reshape(16)

    # Helper bones (no skin influence) appended for contract
    bone_names = list(SMPLX_JOINT_NAMES)
    helper_local = {
        "Spine3_Breast_L": np.array([0.06, 0.04, 0.06], dtype=np.float32),
        "Spine3_Breast_R": np.array([-0.06, 0.04, 0.06], dtype=np.float32),
        "Pelvis_Glute_L": np.array([0.06, -0.02, -0.05], dtype=np.float32),
        "Pelvis_Glute_R": np.array([-0.06, -0.02, -0.05], dtype=np.float32),
    }
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    for hname, parent in HELPER_BONES:
        name_to_idx[hname] = len(bone_names)
        bone_names.append(hname)
        # extend ibm with identity-ish
        m = np.eye(4, dtype=np.float32)
        pidx = name_to_idx[parent]
        m[:3, 3] = -(rest_joints[pidx] + helper_local[hname])
        ibm = np.vstack([ibm, m.T.reshape(1, 16)])

    # Build binary
    blobs: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add_array(arr: np.ndarray, type_: str, component_type: int, target: int | None = None) -> int:
        data = arr.tobytes()
        byte_offset = sum(len(b) for b in blobs)
        pad = (4 - (byte_offset % 4)) % 4
        if pad:
            blobs.append(b"\x00" * pad)
            byte_offset += pad
        blobs.append(data)
        bv = len(views)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        acc = len(accessors)
        mins = maxs = None
        if arr.dtype == np.float32 and arr.ndim == 2 and arr.shape[1] in (2, 3, 4):
            mins = arr.min(axis=0).tolist()
            maxs = arr.max(axis=0).tolist()
        accessors.append(
            {
                "bufferView": bv,
                "byteOffset": 0,
                "componentType": component_type,
                "count": int(arr.shape[0]),
                "type": type_,
                **({"min": mins, "max": maxs} if mins is not None else {}),
            }
        )
        return acc

    acc_pos = add_array(vertices, "VEC3", 5126, 34962)
    acc_norm = add_array(normals.astype(np.float32), "VEC3", 5126, 34962)
    acc_uv = add_array(uvs, "VEC2", 5126, 34962)
    acc_col = add_array(colors, "VEC4", 5126, 34962)
    acc_part = add_array(part_ids.reshape(-1, 1), "SCALAR", 5126, 34962)
    acc_joints = add_array(joints_attr, "VEC4", 5123, 34962)
    acc_weights = add_array(weights_attr, "VEC4", 5126, 34962)
    # indices as uint32 for large mesh
    acc_idx = add_array(faces.reshape(-1).astype(np.uint32), "SCALAR", 5125, 34963)
    acc_ibm = add_array(ibm.astype(np.float32), "MAT4", 5126)
    morph_accs = [add_array(morphs[n], "VEC3", 5126, 34962) for n in MORPH_NAMES]

    bin_blob = _pad4(b"".join(blobs))

    # Nodes
    nodes: list[dict[str, Any]] = []
    children: dict[int, list[int]] = {i: [] for i in range(len(bone_names))}
    for j, name in enumerate(SMPLX_JOINT_NAMES):
        nodes.append({"name": name, "translation": local_t[j].tolist()})
        p = parents[j]
        if p >= 0:
            children[p].append(j)
    for hname, parent in HELPER_BONES:
        idx = name_to_idx[hname]
        nodes.append({"name": hname, "translation": helper_local[hname].tolist()})
        children[name_to_idx[parent]].append(idx)
    for i, kids in children.items():
        if kids and i < len(nodes):
            nodes[i]["children"] = kids

    mesh_node = len(nodes)
    nodes.append({"name": "H3DBody", "mesh": 0, "skin": 0})
    root_node = len(nodes)
    nodes.append({"name": "H3DRoot", "children": [0, mesh_node]})

    gltf = {
        "asset": {"version": "2.0", "generator": "h3d-smplx-real"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [root_node]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "SMPL-X",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": acc_pos,
                            "NORMAL": acc_norm,
                            "TEXCOORD_0": acc_uv,
                            "COLOR_0": acc_col,
                            "JOINTS_0": acc_joints,
                            "WEIGHTS_0": acc_weights,
                            "_PART_ID": acc_part,
                        },
                        "indices": acc_idx,
                        "mode": 4,
                        "targets": [{"POSITION": a} for a in morph_accs],
                    }
                ],
                "weights": [0.0] * len(MORPH_NAMES),
                "extras": {
                    "targetNames": MORPH_NAMES,
                    "h3d": {
                        "topology": "smplx",
                        "vertexCount": int(vertices.shape[0]),
                        "boneCount": len(bone_names),
                        "detailRegions": detail_regions or [],
                        "beta": [float(x) for x in np.asarray(betas).reshape(-1)[:10]],
                        **(extra_meta or {}),
                    },
                },
            }
        ],
        "skins": [
            {
                "name": "H3DSkin",
                "inverseBindMatrices": acc_ibm,
                "skeleton": 0,
                "joints": list(range(len(bone_names))),
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    glb = bytearray()
    glb += struct.pack("<III", 0x46546C67, 2, total)
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A)
    glb += json_bytes
    glb += struct.pack("<II", len(bin_blob), 0x004E4942)
    glb += bin_blob

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(glb)

    return {
        "path": str(out_path),
        "vertexCount": int(vertices.shape[0]),
        "triangleCount": int(faces.shape[0]),
        "bones": bone_names,
        "morphTargets": MORPH_NAMES,
        "partIdRange": [0, 7],
        "skinTone": list(skin_tone),
        "detailRegions": detail_regions or [],
        "topology": "smplx",
        "posedJoints": joints.tolist(),
        "hasNormalMap": normals_map is not None,
    }
