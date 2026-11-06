"""Generate a demo-valid H3D GLB matching the TDD asset contract.

Produces a skinned humanoid (simplified topology, not full SMPL-X 10,475 verts),
with 59 joints (55 + 4 helpers), morph targets, and `_PART_ID` vertex attribute.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .bones import BONE_DEFS, MORPH_NAMES

# Part IDs per TDD §3.4
PART_HEAD = 0
PART_TORSO = 1
PART_ARMS_UPPER = 2
PART_ARMS_LOWER = 3
PART_HANDS = 4
PART_THIGHS = 5
PART_CALVES = 6
PART_FEET = 7


def _ellipsoid(
    center: np.ndarray,
    radii: tuple[float, float, float],
    seg_u: int,
    seg_v: int,
    part_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, normals, uvs, part_ids, indices for a UV-sphere ellipsoid."""
    us = np.linspace(0, 2 * np.pi, seg_u, endpoint=False)
    vs = np.linspace(0, np.pi, seg_v)
    positions = []
    normals = []
    uvs = []
    parts = []
    for iv, v in enumerate(vs):
        for iu, u in enumerate(us):
            x = radii[0] * np.sin(v) * np.cos(u)
            y = radii[1] * np.cos(v)
            z = radii[2] * np.sin(v) * np.sin(u)
            p = center + np.array([x, y, z], dtype=np.float32)
            n = np.array([x / radii[0], y / radii[1], z / radii[2]], dtype=np.float32)
            n /= max(np.linalg.norm(n), 1e-6)
            positions.append(p)
            normals.append(n)
            uvs.append([iu / seg_u, iv / max(seg_v - 1, 1)])
            parts.append(part_id)

    positions_a = np.asarray(positions, dtype=np.float32)
    normals_a = np.asarray(normals, dtype=np.float32)
    uvs_a = np.asarray(uvs, dtype=np.float32)
    parts_a = np.asarray(parts, dtype=np.float32)

    indices = []
    for iv in range(seg_v - 1):
        for iu in range(seg_u):
            i0 = iv * seg_u + iu
            i1 = iv * seg_u + (iu + 1) % seg_u
            i2 = (iv + 1) * seg_u + iu
            i3 = (iv + 1) * seg_u + (iu + 1) % seg_u
            indices.extend([i0, i2, i1, i1, i2, i3])
    return positions_a, normals_a, uvs_a, parts_a, np.asarray(indices, dtype=np.uint16)


def _build_mesh(
    beta: list[float], skin_tone: tuple[float, float, float]
) -> dict[str, Any]:
    """Assemble a capsule-humanoid and skin weights."""
    # Shape-ish scaling from beta[0]/height-ish), beta[1] (girth)
    height_s = 1.0 + 0.05 * float(beta[0] if beta else 0.0)
    girth_s = 1.0 + 0.08 * float(beta[1] if len(beta) > 1 else 0.0)

    segments: list[tuple[np.ndarray, tuple[float, float, float], int, int, str]] = [
        # center, radii, seg_u, seg_v, bone_name, part
    ]

    # We'll define segments with bone binding
    defs = [
        ((0.0, 1.55 * height_s, 0.0), (0.09, 0.11, 0.1), 10, 8, "Head", PART_HEAD),
        ((0.0, 1.38 * height_s, 0.0), (0.05, 0.05, 0.05), 8, 6, "Neck", PART_HEAD),
        ((0.0, 1.18 * height_s, 0.0), (0.14 * girth_s, 0.16, 0.1 * girth_s), 12, 10, "Spine3", PART_TORSO),
        ((0.0, 1.0 * height_s, 0.0), (0.13 * girth_s, 0.12, 0.09 * girth_s), 12, 8, "Spine2", PART_TORSO),
        ((0.0, 0.88 * height_s, 0.0), (0.12 * girth_s, 0.1, 0.08 * girth_s), 12, 8, "Spine1", PART_TORSO),
        ((0.0, 0.95 * height_s, 0.0), (0.14 * girth_s, 0.08, 0.1 * girth_s), 12, 8, "Pelvis", PART_TORSO),
        # arms
        ((0.28, 1.28 * height_s, 0.0), (0.05, 0.12, 0.05), 8, 6, "LeftShoulder", PART_ARMS_UPPER),
        ((0.5, 1.28 * height_s, 0.0), (0.045, 0.11, 0.045), 8, 6, "LeftElbow", PART_ARMS_LOWER),
        ((0.7, 1.28 * height_s, 0.0), (0.035, 0.05, 0.05), 8, 6, "LeftWrist", PART_HANDS),
        ((-0.28, 1.28 * height_s, 0.0), (0.05, 0.12, 0.05), 8, 6, "RightShoulder", PART_ARMS_UPPER),
        ((-0.5, 1.28 * height_s, 0.0), (0.045, 0.11, 0.045), 8, 6, "RightElbow", PART_ARMS_LOWER),
        ((-0.7, 1.28 * height_s, 0.0), (0.035, 0.05, 0.05), 8, 6, "RightWrist", PART_HANDS),
        # legs
        ((0.1, 0.6 * height_s, 0.0), (0.07 * girth_s, 0.18, 0.07 * girth_s), 10, 8, "LeftHip", PART_THIGHS),
        ((0.1, 0.28 * height_s, 0.0), (0.055, 0.16, 0.055), 10, 8, "LeftKnee", PART_CALVES),
        ((0.1, 0.04, 0.04), (0.05, 0.03, 0.09), 8, 6, "LeftFoot", PART_FEET),
        ((-0.1, 0.6 * height_s, 0.0), (0.07 * girth_s, 0.18, 0.07 * girth_s), 10, 8, "RightHip", PART_THIGHS),
        ((-0.1, 0.28 * height_s, 0.0), (0.055, 0.16, 0.055), 10, 8, "RightKnee", PART_CALVES),
        ((-0.1, 0.04, 0.04), (0.05, 0.03, 0.09), 8, 6, "RightFoot", PART_FEET),
    ]

    all_pos = []
    all_norm = []
    all_uv = []
    all_part = []
    all_idx = []
    joints0 = []
    joints1 = []
    weights0 = []
    weights1 = []
    bone_index = {name: i for i, (name, _, _) in enumerate(BONE_DEFS)}
    offset = 0

    for center, radii, su, sv, bone_name, part in defs:
        pos, norm, uv, part_ids, idx = _ellipsoid(
            np.array(center, dtype=np.float32), radii, su, sv, part
        )
        all_pos.append(pos)
        all_norm.append(norm)
        all_uv.append(uv)
        all_part.append(part_ids)
        all_idx.append(idx + offset)
        j = bone_index[bone_name]
        # secondary bone: nearby spine/parent for soft blend
        j2 = bone_index.get("Spine3", j) if part == PART_TORSO else j
        if bone_name.startswith("Left") and "Hip" in bone_name:
            j2 = bone_index["Pelvis"]
        n = len(pos)
        joints0.append(np.full(n, j, dtype=np.uint16))
        joints1.append(np.full(n, j2, dtype=np.uint16))
        weights0.append(np.full(n, 0.85, dtype=np.float32))
        weights1.append(np.full(n, 0.15, dtype=np.float32))
        offset += n

    positions = np.concatenate(all_pos, axis=0)
    normals = np.concatenate(all_norm, axis=0)
    uvs = np.concatenate(all_uv, axis=0)
    part_ids = np.concatenate(all_part, axis=0)
    indices = np.concatenate(all_idx, axis=0)
    j0 = np.concatenate(joints0)
    j1 = np.concatenate(joints1)
    w0 = np.concatenate(weights0)
    w1 = np.concatenate(weights1)

    # joints/weights as VEC4
    n_verts = len(positions)
    joints = np.zeros((n_verts, 4), dtype=np.uint16)
    weights = np.zeros((n_verts, 4), dtype=np.float32)
    joints[:, 0] = j0
    joints[:, 1] = j1
    weights[:, 0] = w0
    weights[:, 1] = w1

    # Morph target deltas
    morphs: dict[str, np.ndarray] = {}
    for name in MORPH_NAMES:
        delta = np.zeros_like(positions)
        if name == "breast_scale":
            mask = (positions[:, 1] > 1.1) & (np.abs(positions[:, 0]) < 0.2) & (positions[:, 2] > 0)
            delta[mask, 2] += 0.08
            delta[mask, 0] *= 0.02
        elif name == "waist_scale":
            mask = (positions[:, 1] > 0.9) & (positions[:, 1] < 1.1)
            delta[mask, 0] = positions[mask, 0] * 0.25
            delta[mask, 2] = positions[mask, 2] * 0.25
        elif name == "glute_scale":
            mask = (positions[:, 1] > 0.85) & (positions[:, 1] < 1.0) & (positions[:, 2] < 0)
            delta[mask, 2] -= 0.06
        elif name == "thigh_scale":
            mask = (part_ids == PART_THIGHS)
            delta[mask, 0] = positions[mask, 0] * 0.2
            delta[mask, 2] = positions[mask, 2] * 0.2
        elif name == "mouthSmile":
            mask = (part_ids == PART_HEAD) & (positions[:, 1] < 1.52) & (positions[:, 2] > 0.05)
            delta[mask, 0] = np.sign(positions[mask, 0]) * 0.015
            delta[mask, 1] += 0.01
        elif name == "jawOpen":
            mask = (part_ids == PART_HEAD) & (positions[:, 1] < 1.5)
            delta[mask, 1] -= 0.04
            delta[mask, 2] += 0.01
        elif name == "eyeBlinkLeft":
            mask = (part_ids == PART_HEAD) & (positions[:, 0] > 0.01) & (positions[:, 1] > 1.55)
            delta[mask, 1] -= 0.01
        elif name == "eyeBlinkRight":
            mask = (part_ids == PART_HEAD) & (positions[:, 0] < -0.01) & (positions[:, 1] > 1.55)
            delta[mask, 1] -= 0.01
        elif name == "browInnerUp":
            mask = (part_ids == PART_HEAD) & (positions[:, 1] > 1.58) & (np.abs(positions[:, 0]) < 0.05)
            delta[mask, 1] += 0.015
        morphs[name] = delta.astype(np.float32)

    # Vertex colors from skin tone (slight underwear tint on torso lower)
    colors = np.tile(np.array([*skin_tone, 1.0], dtype=np.float32), (n_verts, 1))
    underwear = (part_ids == PART_TORSO) & (positions[:, 1] < 1.05) & (positions[:, 1] > 0.9)
    colors[underwear, :3] = np.array([0.15, 0.15, 0.2], dtype=np.float32)

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "colors": colors,
        "part_ids": part_ids.reshape(-1, 1).astype(np.float32),
        "indices": indices,
        "joints": joints,
        "weights": weights,
        "morphs": morphs,
        "skin_tone": skin_tone,
    }


def _pad4(b: bytes, pad_byte: bytes = b"\x00") -> bytes:
    """Pad to 4-byte alignment. JSON chunks must use 0x20; BIN uses 0x00."""
    pad = (4 - (len(b) % 4)) % 4
    return b + (pad_byte * pad)


def _accessor(
    buffer_view: int,
    byte_offset: int,
    component_type: int,
    count: int,
    type_: str,
    mins: list[float] | None = None,
    maxs: list[float] | None = None,
) -> dict[str, Any]:
    acc: dict[str, Any] = {
        "bufferView": buffer_view,
        "byteOffset": byte_offset,
        "componentType": component_type,
        "count": count,
        "type": type_,
    }
    if mins is not None:
        acc["min"] = mins
    if maxs is not None:
        acc["max"] = maxs
    return acc


def write_mock_glb(
    out_path: Path,
    beta: list[float] | None = None,
    skin_tone: tuple[float, float, float] = (0.78, 0.62, 0.52),
    detail_regions: list[str] | None = None,
) -> dict[str, Any]:
    beta = beta or [0.0] * 10
    mesh = _build_mesh(beta, skin_tone)

    # Build inverse bind matrices and node hierarchy
    bone_names = [b[0] for b in BONE_DEFS]
    parent_of = {name: parent for name, parent, _ in BONE_DEFS}
    local_t = {name: np.array(t, dtype=np.float64) for name, _, t in BONE_DEFS}

    # World transforms
    world: dict[str, np.ndarray] = {}

    def world_pos(name: str) -> np.ndarray:
        if name in world:
            return world[name]
        p = parent_of[name]
        if p is None:
            world[name] = local_t[name].copy()
        else:
            world[name] = world_pos(p) + local_t[name]
        return world[name]

    for n in bone_names:
        world_pos(n)

    # Inverse bind: translate -world
    ibm = np.zeros((len(bone_names), 16), dtype=np.float32)
    for i, name in enumerate(bone_names):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -world[name].astype(np.float32)
        ibm[i] = m.T.reshape(16)  # column-major

    # Pack binary blob
    blobs: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add_array(arr: np.ndarray, type_: str, component_type: int, target: int | None = None) -> int:
        data = arr.tobytes()
        byte_offset = sum(len(b) for b in blobs)
        # align
        pad = (4 - (byte_offset % 4)) % 4
        if pad:
            blobs.append(b"\x00" * pad)
            byte_offset += pad
        blobs.append(data)
        bv_index = len(views)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        acc_index = len(accessors)
        mins = maxs = None
        if arr.ndim == 2 and arr.shape[1] in (2, 3) and arr.dtype == np.float32:
            mins = arr.min(axis=0).tolist()
            maxs = arr.max(axis=0).tolist()
        elif arr.ndim == 1 and arr.dtype == np.float32:
            mins = [float(arr.min())]
            maxs = [float(arr.max())]
        accessors.append(
            _accessor(bv_index, 0, component_type, arr.shape[0], type_, mins, maxs)
        )
        return acc_index

    # FLOAT=5126, UNSIGNED_SHORT=5123
    acc_pos = add_array(mesh["positions"], "VEC3", 5126, 34962)
    acc_norm = add_array(mesh["normals"], "VEC3", 5126, 34962)
    acc_uv = add_array(mesh["uvs"], "VEC2", 5126, 34962)
    acc_col = add_array(mesh["colors"], "VEC4", 5126, 34962)
    acc_part = add_array(mesh["part_ids"], "SCALAR", 5126, 34962)
    acc_joints = add_array(mesh["joints"], "VEC4", 5123, 34962)
    acc_weights = add_array(mesh["weights"], "VEC4", 5126, 34962)
    acc_idx = add_array(mesh["indices"], "SCALAR", 5123, 34963)
    acc_ibm = add_array(ibm, "MAT4", 5126)

    morph_accs = []
    for name in MORPH_NAMES:
        morph_accs.append(add_array(mesh["morphs"][name], "VEC3", 5126, 34962))

    bin_blob = b"".join(blobs)
    bin_blob = _pad4(bin_blob)

    # Nodes: bone nodes first, then mesh node, then root scene node
    nodes: list[dict[str, Any]] = []
    children_map: dict[str, list[int]] = {n: [] for n in bone_names}
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    for name, parent, t in BONE_DEFS:
        nodes.append({"name": name, "translation": list(t)})
        if parent is not None:
            children_map[parent].append(name_to_idx[name])
    for name, kids in children_map.items():
        if kids:
            nodes[name_to_idx[name]]["children"] = kids

    mesh_node_index = len(nodes)
    nodes.append(
        {
            "name": "H3DBody",
            "mesh": 0,
            "skin": 0,
        }
    )
    root_index = len(nodes)
    # Scene root holds pelvis + skinned mesh
    nodes.append(
        {
            "name": "H3DRoot",
            "children": [name_to_idx["Pelvis"], mesh_node_index],
        }
    )

    gltf = {
        "asset": {"version": "2.0", "generator": "h3d-mock-glb"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [root_index]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "H3DMesh",
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
                        "topology": "mock-smplx-like",
                        "vertexCount": int(mesh["positions"].shape[0]),
                        "boneCount": len(bone_names),
                        "detailRegions": detail_regions or [],
                        "beta": beta,
                    },
                },
            }
        ],
        "skins": [
            {
                "name": "H3DSkin",
                "inverseBindMatrices": acc_ibm,
                "skeleton": name_to_idx["Pelvis"],
                "joints": [name_to_idx[n] for n in bone_names],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    # glTF 2.0: JSON chunk padded with spaces (0x20); BIN already null-padded above.
    json_bytes = _pad4(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad_byte=b" "
    )
    # GLB: header + JSON chunk + BIN chunk
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    glb = bytearray()
    glb += struct.pack("<III", 0x46546C67, 2, total_len)
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A)  # JSON
    glb += json_bytes
    glb += struct.pack("<II", len(bin_blob), 0x004E4942)  # BIN
    glb += bin_blob

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(glb)

    return {
        "path": str(out_path),
        "vertexCount": int(mesh["positions"].shape[0]),
        "triangleCount": int(mesh["indices"].shape[0] // 3),
        "bones": bone_names,
        "morphTargets": MORPH_NAMES,
        "partIdRange": [0, 7],
        "skinTone": list(skin_tone),
        "detailRegions": detail_regions or [],
    }
