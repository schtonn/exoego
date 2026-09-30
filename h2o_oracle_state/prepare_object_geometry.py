#!/usr/bin/env python3
"""Audit H2O object meshes and create deterministic area-uniform surface samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from h2o_oracle_state import DEFAULT_ROOT, OBJECTS


DEFAULT_OUTPUT = Path("datasets/H2O/oracle_state/object_geometry")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                polygon = [int(token.split("/")[0]) for token in line.split()[1:]]
                polygon = [index - 1 if index > 0 else len(vertices) + index for index in polygon]
                for offset in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable triangle geometry: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def edge_audit(faces: np.ndarray) -> tuple[int, int]:
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in faces:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            counts[tuple(sorted((int(first), int(second))))] += 1
    return sum(count == 1 for count in counts.values()), sum(count > 2 for count in counts.values())


def surface_samples(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangle_vertices = vertices[faces]
    cross = np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    if not np.any(valid):
        raise ValueError("Mesh contains no non-degenerate triangle")
    probabilities = np.where(valid, double_area, 0.0)
    probabilities /= probabilities.sum()
    generator = np.random.default_rng(seed)
    face_indices = generator.choice(len(faces), size=count, p=probabilities)
    selected = triangle_vertices[face_indices]
    uv = generator.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    points = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (
        selected[:, 2] - selected[:, 0]
    )
    normals = cross[face_indices] / double_area[face_indices, None]
    return points, normals, face_indices, double_area * 0.5


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for object_id, (object_name, relative_path) in sorted(OBJECTS.items()):
        mesh_path = args.raw_root / "object" / relative_path
        vertices, faces = read_obj(mesh_path)
        points, normals, face_indices, triangle_area = surface_samples(
            vertices, faces, args.sample_count, args.seed + object_id
        )
        boundary_edges, nonmanifold_edges = edge_audit(faces)
        watertight = boundary_edges == 0 and nonmanifold_edges == 0
        degenerate_faces = int(np.count_nonzero(triangle_area <= 1e-12))
        output_path = args.output / f"{object_name}_surface_{args.sample_count}.npz"
        np.savez_compressed(
            output_path,
            object_id=np.asarray(object_id, dtype=np.int16),
            object_name=np.asarray(object_name),
            points_object_m=points.astype(np.float32),
            normals_object=normals.astype(np.float32),
            sampled_face_index=face_indices.astype(np.int32),
            vertices_object_m=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
        )
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        rows.append(
            {
                "object_id": object_id,
                "object_name": object_name,
                "mesh_path": str(mesh_path),
                "vertex_count": len(vertices),
                "triangle_count": len(faces),
                "surface_sample_count": args.sample_count,
                "extent_x_m": float(extent[0]),
                "extent_y_m": float(extent[1]),
                "extent_z_m": float(extent[2]),
                "surface_area_m2": float(triangle_area.sum()),
                "degenerate_face_count": degenerate_faces,
                "boundary_edge_count": boundary_edges,
                "nonmanifold_edge_count": nonmanifold_edges,
                "watertight": int(watertight),
                "signed_distance_candidate": int(watertight),
                "sample_path": str(output_path),
            }
        )

    csv_path = args.output / "mesh_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mesh_count": len(rows),
        "surface_sample_count_per_mesh": args.sample_count,
        "sampling": "deterministic triangle-area-uniform barycentric sampling",
        "normal_semantics": "OBJ face orientation; outward direction not independently verified",
        "watertight_count": sum(row["watertight"] for row in rows),
        "watertight_objects": [row["object_name"] for row in rows if row["watertight"]],
        "non_watertight_objects": [row["object_name"] for row in rows if not row["watertight"]],
        "signed_distance_policy": "Watertight edge topology is necessary but not sufficient. Validate orientation and self-intersections before signed SDF; use unsigned surface distance otherwise.",
        "mesh_audit_csv": str(csv_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
