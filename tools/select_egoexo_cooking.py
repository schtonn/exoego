#!/usr/bin/env python3
"""Build reproducible Ego-Exo4D cooking selections from takes.json."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


DEFAULT_SOURCE = Path("datasets/EgoExo4D/v2/takes.json")
DEFAULT_OUTPUT = Path("datasets/EgoExo4D/selections")
PILOT_TASKS = (
    "Making Cucumber & Tomato Salad",
    "Cooking an Omelet",
    "Cooking Scrambled Eggs",
    "Making Coffee latte",
    "Making Milk Tea",
    "Cooking Tomato & Eggs",
)


def cameras(take: dict) -> tuple[list[str], list[str]]:
    names = (take.get("frame_aligned_videos") or {}).keys()
    ego = sorted(name for name in names if name.startswith("aria"))
    exo = sorted(name for name in names if name.startswith("cam"))
    return ego, exo


def task_coverage(take: dict) -> float:
    start = take.get("task_start_sec")
    end = take.get("task_end_sec")
    duration = take.get("duration_sec") or 0
    if start is None or end is None or duration <= 0:
        return 0.0
    return max(0.0, min(1.0, (end - start) / duration))


def geometry_ready(take: dict) -> bool:
    ego, exo = cameras(take)
    return bool(
        take.get("parent_task_name") == "Cooking"
        and not take.get("is_dropped")
        and take.get("validated")
        and take.get("has_trimmed_trajectory")
        and take.get("best_exo")
        and ego
        and exo
    )


def pilot_eligible(take: dict) -> bool:
    duration = take.get("duration_sec") or 0
    return bool(
        geometry_ready(take)
        and take.get("task_name") in PILOT_TASKS
        and 120 <= duration <= 1200
        and task_coverage(take) >= 0.9
        and take.get("objects")
    )


def select_diverse_short_takes(takes: list[dict], per_task: int = 4) -> list[dict]:
    """Round-robin universities, taking the shortest complete clips first."""
    selected: list[dict] = []
    for task in PILOT_TASKS:
        by_university: dict[str, deque[dict]] = defaultdict(deque)
        candidates = sorted(
            (take for take in takes if take.get("task_name") == task and pilot_eligible(take)),
            key=lambda take: (take["duration_sec"], take["take_uid"]),
        )
        for take in candidates:
            by_university[take["university_name"]].append(take)
        universities = sorted(
            by_university,
            key=lambda name: (by_university[name][0]["duration_sec"], name),
        )
        while len([take for take in selected if take["task_name"] == task]) < per_task:
            made_progress = False
            for university in universities:
                if by_university[university]:
                    selected.append(by_university[university].popleft())
                    made_progress = True
                    if len([take for take in selected if take["task_name"] == task]) == per_task:
                        break
            if not made_progress:
                raise RuntimeError(f"Not enough eligible takes for {task!r}")
    return selected


def write_selection(name: str, takes: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    uid_path = output_dir / f"{name}_uids.txt"
    csv_path = output_dir / f"{name}.csv"
    uid_path.write_text("\n".join(take["take_uid"] for take in takes) + "\n", encoding="utf-8")
    fields = (
        "take_uid",
        "take_name",
        "task_name",
        "university_name",
        "duration_sec",
        "task_coverage",
        "ego_cameras",
        "exo_cameras",
        "best_exo",
        "capture_uid",
        "object_count",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for take in takes:
            ego, exo = cameras(take)
            writer.writerow(
                {
                    "take_uid": take["take_uid"],
                    "take_name": take["take_name"],
                    "task_name": take["task_name"],
                    "university_name": take["university_name"],
                    "duration_sec": f"{take['duration_sec']:.3f}",
                    "task_coverage": f"{task_coverage(take):.4f}",
                    "ego_cameras": ";".join(ego),
                    "exo_cameras": ";".join(exo),
                    "best_exo": take["best_exo"],
                    "capture_uid": take["capture_uid"],
                    "object_count": len(take.get("objects") or []),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=4)
    args = parser.parse_args()
    takes = json.loads(args.source.read_text(encoding="utf-8"))
    ready = sorted((take for take in takes if geometry_ready(take)), key=lambda take: take["take_uid"])
    pilot = select_diverse_short_takes(takes, args.per_task)
    write_selection("cooking_geometry_ready", ready, args.output_dir)
    write_selection("cooking_pilot_24", pilot, args.output_dir)
    print(f"Cooking geometry-ready: {len(ready)} takes, {sum(t['duration_sec'] for t in ready)/3600:.2f} h")
    print(f"Pilot: {len(pilot)} takes, {sum(t['duration_sec'] for t in pilot)/3600:.2f} h")
    for task in PILOT_TASKS:
        chosen = [take for take in pilot if take["task_name"] == task]
        print(f"  {task}: {len(chosen)} takes, {sum(t['duration_sec'] for t in chosen)/60:.1f} min")


if __name__ == "__main__":
    main()
