#!/usr/bin/env python3
"""Plan, download, extract, and verify the official ETH H2O dataset.

Credentials are read from H2O_USERNAME/H2O_PASSWORD or prompted for. They are
never written to disk. Downloads use .part files and HTTP Range requests, so an
interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://h2odataset.ethz.ch/data/dataset"
COMMON_FILES = ("object.zip", "label_split.zip")
MODE_FILES = {
    "all": tuple(f"subject{i}_v1_1.tar.gz" for i in range(1, 5)),
    "ego": tuple(f"subject{i}_ego_v1_1.tar.gz" for i in range(1, 5)),
    "pose": tuple(f"subject{i}_pose_v1_1.tar.gz" for i in range(1, 5)),
}
DEFAULT_ROOT = Path("datasets/H2O")
CHUNK_SIZE = 1 * 1024 * 1024
READ_TIMEOUT_SECONDS = 60


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def build_session(username: str, password: str) -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("HEAD", "GET")),
    )
    session = requests.Session()
    session.auth = (username, password)
    session.headers["User-Agent"] = "H2O-research-downloader/1.0"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def credentials() -> tuple[str, str]:
    username = os.environ.get("H2O_USERNAME")
    password = os.environ.get("H2O_PASSWORD")
    if not username and sys.stdin.isatty():
        username = input("H2O username: ").strip()
    if not password and sys.stdin.isatty():
        password = getpass.getpass("H2O password: ")
    if not username or not password:
        raise SystemExit(
            "Set H2O_USERNAME and H2O_PASSWORD, or run this command in an "
            "interactive terminal. Credentials are emailed after registration."
        )
    return username, password


def selected_files(mode: str) -> tuple[str, ...]:
    return COMMON_FILES + MODE_FILES[mode]


def remote_size(session: requests.Session, filename: str) -> int:
    url = f"{BASE_URL}/{filename}"
    response = session.head(url, allow_redirects=True, timeout=(30, 120))
    if response.status_code == 401:
        raise RuntimeError("H2O credentials were rejected (HTTP 401)")
    response.raise_for_status()
    length = response.headers.get("content-length")
    if not length:
        raise RuntimeError(f"Server did not report the size of {filename}")
    return int(length)


def build_plan(
    session: requests.Session, filenames: Iterable[str], archive_dir: Path
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    print("Remote files:")
    for filename in filenames:
        size = remote_size(session, filename)
        sizes[filename] = size
        local = archive_dir / filename
        part = local.with_name(local.name + ".part")
        if local.exists() and local.stat().st_size == size:
            state = "complete"
        elif part.exists():
            state = f"partial: {human_size(part.stat().st_size)}"
        else:
            state = "missing"
        print(f"  {filename:34} {human_size(size):>12}  {state}")

    total = sum(sizes.values())
    remaining = sum(
        max(0, size - (archive_dir / f"{name}.part").stat().st_size)
        if (archive_dir / f"{name}.part").exists()
        else (0 if (archive_dir / name).exists() and (archive_dir / name).stat().st_size == size else size)
        for name, size in sizes.items()
    )
    free = shutil.disk_usage(archive_dir).free
    print(f"Total compressed size: {human_size(total)}")
    print(f"Remaining download:     {human_size(remaining)}")
    print(f"Free disk space:        {human_size(free)}")
    if remaining > free:
        raise RuntimeError("Not enough free disk space for the remaining archives")
    return sizes


def download_file(
    session: requests.Session,
    filename: str,
    expected_size: int,
    archive_dir: Path,
) -> None:
    destination = archive_dir / filename
    partial = archive_dir / f"{filename}.part"
    if destination.exists():
        actual = destination.stat().st_size
        if actual == expected_size:
            print(f"[skip] {filename} ({human_size(actual)})")
            return
        destination.replace(partial)

    url = f"{BASE_URL}/{filename}"
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            partial.replace(destination)
            print(f"[done] {filename} ({human_size(expected_size)})")
            return
        if offset > expected_size:
            raise RuntimeError(
                f"Partial file is larger than the server copy: {partial}"
            )

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, READ_TIMEOUT_SECONDS),
            ) as response:
                if response.status_code == 401:
                    raise RuntimeError("H2O credentials were rejected (HTTP 401)")
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    print(f"[restart] Server ignored Range for {filename}")
                    offset = 0

                completed = offset
                last_report = time.monotonic()
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        completed += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            pct = 100 * completed / expected_size
                            print(
                                f"[recv] {filename}: {human_size(completed)} / "
                                f"{human_size(expected_size)} ({pct:.1f}%)",
                                flush=True,
                            )
                            last_report = now

            actual = partial.stat().st_size
            if actual == expected_size:
                partial.replace(destination)
                print(f"[done] {filename} ({human_size(actual)})")
                return
            if actual > expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {filename}: {actual} > {expected_size}"
                )
            failures += 1
            print(f"[resume] {filename} stopped at {human_size(actual)}")
        except (requests.RequestException, OSError) as exc:
            # Count consecutive failures without progress. A long transfer may
            # reconnect many times, but successful new bytes should reset the
            # retry budget.
            progressed = partial.exists() and partial.stat().st_size > offset
            failures = 1 if progressed else failures + 1
            print(f"[retry {failures}/10] {filename}: {exc}", file=sys.stderr)

        if failures >= 10:
            raise RuntimeError(f"Download repeatedly failed for {filename}")
        time.sleep(min(60, 2**failures))


def download_segment(
    username: str,
    password: str,
    filename: str,
    expected_size: int,
    base_offset: int,
    segment_index: int,
    segment_count: int,
    start: int,
    end: int,
    segment_paths: list[Path],
) -> None:
    segment_path = segment_paths[segment_index]
    segment_size = end - start + 1
    failures = 0
    url = f"{BASE_URL}/{filename}"

    while True:
        present = segment_path.stat().st_size if segment_path.exists() else 0
        if present == segment_size:
            return
        if present > segment_size:
            raise RuntimeError(f"Oversized segment: {segment_path}")

        attempt_start_present = present
        range_start = start + present
        try:
            session = build_session(username, password)
            with session.get(
                url,
                headers={"Range": f"bytes={range_start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(30, READ_TIMEOUT_SECONDS),
            ) as response:
                if response.status_code == 401:
                    raise RuntimeError("H2O credentials were rejected (HTTP 401)")
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored byte range for {filename}: HTTP {response.status_code}"
                    )

                last_report = time.monotonic()
                with segment_path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        remaining = segment_size - present
                        if len(chunk) > remaining:
                            chunk = chunk[:remaining]
                        handle.write(chunk)
                        present += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            downloaded = base_offset + sum(
                                min(path.stat().st_size, range_end - range_start_ + 1)
                                if path.exists()
                                else 0
                                for path, (range_start_, range_end) in zip(
                                    segment_paths,
                                    split_ranges(base_offset, expected_size, segment_count),
                                )
                            )
                            pct = 100 * downloaded / expected_size
                            print(
                                f"[recv] {filename} segment {segment_index + 1}/{segment_count}: "
                                f"{human_size(downloaded)} / {human_size(expected_size)} "
                                f"({pct:.1f}%)",
                                flush=True,
                            )
                            last_report = now
                        if present == segment_size:
                            break

            if present == segment_size:
                return
            failures = 1
        except (requests.RequestException, OSError) as exc:
            after = segment_path.stat().st_size if segment_path.exists() else 0
            failures = 1 if after > attempt_start_present else failures + 1
            print(
                f"[retry {failures}/10] {filename} segment "
                f"{segment_index + 1}/{segment_count}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if failures >= 10:
            raise RuntimeError(
                f"Segment repeatedly failed without progress: {filename} "
                f"{segment_index + 1}/{segment_count}"
            )
        time.sleep(min(30, 2**failures))


def split_even_ranges(start: int, end_exclusive: int, count: int) -> list[tuple[int, int]]:
    remaining = end_exclusive - start
    count = max(1, min(count, remaining))
    base, extra = divmod(remaining, count)
    ranges = []
    cursor = start
    for index in range(count):
        length = base + (1 if index < extra else 0)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def split_ranges(start: int, end_exclusive: int, count: int) -> list[tuple[int, int]]:
    # Sixteen ranges are defined by bisecting the established eight-range
    # layout. This lets an in-progress eight-range download migrate without
    # moving or redownloading any bytes.
    if count == 16:
        coarse = split_even_ranges(start, end_exclusive, 8)
        return [
            fine
            for coarse_start, coarse_end in coarse
            for fine in split_even_ranges(coarse_start, coarse_end + 1, 2)
        ]
    return split_even_ranges(start, end_exclusive, count)


def download_file_segmented(
    username: str,
    password: str,
    filename: str,
    expected_size: int,
    archive_dir: Path,
    segment_count: int,
    active_connections: int,
) -> None:
    destination = archive_dir / filename
    partial = archive_dir / f"{filename}.part"
    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"[skip] {filename} ({human_size(expected_size)})")
        return
    if destination.exists():
        if partial.exists():
            raise RuntimeError(f"Both incomplete destination and partial exist: {filename}")
        destination.replace(partial)

    base_offset = partial.stat().st_size if partial.exists() else 0
    if base_offset == expected_size:
        partial.replace(destination)
        print(f"[done] {filename} ({human_size(expected_size)})")
        return
    if base_offset > expected_size:
        raise RuntimeError(f"Partial file is larger than server copy: {partial}")

    ranges = split_ranges(base_offset, expected_size, segment_count)
    segment_paths = [
        archive_dir / f"{filename}.part.seg{index:02d}-{start}-{end}"
        for index, (start, end) in enumerate(ranges)
    ]
    if segment_count == 16:
        old_ranges = split_even_ranges(base_offset, expected_size, 8)
        old_paths = [
            archive_dir / f"{filename}.part.seg{index:02d}-{start}-{end}"
            for index, (start, end) in enumerate(old_ranges)
        ]
        for old_index, (old_path, new_path) in enumerate(
            zip(old_paths, segment_paths[::2])
        ):
            if not old_path.exists() or old_path == new_path:
                continue
            if new_path.exists():
                raise RuntimeError(
                    f"Both old and migrated segment exist for {filename} range {old_index}"
                )
            new_range = ranges[2 * old_index]
            if old_path.stat().st_size > new_range[1] - new_range[0] + 1:
                raise RuntimeError(
                    f"Existing segment has crossed the new split point: {old_path}"
                )
            old_path.replace(new_path)
        print(f"[migrate] {filename}: 8-range layout -> 16-range layout", flush=True)
    print(
        f"[segments] {filename}: preserving {human_size(base_offset)} prefix, "
        f"downloading the remaining {human_size(expected_size - base_offset)} "
        f"as {len(ranges)} ranges with {active_connections} active connections",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(active_connections, len(ranges))) as pool:
        futures = [
            pool.submit(
                download_segment,
                username,
                password,
                filename,
                expected_size,
                base_offset,
                index,
                len(ranges),
                start,
                end,
                segment_paths,
            )
            for index, (start, end) in enumerate(ranges)
        ]
        for future in as_completed(futures):
            future.result()

    merge_path = archive_dir / f"{filename}.merge"
    merge_path.unlink(missing_ok=True)
    with merge_path.open("wb") as output:
        if partial.exists():
            with partial.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=CHUNK_SIZE)
        for segment_path in segment_paths:
            with segment_path.open("rb") as segment:
                shutil.copyfileobj(segment, output, length=CHUNK_SIZE)
    if merge_path.stat().st_size != expected_size:
        raise RuntimeError(f"Merged size mismatch for {filename}")
    merge_path.replace(destination)
    partial.unlink(missing_ok=True)
    for segment_path in segment_paths:
        segment_path.unlink(missing_ok=True)
    print(f"[done] {filename} ({human_size(expected_size)})", flush=True)


def download_files(
    username: str,
    password: str,
    sizes: dict[str, int],
    archive_dir: Path,
    jobs: int,
    connections_per_file: int,
    segments_per_file: int,
) -> None:
    if jobs == 1:
        for filename, size in sizes.items():
            if segments_per_file == 1:
                download_file(build_session(username, password), filename, size, archive_dir)
            else:
                download_file_segmented(
                    username,
                    password,
                    filename,
                    size,
                    archive_dir,
                    segments_per_file,
                    connections_per_file,
                )
        return

    def run_one(filename: str, size: int) -> None:
        if segments_per_file == 1:
            download_file(build_session(username, password), filename, size, archive_dir)
        else:
            download_file_segmented(
                username,
                password,
                filename,
                size,
                archive_dir,
                segments_per_file,
                connections_per_file,
            )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_one, filename, size): filename
            for filename, size in sizes.items()
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Download failed for {filename}: {exc}") from exc


def extract_archive(archive: Path, raw_dir: Path) -> None:
    marker_dir = raw_dir / ".extracted"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{archive.name}.done"
    if marker.exists():
        print(f"[skip] already extracted: {archive.name}")
        return

    print(f"[extract] {archive.name}")
    if archive.name.endswith(".tar.gz"):
        command = ["tar", "-xzf", str(archive), "-C", str(raw_dir)]
    elif archive.suffix == ".zip":
        command = ["unzip", "-q", "-o", str(archive), "-d", str(raw_dir)]
    else:
        raise RuntimeError(f"Unsupported archive: {archive}")
    subprocess.run(command, check=True)
    marker.touch()


def verify(mode: str, root: Path, sizes: dict[str, int] | None = None) -> bool:
    archive_dir = root / "archives"
    raw_dir = root / "raw"
    ok = True

    print("Archive verification:")
    for filename in selected_files(mode):
        path = archive_dir / filename
        if not path.exists():
            print(f"  [missing] {path}")
            ok = False
            continue
        actual = path.stat().st_size
        if sizes and actual != sizes[filename]:
            print(f"  [bad size] {filename}: {actual} != {sizes[filename]}")
            ok = False
        else:
            print(f"  [ok] {filename}: {human_size(actual)}")

    if raw_dir.exists():
        subjects = sorted(p.name for p in raw_dir.glob("subject[1-4]") if p.is_dir())
        objects = raw_dir / "object"
        print(f"Extracted subjects: {', '.join(subjects) if subjects else 'none'}")
        print(f"Object meshes:      {'present' if objects.is_dir() else 'missing'}")
        if mode == "all" and len(subjects) == 4:
            cameras = {p.name for p in (raw_dir / "subject1").glob("*/*/cam*")}
            expected = {f"cam{i}" for i in range(5)}
            missing = expected - cameras
            if missing:
                print(f"Camera folders not observed in subject1: {sorted(missing)}")
                ok = False
            else:
                print("Camera folders:     cam0-cam4 present")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download", "extract", "verify", "all"))
    parser.add_argument("--mode", choices=tuple(MODE_FILES), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        choices=range(1, 7),
        metavar="1-6",
        help="Parallel archive downloads (default: 1)",
    )
    parser.add_argument(
        "--connections-per-file",
        type=int,
        default=1,
        choices=range(1, 17),
        metavar="1-16",
        help="Simultaneous HTTP byte-range connections per archive (default: 1)",
    )
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=None,
        choices=range(1, 17),
        metavar="1-16",
        help="Fixed resumable ranges per archive (default: connections per file)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.segments_per_file is None:
        args.segments_per_file = args.connections_per_file
    if args.connections_per_file > args.segments_per_file:
        raise SystemExit("--connections-per-file cannot exceed --segments-per-file")
    root: Path = args.root.expanduser().resolve()
    archive_dir = root / "archives"
    raw_dir = root / "raw"
    archive_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    needs_network = args.action in ("plan", "download", "all")
    sizes: dict[str, int] | None = None
    if needs_network:
        username, password = credentials()
        session = build_session(username, password)
        sizes = build_plan(session, selected_files(args.mode), archive_dir)
        manifest = {
            "source": BASE_URL,
            "mode": args.mode,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": sizes,
        }
        (root / "remote_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if args.action in ("download", "all"):
            download_files(
                username,
                password,
                sizes,
                archive_dir,
                args.jobs,
                args.connections_per_file,
                args.segments_per_file,
            )

    if args.action in ("extract", "all"):
        for filename in selected_files(args.mode):
            archive = archive_dir / filename
            if not archive.exists():
                raise SystemExit(f"Missing archive: {archive}")
            extract_archive(archive, raw_dir)

    if args.action in ("verify", "all"):
        manifest_path = root / "remote_manifest.json"
        if sizes is None and manifest_path.exists():
            sizes = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        return 0 if verify(args.mode, root, sizes) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
