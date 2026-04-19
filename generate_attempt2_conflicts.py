from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString

HEADER_LINES = [
    "00:00:00.00>TRAIL ON",
    "00:00:00.00>DT 10",
    "00:00:00.00>PLUGIN RIFLOG_THUNDER",
    "00:00:00.00>ASAS ON",
]

def extract_callsigns_from_master_scn(file_path: Path) -> set[str]:
    callsigns: set[str] = set()
    call_re = re.compile(r"^.*>\s*CALL\s+Corrected__([^_\s]+)_.*\.scn\s*$", re.IGNORECASE)
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = call_re.match(line.strip())
            if match:
                callsigns.add(match.group(1).strip())
    return callsigns


def build_allowed_callsigns(master_batch_files: list[Path]) -> set[str]:
    allowed: set[str] = set()
    for batch_file in master_batch_files:
        if batch_file.exists():
            allowed.update(extract_callsigns_from_master_scn(batch_file))
    return allowed


def extract_aircraft_pair_from_filename(file_path: Path) -> tuple[str, str]:
    parts = file_path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid pair filename: {file_path.name}. Expected <idx>_<AC1>_<AC2>.scn"
        )
    return parts[-2].strip(), parts[-1].strip()


def date_score(file_path: Path) -> int:
    match = re.search(r"_(\d{8})$", file_path.stem)
    if not match:
        return -1
    return int(match.group(1))


def choose_preferred_file(candidates: list[Path]) -> Path:
    return sorted(
        candidates,
        key=lambda path: (date_score(path), -len(path.parts), path.name),
        reverse=True,
    )[0]


def extract_callsigns_from_scn(file_path: Path) -> set[str]:
    callsigns: set[str] = set()
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                addwpt_match = re.search(r"ADDWPT\s+(\w+)\s+([\d.-]+)\s+([\d.-]+)", line)
                if addwpt_match:
                    callsigns.add(addwpt_match.group(1).strip())

                # Fallback extraction from CRE lines if needed.
                cre_match = re.search(r"\bCRE\s+(\w+)\b", line)
                if cre_match:
                    callsigns.add(cre_match.group(1).strip())
    except OSError:
        return set()

    return callsigns


def build_trajectory_index(trajectory_dir: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for scn_file in trajectory_dir.rglob("*.scn"):
        callsigns = extract_callsigns_from_scn(scn_file)
        for callsign in callsigns:
            grouped.setdefault(callsign, []).append(scn_file)

    return {
        callsign: choose_preferred_file(paths)
        for callsign, paths in grouped.items()
        if paths
    }


def get_csv_acids(csv_path: Path) -> set[str]:
    acids: set[str] = set()
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if row[0].strip().lower() == "sim_time" and row[1].strip().lower() == "acid":
                    continue
                acid = row[1].strip()
                if acid:
                    acids.add(acid)
    except Exception:
        return set()

    return acids


def build_log_pool_index(log_dir: Path) -> list[tuple[Path, set[str]]]:
    indexed: list[tuple[Path, set[str]]] = []
    for csv_file in sorted(log_dir.glob("*.csv")):
        acids = get_csv_acids(csv_file)
        if acids:
            indexed.append((csv_file, acids))
    return indexed


def resolve_logs_for_pair(
    flight1_id: str,
    flight2_id: str,
    log_index: list[tuple[Path, set[str]]],
) -> tuple[Path, Path]:
    # Prefer one shared CSV containing both aircraft.
    for log_path, acids in log_index:
        if flight1_id in acids and flight2_id in acids:
            return log_path, log_path

    # Fall back to separate logs if needed.
    log1 = None
    log2 = None
    for log_path, acids in log_index:
        if log1 is None and flight1_id in acids:
            log1 = log_path
        if log2 is None and flight2_id in acids:
            log2 = log_path
        if log1 is not None and log2 is not None:
            break

    if log1 is None or log2 is None:
        missing = []
        if log1 is None:
            missing.append(flight1_id)
        if log2 is None:
            missing.append(flight2_id)
        raise ValueError(f"No matching log CSV found for aircraft: {', '.join(missing)}")

    return log1, log2


def load_clean_log_df(log_path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []

    with log_path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or len(row) < 4:
                continue
            if row[0].strip().lower() == "sim_time" and len(row) >= 2 and row[1].strip().lower() == "acid":
                continue

            # Keep only the first five fields if a row has extras.
            sim_time = row[0].strip()
            acid = row[1].strip() if len(row) > 1 else ""
            lat = row[2].strip() if len(row) > 2 else ""
            lon = row[3].strip() if len(row) > 3 else ""
            alt = row[4].strip() if len(row) > 4 else ""

            records.append(
                {
                    "sim_time": sim_time,
                    "acid": acid,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                }
            )

    raw = pd.DataFrame.from_records(records, columns=["sim_time", "acid", "lat", "lon", "alt"])
    raw["acid"] = raw["acid"].astype(str).str.strip()
    for col in ["sim_time", "lat", "lon", "alt"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    cleaned = raw.dropna(subset=["acid", "sim_time", "lat", "lon"]).copy()
    return cleaned.sort_values("sim_time").reset_index(drop=True)


def load_flight_log_slice(log_df: pd.DataFrame, flight_id: str) -> pd.DataFrame:
    filtered = log_df[log_df["acid"] == flight_id].copy()
    return filtered.sort_values("sim_time").reset_index(drop=True)


def build_linestring_from_latlon(waypoints_latlon: list[tuple[float, float]]) -> LineString:
    return LineString([(lon, lat) for lat, lon in waypoints_latlon])


def get_intersection_candidate_points(intersection_geom):
    if intersection_geom.is_empty:
        return []
    if intersection_geom.geom_type == "Point":
        return [intersection_geom]
    if hasattr(intersection_geom, "geoms"):
        points = [geom for geom in intersection_geom.geoms if geom.geom_type == "Point"]
        if points:
            return points
    return [intersection_geom.centroid]


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_delay_from_logs(
    flight1_data: pd.DataFrame,
    flight2_data: pd.DataFrame,
    flight1_id: str,
    flight2_id: str,
) -> tuple[str, str, float, float | None, float | None]:
    waypoints_flight1 = list(zip(flight1_data["lat"], flight1_data["lon"]))
    waypoints_flight2 = list(zip(flight2_data["lat"], flight2_data["lon"]))

    line_flight1 = build_linestring_from_latlon(waypoints_flight1)
    line_flight2 = build_linestring_from_latlon(waypoints_flight2)

    exact_intersection = line_flight1.intersection(line_flight2)
    intersection_candidates = get_intersection_candidate_points(exact_intersection)

    if not intersection_candidates:
        return flight1_id, flight2_id, 0.0, None, None

    candidate_results = []
    for candidate in intersection_candidates:
        candidate_latlon = (candidate.y, candidate.x)

        flight1_distances = flight1_data.apply(
            lambda row: euclidean_distance((row["lat"], row["lon"]), candidate_latlon),
            axis=1,
        )
        flight2_distances = flight2_data.apply(
            lambda row: euclidean_distance((row["lat"], row["lon"]), candidate_latlon),
            axis=1,
        )

        nearest_index_flight1 = flight1_distances.idxmin()
        nearest_index_flight2 = flight2_distances.idxmin()

        sim_time_flight1 = float(flight1_data.loc[nearest_index_flight1, "sim_time"])
        sim_time_flight2 = float(flight2_data.loc[nearest_index_flight2, "sim_time"])
        time_gap = abs(sim_time_flight1 - sim_time_flight2)

        candidate_results.append(
            {
                "coords": candidate_latlon,
                "sim_time_flight1": sim_time_flight1,
                "sim_time_flight2": sim_time_flight2,
                "time_gap": time_gap,
            }
        )

    best_candidate = min(candidate_results, key=lambda item: item["time_gap"])
    intersection_lat, intersection_lon = best_candidate["coords"]
    sim_time_flight1 = best_candidate["sim_time_flight1"]
    sim_time_flight2 = best_candidate["sim_time_flight2"]

    if sim_time_flight1 < sim_time_flight2:
        delayed_flight = flight1_id
        no_delay_flight = flight2_id
        delay_time = sim_time_flight2 - sim_time_flight1
    else:
        delayed_flight = flight2_id
        no_delay_flight = flight1_id
        delay_time = sim_time_flight1 - sim_time_flight2

    return no_delay_flight, delayed_flight, float(delay_time), intersection_lat, intersection_lon


def format_delay_to_timestamp(delay_seconds: float) -> str:
    total_seconds = max(0, int(round(float(delay_seconds))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.00"


def build_conflict_scenario_content(
    no_delay_flight: str,
    delayed_flight: str,
    delay_time: float,
    callsign_to_file: dict[str, Path],
    intersection_lat: float | None,
    intersection_lon: float | None,
    flight1_id: str,
    flight2_id: str,
    flight1_max_sim_time: float,
    flight2_max_sim_time: float,
) -> str:
    no_delay_file = callsign_to_file[no_delay_flight].name
    delayed_file = callsign_to_file[delayed_flight].name

    lines = [f"{item}\n" for item in HEADER_LINES]
    lines.append("\n")

    intersection_lat_text = "N/A" if intersection_lat is None else f"{intersection_lat:.6f}"
    intersection_lon_text = "N/A" if intersection_lon is None else f"{intersection_lon:.6f}"
    lines.append(f"# Intersection lat={intersection_lat_text}, lon={intersection_lon_text}, alt=N/A\n")

    delayed_start_time_formatted = format_delay_to_timestamp(delay_time)
    lines.append(f"00:00:00.00> CALL {no_delay_file}\n")
    lines.append(f"{delayed_start_time_formatted}> CALL {delayed_file}\n")

    return "".join(lines)


def process_pair_file(
    pair_file: Path,
    trajectory_index: dict[str, Path],
    log_pool_index: list[tuple[Path, set[str]]],
    log_df_cache: dict[Path, pd.DataFrame],
    output_dir: Path,
) -> tuple[bool, str, str]:
    flight1_id, flight2_id = extract_aircraft_pair_from_filename(pair_file)

    if flight1_id not in trajectory_index or flight2_id not in trajectory_index:
        missing = [
            flight_id
            for flight_id in [flight1_id, flight2_id]
            if flight_id not in trajectory_index
        ]
        return False, f"{pair_file.name}: missing trajectory .scn for {', '.join(missing)}", ""

    try:
        flight1_log_path, flight2_log_path = resolve_logs_for_pair(
            flight1_id,
            flight2_id,
            log_pool_index,
        )
    except ValueError as exc:
        return False, f"{pair_file.name}: {exc}", ""

    if flight1_log_path not in log_df_cache:
        log_df_cache[flight1_log_path] = load_clean_log_df(flight1_log_path)
    if flight2_log_path not in log_df_cache:
        log_df_cache[flight2_log_path] = load_clean_log_df(flight2_log_path)

    flight1_data = load_flight_log_slice(log_df_cache[flight1_log_path], flight1_id)
    flight2_data = load_flight_log_slice(log_df_cache[flight2_log_path], flight2_id)

    if flight1_data.empty or flight2_data.empty:
        missing_logs = []
        if flight1_data.empty:
            missing_logs.append(flight1_id)
        if flight2_data.empty:
            missing_logs.append(flight2_id)
        return False, f"{pair_file.name}: missing log rows for {', '.join(missing_logs)}", ""

    no_delay_flight, delayed_flight, delay_time, intersection_lat, intersection_lon = compute_delay_from_logs(
        flight1_data=flight1_data,
        flight2_data=flight2_data,
        flight1_id=flight1_id,
        flight2_id=flight2_id,
    )

    flight1_max_sim_time = float(flight1_data["sim_time"].max())
    flight2_max_sim_time = float(flight2_data["sim_time"].max())

    content = build_conflict_scenario_content(
        no_delay_flight=no_delay_flight,
        delayed_flight=delayed_flight,
        delay_time=delay_time,
        callsign_to_file=trajectory_index,
        intersection_lat=intersection_lat,
        intersection_lon=intersection_lon,
        flight1_id=flight1_id,
        flight2_id=flight2_id,
        flight1_max_sim_time=flight1_max_sim_time,
        flight2_max_sim_time=flight2_max_sim_time,
    )

    output_path = output_dir / f"{pair_file.stem}_Conflict_updated.scn"
    output_path.write_text(content, encoding="utf-8")

    return (
        True,
        (
            f"{pair_file.name}: created {output_path.name} "
            f"(delay={delay_time:.2f}s, delayed={delayed_flight}, "
            f"logs={flight1_log_path.name}/{flight2_log_path.name})"
        ),
        output_path.name,
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_batches = [
        base_dir / "2D_batches" / "master_batch1.scn",
        base_dir / "2D_batches" / "master_batch2.scn",
        base_dir / "2D_batches" / "master_batch3.scn",
        base_dir / "2D_batches" / "master_batch4.scn",
        base_dir / "2D_batches" / "master_batch5.scn",
        base_dir / "2D_batches" / "master_batch6.scn",
        base_dir / "2D_batches" / "master_batch7.scn",
        base_dir / "2D_batches" / "master_batch8.scn",
        base_dir / "2D_batches" / "master_batch9.scn",
        base_dir / "2D_batches" / "master_batch11.scn",
    ]
    parser = argparse.ArgumentParser(
        description=(
            "Generate one conflict scenario per point-intersection pair in attempt2, "
            "using DT 10 and delay derived from simulation logs."
        )
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=base_dir / "2d_intersections" / "point_intersections",
        help="Directory containing pair .scn files (e.g., 001_AAR384_QTR946.scn)",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=base_dir / "2D_trajectory",
        help="Directory containing aircraft trajectory .scn files used by CALL",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=base_dir / "log_files",
        help="Directory containing the pool of log CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "2d_intersections" / "conflict_files",
        help="Output folder for generated conflict scenarios",
    )
    parser.add_argument(
        "--allowed-master-batches",
        type=Path,
        nargs="*",
        default=default_batches,
        help="Master batch .scn files whose CALL aircraft IDs are allowed for generation",
    )
    parser.add_argument(
        "--progress-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-intersection generation progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs_dir = args.pairs_dir.resolve()
    trajectory_dir = args.trajectory_dir.resolve()
    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir.resolve()
    allowed_master_batches = [path.resolve() for path in args.allowed_master_batches]
    progress_csv = args.progress_csv.resolve() if args.progress_csv else output_dir / "conflict_generation_progress.csv"

    if not pairs_dir.exists():
        raise FileNotFoundError(f"Pairs directory not found: {pairs_dir}")
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pair_files = sorted([file for file in pairs_dir.glob("*.scn") if not file.name.startswith("master_")])
    if not pair_files:
        raise ValueError(f"No pair .scn files found in {pairs_dir}")

    trajectory_index = build_trajectory_index(trajectory_dir)
    if not trajectory_index:
        raise ValueError(f"No trajectory .scn files indexed under {trajectory_dir}")

    log_pool_index = build_log_pool_index(log_dir)
    if not log_pool_index:
        raise ValueError(f"No usable CSV logs found in {log_dir}")

    allowed_callsigns = build_allowed_callsigns(allowed_master_batches)
    if not allowed_callsigns:
        raise ValueError("No allowed aircraft IDs found from --allowed-master-batches")

    print(f"Pairs found: {len(pair_files)}")
    print(f"Trajectory index size: {len(trajectory_index)} aircraft")
    print(f"Log pool size: {len(log_pool_index)} CSV files")
    print(f"Allowed aircraft IDs from master batches: {len(allowed_callsigns)}")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    failure_messages: list[str] = []
    progress_rows: list[dict[str, str]] = []
    log_df_cache: dict[Path, pd.DataFrame] = {}

    for pair_file in pair_files:
        try:
            flight1_id, flight2_id = extract_aircraft_pair_from_filename(pair_file)
        except ValueError as exc:
            message = f"{pair_file.name}: {exc}"
            print(message)
            failure_count += 1
            failure_messages.append(message)
            progress_rows.append(
                {
                    "pair_file": pair_file.name,
                    "flight1_id": "",
                    "flight2_id": "",
                    "status": "error",
                    "output_file": "",
                    "details": message,
                }
            )
            continue

        if flight1_id not in allowed_callsigns or flight2_id not in allowed_callsigns:
            message = f"{pair_file.name}: skipped (aircraft outside allowed master batches)"
            print(message)
            skipped_count += 1
            progress_rows.append(
                {
                    "pair_file": pair_file.name,
                    "flight1_id": flight1_id,
                    "flight2_id": flight2_id,
                    "status": "skipped",
                    "output_file": "",
                    "details": "aircraft outside allowed master batches",
                }
            )
            continue

        try:
            ok, message, output_name = process_pair_file(
                pair_file=pair_file,
                trajectory_index=trajectory_index,
                log_pool_index=log_pool_index,
                log_df_cache=log_df_cache,
                output_dir=output_dir,
            )
        except Exception as exc:
            ok = False
            output_name = ""
            message = f"{pair_file.name}: unhandled error: {exc}"

        print(message)
        if ok:
            success_count += 1
            progress_rows.append(
                {
                    "pair_file": pair_file.name,
                    "flight1_id": flight1_id,
                    "flight2_id": flight2_id,
                    "status": "generated",
                    "output_file": output_name,
                    "details": message,
                }
            )
        else:
            failure_count += 1
            failure_messages.append(message)
            progress_rows.append(
                {
                    "pair_file": pair_file.name,
                    "flight1_id": flight1_id,
                    "flight2_id": flight2_id,
                    "status": "error",
                    "output_file": "",
                    "details": message,
                }
            )

    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(progress_rows).to_csv(progress_csv, index=False)

    print("\nBatch processing completed.")
    print(f"Total pairs: {len(pair_files)}")
    print(f"Generated: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failure_count}")
    print(f"Progress CSV: {progress_csv}")

    if failure_messages:
        print("\nFailures:")
        for item in failure_messages:
            print(f" - {item}")


if __name__ == "__main__":
    main()
