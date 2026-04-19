from __future__ import annotations

import tempfile
import zipfile
import shutil
from pathlib import Path


def is_conflict_file(file_path: Path) -> bool:
    return "conflict" in file_path.stem.lower()


def unique_destination(dest_dir: Path, file_name: str, folder_name: str) -> Path:
    candidate = dest_dir / file_name
    if not candidate.exists():
        return candidate

    prefixed = dest_dir / f"{folder_name}__{file_name}"
    if not prefixed.exists():
        return prefixed

    stem = prefixed.stem
    suffix = prefixed.suffix
    index = 1
    while True:
        alt = dest_dir / f"{stem}_{index}{suffix}"
        if not alt.exists():
            return alt
        index += 1


def classify_scn_files(case_dir: Path) -> tuple[list[Path], list[Path]]:
    scn_files = sorted(case_dir.glob("*.scn"))
    conflict_files = [file for file in scn_files if is_conflict_file(file)]
    trajectory_files = [file for file in scn_files if file not in conflict_files]
    return trajectory_files, conflict_files


def resolve_input_root(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        return input_path.resolve(), None

    if input_path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Input path does not exist or is not supported: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input zip does not exist: {input_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="gan_output_extract_")
    with zipfile.ZipFile(input_path) as archive:
        archive.extractall(temp_dir.name)

    extracted_root = Path(temp_dir.name)
    children = [item for item in extracted_root.iterdir() if item.is_dir()]
    if len(children) == 1 and not any(extracted_root.glob("*.scn")):
        return children[0].resolve(), temp_dir

    return extracted_root.resolve(), temp_dir


def main() -> None:
    input_path = Path(r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\gan_output_3.zip")
    output_root = Path(r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\gan_data3").resolve()

    input_root, temp_dir = resolve_input_root(input_path)

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a folder: {input_root}")

    traj_dir = output_root / "GAN_Trajectory"
    conflict_dir = output_root / "GAN_conflict"
    traj_dir.mkdir(parents=True, exist_ok=True)
    conflict_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([item for item in input_root.iterdir() if item.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subfolders found in input root: {input_root}")

    copied_trajectory = 0
    copied_conflict = 0
    skipped_cases: list[str] = []

    for case_dir in case_dirs:
        trajectory_files, conflict_files = classify_scn_files(case_dir)

        valid = len(trajectory_files) == 2 and len(conflict_files) == 1
        if not valid:
            message = (
                f"{case_dir.name}: expected 2 trajectory + 1 conflict SCN, got "
                f"{len(trajectory_files)} trajectory and {len(conflict_files)} conflict"
            )
            skipped_cases.append(message)
            continue

        for src in trajectory_files:
            dst = unique_destination(traj_dir, src.name, case_dir.name)
            shutil.copy2(src, dst)
            copied_trajectory += 1

        for src in conflict_files:
            dst = unique_destination(conflict_dir, src.name, case_dir.name)
            shutil.copy2(src, dst)
            copied_conflict += 1

    if copied_trajectory != 2 * copied_conflict:
        raise RuntimeError(
            "Output count check failed: trajectory file count is not exactly twice "
            f"conflict count ({copied_trajectory} vs {copied_conflict})."
        )

    print("Collection completed.")
    print(f"Input root:   {input_root}")
    print(f"Output root:  {output_root}")
    print(f"Trajectories: {copied_trajectory} -> {traj_dir}")
    print(f"Conflicts:    {copied_conflict} -> {conflict_dir}")

    if skipped_cases:
        print("\nSkipped subfolders:")
        for entry in skipped_cases:
            print(f"- {entry}")

    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()