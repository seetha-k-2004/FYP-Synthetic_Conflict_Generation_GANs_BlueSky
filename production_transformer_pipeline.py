from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = Path(r"D:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\GAN_output_1\log")
OUTPUT_DIR = Path(r"D:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\GAN_output_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional switches
ADD_RELATIVE_FEATURES = False   # True -> adds dlat, dlon, dalt, dgs
INCLUDE_TIME_FEATURE = False    # Usually False if sampling interval is uniform


# ============================================================
# HELPERS
# ============================================================
def standardise_wide_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise already-restructured files so they all use:
    sim_time, acid1, lat1, lon1, alt1, hdg1, gs1, acid2, lat2, lon2, alt2, hdg2, gs2
    """
    rename_map = {
        "long1": "lon1",
        "long2": "lon2",
        "heading1": "hdg1",
        "heading2": "hdg2",
        "ground_speed1": "gs1",
        "ground_speed2": "gs2",
    }
    df = df.rename(columns=rename_map).copy()

    ordered_cols = [
        "sim_time",
        "acid1", "lat1", "lon1", "alt1", "hdg1", "gs1",
        "acid2", "lat2", "lon2", "alt2", "hdg2", "gs2",
    ]

    missing = set(ordered_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing wide-format columns: {missing}")

    return df[ordered_cols].sort_values("sim_time").reset_index(drop=True)


def long_to_wide_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long format:
        sim_time, acid, lat, lon, alt, heading, ground_speed
    into wide pair format:
        sim_time, acid1, lat1, lon1, alt1, hdg1, gs1, acid2, lat2, lon2, alt2, hdg2, gs2
    Assumes one file contains exactly one aircraft pair.
    """
    required = {"sim_time", "acid", "lat", "lon", "alt", "heading", "ground_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing long-format columns: {missing}")

    df = df.copy().sort_values(["sim_time", "acid"]).reset_index(drop=True)

    aircraft_ids = df["acid"].dropna().astype(str).unique().tolist()
    if len(aircraft_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 aircraft in one file, found {len(aircraft_ids)}: {aircraft_ids}"
        )

    acid1, acid2 = aircraft_ids[0], aircraft_ids[1]

    df1 = (
        df[df["acid"] == acid1][["sim_time", "lat", "lon", "alt", "heading", "ground_speed"]]
        .rename(columns={
            "lat": "lat1",
            "lon": "lon1",
            "alt": "alt1",
            "heading": "hdg1",
            "ground_speed": "gs1",
        })
    )

    df2 = (
        df[df["acid"] == acid2][["sim_time", "lat", "lon", "alt", "heading", "ground_speed"]]
        .rename(columns={
            "lat": "lat2",
            "lon": "lon2",
            "alt": "alt2",
            "heading": "hdg2",
            "ground_speed": "gs2",
        })
    )

    wide = df1.merge(df2, on="sim_time", how="inner", validate="one_to_one")
    wide["acid1"] = acid1
    wide["acid2"] = acid2

    ordered_cols = [
        "sim_time",
        "acid1", "lat1", "lon1", "alt1", "hdg1", "gs1",
        "acid2", "lat2", "lon2", "alt2", "hdg2", "gs2",
    ]
    return wide[ordered_cols].sort_values("sim_time").reset_index(drop=True)


def load_any_csv(path: Path) -> pd.DataFrame:
    """
    Detect whether file is long-format or already restructured wide-format.
    """
    df = pd.read_csv(path)
    cols = set(df.columns)

    # Already wide / restructured
    if {"acid1", "acid2", "lat1", "lat2"}.issubset(cols) or {"acid1", "acid2", "lat1", "long1", "lat2", "long2"}.issubset(cols):
        return standardise_wide_columns(df)

    # Raw long format
    if {"acid", "lat", "lon", "alt", "heading", "ground_speed"}.issubset(cols):
        return long_to_wide_pair(df)

    raise ValueError(f"Unrecognized file structure in {path.name}. Columns: {list(df.columns)}")


def build_numeric_sequence(wide_df: pd.DataFrame,
                           add_relative: bool = False,
                           include_time_feature: bool = False):
    """
    Convert one wide-format pair trajectory into numeric Transformer features.
    Output shape: (T, F)
    """
    df = wide_df.copy().sort_values("sim_time").reset_index(drop=True)

    # Convert heading to sin/cos
    for k in [1, 2]:
        df[f"sin_hdg{k}"] = np.sin(np.deg2rad(df[f"hdg{k}"].astype(float)))
        df[f"cos_hdg{k}"] = np.cos(np.deg2rad(df[f"hdg{k}"].astype(float)))

    # Base features: 12 total
    feature_cols = [
        "lat1", "lon1", "alt1", "gs1", "sin_hdg1", "cos_hdg1",
        "lat2", "lon2", "alt2", "gs2", "sin_hdg2", "cos_hdg2",
    ]

    # Optional relative features
    if add_relative:
        df["dlat"] = df["lat2"] - df["lat1"]
        df["dlon"] = df["lon2"] - df["lon1"]
        df["dalt"] = df["alt2"] - df["alt1"]
        df["dgs"] = df["gs2"] - df["gs1"]
        feature_cols += ["dlat", "dlon", "dalt", "dgs"]

    # Optional time feature
    # Usually unnecessary if your sequence is uniformly sampled and the transformer uses positional encoding
    if include_time_feature:
        df["t_rel"] = df["sim_time"] - df["sim_time"].iloc[0]
        feature_cols = ["t_rel"] + feature_cols

    X_seq = df[feature_cols].to_numpy(dtype=np.float32)

    metadata = {
        "acid1": str(df["acid1"].iloc[0]),
        "acid2": str(df["acid2"].iloc[0]),
        "timesteps": int(len(df)),
        "feature_dim": int(len(feature_cols)),
    }

    return X_seq, feature_cols, metadata, df


def pad_and_stack(sequences, pad_value=0.0):
    """
    Convert list of variable-length arrays [(T1,F), (T2,F), ...]
    into:
        X    -> (N, T_max, F)
        mask -> (N, T_max), 1 = valid timestep, 0 = padding
        lengths -> (N,)
    """
    lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int32)
    num_samples = len(sequences)
    max_len = int(lengths.max())
    feature_dim = sequences[0].shape[1]

    X = np.full((num_samples, max_len, feature_dim), pad_value, dtype=np.float32)
    mask = np.zeros((num_samples, max_len), dtype=np.float32)

    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        X[i, :T, :] = seq
        mask[i, :T] = 1.0

    return X, mask, lengths


# ============================================================
# MAIN PIPELINE
# ============================================================
all_csvs = sorted(INPUT_DIR.glob("*.csv"))
if not all_csvs:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"Found {len(all_csvs)} CSV files in {INPUT_DIR}\n")

sequences = []
metadata_rows = []
sequence_rows = []   # optional one-row-per-sequence storage

for path in all_csvs:
    try:
        wide_df = load_any_csv(path)

        X_seq, feature_cols, meta, seq_df = build_numeric_sequence(
            wide_df,
            add_relative=ADD_RELATIVE_FEATURES,
            include_time_feature=INCLUDE_TIME_FEATURE
        )

        sequences.append(X_seq)

        metadata_rows.append({
            "file_name": path.name,
            "acid1": meta["acid1"],
            "acid2": meta["acid2"],
            "timesteps": meta["timesteps"],
            "feature_dim": meta["feature_dim"],
        })

        # Optional: store as one row with list-valued columns
        row = {
            "file_name": path.name,
            "acid1": meta["acid1"],
            "acid2": meta["acid2"],
            "timesteps": meta["timesteps"],
        }
        for col in feature_cols:
            row[col] = seq_df[col].tolist()
        sequence_rows.append(row)
        
        print(f"✓ {path.name:30s} → {X_seq.shape[0]:3d} timesteps × {X_seq.shape[1]} features")
    
    except Exception as e:
        print(f"✗ {path.name:30s} → ERROR: {e}")

print(f"\nTotal sequences processed: {len(sequences)}\n")

# If all files have same length, stack directly
same_length = len(set(seq.shape[0] for seq in sequences)) == 1

if same_length:
    X = np.stack(sequences).astype(np.float32)                 # (N, T, F)
    mask = np.ones((X.shape[0], X.shape[1]), dtype=np.float32)
    lengths = np.full((X.shape[0],), X.shape[1], dtype=np.int32)
else:
    X, mask, lengths = pad_and_stack(sequences)

# ============================================================
# SAVE OUTPUTS
# ============================================================
np.save(OUTPUT_DIR / "X_transformer.npy", X)
np.save(OUTPUT_DIR / "attention_mask.npy", mask)
np.save(OUTPUT_DIR / "sequence_lengths.npy", lengths)

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(OUTPUT_DIR / "sequence_metadata.csv", index=False)

# Better than CSV for list-columns
sequence_rows_df = pd.DataFrame(sequence_rows)
sequence_rows_df.to_pickle(OUTPUT_DIR / "sequence_rows.pkl")

print("="*70)
print("OUTPUTS SAVED to:", OUTPUT_DIR)
print("="*70)
print(f"X shape: {X.shape}")              # (N, T, F)
print(f"Mask shape: {mask.shape}")        # (N, T)
print(f"Lengths shape: {lengths.shape}")  # (N,)
print(f"\nFeature columns used:")
print(feature_cols)
print(f"\nFiles generated:")
print(f"  1. X_transformer.npy         — (N, T, F) array ready for Transformer")
print(f"  2. attention_mask.npy        — (N, T) mask for variable-length sequences")
print(f"  3. sequence_lengths.npy      — (N,) actual length per sequence")
print(f"  4. sequence_metadata.csv     — metadata per file")
print(f"  5. sequence_rows.pkl         — list-based columns (for inspection)")
