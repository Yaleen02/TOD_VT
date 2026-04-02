import os
import numpy as np
import pandas as pd


TRAJ_PATH = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04"
    r"\TOD_VT_01_01_01_Huanle Avenue at Yangchunhu Road\4 Trajectory Data.csv"
)
LANE_PATH = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04"
    r"\TOD_VT_01_01_01_Huanle Avenue at Yangchunhu Road\5 Lane Division Data.csv"
)
OUTPUT_PATH = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_data_lane_id.csv"
)
OUTPUT_BASIC_PATH = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_data_lane_id_basic_columns.csv"
)

SMOOTH_POS_WINDOW = 5
LANE_TOL = 3.0
MAINLINE_LANE_LINES = [1, 2, 3, 4]

TRAJ_COLS = [
    "Vehicle_ID",
    "Frame_ID",
    "Local_X_RT",
    "Local_Y_RT",
    "Local_X_RB",
    "Local_Y_RB",
    "Local_X_LB",
    "Local_Y_LB",
    "Local_X_LT",
    "Local_Y_LT",
    "Local_X",
    "Local_Y",
    "V_Class",
    "V_Direction",
    "Speed_X",
    "Speed_Y",
    "V_Length",
    "V_Width",
    "V_Heading",
]

LANE_COLS = ["LaneLine_ID", "X", "Y"]

BASIC_OUTPUT_COLS = [
    "Vehicle_ID",
    "Frame_ID",
    "Local_X",
    "Local_Y",
    "Speed_X",
    "Speed_Y",
    "V_Length",
    "V_Width",
    "V_Heading",
    "Lane_ID",
]


def load_csv_no_header(path: str, expected_cols: list[str]) -> pd.DataFrame:
    for read_kwargs in (
        {"header": None},
        {"sep": r"\s+", "engine": "python", "header": None},
    ):
        try:
            df = pd.read_csv(path, **read_kwargs)
            if df.shape[1] == len(expected_cols):
                df.columns = expected_cols
                return df
        except Exception:
            continue

    raise ValueError(f"Failed to read file or column count mismatch: {path}")


def eval_line_y(line_models: dict[int, np.poly1d], line_id: int, x: float) -> float:
    return float(line_models[line_id](x))


def in_band_with_tol(y: float, a: float, b: float, tol: float = 0.0) -> bool:
    low = min(a, b) - tol
    high = max(a, b) + tol
    return low <= y < high


def assign_mainline_lane(
    x: float,
    y: float,
    line_models: dict[int, np.poly1d],
    ramp_x_min: float,
    tol: float = LANE_TOL,
):
    y1 = eval_line_y(line_models, 1, x)
    y2 = eval_line_y(line_models, 2, x)
    y3 = eval_line_y(line_models, 3, x)
    y4 = eval_line_y(line_models, 4, x)

    bands = [
        {"lane_id": 1, "low_y": min(y1, y2), "high_y": max(y1, y2)},
        {"lane_id": 2, "low_y": min(y2, y3), "high_y": max(y2, y3)},
        {"lane_id": 3, "low_y": min(y3, y4), "high_y": max(y3, y4)},
    ]

    for band in bands:
        if in_band_with_tol(y, band["low_y"], band["high_y"], tol):
            return band["lane_id"]

    if (x >= ramp_x_min) and (y >= max(y3, y4) + tol):
        return "ramp"

    nearest_band = min(
        bands,
        key=lambda band: abs(y - ((band["low_y"] + band["high_y"]) / 2.0)),
    )
    return nearest_band["lane_id"]


def main() -> None:
    traj = load_csv_no_header(TRAJ_PATH, TRAJ_COLS)
    lane = load_csv_no_header(LANE_PATH, LANE_COLS)

    for col in TRAJ_COLS:
        traj[col] = pd.to_numeric(traj[col], errors="coerce")
    for col in LANE_COLS:
        lane[col] = pd.to_numeric(lane[col], errors="coerce")

    traj = traj.dropna(subset=["Local_X", "Local_Y"]).copy()
    lane = lane.dropna(subset=["LaneLine_ID", "X", "Y"]).copy()
    lane["LaneLine_ID"] = lane["LaneLine_ID"].astype(int)

    print("Original trajectory shape:", traj.shape)
    print("Lane division shape:", lane.shape)
    print("Lane line ids:", sorted(lane["LaneLine_ID"].unique().tolist()))

    traj = traj.sort_values(["Vehicle_ID", "Frame_ID"]).copy()
    traj["Smooth_X"] = traj.groupby("Vehicle_ID")["Local_X"].transform(
        lambda s: s.rolling(window=SMOOTH_POS_WINDOW, center=True, min_periods=1).mean()
    )
    traj["Smooth_Y"] = traj.groupby("Vehicle_ID")["Local_Y"].transform(
        lambda s: s.rolling(window=SMOOTH_POS_WINDOW, center=True, min_periods=1).mean()
    )
    print("Smoothed trajectory coordinates.")

    lane_main = lane[lane["LaneLine_ID"].isin(MAINLINE_LANE_LINES)].copy()
    line_models: dict[int, np.poly1d] = {}
    for line_id, group in lane_main.groupby("LaneLine_ID"):
        coeff = np.polyfit(group["X"].to_numpy(dtype=float), group["Y"].to_numpy(dtype=float), deg=1)
        line_models[int(line_id)] = np.poly1d(coeff)

    ramp_boundary = lane[lane["LaneLine_ID"].isin([5, 6])].copy()
    ramp_x_min = float(ramp_boundary["X"].min()) if not ramp_boundary.empty else float("inf")

    traj["Lane_ID"] = traj.apply(
        lambda row: assign_mainline_lane(row["Smooth_X"], row["Smooth_Y"], line_models, ramp_x_min),
        axis=1,
    ).astype(str)

    print("\nLane_ID distribution:")
    print(traj["Lane_ID"].value_counts(dropna=False).sort_index())

    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    traj_full = traj[TRAJ_COLS + ["Smooth_X", "Smooth_Y", "Lane_ID"]].copy()
    traj_full.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved full output: {OUTPUT_PATH}")

    traj_basic = traj_full[BASIC_OUTPUT_COLS].copy()
    traj_basic.to_csv(OUTPUT_BASIC_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved basic output: {OUTPUT_BASIC_PATH}")


if __name__ == "__main__":
    main()
