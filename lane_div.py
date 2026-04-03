import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = [
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

DEFAULT_INPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_origin\4 Trajectory Data.csv"
)
DEFAULT_LANE_DIVISION = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_origin\5 Lane Division Data.csv"
)
DEFAULT_OUTPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_lane_id.csv"
)
DEFAULT_MAINLINE_OUTPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\Local_X_2000_straight.csv"
)


def ensure_csv_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    return path


def interpolate_boundary(boundary_df: pd.DataFrame, x_values: np.ndarray) -> np.ndarray:
    boundary_df = boundary_df.sort_values("Local_X")
    x = boundary_df["Local_X"].to_numpy(dtype=float)
    y = boundary_df["Local_Y"].to_numpy(dtype=float)
    clipped = np.clip(x_values, x.min(), x.max())
    return np.interp(clipped, x, y)


def load_lane_boundaries(lane_division_file: Path) -> dict[int, pd.DataFrame]:
    boundary_df = pd.read_csv(
        lane_division_file,
        header=None,
        names=["Boundary_ID", "Local_X", "Local_Y"],
    )
    return {
        int(boundary_id): group[["Local_X", "Local_Y"]].copy()
        for boundary_id, group in boundary_df.groupby("Boundary_ID")
    }


def assign_lane_ids(
    trajectory_df: pd.DataFrame,
    boundaries: dict[int, pd.DataFrame],
    coeff_ramp: float = 0.56,
    coeff_lane3: float = 1.56,
    coeff_lane2: float = 2.48,
) -> pd.DataFrame:
    x_values = trajectory_df["Local_X"].to_numpy(dtype=float)
    y_values = trajectory_df["Local_Y"].to_numpy(dtype=float)

    b1 = interpolate_boundary(boundaries[1], x_values)
    b2 = interpolate_boundary(boundaries[2], x_values)
    b3 = interpolate_boundary(boundaries[3], x_values)
    b4 = interpolate_boundary(boundaries[4], x_values)

    # The historical result aligns best when the upper lanes are obtained by
    # extending the spacing of boundaries 1-4 upward from boundary 4.
    lane_width = ((b2 - b1) + (b3 - b2) + (b4 - b3)) / 3.0

    threshold_ramp = b4 + coeff_ramp * lane_width
    threshold_lane3 = b4 + coeff_lane3 * lane_width
    threshold_lane2 = b4 + coeff_lane2 * lane_width

    lane_ids = np.full(len(trajectory_df), "1", dtype=object)
    lane_ids[y_values <= threshold_lane2] = "2"
    lane_ids[y_values <= threshold_lane3] = "3"
    lane_ids[y_values <= threshold_ramp] = "ramp1"

    out = trajectory_df.copy()
    out["Lane_ID"] = lane_ids
    return out


def build_mainline_straight_dataset(
    trajectory_with_lane_df: pd.DataFrame,
    max_local_x: float = 2000.0,
) -> pd.DataFrame:
    mainline_lanes = {"1", "2", "3"}
    lane_str = trajectory_with_lane_df["Lane_ID"].astype(str)
    mask = (trajectory_with_lane_df["Local_X"].astype(float) < max_local_x) & lane_str.isin(mainline_lanes)
    return trajectory_with_lane_df.loc[mask].copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign lane IDs from trajectory and lane division data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input trajectory CSV path.")
    parser.add_argument("--lane-division", default=DEFAULT_LANE_DIVISION, help="Lane division CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--mainline-output",
        default=DEFAULT_MAINLINE_OUTPUT,
        help="Output CSV path for mainline straight-segment data.",
    )
    parser.add_argument("--max-local-x", type=float, default=2000.0, help="Upper Local_X bound for straight mainline data.")
    parser.add_argument("--coeff-ramp", type=float, default=0.56, help="Ramp threshold coefficient.")
    parser.add_argument("--coeff-lane3", type=float, default=1.56, help="Lane 3 threshold coefficient.")
    parser.add_argument("--coeff-lane2", type=float, default=2.48, help="Lane 2 threshold coefficient.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_file = Path(args.input)
    lane_division_file = Path(args.lane_division)
    output_file = ensure_csv_path(args.output)
    mainline_output_file = ensure_csv_path(args.mainline_output)

    print(f"读取轨迹文件: {input_file}")
    trajectory_df = pd.read_csv(input_file, header=None, names=TRAJECTORY_COLUMNS)
    print(f"轨迹数据行数: {len(trajectory_df)}")

    print(f"读取车道分界文件: {lane_division_file}")
    boundaries = load_lane_boundaries(lane_division_file)
    missing = sorted(set([1, 2, 3, 4]) - set(boundaries.keys()))
    if missing:
        raise ValueError(f"缺少关键车道分界线: {missing}")

    output_df = assign_lane_ids(
        trajectory_df,
        boundaries,
        coeff_ramp=args.coeff_ramp,
        coeff_lane3=args.coeff_lane3,
        coeff_lane2=args.coeff_lane2,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"已输出到: {output_file}")
    print("Lane_ID 分布:")
    print(output_df["Lane_ID"].value_counts().to_string())

    mainline_df = build_mainline_straight_dataset(output_df, max_local_x=args.max_local_x)
    mainline_output_file.parent.mkdir(parents=True, exist_ok=True)
    mainline_df.to_csv(mainline_output_file, index=False)
    print(f"主车道直线段数据已输出到: {mainline_output_file}")
    print(f"主车道直线段数据行数: {len(mainline_df)}")


if __name__ == "__main__":
    main()
