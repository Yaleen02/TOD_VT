import os
import numpy as np
import pandas as pd


TRAJ_PATH = r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT_01_01_01_Huanle Avenue at Yangchunhu Road\4 Trajectory Data.csv"
LANE_PATH = r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT_01_01_01_Huanle Avenue at Yangchunhu Road\5 Lane Division Data.csv"
OUTPUT_PATH = r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT\dataset_processed\trajectory_data_lane_id.csv"

SMOOTH_POS_WINDOW = 5
LANE_TOL = 3.0
MAINLINE_LANE_LINES = [1, 2, 3, 4]

TRAJ_COLS = [
    "Vehicle_ID", "Frame_ID",
    "Local_X_RT", "Local_Y_RT",
    "Local_X_RB", "Local_Y_RB",
    "Local_X_LB", "Local_Y_LB",
    "Local_X_LT", "Local_Y_LT",
    "Local_X", "Local_Y",
    "V_Class", "V_Direction",
    "Speed_X", "Speed_Y",
    "V_Length", "V_Width", "V_Heading",
]

LANE_COLS = ["LaneLine_ID", "X", "Y"]


def load_csv_no_header(path, expected_cols):
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == len(expected_cols):
            df.columns = expected_cols
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
        if df.shape[1] == len(expected_cols):
            df.columns = expected_cols
            return df
    except Exception:
        pass

    raise ValueError(f"读取失败或列数不匹配: {path}")


traj = load_csv_no_header(TRAJ_PATH, TRAJ_COLS)
lane = load_csv_no_header(LANE_PATH, LANE_COLS)

for col in TRAJ_COLS:
    traj[col] = pd.to_numeric(traj[col], errors="coerce")

for col in LANE_COLS:
    lane[col] = pd.to_numeric(lane[col], errors="coerce")

traj = traj.dropna(subset=["Local_X", "Local_Y"]).copy()
lane = lane.dropna(subset=["LaneLine_ID", "X", "Y"]).copy()
lane["LaneLine_ID"] = lane["LaneLine_ID"].astype(int)

print("原始轨迹数据维度：", traj.shape)
print("车道线数据维度：", lane.shape)
print("车道线编号：", sorted(lane["LaneLine_ID"].unique().tolist()))

traj = traj.sort_values(["Vehicle_ID", "Frame_ID"]).copy()
traj["Smooth_X"] = traj.groupby("Vehicle_ID")["Local_X"].transform(
    lambda s: s.rolling(window=SMOOTH_POS_WINDOW, center=True, min_periods=1).mean()
)
traj["Smooth_Y"] = traj.groupby("Vehicle_ID")["Local_Y"].transform(
    lambda s: s.rolling(window=SMOOTH_POS_WINDOW, center=True, min_periods=1).mean()
)
print("已完成轨迹坐标平滑。")

lane_main = lane[lane["LaneLine_ID"].isin(MAINLINE_LANE_LINES)].copy()
print("参与拟合的直线车道线编号：", sorted(lane_main["LaneLine_ID"].unique().tolist()))
print(
    "直线车道线 Y 范围：",
    (lane_main["Y"].min(), lane_main["Y"].max()),
    "；轨迹 Smooth_Y 范围：",
    (traj["Smooth_Y"].min(), traj["Smooth_Y"].max()),
)

line_models = {}
for line_id, g in lane_main.groupby("LaneLine_ID"):
    x = g["X"].to_numpy(dtype=float)
    y = g["Y"].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=1)
    line_models[line_id] = np.poly1d(coeff)

ramp_boundary = lane[lane["LaneLine_ID"].isin([5, 6])].copy()
RAMP_X_MIN = float(ramp_boundary["X"].min()) if not ramp_boundary.empty else float("inf")
print("匝道判定起始 X：", RAMP_X_MIN)


def eval_line_y(line_id, x):
    return float(line_models[line_id](x))


def in_band_with_tol(y, a, b, tol=0.0):
    low = min(a, b) - tol
    high = max(a, b) + tol
    return low <= y < high


def assign_mainline_lane(x, y, tol=LANE_TOL):
    y1 = eval_line_y(1, x)
    y2 = eval_line_y(2, x)
    y3 = eval_line_y(3, x)
    y4 = eval_line_y(4, x)

    bands = [
        {"lane_id": 1, "low_y": min(y1, y2), "high_y": max(y1, y2)},
        {"lane_id": 2, "low_y": min(y2, y3), "high_y": max(y2, y3)},
        {"lane_id": 3, "low_y": min(y3, y4), "high_y": max(y3, y4)},
    ]

    for band in bands:
        if in_band_with_tol(y, band["low_y"], band["high_y"], tol):
            return band["lane_id"]

    if (x >= RAMP_X_MIN) and (y >= max(y3, y4) + tol):
        return "ramp"

    nearest_band = min(
        bands,
        key=lambda band: abs(y - ((band["low_y"] + band["high_y"]) / 2.0)),
    )
    return nearest_band["lane_id"]


traj["Lane_ID"] = traj.apply(
    lambda row: assign_mainline_lane(row["Smooth_X"], row["Smooth_Y"]),
    axis=1,
).astype(str)

print("\nLane_ID 统计（全部轨迹）：")
print(traj["Lane_ID"].value_counts(dropna=False).sort_index())

traj = traj[TRAJ_COLS + ["Smooth_X", "Smooth_Y", "Lane_ID"]].copy()
print("精简后输出数据维度：", traj.shape)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
traj.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n已输出文件：{OUTPUT_PATH}")
