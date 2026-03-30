# 将原始轨迹数据根据提供的车道线数据进行划分，对每条轨迹进行车道线ID的标注，输出新的轨迹数据文件。
import json
import os
import pandas as pd

import pandas as pd
import numpy as np

# ========== 1. 读取轨迹 ==========
traj_cols = [
    "Vehicle_ID", "Frame_ID",
    "Local_X_RT", "Local_Y_RT",
    "Local_X_RB", "Local_Y_RB",
    "Local_X_LB", "Local_Y_LB",
    "Local_X_LT", "Local_Y_LT",
    "Local_X", "Local_Y",
    "V_Class", "V_Direction",
    "Speed_X", "Speed_Y",
    "V_Length", "V_Width", "V_Heading"
]

traj = pd.read_csv("trajectory.txt", sep=r"\s+", header=None, names=traj_cols)

# ========== 2. 读取车道线 ==========
lane_cols = ["LaneLine_ID", "X", "Y"]
lane = pd.read_csv("lane_division.txt", sep=r"\s+", header=None, names=lane_cols)

# ========== 3. 每条车道线拟合 ==========
lane_models = {}
for line_id, group in lane.groupby("LaneLine_ID"):
    x = group["X"].values
    y = group["Y"].values
    coeff = np.polyfit(x, y, deg=1)   # 主线可先用一次拟合
    lane_models[line_id] = coeff

def eval_line(coeff, x):
    a, b = coeff
    return a * x + b

def assign_lane(x, y, lane_models, boundaries):
    values = []
    for bid in boundaries:
        yb = eval_line(lane_models[bid], x)
        values.append((bid, yb))
    
    values.sort(key=lambda t: t[1])
    ys = [t[1] for t in values]
    
    for i in range(len(ys) - 1):
        if ys[i] <= y < ys[i + 1]:
            return i + 1
    return np.nan

# 假设主线边界线编号为 1,2,3,4
mainline_boundaries = [1, 2, 3, 4]

traj["Lane_ID"] = traj.apply(
    lambda row: assign_lane(row["Local_X"], row["Local_Y"], lane_models, mainline_boundaries),
    axis=1
)

traj.to_csv("trajectory_with_lane_id.csv", index=False, encoding="utf-8-sig")
print(traj.head())