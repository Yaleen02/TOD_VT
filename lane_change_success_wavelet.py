import os
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks


# =========================
# 1. Parameters
# =========================
POSITION_SCALE = 10.0
DATA_FILE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_data_lane_id.csv"
)
OUTPUT_FILE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_results_mainline_only.csv"
)
OUTPUT_EVENT_ROWS_FILE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_event_rows_mainline_only.csv"
)
OUTPUT_FIG_DIR = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_figures"
)

FPS = 10
BOUNDARY_SECONDS = 6.5
BOUNDARY_FRAMES = int(BOUNDARY_SECONDS * FPS)
SINGULARITY_FILTER_FRAMES = 10

MIN_Y_DISPLACEMENT = 0.8
MAX_TIME_DURATION = 8
MAX_FRAMES_DURATION = round(MAX_TIME_DURATION * FPS)
MIN_TRAJECTORY_LENGTH = 20

WAVELET_NAME = "mexh"
WAVELET_SCALES = np.arange(1, 17)
SAMPLING_PERIOD = 0.1
ENERGY_THRESHOLD_FACTOR = 0.5

POST_EVENT_CHECK_SECONDS = 5
POST_EVENT_CHECK_FRAMES = int(POST_EVENT_CHECK_SECONDS * FPS)
EVENT_CONTEXT_SECONDS = 5
EVENT_CONTEXT_FRAMES = int(EVENT_CONTEXT_SECONDS * FPS)

SAVE_VISUALIZATIONS = True
MAX_VISUALIZE_VEHICLES = None

matplotlib.use("Agg")


def resolve_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"未找到列，候选列名: {candidates}")
    return None


def calculate_energy_exclude_boundary(energy: np.ndarray, boundary_frames: int) -> np.ndarray:
    num_scales, signal_length = energy.shape
    if signal_length > 2 * boundary_frames:
        valid_start = boundary_frames
        valid_end = signal_length - boundary_frames
    else:
        margin = round(0.1 * signal_length)
        valid_start = max(0, margin)
        valid_end = min(signal_length, signal_length - margin)

    mask = np.zeros((num_scales, signal_length))
    mask[:, valid_start:valid_end] = 1
    masked_energy = energy * mask
    return masked_energy.sum(axis=0)


def detect_singularities_by_energy(
    total_energy: np.ndarray,
    boundary_frames: int,
    energy_threshold_factor: float,
) -> np.ndarray:
    signal_length = len(total_energy)
    if signal_length > 2 * boundary_frames:
        valid_start = boundary_frames
        valid_end = signal_length - boundary_frames
    else:
        margin = round(0.1 * signal_length)
        valid_start = max(0, margin)
        valid_end = min(signal_length, signal_length - margin)

    valid_energy = total_energy[valid_start:valid_end]
    if len(valid_energy) < 5:
        return np.array([], dtype=int)

    mean_energy = valid_energy.mean()
    std_energy = valid_energy.std()
    threshold = mean_energy + energy_threshold_factor * std_energy
    peak_locs, _ = find_peaks(total_energy, height=threshold)
    valid_peaks = peak_locs[(peak_locs >= valid_start) & (peak_locs < valid_end)]
    return np.sort(valid_peaks.astype(int))


def filter_singularity_points(raw_points: np.ndarray, filter_frames: int) -> np.ndarray:
    if len(raw_points) == 0:
        return np.array([], dtype=int)

    filtered_points: List[int] = []
    for current_point in raw_points:
        should_keep = True
        for kept_point in filtered_points:
            if abs(int(current_point) - int(kept_point)) <= filter_frames:
                should_keep = False
                break
        if should_keep:
            filtered_points.append(int(current_point))

    return np.array(filtered_points, dtype=int)


def is_potential_success_pair(
    y_coords: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_y_displacement: float,
) -> bool:
    if start_idx >= end_idx or start_idx < 0 or end_idx >= len(y_coords):
        return False

    segment_y = y_coords[start_idx : end_idx + 1]
    return (segment_y.max() - segment_y.min()) >= min_y_displacement


def find_adjacent_success_pairs(
    y_coords: np.ndarray,
    singularity_points: np.ndarray,
    min_y_displacement: float,
    max_frames_duration: int,
) -> np.ndarray:
    success_pairs = []
    for i in range(len(singularity_points) - 1):
        start_idx = int(singularity_points[i])
        end_idx = int(singularity_points[i + 1])
        if (end_idx - start_idx) > max_frames_duration:
            continue
        if is_potential_success_pair(y_coords, start_idx, end_idx, min_y_displacement):
            success_pairs.append([start_idx, end_idx])

    if not success_pairs:
        return np.empty((0, 2), dtype=int)
    return np.array(success_pairs, dtype=int)


def validate_success_pair(
    segment_data: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_y_displacement: float,
    post_event_check_frames: int,
) -> bool:
    y_coords = segment_data["Local_Y"].to_numpy(dtype=float)
    lane_ids = segment_data["Lane_ID"].astype(str).to_numpy()

    segment_y = y_coords[start_idx : end_idx + 1]
    max_displacement = segment_y.max() - segment_y.min()
    if max_displacement < min_y_displacement:
        return False

    start_lane = lane_ids[start_idx]
    end_lane = lane_ids[end_idx]

    # Keep the MATLAB logic: a successful lane change must start and end in different lanes.
    if start_lane == end_lane:
        return False

    # Exclude events that enter the ramp because the current study focuses on
    # interactions during lane changes on the mainline, not topology-driven exits.
    if end_lane == "ramp":
        return False

    post_end_idx = min(end_idx + post_event_check_frames, len(lane_ids) - 1)
    if post_end_idx > end_idx:
        post_event_lanes = lane_ids[end_idx : post_end_idx + 1]
        if np.any(post_event_lanes != end_lane):
            return False

    return True


def compress_lane_path(lane_ids: np.ndarray) -> List[str]:
    path: List[str] = []
    for lane in lane_ids.astype(str):
        if not path or path[-1] != lane:
            path.append(lane)
    return path


def is_adjacent_lane_transition(from_lane: str, to_lane: str) -> bool:
    if from_lane == to_lane:
        return False
    if "ramp" in (from_lane, to_lane):
        return False
    try:
        return abs(int(from_lane) - int(to_lane)) == 1
    except ValueError:
        return False


def split_event_by_lane_transitions(
    segment_data: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_y_displacement: float,
) -> List[List[object]]:
    y_coords = segment_data["Local_Y"].to_numpy(dtype=float)
    frames = segment_data["Frame_ID"].to_numpy()
    lane_ids = segment_data["Lane_ID"].astype(str).to_numpy()

    sub_lane_ids = lane_ids[start_idx : end_idx + 1]
    runs = []
    run_start = start_idx
    current_lane = lane_ids[start_idx]

    for idx in range(start_idx + 1, end_idx + 1):
        if lane_ids[idx] != current_lane:
            runs.append((run_start, idx - 1, current_lane))
            run_start = idx
            current_lane = lane_ids[idx]
    runs.append((run_start, end_idx, current_lane))

    split_events: List[List[object]] = []
    for i in range(len(runs) - 1):
        from_start, from_end, from_lane = runs[i]
        to_start, to_end, to_lane = runs[i + 1]

        if not is_adjacent_lane_transition(from_lane, to_lane):
            continue

        sub_start = from_start
        sub_end = to_start
        segment_y = y_coords[sub_start : sub_end + 1]
        max_disp = segment_y.max() - segment_y.min()
        if max_disp < min_y_displacement:
            continue

        split_events.append(
            [
                frames[sub_start],
                frames[sub_end],
                max_disp,
                sub_end - sub_start + 1,
                from_lane,
                to_lane,
                f"{from_lane}->{to_lane}",
            ]
        )

    return split_events


def remove_overlapping_events(events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.empty((0, 7), dtype=object)

    events = events[np.argsort(events[:, 0])]
    non_overlapping = [events[0]]

    for i in range(1, len(events)):
        current_start = events[i, 0]
        has_overlap = False
        for prev_event in non_overlapping:
            prev_end = prev_event[1]
            if current_start < prev_end:
                has_overlap = True
                break
        if not has_overlap:
            non_overlapping.append(events[i])

    return np.array(non_overlapping, dtype=object)


def detect_lane_change_success_wavelet_energy(
    segment_data: pd.DataFrame,
    min_y_displacement: float,
    max_frames_duration: int,
    wavelet_name: str,
    wavelet_scales: np.ndarray,
    boundary_frames: int,
    energy_threshold_factor: float,
    singularity_filter_frames: int,
    post_event_check_frames: int,
) -> np.ndarray:
    if len(segment_data) < 20:
        return np.empty((0, 7), dtype=object)

    y_coords = segment_data["Local_Y"].to_numpy(dtype=float)
    frames = segment_data["Frame_ID"].to_numpy()

    if np.std(y_coords) < 0.1:
        return np.empty((0, 7), dtype=object)

    try:
        coeffs, _ = pywt.cwt(y_coords, wavelet_scales, wavelet_name, sampling_period=SAMPLING_PERIOD)
        energy = np.abs(coeffs) ** 2
        total_energy = calculate_energy_exclude_boundary(energy, boundary_frames)

        singularity_points = detect_singularities_by_energy(
            total_energy,
            boundary_frames,
            energy_threshold_factor,
        )
        if len(singularity_points) < 2:
            return np.empty((0, 7), dtype=object)

        filtered_points = filter_singularity_points(singularity_points, singularity_filter_frames)
        if len(filtered_points) < 2:
            return np.empty((0, 7), dtype=object)

        candidate_pairs = find_adjacent_success_pairs(
            y_coords,
            filtered_points,
            min_y_displacement,
            max_frames_duration,
        )
        if len(candidate_pairs) == 0:
            return np.empty((0, 7), dtype=object)

        valid_events = []
        for start_idx, end_idx in candidate_pairs:
            if validate_success_pair(
                segment_data,
                int(start_idx),
                int(end_idx),
                min_y_displacement,
                post_event_check_frames,
            ):
                start_frame = frames[start_idx]
                end_frame = frames[end_idx]
                duration_frames = end_idx - start_idx + 1
                segment_y = y_coords[start_idx : end_idx + 1]
                max_disp = segment_y.max() - segment_y.min()
                split_events = split_event_by_lane_transitions(
                    segment_data,
                    int(start_idx),
                    int(end_idx),
                    min_y_displacement,
                )
                if split_events:
                    valid_events.extend(split_events)
                else:
                    lane_ids = segment_data["Lane_ID"].astype(str).to_numpy()
                    start_lane = lane_ids[start_idx]
                    end_lane = lane_ids[end_idx]
                    if not is_adjacent_lane_transition(start_lane, end_lane):
                        continue
                    valid_events.append(
                        [start_frame, end_frame, max_disp, duration_frames, start_lane, end_lane, f"{start_lane}->{end_lane}"]
                    )

        if not valid_events:
            return np.empty((0, 7), dtype=object)

        return remove_overlapping_events(np.array(valid_events, dtype=object))
    except Exception:
        return np.empty((0, 7), dtype=object)


def visualize_wavelet_energy_results(
    vehicle_data: pd.DataFrame,
    events: np.ndarray,
    vehicle_id: Union[int, str],
    wavelet_name: str,
    wavelet_scales: np.ndarray,
    boundary_frames: int,
    energy_threshold_factor: float,
    save_path: str,
) -> None:
    y_coords = vehicle_data["Local_Y"].to_numpy(dtype=float)
    frames = vehicle_data["Frame_ID"].to_numpy()
    lane_ids = vehicle_data["Lane_ID"].astype(str).to_numpy()

    coeffs, _ = pywt.cwt(y_coords, wavelet_scales, wavelet_name, sampling_period=SAMPLING_PERIOD)
    energy = np.abs(coeffs) ** 2
    total_energy = calculate_energy_exclude_boundary(energy, boundary_frames)
    singularity_points = detect_singularities_by_energy(total_energy, boundary_frames, energy_threshold_factor)
    x_min = float(frames.min())
    x_max = float(frames.max())

    fig, axes = plt.subplots(5, 1, figsize=(8, 8))

    ax = axes[0]
    ax.plot(frames, y_coords, "b-", linewidth=2)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Vehicle ID= {vehicle_id}", pad=4)
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()

    signal_length = len(frames)
    if signal_length > 2 * boundary_frames:
        left_boundary_idx = min(boundary_frames, signal_length - 1)
        right_boundary_idx = max(0, signal_length - boundary_frames - 1)
        boundary_start = frames[left_boundary_idx]
        boundary_end = frames[right_boundary_idx]
        ylim_vals = ax.get_ylim()
        ax.fill_between(
            [frames[0], boundary_start],
            ylim_vals[0],
            ylim_vals[1],
            color="red",
            alpha=0.18,
            zorder=0,
        )
        ax.fill_between(
            [boundary_end, frames[-1]],
            ylim_vals[0],
            ylim_vals[1],
            color="red",
            alpha=0.18,
            zorder=0,
        )

    for pos in singularity_points:
        if 0 <= pos < len(frames):
            ax.plot(frames[pos], y_coords[pos], "ro", markersize=6)
    for idx, pos in enumerate(singularity_points[:10], start=1):
        if 0 <= pos < len(frames):
            ax.text(frames[pos], y_coords[pos], f" S{idx}", fontsize=8, color="black")

    colors = ["g", "m", "c", "k", "y"]
    for i in range(len(events)):
        start_frame, end_frame = int(events[i, 0]), int(events[i, 1])
        start_idx = np.where(frames == start_frame)[0]
        end_idx = np.where(frames == end_frame)[0]
        if len(start_idx) == 0 or len(end_idx) == 0:
            continue
        start_idx = int(start_idx[0])
        end_idx = int(end_idx[0])
        color = colors[i % len(colors)]
        ax.plot(
            [start_frame, end_frame],
            [y_coords[start_idx], y_coords[end_idx]],
            "--",
            color=color,
            linewidth=3,
        )
        ax.plot(start_frame, y_coords[start_idx], "o", color=color, markersize=8, markerfacecolor=color)
        ax.plot(end_frame, y_coords[end_idx], "d", color=color, markersize=8, markerfacecolor=color)

    ax = axes[1]
    lane_order = ["1", "2", "3", "ramp"]
    lane_to_num = {label: idx + 1 for idx, label in enumerate(lane_order)}
    numeric_lane = np.array([lane_to_num.get(label, len(lane_order) + 1) for label in lane_ids])
    ax.plot(frames, numeric_lane, "g-", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Lane ID")
    ax.set_title("LC information", pad=4)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(lane_order)
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    for i in range(len(events)):
        start_frame, end_frame = int(events[i, 0]), int(events[i, 1])
        start_idx = np.where(frames == start_frame)[0]
        end_idx = np.where(frames == end_frame)[0]
        if len(start_idx) == 0 or len(end_idx) == 0:
            continue
        start_idx = int(start_idx[0])
        end_idx = int(end_idx[0])
        color = colors[i % len(colors)]
        ax.plot(start_frame, numeric_lane[start_idx], "o", color=color, markersize=8, markerfacecolor=color)
        ax.plot(end_frame, numeric_lane[end_idx], "d", color=color, markersize=8, markerfacecolor=color)

    ax = axes[2]
    ax.plot(frames, energy.sum(axis=0), "k-", linewidth=1)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Original Total Energy")
    ax.set_title("The original total energy (including boundary points)", pad=4)
    ax.grid(True)
    ax.set_xlim(x_min, x_max)

    ax = axes[3]
    ax.plot(frames, total_energy, "b-", linewidth=2, label="Total energy (excluding boundaries)")
    valid_energy = total_energy[boundary_frames : len(total_energy) - boundary_frames] if len(total_energy) > 2 * boundary_frames else total_energy
    if len(valid_energy) >= 5:
        threshold = valid_energy.mean() + energy_threshold_factor * valid_energy.std()
        ax.plot(
            frames,
            np.full_like(frames, threshold, dtype=float),
            "r--",
            linewidth=2,
            label="Detection threshold",
        )
    for pos in singularity_points:
        if 0 <= pos < len(frames):
            ax.plot(frames[pos], total_energy[pos], "ro", markersize=8, label="Singular points" if pos == singularity_points[0] else "")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Total Energy (No Boundary)")
    ax.set_title("Total energy (excluding boundaries) & Detection threshold", pad=4)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(x_min, x_max)

    ax = axes[4]
    im = ax.imshow(
        np.abs(coeffs),
        aspect="auto",
        extent=[x_min, x_max, wavelet_scales.max(), wavelet_scales.min()],
        cmap="afmhot",
    )
    ax.set_xlabel("Time b")
    ax.set_ylabel("Scales a")
    ax.set_title("Absolute Values for a = 1", pad=4)
    ax.set_xlim(x_min, x_max)
    for pos in singularity_points:
        if 0 <= pos < len(frames):
            ax.axvline(frames[pos], color="red", linestyle="--", linewidth=2)

    cax = inset_axes(
        ax,
        width="2.5%",
        height="85%",
        loc="center right",
        bbox_to_anchor=(0.06, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(im, cax=cax)

    fig.subplots_adjust(left=0.11, right=0.95, top=0.96, bottom=0.06, hspace=0.95)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("正在加载数据...")
    data = pd.read_csv(DATA_FILE)
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

    lane_col = resolve_column(data, ["Lane_ID"])
    y_col = resolve_column(data, ["Local_Y", "y"])
    vehicle_col = resolve_column(data, ["Vehicle_ID", "vehicle_id", "id"])
    frame_col = resolve_column(data, ["Frame_ID", "frame", "time"])

    data["Lane_ID"] = data[lane_col].astype(str)
    data["Local_Y"] = pd.to_numeric(data[y_col], errors="coerce") / POSITION_SCALE
    data["Vehicle_ID"] = data[vehicle_col]
    data["Frame_ID"] = data[frame_col]

    data = data.dropna(subset=["Local_Y", "Vehicle_ID", "Frame_ID", "Lane_ID"]).copy()
    unique_vehicles = pd.unique(data["Vehicle_ID"])

    print(f"数据加载完成，共 {len(unique_vehicles)} 个车辆，{len(data)} 行数据")

    results = []
    event_rows = []
    visualized = 0

    for vehicle_id in unique_vehicles:
        vehicle_data = data[data["Vehicle_ID"] == vehicle_id].copy()
        if len(vehicle_data) < MIN_TRAJECTORY_LENGTH:
            continue

        vehicle_data = vehicle_data.sort_values("Frame_ID").reset_index(drop=True)
        lanechange_events = detect_lane_change_success_wavelet_energy(
            vehicle_data,
            MIN_Y_DISPLACEMENT,
            MAX_FRAMES_DURATION,
            WAVELET_NAME,
            WAVELET_SCALES,
            BOUNDARY_FRAMES,
            ENERGY_THRESHOLD_FACTOR,
            SINGULARITY_FILTER_FRAMES,
            POST_EVENT_CHECK_FRAMES,
        )

        if len(lanechange_events) > 0:
            for event_idx, event in enumerate(lanechange_events, start=1):
                start_frame = event[0]
                end_frame = event[1]
                max_disp = event[2]
                duration_frames = event[3]
                from_lane = event[4]
                to_lane = event[5]
                lane_path = event[6]
                results.append([vehicle_id, start_frame, end_frame, max_disp, duration_frames, from_lane, to_lane, lane_path])

                window_start_frame = max(vehicle_data["Frame_ID"].min(), start_frame - EVENT_CONTEXT_FRAMES)
                window_end_frame = min(vehicle_data["Frame_ID"].max(), end_frame + EVENT_CONTEXT_FRAMES)

                event_segment = vehicle_data[
                    (vehicle_data["Frame_ID"] >= window_start_frame) & (vehicle_data["Frame_ID"] <= window_end_frame)
                ].copy()
                event_segment.insert(0, "Event_Index", event_idx)
                event_segment.insert(0, "Vehicle_ID_Event", vehicle_id)
                event_segment.insert(2, "Start_Frame", start_frame)
                event_segment.insert(3, "End_Frame", end_frame)
                event_segment.insert(4, "From_Lane", from_lane)
                event_segment.insert(5, "To_Lane", to_lane)
                event_segment.insert(6, "Lane_Path", lane_path)
                event_segment.insert(7, "Window_Start_Frame", window_start_frame)
                event_segment.insert(8, "Window_End_Frame", window_end_frame)
                event_rows.append(event_segment)

            should_visualize = SAVE_VISUALIZATIONS and (
                MAX_VISUALIZE_VEHICLES is None or visualized < MAX_VISUALIZE_VEHICLES
            )
            if should_visualize:
                fig_path = os.path.join(OUTPUT_FIG_DIR, f"vehicle_{vehicle_id}_lane_change.png")
                visualize_wavelet_energy_results(
                    vehicle_data,
                    lanechange_events,
                    vehicle_id,
                    WAVELET_NAME,
                    WAVELET_SCALES,
                    BOUNDARY_FRAMES,
                    ENERGY_THRESHOLD_FACTOR,
                    fig_path,
                )
                visualized += 1

    print("\n=== 换道成功检测结果总结 ===")
    if results:
        results_array = np.array(results, dtype=object)
        print(f"总共检测到 {len(results_array)} 个换道成功事件")
        print(f"涉及 {len(pd.unique(results_array[:, 0]))} 个车辆")
        result_table = pd.DataFrame(
            results_array,
            columns=[
                "Vehicle_ID",
                "Start_Frame",
                "End_Frame",
                "Max_Displacement",
                "Duration_Frames",
                "From_Lane",
                "To_Lane",
                "Lane_Path",
            ],
        )
        result_table.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(result_table)
        print(f"\n结果已输出到：{OUTPUT_FILE}")

        if event_rows:
            event_rows_table = pd.concat(event_rows, ignore_index=True)
            event_rows_table.to_csv(OUTPUT_EVENT_ROWS_FILE, index=False, encoding="utf-8-sig")
            print(f"换道事件轨迹明细已输出到：{OUTPUT_EVENT_ROWS_FILE}")

        if SAVE_VISUALIZATIONS:
            print(f"可视化图片已输出到：{OUTPUT_FIG_DIR}")
    else:
        print("未检测到任何换道成功事件")


if __name__ == "__main__":
    main()
