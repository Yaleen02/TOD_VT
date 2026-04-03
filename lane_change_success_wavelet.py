import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
from matplotlib.ticker import FixedLocator, FormatStrFormatter


matplotlib.use("Agg")

DEFAULT_INPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\Local_X_2000_straight.csv"
)
DEFAULT_WINDOW_SOURCE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_lane_id.csv"
)
DEFAULT_EVENTS_OUTPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_events_wavelet.csv"
)
DEFAULT_WINDOWS_OUTPUT = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_event_windows_wavelet.csv"
)
DEFAULT_FIGURE_DIR = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\lane_change_success_wavelet_figures"
)

FPS = 10
BOUNDARY_SECONDS = 6.5
BOUNDARY_FRAMES = int(BOUNDARY_SECONDS * FPS)
SINGULARITY_FILTER_FRAMES = 10
EVENT_CONTEXT_SECONDS = 5
EVENT_CONTEXT_FRAMES = int(EVENT_CONTEXT_SECONDS * FPS)
POST_EVENT_CHECK_SECONDS = 5
POST_EVENT_CHECK_FRAMES = int(POST_EVENT_CHECK_SECONDS * FPS)
MIN_POST_STABLE_SECONDS = 1.0
MIN_POST_STABLE_FRAMES = int(MIN_POST_STABLE_SECONDS * FPS)
MAX_TIME_DURATION = 8.0
MAX_FRAMES_DURATION = round(MAX_TIME_DURATION * FPS)
MIN_TRAJECTORY_LENGTH = 20
MIN_Y_DISPLACEMENT = 15.0
WAVELET_NAME = "mexh"
WAVELET_SCALES = np.arange(1, 17)
SAMPLING_PERIOD = 0.1
ENERGY_THRESHOLD_FACTOR = 0.5


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
    return (energy * mask).sum(axis=0)


def detect_singularities_by_energy(
    total_energy: np.ndarray,
    boundary_frames: int,
    energy_threshold_factor: float,
):
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
        return np.array([], dtype=int), np.nan

    mean_energy = valid_energy.mean()
    std_energy = valid_energy.std()
    threshold = mean_energy + energy_threshold_factor * std_energy
    peak_locs, _ = find_peaks(total_energy, height=threshold)
    valid_peaks = peak_locs[(peak_locs >= valid_start) & (peak_locs < valid_end)]
    return np.sort(valid_peaks.astype(int)), threshold


def filter_singularity_points(raw_points: np.ndarray, filter_frames: int) -> np.ndarray:
    if len(raw_points) == 0:
        return np.array([], dtype=int)

    filtered_points = []
    for current_point in raw_points:
        if not filtered_points or abs(int(current_point) - filtered_points[-1]) > filter_frames:
            filtered_points.append(int(current_point))
    return np.array(filtered_points, dtype=int)


def extract_valid_mainline_lanes(lane_ids: np.ndarray) -> List[str]:
    valid_lanes = []
    for lane in lane_ids:
        lane_str = str(lane)
        if lane_str in {"1", "2", "3"}:
            valid_lanes.append(lane_str)
    return valid_lanes


def get_stable_lane(lane_ids: np.ndarray, center_idx: int, window: int = 3) -> Optional[str]:
    start_idx = max(0, center_idx - window)
    end_idx = min(len(lane_ids), center_idx + window + 1)
    valid_lanes = extract_valid_mainline_lanes(lane_ids[start_idx:end_idx])
    if not valid_lanes:
        return None
    return pd.Series(valid_lanes).mode().iloc[0]


def has_min_stable_target_lane(
    lane_ids: np.ndarray,
    end_idx: int,
    target_lane: str,
    min_stable_frames: int,
) -> bool:
    post_lanes = extract_valid_mainline_lanes(lane_ids[end_idx:])
    if not post_lanes:
        return False

    consecutive = 0
    for lane in post_lanes:
        if lane == target_lane:
            consecutive += 1
            if consecutive >= min_stable_frames:
                return True
        else:
            consecutive = 0
    return False


def is_adjacent_lane_transition(from_lane: str, to_lane: str) -> bool:
    try:
        return abs(int(from_lane) - int(to_lane)) == 1
    except ValueError:
        return False


def find_lane_change_frame(
    frames: np.ndarray,
    lane_ids: np.ndarray,
    start_idx: int,
    end_idx: int,
    from_lane: str,
    to_lane: str,
) -> int:
    for idx in range(start_idx, end_idx + 1):
        if str(lane_ids[idx]) == to_lane:
            return int(frames[idx])
    for idx in range(start_idx, end_idx + 1):
        if str(lane_ids[idx]) != from_lane:
            return int(frames[idx])
    return int(frames[end_idx])


def detect_lane_change_success_wavelet_energy(
    vehicle_data: pd.DataFrame,
    min_y_displacement: float,
    max_frames_duration: int,
    wavelet_name: str,
    wavelet_scales: np.ndarray,
    boundary_frames: int,
    energy_threshold_factor: float,
    singularity_filter_frames: int,
    post_event_check_frames: int,
):
    if len(vehicle_data) < MIN_TRAJECTORY_LENGTH:
        return [], None

    segment_data = vehicle_data.sort_values("Frame_ID").reset_index(drop=True)
    y_coords = segment_data["Local_Y"].to_numpy(dtype=float)
    frames = segment_data["Frame_ID"].to_numpy(dtype=int)
    lane_ids = segment_data["Lane_ID"].astype(str).to_numpy()

    coeffs, _ = pywt.cwt(y_coords, wavelet_scales, wavelet_name, sampling_period=SAMPLING_PERIOD)
    energy = np.abs(coeffs) ** 2
    total_energy_original = energy.sum(axis=0)
    total_energy = calculate_energy_exclude_boundary(energy, boundary_frames)
    singularity_points, threshold = detect_singularities_by_energy(
        total_energy,
        boundary_frames,
        energy_threshold_factor,
    )
    filtered_points = filter_singularity_points(singularity_points, singularity_filter_frames)

    lanechange_events = []
    for i in range(len(filtered_points) - 1):
        start_idx = int(filtered_points[i])
        end_idx = int(filtered_points[i + 1])
        if (end_idx - start_idx) > max_frames_duration:
            continue

        segment_y = y_coords[start_idx : end_idx + 1]
        if (segment_y.max() - segment_y.min()) < min_y_displacement:
            continue

        start_lane = get_stable_lane(lane_ids, start_idx)
        end_lane = get_stable_lane(lane_ids, end_idx)
        if start_lane is None or end_lane is None:
            continue
        if start_lane == end_lane:
            continue
        if not is_adjacent_lane_transition(start_lane, end_lane):
            continue

        post_end_idx = min(end_idx + post_event_check_frames, len(lane_ids))
        if not has_min_stable_target_lane(
            lane_ids[:post_end_idx],
            end_idx,
            end_lane,
            MIN_POST_STABLE_FRAMES,
        ):
            continue

        lane_change_frame = find_lane_change_frame(frames, lane_ids, start_idx, end_idx, start_lane, end_lane)
        lanechange_events.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_frame": int(frames[start_idx]),
                "end_frame": int(frames[end_idx]),
                "lane_change_frame": lane_change_frame,
                "max_displacement": float(segment_y.max() - segment_y.min()),
                "duration_frames": int(end_idx - start_idx + 1),
                "from_lane": start_lane,
                "to_lane": end_lane,
                "lane_path": f"{start_lane}->{end_lane}",
            }
        )

    debug = {
        "frames": frames,
        "lane_ids": lane_ids,
        "y_coords": y_coords,
        "coeffs": coeffs,
        "energy": energy,
        "total_energy_original": total_energy_original,
        "total_energy": total_energy,
        "threshold": threshold,
        "singularity_points": singularity_points,
        "filtered_points": filtered_points,
    }
    return lanechange_events, debug


def visualize_wavelet_energy_results(
    vehicle_data: pd.DataFrame,
    events: List[Dict],
    debug: Dict,
    vehicle_id: str,
    figure_path: Path,
):
    frames = debug["frames"]
    y_coords = debug["y_coords"]
    lane_ids = pd.to_numeric(pd.Series(debug["lane_ids"]), errors="coerce").to_numpy(dtype=float)
    y_display = float(np.nanmax(y_coords) + np.nanmin(y_coords)) - y_coords
    total_energy_original = debug["total_energy_original"]
    total_energy = debug["total_energy"]
    threshold = debug["threshold"]
    singularity_points = debug["singularity_points"]
    coeff_energy = debug["energy"]

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
    fig.suptitle(f"Vehicle_ID= {vehicle_id}")

    axes[0].plot(frames, y_display, color="blue", linewidth=2)
    axes[0].set_title("Adjusted Y Coordinate")
    axes[0].set_ylabel("Adjusted Y")
    axes[0].grid(True, alpha=0.3)
    for event in events:
        axes[0].axvspan(event["start_frame"], event["end_frame"], color="red", alpha=0.18)
        start_x = event["start_frame"]
        end_x = event["end_frame"]
        y_start = y_display[event["start_idx"]]
        y_end = y_display[event["end_idx"]]
        axes[0].plot(start_x, y_start, "o", color="limegreen", markersize=9)
        axes[0].plot(end_x, y_end, "D", color="limegreen", markersize=8)
        axes[0].plot([start_x, end_x], [y_start, y_end], linestyle=(0, (5, 5)), color="limegreen", linewidth=2.5)
        axes[0].text(start_x + 1, y_start, "s1", color="navy", fontsize=10)
        axes[0].text(end_x + 1, y_end, "s2", color="navy", fontsize=10)
    axes[0].invert_yaxis()

    axes[1].plot(frames, lane_ids, color="lime", linewidth=2.5, marker="o", markersize=4)
    axes[1].set_title("LC information")
    axes[1].set_ylabel("Lane ID")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(3.05, 0.95)
    axes[1].yaxis.set_major_locator(FixedLocator([1, 2, 3]))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%d"))
    for event in events:
        start_x = event["start_frame"]
        end_x = event["end_frame"]
        start_lane = float(event["from_lane"])
        end_lane = float(event["to_lane"])
        axes[1].plot(start_x, start_lane, "o", color="lime", markersize=9)
        axes[1].plot(end_x, end_lane, "D", color="lime", markersize=8)
        axes[1].plot([start_x, end_x], [start_lane, end_lane], linestyle=(0, (5, 5)), color="lime", linewidth=2.5)

    axes[2].plot(frames, total_energy_original, color="black", linewidth=1.5)
    axes[2].set_title("The original total energy (including boundary points)")
    axes[2].set_ylabel("Original Total Energy")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(frames, total_energy, color="blue", linewidth=2, label="Total energy (excluding boundaries)")
    if np.isfinite(threshold):
        axes[3].axhline(threshold, color="red", linestyle=(0, (5, 4)), linewidth=1.8, label="Detection threshold")
    if len(singularity_points) > 0:
        axes[3].plot(
            frames[singularity_points],
            total_energy[singularity_points],
            "o",
            color="red",
            markersize=8,
            label="Singular points",
        )
    axes[3].set_title("Total energy (excluding boundaries) & Detection threshold")
    axes[3].set_ylabel("Total Energy (No Boundary)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    image = axes[4].imshow(
        coeff_energy,
        aspect="auto",
        origin="lower",
        extent=[frames.min(), frames.max(), WAVELET_SCALES.min(), WAVELET_SCALES.max()],
        cmap="pink",
    )
    axes[4].set_title("Absolute Values for a = 1 2 3...")
    axes[4].set_ylabel("Scales a")
    axes[4].set_xlabel("Time b")
    fig.colorbar(image, ax=axes[4], fraction=0.03, pad=0.02)

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_event_window_rows(
    vehicle_data: pd.DataFrame,
    event: dict,
    event_index: int,
):
    start_frame = event["lane_change_frame"] - EVENT_CONTEXT_FRAMES
    end_frame = event["lane_change_frame"] + EVENT_CONTEXT_FRAMES
    event_window = vehicle_data[
        (vehicle_data["Frame_ID"] >= start_frame) & (vehicle_data["Frame_ID"] <= end_frame)
    ].copy()
    event_window["Event_Index"] = event_index
    event_window["Vehicle_ID_Event"] = event["Vehicle_ID"]
    event_window["Start_Frame"] = event["start_frame"]
    event_window["End_Frame"] = event["end_frame"]
    event_window["Lane_Change_Frame"] = event["lane_change_frame"]
    event_window["From_Lane"] = event["from_lane"]
    event_window["To_Lane"] = event["to_lane"]
    event_window["Lane_Path"] = event["lane_path"]
    event_window["Window_Start_Frame"] = start_frame
    event_window["Window_End_Frame"] = end_frame
    event_window["Time_Offset"] = (event_window["Frame_ID"] - event["lane_change_frame"]) / FPS
    return event_window


def build_parser():
    parser = argparse.ArgumentParser(description="Detect successful lane changes using wavelet energy.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input trajectory CSV.")
    parser.add_argument("--window-source", default=DEFAULT_WINDOW_SOURCE, help="Full trajectory CSV used to extract +/-5s event windows.")
    parser.add_argument("--events-output", default=DEFAULT_EVENTS_OUTPUT, help="Event summary CSV.")
    parser.add_argument("--windows-output", default=DEFAULT_WINDOWS_OUTPUT, help="Event window rows CSV.")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR, help="Figure output directory.")
    return parser


def main():
    args = build_parser().parse_args()
    input_file = Path(args.input)
    window_source_file = Path(args.window_source)
    events_output = Path(args.events_output)
    windows_output = Path(args.windows_output)
    figure_dir = Path(args.figure_dir)

    print(f"读取输入数据: {input_file}")
    df = pd.read_csv(input_file)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(str)
    df["Lane_ID"] = pd.to_numeric(df["Lane_ID"], errors="coerce").astype("Int64")
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    print(f"读取窗口源数据: {window_source_file}")
    full_df = pd.read_csv(window_source_file)
    full_df["Vehicle_ID"] = full_df["Vehicle_ID"].astype(str)
    full_df["Lane_ID"] = pd.to_numeric(full_df["Lane_ID"], errors="coerce").astype("Int64")
    full_df = full_df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
    full_lookup = {vehicle_id: group.copy() for vehicle_id, group in full_df.groupby("Vehicle_ID", sort=False)}

    events = []
    event_windows = []
    visualized = 0

    grouped = df.groupby("Vehicle_ID", sort=False)
    for vehicle_id, vehicle_data in grouped:
        vehicle_data = vehicle_data.sort_values("Frame_ID").reset_index(drop=True)
        if len(vehicle_data) < MIN_TRAJECTORY_LENGTH:
            continue

        vehicle_events, debug = detect_lane_change_success_wavelet_energy(
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

        if not vehicle_events:
            continue

        print(f"车辆 {vehicle_id} 检测到 {len(vehicle_events)} 个换道成功事件")
        for event_index, event in enumerate(vehicle_events, start=1):
            full_vehicle_data = full_lookup.get(vehicle_id)
            if full_vehicle_data is None:
                continue

            event["Vehicle_ID"] = vehicle_id
            events.append(
                {
                    "Vehicle_ID": vehicle_id,
                    "Event_Index": event_index,
                    "Start_Frame": event["start_frame"],
                    "End_Frame": event["end_frame"],
                    "Lane_Change_Frame": event["lane_change_frame"],
                    "Max_Displacement": event["max_displacement"],
                    "Duration_Frames": event["duration_frames"],
                    "Duration_Seconds": event["duration_frames"] / FPS,
                    "From_Lane": event["from_lane"],
                    "To_Lane": event["to_lane"],
                    "Lane_Path": event["lane_path"],
                }
            )
            event_windows.append(build_event_window_rows(full_vehicle_data, event, event_index))

        visualize_wavelet_energy_results(
            vehicle_data,
            vehicle_events,
            debug,
            vehicle_id,
            figure_dir / f"vehicle_{vehicle_id}_wavelet.png",
        )
        visualized += 1

    events_df = pd.DataFrame(events)
    windows_df = pd.concat(event_windows, ignore_index=True) if event_windows else pd.DataFrame()

    events_output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(events_output, index=False)
    windows_df.to_csv(windows_output, index=False)

    print(f"换道成功事件已输出到: {events_output}")
    print(f"前后5s窗口轨迹已输出到: {windows_output}")
    print(f"可视化图片目录: {figure_dir}")
    print(f"共检测到 {len(events_df)} 个成功事件，生成 {visualized} 张图片")


if __name__ == "__main__":
    main()
