import pandas as pd


INPUT_FILE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_data_lane_id.csv"
)
OUTPUT_FILE = (
    r"D:\WPS云盘\1634812337\WPS企业云盘\东南大学\我的企业文档\2026\2026.04\TOD_VT"
    r"\dataset_processed\trajectory_data_lane_id_basic_columns.csv"
)

COLUMNS = [
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


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = df[COLUMNS].copy()
    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved trimmed file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
