import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

H5_PATH = "/data/Half_GazeCapture.h5"
OUT_DIR = Path("/data/gazecapture_parquet_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Controls
BATCH_FRAMES = 512
ROWS_PER_FILE = 100_000
COMPRESSION = "snappy"

def safe_mean_std_uint8(batch: np.ndarray):
    x = batch.astype(np.float32)
    mean = x.mean(axis=(1, 2, 3))
    std = x.std(axis=(1, 2, 3))
    return mean, std

def write_part(rows, part_idx: int):
    df = pd.DataFrame(rows)

    # dtypes
    df["sample_id"] = df["sample_id"].astype("string")
    df["subject_id"] = df["subject_id"].astype("string")
    df["frame_id"] = df["frame_id"].astype(np.int32)

    for c in [
        "PoG_x","PoG_y",
        "left_eye_mean","left_eye_std",
        "right_eye_mean","right_eye_std",
        "face_mean","face_std"
    ]:
        df[c] = df[c].astype(np.float32)

    table = pa.Table.from_pandas(df, preserve_index=False)
    out_path = OUT_DIR / f"gazecapture_features_part_{part_idx:05d}.parquet"
    pq.write_table(table, out_path, compression=COMPRESSION)
    print(f"✔ wrote {len(df):,} rows → {out_path.name}")
    return out_path

def main():
    part_idx = 0
    rows = []

    with h5py.File(H5_PATH, "r") as h5:
        subject_ids = list(h5.keys())
        print(f"Subjects: {len(subject_ids)}")

        for s_i, subject_id in enumerate(subject_ids, start=1):
            grp = h5[subject_id]

            required = ["face", "left_eye", "right_eye", "PoG"]
            if not all(k in grp for k in required):
                continue

            face = grp["face"]
            left = grp["left_eye"]
            right = grp["right_eye"]
            pog = grp["PoG"]

            n = min(len(face), len(left), len(right), len(pog))
            if n == 0:
                continue

            for start in range(0, n, BATCH_FRAMES):
                end = min(start + BATCH_FRAMES, n)

                face_b = face[start:end]
                left_b = left[start:end]
                right_b = right[start:end]
                pog_b = pog[start:end]

                face_mean, face_std = safe_mean_std_uint8(face_b)
                left_mean, left_std = safe_mean_std_uint8(left_b)
                right_mean, right_std = safe_mean_std_uint8(right_b)

                for j in range(end - start):
                    frame_id = start + j
                    sample_id = f"{subject_id}_{frame_id}"

                    rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "frame_id": frame_id,
                        "PoG_x": float(pog_b[j][0]),
                        "PoG_y": float(pog_b[j][1]),
                        "left_eye_mean": float(left_mean[j]),
                        "left_eye_std": float(left_std[j]),
                        "right_eye_mean": float(right_mean[j]),
                        "right_eye_std": float(right_std[j]),
                        "face_mean": float(face_mean[j]),
                        "face_std": float(face_std[j]),
                    })

                if len(rows) >= ROWS_PER_FILE:
                    write_part(rows, part_idx)
                    part_idx += 1
                    rows = []

            print(f"Done subject {subject_id} ({s_i}/{len(subject_ids)}) → {n} frames")

    if rows:
        write_part(rows, part_idx)

    print("✅ Conversion complete")
    print(f"Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
