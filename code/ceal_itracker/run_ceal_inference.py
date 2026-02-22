"""
run_ceal_inference.py

Run pretrained iTracker inference on the CEAL dataset.

Prerequisites:
  - manifest.csv produced by ceal_data_preprocessing.ipynb (Step 2 pipeline)
  - Pretrained iTracker checkpoint (checkpoint.pth.tar)
  - Mean image .mat files (mean_face_224.mat, mean_left_224.mat, mean_right_224.mat)

Usage:
  python run_ceal_inference.py \
      --manifest_csv /path/to/ceal_itracker_artifacts/manifest.csv \
      --itracker_dir /path/to/iTracker-pytorch \
      --output_csv  /path/to/ceal_predictions.csv \
      --batch_size 64

Output:
  CSV with columns:
    sample_id, subject, aug_id,
    head_pose_deg, gaze_vertical_deg, gaze_horizontal_deg, gt_label,
    pred_x, pred_y
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------- project imports (must be on PYTHONPATH or in same directory) ------
from ITrackerModel import ITrackerModel
from ceal_data import CEALManifestDataset


# ──────────────────────────────────────────────────────────────────────────────
# Label mapping (mirrors ceal_data_preprocessing.ipynb create_labels logic)
# ──────────────────────────────────────────────────────────────────────────────
def degrees_to_label(gaze_h: float, gaze_v: float) -> str:
    """
    Convert gaze horizontal/vertical degrees to a directional intent label.
    Matches the create_labels() function from the preprocessing notebook.
    """
    if gaze_v == 0 and gaze_h == 0:
        return "straight"

    if abs(gaze_h) > abs(gaze_v):
        return "left" if gaze_h < 0 else "right"

    if abs(gaze_v) > abs(gaze_h):
        return "down" if gaze_v < 0 else "up"

    # tie → horizontal wins
    return "left" if gaze_h < 0 else "right"


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader (handles DataParallel 'module.' prefix)
# ──────────────────────────────────────────────────────────────────────────────
def load_itracker_checkpoint(model: ITrackerModel, ckpt_path: str, device: torch.device):
    """
    Load a pretrained iTracker checkpoint, handling both DataParallel and
    bare state dicts.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract state dict from wrapper if needed
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "?")
        best_loss = ckpt.get("best_prec1", "?")
        print(f"  Checkpoint from epoch {epoch}, best loss {best_loss}")
    else:
        state = ckpt

    # Strip 'module.' prefix if saved from DataParallel
    cleaned_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned_state[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"  Warning — missing keys: {missing}")
    if unexpected:
        print(f"  Warning — unexpected keys: {unexpected}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataloader, device):
    """
    Run iTracker inference over the entire dataloader.

    Returns:
        all_sample_ids : list[str]
        all_preds      : np.ndarray of shape (N, 2) — predicted (x, y) gaze coords
    """
    model.eval()

    all_sample_ids = []
    all_preds = []

    total_batches = len(dataloader)
    t0 = time.time()

    for batch_idx, (sample_ids, faces, lefts, rights, facegrids) in enumerate(dataloader):
        # Move to device
        faces = faces.to(device)
        lefts = lefts.to(device)
        rights = rights.to(device)
        facegrids = facegrids.to(device)

        # Forward pass → (B, 2)
        outputs = model(faces, lefts, rights, facegrids)

        # Collect
        all_sample_ids.extend(sample_ids)
        all_preds.append(outputs.cpu().numpy())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            print(f"  Batch [{batch_idx + 1}/{total_batches}]  "
                  f"({elapsed:.1f}s elapsed)")

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 2)
    return all_sample_ids, all_preds


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run iTracker inference on CEAL dataset artifacts."
    )
    parser.add_argument(
        "--manifest_csv", type=str, required=True,
        help="Path to manifest.csv from the preprocessing pipeline."
    )
    parser.add_argument(
        "--itracker_dir", type=str, required=True,
        help="Path to iTracker-pytorch directory containing checkpoint.pth.tar "
             "and mean_*.mat files."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pth.tar",
        help="Checkpoint filename inside --itracker_dir (default: checkpoint.pth.tar)."
    )
    parser.add_argument(
        "--output_csv", type=str, default="ceal_predictions.csv",
        help="Path to write the predictions CSV."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference (default: 64)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device ('cpu' or 'cuda'). Default: auto-detect."
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    itracker_dir = Path(args.itracker_dir)
    manifest_csv = Path(args.manifest_csv)

    print(f"Loading dataset from: {manifest_csv}")
    dataset = CEALManifestDataset(
        manifest_csv=manifest_csv,
        itracker_pytorch_dir=itracker_dir,
        image_size=(224, 224),
        grid_size=25,
        only_ok=True,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,       # preserve order for matching predictions
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ITrackerModel()
    ckpt_path = str(itracker_dir / args.checkpoint)
    model = load_itracker_checkpoint(model, ckpt_path, device)
    model = model.to(device)

    # ── Inference ─────────────────────────────────────────────────────────
    print("Running inference...")
    sample_ids, preds = run_inference(model, dataloader, device)
    print(f"  Inference complete: {len(sample_ids)} predictions")

    # ── Build results DataFrame ───────────────────────────────────────────
    # Read manifest to join ground-truth metadata
    manifest_df = pd.read_csv(manifest_csv)
    manifest_df = manifest_df[manifest_df["status"] == "ok"].reset_index(drop=True)

    # Create predictions DataFrame keyed by sample_id
    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "pred_x": preds[:, 0],
        "pred_y": preds[:, 1],
    })

    # Merge predictions with manifest metadata
    results = manifest_df.merge(pred_df, on="sample_id", how="inner")

    # Add ground-truth directional label
    results["gt_label"] = results.apply(
        lambda r: degrees_to_label(r["gaze_horizontal_deg"], r["gaze_vertical_deg"]),
        axis=1,
    )

    # Select output columns
    output_cols = [
        "sample_id",
        "subject",
        "aug_id",
        "head_pose_deg",
        "gaze_vertical_deg",
        "gaze_horizontal_deg",
        "gt_label",
        "pred_x",
        "pred_y",
    ]
    results = results[output_cols]

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"  Shape: {results.shape}")
    print(f"  Label distribution:\n{results['gt_label'].value_counts().to_string()}")

    # ── Quick summary stats ───────────────────────────────────────────────
    print(f"\n  pred_x range: [{results['pred_x'].min():.4f}, {results['pred_x'].max():.4f}]")
    print(f"  pred_y range: [{results['pred_y'].min():.4f}, {results['pred_y'].max():.4f}]")


if __name__ == "__main__":
    main()