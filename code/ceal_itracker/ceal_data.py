import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import scipy.io
import pandas as pd


class SubtractMean(object):
    """
    Subtract a per-pixel mean image (loaded from iTracker .mat mean files).

    meanImg expected as HWC float32 in 0..255.
    We convert to CHW float32 in 0..1 to match torchvision ToTensor output.
    """
    def __init__(self, meanImg: np.ndarray):
        mean = (meanImg / 255.0).astype(np.float32)     # HWC, 0..1
        mean = torch.from_numpy(mean).permute(2, 0, 1)  # CHW
        self.meanImg = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.meanImg.to(device=tensor.device, dtype=tensor.dtype)
        return tensor.sub(mean)


def _load_mean_image(mat_path: Path) -> np.ndarray:
    """Loads 'image_mean' from a .mat file as float32."""
    mat = scipy.io.loadmat(str(mat_path))
    return mat["image_mean"].astype(np.float32)


class CEALManifestDataset(Dataset):
    """
    Dataset that reads iTracker-ready artifacts from a manifest.csv produced by Step 2 pipeline.

    Each item returns:
      (sample_id, face, left, right, facegrid)

    Where:
      - face/left/right: torch.float32 (3,224,224)
      - facegrid: torch.float32 (625,)  (flattened 25x25)
    """
    def __init__(
        self,
        manifest_csv: str | Path,
        itracker_pytorch_dir: str | Path,
        image_size: tuple[int, int] = (224, 224),
        grid_size: int = 25,
        only_ok: bool = True,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.itracker_pytorch_dir = Path(itracker_pytorch_dir)
        self.grid_size = grid_size

        # Load manifest
        df = pd.read_csv(self.manifest_csv)

        # Keep only successful rows unless you explicitly want failures
        if only_ok:
            df = df[df["status"] == "ok"].reset_index(drop=True)

        # Store as DataFrame for easy column access in __getitem__
        self.df = df

        # Load mean images (HWC float32, 0..255)
        self.face_mean = _load_mean_image(self.itracker_pytorch_dir / "mean_face_224.mat")
        self.left_mean = _load_mean_image(self.itracker_pytorch_dir / "mean_left_224.mat")
        self.right_mean = _load_mean_image(self.itracker_pytorch_dir / "mean_right_224.mat")

        # Preprocessing expected by pretrained iTracker checkpoint:
        # Resize -> ToTensor (0..1) -> subtract mean image (also 0..1)
        self.transform_face = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.face_mean),
        ])
        self.transform_left = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.left_mean),
        ])
        self.transform_right = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.right_mean),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, path: str) -> Image.Image:
        """Load an RGB image from disk as PIL."""
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        sample_id = row["sample_id"]

        # Paths produced by Step 2 pipeline
        face_path = row["face_path"]
        left_path = row["left_path"]
        right_path = row["right_path"]
        facegrid_path = row["facegrid_path"]

        # 1) Load crops as PIL RGB
        face_pil = self._load_rgb(face_path)
        left_pil = self._load_rgb(left_path)
        right_pil = self._load_rgb(right_path)

        # 2) Apply preprocessing transforms (resize is idempotent if already 224x224)
        face = self.transform_face(face_pil)     # (3,224,224)
        left = self.transform_left(left_pil)     # (3,224,224)
        right = self.transform_right(right_pil)  # (3,224,224)

        # 3) Load facegrid (625,) float32
        facegrid_np = np.load(facegrid_path).astype(np.float32)

        # Safety: enforce expected shape
        expected = self.grid_size * self.grid_size
        if facegrid_np.shape != (expected,):
            raise ValueError(f"facegrid has shape {facegrid_np.shape}, expected ({expected},) at {facegrid_path}")

        facegrid = torch.from_numpy(facegrid_np)  # (625,)

        return sample_id, face, left, right, facegrid