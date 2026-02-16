import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import scipy.io


class SubtractMean(object):
    """
    Matches ITrackerData.py behavior, but implemented more robustly:

    - meanImg is stored as HWC float32 in 0..255
    - convert to CHW float32 in 0..1 via (meanImg/255) and permute
    - subtract elementwise from input tensor (C,H,W) in 0..1
    """
    def __init__(self, meanImg: np.ndarray):
        mean = (meanImg / 255.0).astype(np.float32)     # HWC, 0..1
        mean = torch.from_numpy(mean).permute(2, 0, 1)  # CHW
        self.meanImg = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure mean lives on the same device/dtype as the incoming tensor
        mean = self.meanImg.to(device=tensor.device, dtype=tensor.dtype)
        return tensor.sub(mean)


def _load_mean_image(mat_path: Path) -> np.ndarray:
    """Loads 'image_mean' from a .mat file."""
    mat = scipy.io.loadmat(str(mat_path))
    return mat["image_mean"].astype(np.float32)


class CEALDataset(Dataset):
    """
    CEAL -> iTracker input adapter.

    Returns a tuple shaped for ITrackerModel.forward:
        (face, left_eye, right_eye, face_grid, label)

    Notes:
    - face/left/right are torch.float32 tensors shaped (3,224,224)
      produced by Resize -> ToTensor -> SubtractMean(mean image)
    - face_grid is torch.float32 shaped (25,25) (placeholder by default)
    - label is optional; return None or a dummy tensor if you don't have it yet
    """
    def __init__(
        self,
        ceal_root: str | Path,
        samples,  # list of sample records/paths, or a DataFrame-like
        itracker_pytorch_dir: str | Path,
        grid_size: int = 25,
        image_size: tuple[int, int] = (224, 224),
        return_label: bool = False,
    ):
        # --- CEAL-side bookkeeping ---
        self.ceal_root = Path(ceal_root)
        self.samples = samples
        self.return_label = return_label

        # --- iTracker preprocessing assets ---
        itracker_pytorch_dir = Path(itracker_pytorch_dir)

        # mean images (HWC float32, 0..255)
        self.face_mean = _load_mean_image(itracker_pytorch_dir / "mean_face_224.mat")
        self.left_mean = _load_mean_image(itracker_pytorch_dir / "mean_left_224.mat")
        self.right_mean = _load_mean_image(itracker_pytorch_dir / "mean_right_224.mat")

        # transforms: match ITrackerData.py (Resize -> ToTensor -> mean-image subtraction)
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

        # faceGrid config
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self.samples)

    def _make_placeholder_face_grid(self) -> torch.Tensor:
        """
        Placeholder face grid (grid_size x grid_size) until we compute a real face bounding box mapping.
        For now: a centered block of ones.
        """
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        c = self.grid_size // 2
        r = max(1, self.grid_size // 10)
        g[c - r:c + r + 1, c - r:c + r + 1] = 1.0
        return torch.from_numpy(g)

    def __getitem__(self, idx: int):
        # --- 1) load CEAL sample record ---
        # samples can be a list-like OR a DataFrame-like (iloc)
        try:
            sample = self.samples[idx]
        except Exception:
            sample = self.samples.iloc[idx]

        # --- 1) load CEAL image(s) ---
        # TODO: replace with your actual CEAL image path logic.
        #
        # Example if `sample` is a relative path string:
        #   img_path = self.ceal_root / sample
        #   img = Image.open(img_path).convert("RGB")
        #
        # Example if `sample` is a dict/Series with a key:
        #   img_path = self.ceal_root / sample["path"]
        #   img = Image.open(img_path).convert("RGB")

        raise NotImplementedError(
            "Wire CEAL loading: define `img_path` and load a PIL RGB image (or separate face/eye crops)."
        )

        # --- 2) crop face / left / right ---
        # TODO: replace with real cropping. Options:
        # - if CEAL already has crops, load them directly as PIL images.
        # - if you have bounding boxes/landmarks, crop here.
        #
        # face_pil = ...
        # left_pil = ...
        # right_pil = ...

        # --- 3-4) resize + normalize (done via transforms) ---
        # face = self.transform_face(face_pil)      # (3,224,224) float32
        # left = self.transform_left(left_pil)      # (3,224,224) float32
        # right = self.transform_right(right_pil)   # (3,224,224) float32

        # --- faceGrid (placeholder for now) ---
        # face_grid = self._make_placeholder_face_grid()  # (grid_size, grid_size) float32

        # --- 5) return tensors + label ---
        # if self.return_label:
        #     label = ...  # TODO: CEAL label (classification) or regression target
        # else:
        #     label = None
        #
        # return face, left, right, face_grid, label
