"""
inference_dynamic.py
====================
Dynamic GazeNet inference engine for the Eye2Voice testing backend.

Automatically detects which model architecture is encoded in a .pth file by
inspecting the state dict keys and shapes, then instantiates and loads the
correct model class.

Supported architectures
-----------------------
GazeNet (v9, 5-class)
  - State dict keys: eye_cnn.*, face_cnn.*, fc.*
  - Output: fc.6.weight shape [5, 256]
  - Classes: 0=Straight 1=Up 2=Down 3=Left 4=Right

GazeNetM5 (m5c, 4-class + geometric features)
  - State dict keys: eye_cnn.*, face_cnn.*, geo_mlp.*, fc.*
  - Output: fc.6.weight shape [4, 256]
  - Classes: 0=Up 1=Down 2=Left 3=Right
  - Requires 7 geometric landmark features as additional input

Key functions
-------------
  load_model_dynamic(path)           → (model, model_meta)
  predict_dynamic(model, model_meta, face_b64, left_eye_b64, right_eye_b64,
                  geo_features=None) → dict
"""

import io
import base64
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture Definitions
# ─────────────────────────────────────────────────────────────────────────────

class GazeNet(nn.Module):
    """
    5-class gaze direction CNN (GazeNet v9).

    Three parallel input streams:
      - eye_cnn (shared weights): left eye + right eye, each (batch,3,48,48)
      - face_cnn: full face crop (batch,3,112,112)
      - fc: fusion of all three → 5 class logits

    Classes: 0=Straight 1=Up 2=Down 3=Left 4=Right
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # Shared eye CNN: (3,48,48) → (128,6,6) = 4608 values per eye
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → (32,48,48)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # → (32,24,24)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (64,24,24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # → (64,12,12)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# → (128,12,12)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # → (128,6,6)
        )

        # Face CNN: (3,112,112) → (256,3,3) = 2304 values
        self.face_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # → (32,56,56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # → (32,28,28)
            nn.Conv2d(32, 64, kernel_size=5, padding=2),             # → (64,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # → (64,14,14)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),            # → (128,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # → (128,7,7)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),           # → (256,7,7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # → (256,3,3)
        )

        # Fusion: 4608 + 4608 + 2304 = 11520 → num_classes
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6 * 2 + 256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, left_eye, right_eye, face):
        left_flat  = self.eye_cnn(left_eye).flatten(1)
        right_flat = self.eye_cnn(right_eye).flatten(1)
        face_flat  = self.face_cnn(face).flatten(1)
        combined   = torch.cat([left_flat, right_flat, face_flat], dim=1)
        return self.fc(combined)


class GazeNetM5(nn.Module):
    """
    4-class gaze direction CNN with geometric landmark features (m5c).

    Four input streams:
      - eye_cnn (shared): left eye + right eye, each (batch,3,48,48) → 4608 each
      - face_cnn: face crop (batch,3,112,112) → 2304
      - geo_mlp: 7 geometric MediaPipe features → 64
      - fc: fusion of all four → 4 class logits

    Geometric feature vector (7 values, in order):
      0: left_iris_h      — left iris horizontal ratio in eye socket [0,1]
      1: right_iris_h     — right iris horizontal ratio in eye socket [0,1]
      2: iris_h_agreement — right_iris_h - left_iris_h (centered at 0)
      3: head_yaw         — head yaw angle (radians, left-right turn)
      4: head_pitch       — head pitch angle (radians, up-down tilt)
      5: z_tilt           — head roll angle (radians)
      6: z_nose_rel       — nose z depth relative to face centroid

    Default (neutral gaze): [0.5, 0.5, 0.0, 0.0, 0.35, -0.1, -0.26]

    Classes: 0=Up 1=Down 2=Left 3=Right
    """

    def __init__(self, num_classes=4, geo_feat_dim=7):
        super().__init__()

        # Same eye_cnn and face_cnn as GazeNet
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.face_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Geometric feature MLP: 7 → 64 → 64
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion: 4608 + 4608 + 2304 + 64 = 11584 → num_classes
        self.fc = nn.Sequential(
            nn.Linear(4608 * 2 + 2304 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, left_eye, right_eye, face, geo_features):
        left_flat  = self.eye_cnn(left_eye).flatten(1)
        right_flat = self.eye_cnn(right_eye).flatten(1)
        face_flat  = self.face_cnn(face).flatten(1)
        geo_feat   = self.geo_mlp(geo_features)
        combined   = torch.cat([left_flat, right_flat, face_flat, geo_feat], dim=1)
        return self.fc(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture Detection
# ─────────────────────────────────────────────────────────────────────────────

# Neutral geo feature defaults (used when frontend can't compute them)
GEO_DEFAULT = [0.5, 0.5, 0.0, 0.0, 0.35, -0.1, -0.26]

GEO_FEATURE_NAMES = [
    'left_iris_h',
    'right_iris_h',
    'iris_h_agreement',
    'head_yaw',
    'head_pitch',
    'z_tilt',
    'z_nose_rel',
]

# Label maps by num_classes
LABEL_MAP_5 = {0: 'straight', 1: 'up', 2: 'down', 3: 'left', 4: 'right'}
LABEL_MAP_4 = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}


def inspect_state_dict(state_dict: dict) -> dict:
    """
    Examine a state dict to determine which architecture it came from.

    Detection rules:
      - 'geo_mlp.0.weight' present → GazeNetM5
      - Otherwise → GazeNet
      - num_classes: state_dict['fc.6.weight'].shape[0]
      - geo_feat_dim: state_dict['geo_mlp.0.weight'].shape[1] (if present)

    Returns a model_meta dict (no model object — just metadata).
    """
    has_geo = 'geo_mlp.0.weight' in state_dict
    num_classes = state_dict['fc.6.weight'].shape[0]
    geo_feat_dim = int(state_dict['geo_mlp.0.weight'].shape[1]) if has_geo else 0
    total_params = sum(t.numel() for t in state_dict.values())

    if has_geo:
        architecture = 'GazeNetM5'
        label_map = LABEL_MAP_4
    else:
        architecture = 'GazeNet'
        label_map = LABEL_MAP_5 if num_classes == 5 else {i: str(i) for i in range(num_classes)}

    return {
        'architecture': architecture,
        'num_classes': int(num_classes),
        'has_geo': has_geo,
        'geo_feat_dim': geo_feat_dim,
        'geo_feature_names': GEO_FEATURE_NAMES if has_geo else [],
        'geo_default': GEO_DEFAULT if has_geo else [],
        'label_map': label_map,
        'input_shapes': {
            'left_eye':  [1, 3, 48, 48],
            'right_eye': [1, 3, 48, 48],
            'face':      [1, 3, 112, 112],
        },
        'total_params': total_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model_dynamic(model_path: str):
    """
    Load any supported GazeNet variant from a .pth state dict file.

    Inspects the state dict to detect the architecture, instantiates the
    correct model class, loads weights, and sets eval mode.

    Parameters
    ----------
    model_path : str
        Path to a .pth file saved with torch.save(model.state_dict(), path).

    Returns
    -------
    (model, model_meta)
        model     — nn.Module in eval() mode, on CPU
        model_meta — dict from inspect_state_dict() plus the model_path
    """
    logger.info(f'Loading model from {model_path}')

    # Load the raw state dict
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Detect architecture from state dict contents
    meta = inspect_state_dict(state_dict)
    meta['model_path'] = model_path

    logger.info(
        f"Detected: {meta['architecture']} "
        f"({meta['num_classes']} classes, "
        f"geo={'yes' if meta['has_geo'] else 'no'}, "
        f"params={meta['total_params']:,})"
    )

    # Instantiate the correct model class
    if meta['architecture'] == 'GazeNetM5':
        model = GazeNetM5(
            num_classes=meta['num_classes'],
            geo_feat_dim=meta['geo_feat_dim'],
        )
    else:
        model = GazeNet(num_classes=meta['num_classes'])

    # Load weights — strict=True ensures the architecture matches exactly
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info('Model loaded and ready.')
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

EYE_TRANSFORM = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def decode_image(b64_string: str) -> Image.Image:
    """Decode a base64 JPEG/PNG string (with or without data URL prefix) to PIL RGB."""
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def preprocess_inputs(face_b64, left_eye_b64, right_eye_b64, eye_size=48, face_size=112):
    """
    Decode base64 images and apply training-identical preprocessing.

    The eye_size and face_size parameters allow future flexibility if models
    are trained with different input resolutions (configurable via the UI).
    Default values match all current models (48×48 eyes, 112×112 face).

    Returns (face_tensor, left_eye_tensor, right_eye_tensor), each (1,3,H,W).
    """
    eye_tf = transforms.Compose([
        transforms.Resize((eye_size, eye_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    face_tf = transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    face_t     = face_tf(decode_image(face_b64)).unsqueeze(0)
    left_eye_t = eye_tf(decode_image(left_eye_b64)).unsqueeze(0)
    right_eye_t= eye_tf(decode_image(right_eye_b64)).unsqueeze(0)
    return face_t, left_eye_t, right_eye_t


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_dynamic(
    model,
    model_meta: dict,
    face_b64: str,
    left_eye_b64: str,
    right_eye_b64: str,
    geo_features=None,
    eye_size: int = 48,
    face_size: int = 112,
) -> dict:
    """
    Run inference on one frame using the dynamically loaded model.

    Parameters
    ----------
    model         : nn.Module — loaded by load_model_dynamic()
    model_meta    : dict     — metadata from load_model_dynamic()
    face_b64      : str      — base64 face crop
    left_eye_b64  : str      — base64 left eye crop (subject's left = camera right)
    right_eye_b64 : str      — base64 right eye crop
    geo_features  : list[float] | None
                    7 geometric features for GazeNetM5. If None and model has_geo,
                    uses GEO_DEFAULT (neutral). For GazeNet, ignored entirely.
    eye_size      : int      — resize target for eye crops (default 48)
    face_size     : int      — resize target for face crop (default 112)

    Returns
    -------
    dict with keys:
        direction         : str   — predicted direction label
        confidence        : float — probability of predicted class
        probabilities     : dict  — {label: probability} for all classes
        used_geo_default  : bool  — True if GEO_DEFAULT was substituted
    """
    face_t, left_t, right_t = preprocess_inputs(
        face_b64, left_eye_b64, right_eye_b64, eye_size, face_size
    )

    used_geo_default = False

    with torch.no_grad():
        if model_meta['has_geo']:
            if geo_features is None:
                geo_features = GEO_DEFAULT
                used_geo_default = True
            geo_t = torch.tensor([geo_features], dtype=torch.float32)  # (1, 7)
            logits = model(left_eye=left_t, right_eye=right_t, face=face_t, geo_features=geo_t)
        else:
            logits = model(left_eye=left_t, right_eye=right_t, face=face_t)

        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        prob_list = probs[0].tolist()

    label_map = model_meta['label_map']
    direction = label_map[pred_idx]
    prob_dict = {label_map[i]: round(prob_list[i], 4) for i in range(model_meta['num_classes'])}

    logger.info(f'Prediction: {direction} (conf={confidence:.3f}, geo_default={used_geo_default})')

    return {
        'direction':        direction,
        'confidence':       round(confidence, 4),
        'probabilities':    prob_dict,
        'used_geo_default': used_geo_default,
    }
