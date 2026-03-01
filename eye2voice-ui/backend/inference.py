"""
inference.py
============
Core GazeNet v9 inference logic — shared between the local Flask server and
the AWS Lambda handler. Keeping inference isolated here makes it easy to
test the model independently of any web framework.

What this file does:
  1. Defines the GazeNet architecture (must exactly match the training code)
  2. Loads the saved state dict from a .pth file
  3. Provides a single `predict()` function that accepts raw PIL Images and
     returns a gaze direction label and confidence score

Author note: This module has no Flask or Lambda dependencies, so it can be
imported by any Python script for quick testing from the command line.
"""

import io
import base64
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture Definition
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: This class definition MUST be identical to the one used during
# training (eye2voice/notebooks_MIT/Baseline.ipynb). PyTorch saves only the
# weight tensors — it does NOT save the class definition. If these layer
# definitions differ even slightly from training, load_state_dict() will fail
# or produce incorrect predictions.

class GazeNet(nn.Module):
    """
    Multi-stream CNN for 5-class gaze direction classification.

    Three parallel input streams:
      - eye_cnn (shared weights): processes left eye AND right eye independently
        using the exact same convolutional filters. Shared weights enforce the
        prior that both eyes should be analyzed identically.
      - face_cnn: processes the full face crop. The face provides context about
        head pose, which helps disambiguate gaze direction.
      - fc (fusion): concatenates all three flattened feature vectors and
        classifies into 5 directions.

    Input tensor shapes (what the model expects):
      - left_eye:  (batch, 3, 48, 48)   — 3 RGB channels, 48×48 pixels
      - right_eye: (batch, 3, 48, 48)   — same shape as left_eye
      - face:      (batch, 3, 112, 112) — 3 RGB channels, 112×112 pixels

    Output:
      - logits: (batch, 5) — raw scores before softmax
      - Classes: 0=Straight, 1=Up, 2=Down, 3=Left, 4=Right
    """

    def __init__(self, num_classes=5):
        super(GazeNet, self).__init__()

        # ── Eye CNN ───────────────────────────────────────────────────────────
        # Applied identically to both left and right eye crops.
        # Input:  (batch, 3, 48, 48)
        # Output: (batch, 128, 6, 6) → flattened to (batch, 4608)
        #
        # Architecture walkthrough:
        #   Conv1: 3 input channels → 32 feature maps, 3×3 kernel, same padding
        #          keeps spatial size at 48×48
        #   MaxPool: halves spatial dims → 24×24
        #   Conv2: 32 → 64 channels, still 24×24 after padding
        #   MaxPool: → 12×12
        #   Conv3: 64 → 128 channels, → 12×12
        #   MaxPool: → 6×6
        # Final spatial size: 6×6 with 128 channels = 4608 values per eye
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # → (32,48,48)
            nn.ReLU(inplace=True),    # inplace saves memory by reusing the tensor
            nn.MaxPool2d(kernel_size=2),                                           # → (32,24,24)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # → (64,24,24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                           # → (64,12,12)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),# → (128,12,12)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                           # → (128,6,6)
        )

        # ── Face CNN ──────────────────────────────────────────────────────────
        # Processes the full face crop (wider field of view than eye-only).
        # Input:  (batch, 3, 112, 112)
        # Output: (batch, 256, 3, 3) → flattened to (batch, 2304)
        #
        # Architecture walkthrough:
        #   Conv1: 3 → 32 channels, 7×7 kernel (larger to capture coarse features),
        #          stride=2 immediately halves spatial dims: 112 → 56; padding=3 preserves
        #   MaxPool: 56 → 28
        #   Conv2: 32 → 64, 5×5 kernel with same padding → stays 28×28
        #   MaxPool: 28 → 14
        #   Conv3: 64 → 128, 3×3 → 14×14
        #   MaxPool: 14 → 7
        #   Conv4: 128 → 256, 3×3 → 7×7
        #   MaxPool: 7 → 3 (floor division: 7//2 = 3)
        # Final: 3×3 with 256 channels = 2304 values
        self.face_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),  # → (32,56,56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                      # → (32,28,28)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),            # → (64,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                      # → (64,14,14)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),           # → (128,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                      # → (128,7,7)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),          # → (256,7,7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                      # → (256,3,3)
        )

        # ── Fusion FC Layers ──────────────────────────────────────────────────
        # Concatenated input size = 4608 (left eye) + 4608 (right eye) + 2304 (face)
        #                        = 11,520 total features
        #
        # Three linear layers progressively compress 11520 → 512 → 256 → 5
        # Dropout(0.5) during training randomly zeroes 50% of neurons each forward
        # pass, forcing the network to not rely on any single feature.
        # At inference time (model.eval()), dropout is automatically disabled.
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 6 * 6 * 2 + 256 * 3 * 3,  # = 11520
                      out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),    # disabled at eval() time automatically

            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=256, out_features=5),  # 5 gaze direction classes
            # No softmax here — CrossEntropyLoss expects raw logits during training
            # At inference, we apply softmax manually to get probabilities
        )

    def forward(self, left_eye, right_eye, face):
        """
        Forward pass through all three streams and fusion.

        Parameters
        ----------
        left_eye : Tensor (batch, 3, 48, 48)
            Subject's left eye crop (camera right), normalized to [-1, 1].
        right_eye : Tensor (batch, 3, 48, 48)
            Subject's right eye crop (camera left), normalized to [-1, 1].
        face : Tensor (batch, 3, 112, 112)
            Full face crop, normalized to [-1, 1].

        Returns
        -------
        Tensor (batch, 5) — raw logits (NOT probabilities).
        """
        # Process left eye through the eye CNN
        # eye_cnn expects (batch, 3, 48, 48) → produces (batch, 128, 6, 6)
        left_features = self.eye_cnn(left_eye)

        # Process right eye through the SAME eye CNN (shared weights)
        # This is key: the same filters learn to detect the same gaze patterns in both eyes
        right_features = self.eye_cnn(right_eye)

        # Process face crop through the deeper face CNN
        # face_cnn expects (batch, 3, 112, 112) → produces (batch, 256, 3, 3)
        face_features = self.face_cnn(face)

        # Flatten each feature map from (batch, C, H, W) to (batch, C*H*W)
        # .flatten(1) keeps dimension 0 (batch) and flattens everything else
        left_flat  = left_features.flatten(1)   # (batch, 4608)
        right_flat = right_features.flatten(1)  # (batch, 4608)
        face_flat  = face_features.flatten(1)   # (batch, 2304)

        # Concatenate all features along the feature dimension (dim=1)
        # Result: (batch, 4608 + 4608 + 2304) = (batch, 11520)
        combined = torch.cat([left_flat, right_flat, face_flat], dim=1)

        # Pass concatenated features through the classification head
        # fc transforms (batch, 11520) → (batch, 5)
        logits = self.fc(combined)

        return logits  # raw scores; caller applies softmax for probabilities


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label and Preprocessing Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Maps integer class index → human-readable direction string.
# MUST match the label_map used during training:
#   {'Straight': 0, 'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4}
LABEL_MAP = {
    0: 'straight',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
}

# Preprocessing transforms that EXACTLY match training-time transforms.
# Any deviation (wrong size, wrong normalization values) will hurt accuracy.
#
# Eye images: input comes as a PIL RGB image at whatever size MediaPipe
# extracted. We resize to 48×48, convert to tensor (float32 in [0,1]),
# then normalize to [-1, 1] using mean=0.5, std=0.5 for each channel.
EYE_TRANSFORM = transforms.Compose([
    transforms.Resize((48, 48)),         # resize to exactly 48×48 pixels
    transforms.ToTensor(),               # PIL [0,255] uint8 → torch [0,1] float32; also transposes HWC→CHW
    transforms.Normalize(                # [0,1] → [-1,1] via (x - 0.5) / 0.5
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# Face images: same normalization, but target size is 112×112 (4× larger than eye)
FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),       # resize to exactly 112×112 pixels
    transforms.ToTensor(),               # PIL → torch tensor in [0,1]
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> GazeNet:
    """
    Load GazeNet v9 from a .pth state dict file.

    The checkpoint is a plain OrderedDict of tensors (no training metadata).
    We instantiate a fresh GazeNet, load the weights, and switch to eval mode.

    Parameters
    ----------
    model_path : str
        Path to the .pth file (e.g., 'best_gazenet_model_v9.pth').

    Returns
    -------
    GazeNet in eval mode, on CPU.
    """
    logger.info(f"Loading model from {model_path} ...")

    # Instantiate model architecture (random weights at this point)
    model = GazeNet(num_classes=5)

    # Load the saved state dict (OrderedDict of {layer_name: tensor})
    # map_location='cpu' ensures it loads on CPU even if saved on GPU
    # weights_only=True is safer — avoids executing arbitrary pickle code
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Copy the saved weights into the model.
    # strict=True (default) means every key in state_dict must match the model,
    # and vice versa. This will raise an error if the architecture doesn't match.
    model.load_state_dict(state_dict, strict=True)

    # Switch from training mode to evaluation mode.
    # This does two things:
    #   1. Disables Dropout layers (they pass through all values unchanged)
    #   2. Uses running statistics in BatchNorm layers (not used here, but good practice)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {total_params:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing Helper
# ─────────────────────────────────────────────────────────────────────────────

def decode_image(base64_string: str) -> Image.Image:
    """
    Decode a base64-encoded JPEG/PNG string into a PIL RGB Image.

    The React frontend encodes canvas crops as base64 data URLs or plain
    base64 strings. This function handles both formats:
      - 'data:image/jpeg;base64,/9j/4AAQSk...' (data URL)
      - '/9j/4AAQSk...' (plain base64)

    Parameters
    ----------
    base64_string : str
        Base64-encoded image data, with or without the data URL prefix.

    Returns
    -------
    PIL.Image.Image in RGB mode (3 channels, uint8).
    """
    # Strip the data URL prefix if present (e.g., 'data:image/jpeg;base64,')
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 bytes back to raw image bytes
    image_bytes = base64.b64decode(base64_string)

    # Wrap in a BytesIO so PIL can read it like a file
    image_buffer = io.BytesIO(image_bytes)

    # Open and convert to RGB (ensures 3 channels regardless of source format)
    # Convert is needed because JPEG images are sometimes decoded as RGBA or L (grayscale)
    image = Image.open(image_buffer).convert('RGB')

    return image


def preprocess_inputs(face_b64: str, left_eye_b64: str, right_eye_b64: str):
    """
    Decode base64 images and apply training-identical preprocessing.

    Steps:
      1. Decode each base64 string to a PIL Image
      2. Apply the appropriate transform (eye or face)
      3. Add batch dimension (unsqueeze to make shape (1, C, H, W))

    Parameters
    ----------
    face_b64 : str     — base64 face crop
    left_eye_b64 : str — base64 left eye crop (subject's left = camera right)
    right_eye_b64 : str — base64 right eye crop (subject's right = camera left)

    Returns
    -------
    Tuple of three tensors: (face, left_eye, right_eye)
    Each is shape (1, 3, H, W) with values in [-1, 1], on CPU.
    """
    # Decode base64 → PIL Image
    face_pil      = decode_image(face_b64)
    left_eye_pil  = decode_image(left_eye_b64)
    right_eye_pil = decode_image(right_eye_b64)

    # Apply transforms: PIL Image → normalized torch tensor of shape (3, H, W)
    face_tensor      = FACE_TRANSFORM(face_pil)       # (3, 112, 112)
    left_eye_tensor  = EYE_TRANSFORM(left_eye_pil)    # (3, 48, 48)
    right_eye_tensor = EYE_TRANSFORM(right_eye_pil)   # (3, 48, 48)

    # Add batch dimension: (3, H, W) → (1, 3, H, W)
    # The model always expects a batch dimension, even for single-sample inference.
    face_tensor      = face_tensor.unsqueeze(0)
    left_eye_tensor  = left_eye_tensor.unsqueeze(0)
    right_eye_tensor = right_eye_tensor.unsqueeze(0)

    return face_tensor, left_eye_tensor, right_eye_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Predict Function
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: GazeNet, face_b64: str, left_eye_b64: str, right_eye_b64: str) -> dict:
    """
    Run a single inference: base64 images → gaze direction + confidence.

    This is the function called by both local_server.py and lambda_handler.py.

    Parameters
    ----------
    model : GazeNet
        Loaded and eval()-mode model (from load_model()).
    face_b64 : str
        Base64-encoded face crop JPEG.
    left_eye_b64 : str
        Base64-encoded left eye crop JPEG (subject's left = camera right).
    right_eye_b64 : str
        Base64-encoded right eye crop JPEG (subject's right = camera left).

    Returns
    -------
    dict with keys:
        direction   : str — predicted gaze direction ('up', 'down', 'left', 'right', 'straight')
        confidence  : float — probability of the predicted class (0.0 to 1.0)
        probabilities : dict — {direction: probability} for all 5 classes
    """
    # Preprocess the three images into tensors
    face_tensor, left_eye_tensor, right_eye_tensor = preprocess_inputs(
        face_b64, left_eye_b64, right_eye_b64
    )

    # Run inference without computing gradients.
    # torch.no_grad() tells PyTorch not to build a computation graph for
    # backpropagation — unnecessary at inference time and saves memory/compute.
    with torch.no_grad():
        # Forward pass: model takes (face, left_eye, right_eye) and returns logits (1, 5)
        # Note: the training forward() signature is forward(left_eye, right_eye, face)
        # but we've reordered here to be more human-readable; the model doesn't care
        # about argument names, only position order. We call it with explicit kwargs:
        logits = model(
            left_eye=left_eye_tensor,   # (1, 3, 48, 48)
            right_eye=right_eye_tensor, # (1, 3, 48, 48)
            face=face_tensor            # (1, 3, 112, 112)
        )
        # logits shape: (1, 5) — one row, five class scores

        # Convert raw logits to probabilities using softmax.
        # Softmax computes exp(x_i) / sum(exp(x_j)) for each class,
        # producing a valid probability distribution that sums to 1.0.
        # dim=1 means we apply softmax across the class dimension.
        probabilities = torch.softmax(logits, dim=1)  # (1, 5) in [0, 1]

        # Get the index of the highest probability class
        # .argmax(dim=1) returns the index along the class dimension
        # .item() converts the scalar tensor to a plain Python int
        predicted_index = probabilities.argmax(dim=1).item()  # int in {0,1,2,3,4}

        # Get the confidence (probability) of the predicted class
        confidence = probabilities[0, predicted_index].item()  # float in [0.0, 1.0]

        # Convert tensor row to a list of floats for all 5 classes
        prob_list = probabilities[0].tolist()  # [p_straight, p_up, p_down, p_left, p_right]

    # Map the predicted index to a direction string
    direction = LABEL_MAP[predicted_index]

    # Build the probabilities dict: {'straight': 0.05, 'up': 0.82, ...}
    prob_dict = {LABEL_MAP[i]: round(prob_list[i], 4) for i in range(5)}

    logger.info(f"Prediction: {direction} (confidence={confidence:.3f})")

    return {
        'direction':     direction,    # e.g., 'up'
        'confidence':    round(confidence, 4),   # e.g., 0.8213
        'probabilities': prob_dict,    # e.g., {'up': 0.8213, 'straight': 0.0721, ...}
    }
