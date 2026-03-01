# Dockerfile
# ==========
# Builds a Lambda-compatible container image for GazeNet v9 inference.
#
# Build command (from backend/ directory):
#   docker build -t eye2voice-inference .
#
# Test locally before pushing to ECR:
#   docker run -p 9000:8080 eye2voice-inference
#   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -H "Content-Type: application/json" \
#     -d '{"body": "{\"face\":\"<base64>\", \"left_eye\":\"<base64>\", \"right_eye\":\"<base64>\"}"}'

# ── Base image ──────────────────────────────────────────────────────────────
# AWS provides official Lambda base images for Python.
# The 'python3.11' tag matches what is available in the Lambda runtime.
# This image includes the Lambda Runtime Interface Client (RIC) which handles
# communication between Lambda and our handler function.
FROM public.ecr.aws/lambda/python:3.11

# ── Set working directory ───────────────────────────────────────────────────
# LAMBDA_TASK_ROOT is set by the base image to /var/task.
# This is where Lambda expects to find your handler code.
WORKDIR ${LAMBDA_TASK_ROOT}

# ── Install Python dependencies ─────────────────────────────────────────────
# Copy requirements first (separate layer from code) so Docker can cache the
# dependency install step. If only inference.py changes, Docker skips re-installing.
COPY requirements.txt ./requirements.txt

# Install packages to the Lambda task root.
# --no-cache-dir: don't store pip cache (saves image space)
# --target: installs packages to the current directory (Lambda looks here)
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# ── Copy application code ───────────────────────────────────────────────────
# Copy our two Python files into the Lambda task root.
# inference.py contains the model architecture and prediction logic.
# lambda_handler.py contains the handler() function Lambda will call.
COPY inference.py       ./inference.py
COPY lambda_handler.py  ./lambda_handler.py

# Note: We do NOT copy best_gazenet_model_v9.pth into the image.
# The model is stored in S3 and downloaded at cold-start by lambda_handler.py.
# This keeps the image smaller and allows model updates without rebuilding the image.

# ── Lambda handler configuration ────────────────────────────────────────────
# CMD tells Lambda which function to call.
# Format: "module_name.function_name"
# Lambda will call lambda_handler.handler(event, context)
CMD ["lambda_handler.handler"]
