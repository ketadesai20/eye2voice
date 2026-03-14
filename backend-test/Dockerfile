# Dockerfile — Eye2Voice Testing Backend
# ========================================
# Lambda container for the dynamic multi-model testing server.
# Uses inference_dynamic.py instead of the production inference.py.
#
# Build (from backend-test/ directory):
#   docker build -t eye2voice-inference-test .
#
# Test locally:
#   docker run -p 9000:8080 \
#     -e MODEL_BUCKET=eye2voice-models \
#     -e MODEL_KEY=best_gazenet_model_v9.pth \
#     -e LOG_BUCKET=eye2voice-models \
#     eye2voice-inference-test
#
#   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -H "Content-Type: application/json" \
#     -d '{"requestContext":{"http":{"method":"GET","path":"/health"}}}'

FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

# Install dependencies (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Copy testing backend source files
COPY inference_dynamic.py  ./inference_dynamic.py
COPY lambda_handler.py     ./lambda_handler.py

# Models are stored in S3 and downloaded at cold start — not baked into the image.
# This keeps the image small and allows swapping models without a rebuild.

CMD ["lambda_handler.handler"]
