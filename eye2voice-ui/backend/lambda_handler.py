"""
lambda_handler.py
=================
AWS Lambda entry point for GazeNet v9 inference.

When AWS Lambda receives an HTTP request via a Function URL or API Gateway,
it calls the `handler(event, context)` function defined here. This file
handles the AWS-specific parts (event parsing, environment variables, S3
model download) and delegates all ML logic to inference.py.

Lambda execution model:
  - First invocation ("cold start"): Lambda downloads + unpacks your container
    image, Python imports run, model downloads from S3 and loads into memory.
    This takes 5–15 seconds.
  - Subsequent invocations ("warm start"): the Lambda container is reused.
    Python module state (including the loaded MODEL) persists between warm
    invocations. Only the handler() function runs, typically in <500ms.

Environment variables (set in Lambda console or CLI):
    MODEL_BUCKET — S3 bucket where best_gazenet_model_v9.pth is stored
    MODEL_KEY    — S3 object key (e.g., 'best_gazenet_model_v9.pth')
"""

import os
import json
import logging
import tempfile

import boto3        # AWS SDK for Python — used to download model from S3
import botocore     # boto3 exceptions live here

# Import our model logic from inference.py (in the same directory in the container)
from inference import load_model, predict

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
# Lambda automatically captures stdout/stderr into CloudWatch Logs.
# Using the standard logging library ensures structured, level-filtered output.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start: download model and load it into module-level variable
# ─────────────────────────────────────────────────────────────────────────────
# Code outside handler() runs once per Lambda container lifetime ("cold start").
# We use this to load the model into memory so warm invocations don't repeat it.
# MODULE-LEVEL variable MODEL is None until successfully loaded.

MODEL = None  # will be set to a GazeNet instance after the first cold start


def _download_and_load_model() -> object:
    """
    Download best_gazenet_model_v9.pth from S3 to Lambda's /tmp filesystem,
    then load and return the GazeNet model.

    Lambda containers have a read-only filesystem except for /tmp, which
    provides up to 10 GB of ephemeral storage that persists for the lifetime
    of the warm container.

    Returns
    -------
    GazeNet model instance in eval() mode.

    Raises
    ------
    EnvironmentError — if MODEL_BUCKET or MODEL_KEY are not set
    botocore.exceptions.ClientError — if S3 download fails (permissions, key)
    """
    # Read configuration from environment variables.
    # These must be set in the Lambda configuration (console → Configuration → Environment variables).
    bucket = os.environ.get('MODEL_BUCKET')
    key    = os.environ.get('MODEL_KEY', 'best_gazenet_model_v9.pth')

    if not bucket:
        raise EnvironmentError(
            "MODEL_BUCKET environment variable is not set. "
            "Set it to the S3 bucket name containing your model file."
        )

    # Use /tmp for the local model file — the only writable path in Lambda
    local_model_path = f'/tmp/{os.path.basename(key)}'

    # Download only if the file isn't already cached from a previous warm invocation
    if not os.path.exists(local_model_path):
        logger.info(f"Downloading model from s3://{bucket}/{key} → {local_model_path}")

        # boto3 S3 client uses the Lambda execution role's IAM permissions.
        # The role must have s3:GetObject permission on the model bucket.
        s3 = boto3.client('s3')

        try:
            s3.download_file(
                Bucket=bucket,
                Key=key,
                Filename=local_model_path
            )
            logger.info("Model download complete.")
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                raise PermissionError(
                    f"Lambda role cannot access s3://{bucket}/{key}. "
                    "Attach AmazonS3ReadOnlyAccess or a custom policy to the Lambda role."
                ) from e
            elif error_code == '404':
                raise FileNotFoundError(
                    f"Model file not found: s3://{bucket}/{key}"
                ) from e
            raise  # re-raise for other S3 errors
    else:
        logger.info(f"Reusing cached model at {local_model_path} (warm container)")

    # Load the model from the downloaded file using our inference.py loader
    return load_model(local_model_path)


# Attempt to load the model during cold start.
# If this fails (bad S3 config, missing env vars), the Lambda function will
# still deploy successfully but every invocation will fail with a clear error.
try:
    MODEL = _download_and_load_model()
    logger.info("Model loaded successfully during cold start.")
except Exception as e:
    logger.error(f"Cold-start model load failed: {e}")
    MODEL = None  # will cause a 500 error on every request until fixed


# ─────────────────────────────────────────────────────────────────────────────
# CORS headers
# ─────────────────────────────────────────────────────────────────────────────
# These headers are included in every response to allow the React app (hosted
# on CloudFront) to call this Lambda URL from a browser without security errors.
#
# The browser's CORS pre-flight (OPTIONS) request and the actual POST request
# both need these headers.
CORS_HEADERS = {
    'Access-Control-Allow-Origin':  os.environ.get(
        'ALLOWED_ORIGIN',
        '*'   # Change to your CloudFront domain in production: 'https://xxx.cloudfront.net'
    ),
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type':                 'application/json',
}


def _response(status_code: int, body: dict) -> dict:
    """
    Build a Lambda Function URL response dict.

    Lambda Function URLs expect a specific return format:
        { statusCode: int, headers: dict, body: str }
    Note: body must be a JSON string, not a dict.
    """
    return {
        'statusCode': status_code,
        'headers': CORS_HEADERS,
        'body': json.dumps(body),  # must be a string, not a dict
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lambda handler
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point. Called once per HTTP request.

    Parameters
    ----------
    event : dict
        The HTTP request from the Lambda Function URL or API Gateway.
        For Lambda Function URLs, the structure is:
          {
            "requestContext": { "http": { "method": "POST" } },
            "body": '{"face": "base64...", "left_eye": "base64...", "right_eye": "base64..."}',
            "isBase64Encoded": false
          }

    context : LambdaContext
        AWS Lambda runtime information (function name, remaining time, etc.).
        Not used for inference but available if needed (e.g., context.get_remaining_time_in_millis()).

    Returns
    -------
    dict with statusCode, headers, and body (JSON string).
    """

    # ── Handle CORS pre-flight request ─────────────────────────────────────
    # Before the browser sends the actual POST, it sends an OPTIONS request
    # ("pre-flight") to check if cross-origin requests are allowed.
    # We must respond to OPTIONS with 200 and the CORS headers.
    http_method = (
        event.get('requestContext', {})
             .get('http', {})
             .get('method', 'POST')
        or event.get('httpMethod', 'POST')  # API Gateway v1 format
    )

    if http_method == 'OPTIONS':
        return _response(200, {'message': 'CORS pre-flight OK'})

    # ── Validate HTTP method ────────────────────────────────────────────────
    if http_method != 'POST':
        return _response(405, {'error': f'Method {http_method} not allowed. Use POST.'})

    # ── Check model loaded ──────────────────────────────────────────────────
    if MODEL is None:
        # Model failed to load during cold start. Return 503 Service Unavailable.
        logger.error("Inference request received but model is not loaded.")
        return _response(503, {
            'error': 'Model not loaded. Check Lambda logs for cold-start errors.'
        })

    # ── Parse request body ──────────────────────────────────────────────────
    # Lambda Function URL puts the HTTP body in event['body'] as a string.
    # It may be base64-encoded if the request body is binary, but our JSON
    # payload is text so isBase64Encoded should be False.
    raw_body = event.get('body', '')

    if event.get('isBase64Encoded', False):
        # Rare case: if API Gateway base64-encodes the body, decode it first
        import base64 as b64
        raw_body = b64.b64decode(raw_body).decode('utf-8')

    # Parse JSON string to Python dict
    try:
        data = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError):
        return _response(400, {'error': 'Request body must be valid JSON'})

    # ── Validate required fields ────────────────────────────────────────────
    required_fields = ['face', 'left_eye', 'right_eye']
    missing = [f for f in required_fields if not data.get(f)]

    if missing:
        return _response(400, {
            'error': f"Missing required fields: {', '.join(missing)}"
        })

    # ── Run inference ───────────────────────────────────────────────────────
    try:
        # Delegate to the shared predict() function in inference.py.
        # This does: decode base64 → preprocess → forward pass → softmax → argmax
        result = predict(
            model=MODEL,
            face_b64=data['face'],
            left_eye_b64=data['left_eye'],
            right_eye_b64=data['right_eye'],
        )

        logger.info(
            f"Prediction: {result['direction']} "
            f"conf={result['confidence']:.3f} "
            f"remaining_ms={context.get_remaining_time_in_millis()}"
        )

        return _response(200, result)

    except ValueError as e:
        logger.warning(f"Bad image data in request: {e}")
        return _response(400, {'error': f'Invalid image data: {str(e)}'})

    except Exception as e:
        logger.error(f"Unexpected inference error: {e}", exc_info=True)
        return _response(500, {'error': 'Internal inference error. See CloudWatch logs.'})
