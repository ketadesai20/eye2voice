"""
lambda_handler.py — Eye2Voice Testing Lambda
=============================================
AWS Lambda entry point for the testing backend. Handles multiple model variants
dynamically, with per-model caching in /tmp so warm invocations skip S3 downloads.

Key differences from the production lambda_handler.py:
  - Multi-model cache: loads any .pth from MODEL_BUCKET on demand
  - Endpoints: health, models, model-info, predict, log
  - Predict accepts optional model_key to switch models per-request
  - Log saves to S3 at LOG_BUCKET/logs/<session_id>.json

Environment variables:
    MODEL_BUCKET  — S3 bucket containing .pth model files (required)
    MODEL_KEY     — default model key (default: best_gazenet_model_v9.pth)
    LOG_BUCKET    — S3 bucket for session logs (defaults to MODEL_BUCKET)
    ALLOWED_ORIGIN — CORS allowed origin (default: *)
"""

import os
import json
import logging

import boto3
import botocore
import torch

from inference_dynamic import load_model_dynamic, predict_dynamic, inspect_state_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Multi-model cache
# ─────────────────────────────────────────────────────────────────────────────
# Dict of {s3_key: (model, model_meta)} — persists across warm invocations.
# Models are downloaded to /tmp/<key_basename> and cached here.

MODEL_CACHE: dict = {}

MODEL_BUCKET   = os.environ.get('MODEL_BUCKET', '')
DEFAULT_KEY    = os.environ.get('MODEL_KEY', 'best_gazenet_model_v9.pth')
LOG_BUCKET     = os.environ.get('LOG_BUCKET', MODEL_BUCKET)
ALLOWED_ORIGIN = os.environ.get('ALLOWED_ORIGIN', '*')


def _get_local_path(key: str) -> str:
    """Map an S3 key to a /tmp path (Lambda's only writable filesystem)."""
    return f'/tmp/{os.path.basename(key)}'


def _ensure_model(key: str):
    """
    Return (model, model_meta) for the given S3 key.

    Downloads from S3 to /tmp if not already there, then loads and caches.
    Subsequent calls for the same key return immediately from MODEL_CACHE.
    """
    if key in MODEL_CACHE:
        logger.info(f'Cache hit: {key}')
        return MODEL_CACHE[key]

    local_path = _get_local_path(key)

    if not os.path.exists(local_path):
        if not MODEL_BUCKET:
            raise EnvironmentError('MODEL_BUCKET environment variable is not set.')
        logger.info(f'Downloading s3://{MODEL_BUCKET}/{key} → {local_path}')
        s3 = boto3.client('s3')
        try:
            s3.download_file(Bucket=MODEL_BUCKET, Key=key, Filename=local_path)
        except botocore.exceptions.ClientError as e:
            code = e.response['Error']['Code']
            if code == '404':
                raise FileNotFoundError(f'Model not found: s3://{MODEL_BUCKET}/{key}') from e
            if code == '403':
                raise PermissionError(f'Access denied: s3://{MODEL_BUCKET}/{key}') from e
            raise
        logger.info('Download complete.')
    else:
        logger.info(f'Reusing /tmp/{os.path.basename(key)} (warm container)')

    model, meta = load_model_dynamic(local_path)
    meta['model_key'] = key
    MODEL_CACHE[key] = (model, meta)
    return model, meta


# Preload default model at cold start
try:
    _ensure_model(DEFAULT_KEY)
    logger.info(f'Cold-start model loaded: {DEFAULT_KEY}')
except Exception as e:
    logger.error(f'Cold-start model load failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Response helpers
# ─────────────────────────────────────────────────────────────────────────────

CORS_HEADERS = {
    'Access-Control-Allow-Origin':  ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json',
}


def _response(status: int, body: dict) -> dict:
    return {'statusCode': status, 'headers': CORS_HEADERS, 'body': json.dumps(body)}


def _parse_body(event: dict) -> dict:
    raw = event.get('body', '') or ''
    if event.get('isBase64Encoded'):
        import base64
        raw = base64.b64decode(raw).decode('utf-8')
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_method(event: dict) -> str:
    return (
        event.get('requestContext', {}).get('http', {}).get('method')
        or event.get('httpMethod', 'POST')
    )


def _get_path(event: dict) -> str:
    return (
        event.get('requestContext', {}).get('http', {}).get('path')
        or event.get('rawPath', '/predict')
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_health():
    if not MODEL_CACHE:
        return _response(503, {'status': 'error', 'detail': 'No model loaded'})
    # Return info about the default (most recently loaded) model
    _, meta = MODEL_CACHE.get(DEFAULT_KEY, list(MODEL_CACHE.values())[-1])
    return _response(200, {
        'status': 'ok',
        'model_key': meta.get('model_key'),
        'architecture': meta['architecture'],
        'num_classes': meta['num_classes'],
        'has_geo': meta['has_geo'],
        'cached_models': list(MODEL_CACHE.keys()),
    })


def _handle_models():
    if not MODEL_BUCKET:
        return _response(200, {'models': [], 'error': 'MODEL_BUCKET not set'})
    try:
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        models = []
        for page in paginator.paginate(Bucket=MODEL_BUCKET):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.pth'):
                    models.append({
                        'key': obj['Key'],
                        'size_mb': round(obj['Size'] / 1_048_576, 1),
                        'source': 's3',
                        'cached': obj['Key'] in MODEL_CACHE,
                    })
        return _response(200, {'models': models})
    except Exception as e:
        logger.error(f'models list error: {e}')
        return _response(500, {'error': str(e)})


def _handle_model_info(event: dict):
    qs = event.get('queryStringParameters') or {}
    key = qs.get('key', '')
    if not key:
        return _response(400, {'error': 'Missing ?key= parameter'})

    local_path = _get_local_path(key)
    if not os.path.exists(local_path):
        if not MODEL_BUCKET:
            return _response(503, {'error': 'MODEL_BUCKET not set'})
        try:
            s3 = boto3.client('s3')
            s3.download_file(Bucket=MODEL_BUCKET, Key=key, Filename=local_path)
        except botocore.exceptions.ClientError as e:
            code = e.response['Error']['Code']
            return _response(404 if code == '404' else 500, {'error': str(e)})

    try:
        state_dict = torch.load(local_path, map_location='cpu', weights_only=True)
        meta = inspect_state_dict(state_dict)
        meta['model_key'] = key
        del state_dict
        return _response(200, meta)
    except Exception as e:
        logger.error(f'model-info error: {e}')
        return _response(500, {'error': str(e)})


def _handle_predict(event: dict):
    data = _parse_body(event)
    if not data:
        return _response(400, {'error': 'Request body must be valid JSON'})

    missing = [f for f in ('face', 'left_eye', 'right_eye') if not data.get(f)]
    if missing:
        return _response(400, {'error': f'Missing fields: {", ".join(missing)}'})

    # Allow per-request model switching
    key = data.get('model_key', DEFAULT_KEY)
    try:
        model, meta = _ensure_model(key)
    except FileNotFoundError as e:
        return _response(404, {'error': str(e)})
    except Exception as e:
        return _response(500, {'error': str(e)})

    try:
        result = predict_dynamic(
            model=model,
            model_meta=meta,
            face_b64=data['face'],
            left_eye_b64=data['left_eye'],
            right_eye_b64=data['right_eye'],
            geo_features=data.get('geo_features'),
            eye_size=int(data.get('eye_size', 48)),
            face_size=int(data.get('face_size', 112)),
        )
        return _response(200, result)
    except ValueError as e:
        return _response(400, {'error': f'Invalid image data: {e}'})
    except Exception as e:
        logger.error(f'Inference error: {e}', exc_info=True)
        return _response(500, {'error': 'Internal inference error'})


def _handle_log(event: dict):
    data = _parse_body(event)
    if not data:
        return _response(400, {'error': 'Request body must be valid JSON'})

    session_id = data.get('session_id', 'unknown')
    log_key    = f'logs/{session_id}.json'
    bucket     = LOG_BUCKET or MODEL_BUCKET

    if not bucket:
        return _response(503, {'error': 'LOG_BUCKET not set — cannot save log'})

    try:
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket,
            Key=log_key,
            Body=json.dumps(data, indent=2).encode('utf-8'),
            ContentType='application/json',
        )
        location = f's3://{bucket}/{log_key}'
        logger.info(f'Session log saved: {location}')
        return _response(200, {'saved': True, 'location': location})
    except Exception as e:
        logger.error(f'Log save error: {e}', exc_info=True)
        return _response(500, {'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Lambda handler entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """Main Lambda entry point — routes requests to the correct handler."""

    method = _get_method(event)
    path   = _get_path(event)

    # CORS pre-flight
    if method == 'OPTIONS':
        return _response(200, {'message': 'CORS OK'})

    # Route table
    if path == '/health':
        return _handle_health()

    if path == '/models' and method == 'GET':
        return _handle_models()

    if path == '/model-info' and method == 'GET':
        return _handle_model_info(event)

    if path == '/predict' and method == 'POST':
        return _handle_predict(event)

    if path == '/log' and method == 'POST':
        return _handle_log(event)

    return _response(404, {'error': f'Unknown endpoint: {method} {path}'})
