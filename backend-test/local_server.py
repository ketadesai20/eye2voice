"""
local_server.py — Eye2Voice Testing Backend
============================================
Flask development server for testing different GazeNet model variants locally.
Runs on port 5001 (production server uses 5000) so both can run side by side.

Run from the capstone/ root directory:
    source .mp_env/bin/activate          # or any env with torch + flask
    python backend-test/local_server.py

Set in eye2voice-testing-ui/.env.local:
    REACT_APP_TEST_INFERENCE_URL=http://localhost:5001

Environment variables (all optional):
    MODEL_KEY    — default model filename in ../models/ (default: best_gazenet_model_v9.pth)
    MODEL_BUCKET — S3 bucket for model listing (if set, /models also lists S3 keys)
    LOG_BUCKET   — S3 bucket for session logs (if set, /log uploads there)
    PORT         — server port (default: 5001)

Endpoints:
    GET  /health          — liveness + loaded model info
    GET  /models          — list available .pth files (local + S3)
    GET  /model-info      — inspect a model file without keeping it loaded (?key=filename)
    POST /load-model      — hot-reload a different model {key: filename}
    POST /predict         — run inference {face, left_eye, right_eye, geo_features?}
    POST /log             — save session log JSON to file or S3
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference_dynamic import load_model_dynamic, predict_dynamic, inspect_state_dict

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Root of the capstone project (one level up from backend-test/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR     = os.path.join(PROJECT_ROOT, 'logs')

DEFAULT_MODEL_KEY = os.environ.get('MODEL_KEY', 'best_gazenet_model_v9.pth')
MODEL_BUCKET      = os.environ.get('MODEL_BUCKET', '')
LOG_BUCKET        = os.environ.get('LOG_BUCKET', MODEL_BUCKET)
PORT              = int(os.environ.get('PORT', 5001))

# ─────────────────────────────────────────────────────────────────────────────
# App + logging setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, origins='*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('test_server')

# ─────────────────────────────────────────────────────────────────────────────
# Model state (module-level, reloaded on /load-model)
# ─────────────────────────────────────────────────────────────────────────────

MODEL      = None
MODEL_META = None


def _load_model_from_key(key: str):
    """Load model from ../models/<key> and update module-level globals."""
    global MODEL, MODEL_META
    model_path = os.path.join(MODELS_DIR, key)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    MODEL, MODEL_META = load_model_dynamic(model_path)
    MODEL_META['model_key'] = key
    logger.info(f'Active model: {key} ({MODEL_META["architecture"]}, {MODEL_META["num_classes"]} classes)')


# Load default model at startup
try:
    _load_model_from_key(DEFAULT_MODEL_KEY)
except Exception as e:
    logger.error(f'Startup model load failed: {e}')
    logger.error(f'Place .pth files in {MODELS_DIR}/')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: list models
# ─────────────────────────────────────────────────────────────────────────────

def _list_local_models():
    """Return list of .pth files in ../models/ with sizes."""
    results = []
    if os.path.isdir(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            if fname.endswith('.pth'):
                fpath = os.path.join(MODELS_DIR, fname)
                size_mb = round(os.path.getsize(fpath) / 1_048_576, 1)
                results.append({'key': fname, 'size_mb': size_mb, 'source': 'local'})
    return results


def _list_s3_models():
    """Return list of .pth files in MODEL_BUCKET S3 bucket."""
    if not MODEL_BUCKET:
        return []
    try:
        import boto3
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        results = []
        for page in paginator.paginate(Bucket=MODEL_BUCKET):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.pth'):
                    size_mb = round(obj['Size'] / 1_048_576, 1)
                    results.append({
                        'key': obj['Key'],
                        'size_mb': size_mb,
                        'source': 's3',
                    })
        return results
    except Exception as e:
        logger.warning(f'S3 model listing failed: {e}')
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    if MODEL is None:
        return jsonify({'status': 'error', 'detail': 'No model loaded'}), 503
    return jsonify({
        'status': 'ok',
        'model_key': MODEL_META.get('model_key'),
        'architecture': MODEL_META['architecture'],
        'num_classes': MODEL_META['num_classes'],
        'has_geo': MODEL_META['has_geo'],
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List available .pth model files from local disk and optionally S3."""
    local  = _list_local_models()
    remote = _list_s3_models()
    # Merge: prefer local entry if same key exists in both
    seen = {m['key'] for m in local}
    combined = local + [m for m in remote if m['key'] not in seen]
    return jsonify({'models': combined})


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Inspect a model file and return its metadata without keeping it loaded.

    Query param: ?key=m5c.pth
    """
    key = request.args.get('key', '')
    if not key:
        return jsonify({'error': 'Missing ?key= parameter'}), 400

    model_path = os.path.join(MODELS_DIR, key)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file not found: {key}'}), 404

    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        meta = inspect_state_dict(state_dict)
        meta['model_key'] = key
        meta['size_mb'] = round(os.path.getsize(model_path) / 1_048_576, 1)
        del state_dict  # free memory immediately — we're not loading this model
        return jsonify(meta)
    except Exception as e:
        logger.error(f'model-info error for {key}: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Hot-reload a different model. Body: {"key": "m5c.pth"}"""
    data = request.get_json(force=True, silent=True) or {}
    key  = data.get('key', '')
    if not key:
        return jsonify({'error': 'Missing key'}), 400
    try:
        _load_model_from_key(key)
        return jsonify({'status': 'ok', 'model_key': key, 'architecture': MODEL_META['architecture']})
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'load-model error: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Run gaze inference.

    Body:
        {
            "face":         "<base64 JPEG>",
            "left_eye":     "<base64 JPEG>",
            "right_eye":    "<base64 JPEG>",
            "geo_features": [f1, f2, f3, f4, f5, f6, f7],  // optional, for GazeNetM5
            "eye_size":     48,    // optional, default 48
            "face_size":    112    // optional, default 112
        }

    Response:
        {
            "direction":        "up",
            "confidence":       0.82,
            "probabilities":    {"up": 0.82, ...},
            "used_geo_default": false
        }
    """
    if MODEL is None:
        return jsonify({'error': 'No model loaded. Call /load-model first.'}), 503

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    missing = [f for f in ('face', 'left_eye', 'right_eye') if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        result = predict_dynamic(
            model=MODEL,
            model_meta=MODEL_META,
            face_b64=data['face'],
            left_eye_b64=data['left_eye'],
            right_eye_b64=data['right_eye'],
            geo_features=data.get('geo_features'),
            eye_size=int(data.get('eye_size', 48)),
            face_size=int(data.get('face_size', 112)),
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': f'Invalid image data: {e}'}), 400
    except Exception as e:
        logger.error(f'Inference error: {e}', exc_info=True)
        return jsonify({'error': 'Internal inference error. See server logs.'}), 500


@app.route('/log', methods=['POST'])
def save_log():
    """
    Save a test session log.

    If LOG_BUCKET env var is set: uploads to S3 at logs/<session_id>.json
    Otherwise: writes to ../logs/<session_id>.json (creates dir if needed)

    Body: full session log JSON object (see TestPage.js for schema)
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    session_id = data.get('session_id', 'unknown')
    log_key    = f'logs/{session_id}.json'
    log_json   = json.dumps(data, indent=2)

    if LOG_BUCKET:
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket=LOG_BUCKET,
                Key=log_key,
                Body=log_json.encode('utf-8'),
                ContentType='application/json',
            )
            location = f's3://{LOG_BUCKET}/{log_key}'
            logger.info(f'Log saved to {location}')
            return jsonify({'saved': True, 'location': location})
        except Exception as e:
            logger.error(f'S3 log upload failed: {e}')
            # Fall through to local save

    # Local fallback
    os.makedirs(LOGS_DIR, exist_ok=True)
    local_path = os.path.join(LOGS_DIR, f'{session_id}.json')
    with open(local_path, 'w') as f:
        f.write(log_json)
    logger.info(f'Log saved locally: {local_path}')
    return jsonify({'saved': True, 'location': local_path})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info(f'Eye2Voice Test Server starting on port {PORT}')
    logger.info(f'Models directory: {MODELS_DIR}')
    logger.info(f'Endpoints:')
    logger.info(f'  GET  http://localhost:{PORT}/health')
    logger.info(f'  GET  http://localhost:{PORT}/models')
    logger.info(f'  GET  http://localhost:{PORT}/model-info?key=<filename>')
    logger.info(f'  POST http://localhost:{PORT}/load-model')
    logger.info(f'  POST http://localhost:{PORT}/predict')
    logger.info(f'  POST http://localhost:{PORT}/log')
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False, threaded=True)
