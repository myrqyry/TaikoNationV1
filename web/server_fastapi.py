#!/usr/bin/env python3
"""TaikoNation Studio - FastAPI ASGI Server

This file provides a FastAPI-based server with python-socketio AsyncServer
mounted as an ASGI app. It mirrors the main API surface from the Flask server
but uses async handling and lazy imports for heavy ML dependencies.
"""
import os
import io
import json
import zipfile
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import socketio
from web.persistence import StudioStore

# Project paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
UPLOAD_FOLDER = REPO_ROOT / 'input_songs'
CHART_OUTPUT_FOLDER = REPO_ROOT / 'output'
CONFIG_FOLDER = REPO_ROOT / 'config'
MODEL_FOLDER = REPO_ROOT / 'model'

for p in (UPLOAD_FOLDER, CHART_OUTPUT_FOLDER):
    p.mkdir(parents=True, exist_ok=True)

# Config
API_TOKEN = os.environ.get('TAIKONATION_API_TOKEN')

# Socket.IO Async server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Serve static files
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')

# In-memory state (demo)
system_logs = []
generated_charts = []
human_evaluations = []
model_configurations = []
audio_features = []
active_models: Dict[str, Dict[str, Any]] = {}
store = StudioStore(CHART_OUTPUT_FOLDER / "studio.sqlite3")


def _load_persisted_state():
    """Load persisted entities for compatibility with existing callers/tests."""
    system_logs[:] = store.list_logs(limit=200)
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    model_configurations[:] = store.get_json("model_configurations", [])
    audio_features[:] = store.get_json("audio_features", [])


_load_persisted_state()


def require_token(token: Optional[str] = None):
    if API_TOKEN is None:
        return True
    if token == API_TOKEN:
        return True
    raise HTTPException(status_code=401, detail='Unauthorized')


def token_auth(authorization: Optional[str] = Header(None), token_form: Optional[str] = None):
    """Dependency that extracts token from Authorization header or form and validates it."""
    token = token_form or (authorization or '').replace('Bearer ', '')
    return require_token(token)


async def add_system_log(level: str, message: str):
    entry = {'timestamp': datetime.utcnow().isoformat(), 'level': level, 'message': message}
    system_logs.insert(0, entry)
    # Keep last 200
    del system_logs[200:]
    store.append_log(entry)
    await sio.emit('system_log', entry)


def lazy_load_training_utils():
    try:
        # Import when needed to avoid heavy import-time deps (wandb etc.)
        from train_transformer import load_config as load_training_config
        return load_training_config
    except Exception as e:
        # Don't crash server; caller should handle None
        return None


@app.get('/')
async def index():
    index_path = BASE_DIR / 'index.html'
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return JSONResponse({'error': 'index.html not found'}, status_code=404)


@app.get('/api/status')
async def api_status():
    return {'status': 'ready', 'models_loaded': len(active_models), 'training_active': False, 'generation_active': False}


@app.get('/api/dashboard')
async def api_dashboard():
    best_accuracy = max((m.get('accuracy', 0) for m in active_models.values()), default=0)
    return {
        'metrics': {
            'active_models': len([m for m in active_models.values() if m.get('status') == 'ready']),
            'generated_charts': len(generated_charts),
            'best_accuracy': best_accuracy,
            'avg_rating': round(sum((c.get('rating', 0) for c in generated_charts)) / (len(generated_charts) or 1), 1)
        },
        'recent_logs': system_logs[:10]
    }


@app.post('/api/upload-audio')
async def api_upload_audio(filename: str, content: bytes):
    if not filename:
        raise HTTPException(status_code=400, detail='No file provided')

    filename = Path(filename).name
    allowed = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
    if Path(filename).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail='Unsupported file type')

    dest = UPLOAD_FOLDER / filename
    # Stream to disk
    with open(dest, 'wb') as f:
        f.write(content)

    # Minimal processing: return title
    title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    await add_system_log('success', f'Audio uploaded: {filename}')
    return {'success': True, 'filename': filename, 'title': title}


@app.post('/api/generate-chart')
async def api_generate_chart(
    title: str = 'Untitled',
    artist: str = 'Unknown',
    bpm: int = 120,
    genre: str = 'electronic',
    difficulty: str = 'oni',
    pattern_style: str = 'balanced',
    _auth: bool = Depends(token_auth)
):
    await add_system_log('info', f'Chart generation started: {title}')

    async def simulate_generation():
        for p in range(0, 101, 10):
            await sio.emit('generation_progress', {'progress': p})
            await asyncio.sleep(0.5)

        chart = {
            'title': title,
            'artist': artist,
            'difficulty': difficulty,
            'bpm': bpm,
            'genre': genre,
            'rating': 0,
            'plays': 0,
            'created_at': datetime.utcnow().isoformat()
        }
        stored_chart = store.create_chart(chart)
        generated_charts.insert(0, stored_chart)
        await sio.emit('chart_generated', {'chart': stored_chart})
        await add_system_log('success', f'Chart generated: {title}')

    # schedule background task
    asyncio.create_task(simulate_generation())
    return {'success': True, 'message': 'Chart generation started'}


@app.get('/api/charts')
async def api_charts():
    generated_charts[:] = store.list_charts()
    return {'charts': generated_charts}


@app.get('/api/get-chart-for-evaluation')
async def api_get_chart_for_evaluation():
    chart = store.get_unrated_chart()
    if chart:
        return chart
    raise HTTPException(status_code=404, detail='No charts available for evaluation')


@app.post('/api/submit-evaluation')
async def api_submit_evaluation(chart_id: int, fun: int, musicality: int, playability: int, coherence: int, comments: str = ''):
    ok = store.submit_evaluation(
        chart_id=chart_id,
        fun=fun,
        musicality=musicality,
        playability=playability,
        coherence=coherence,
        comments=comments,
        created_at=datetime.utcnow().isoformat(),
    )
    if not ok:
        raise HTTPException(status_code=404, detail='Chart not found')
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    await add_system_log('success', f'Evaluation submitted for chart {chart_id}')
    return {'success': True}




@app.get('/api/research/export-dataset')
async def api_research_export_dataset():
    """Export the current in-memory research dataset as a zip archive."""
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    model_configurations[:] = store.get_json("model_configurations", model_configurations)
    audio_features[:] = store.get_json("audio_features", audio_features)
    export_payload = {
        'generated_charts': generated_charts,
        'human_evaluations': human_evaluations,
        'model_configurations': model_configurations,
        'audio_features': audio_features,
        'exported_at': datetime.utcnow().isoformat()
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('dataset.json', json.dumps(export_payload, indent=2))

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=research_dataset.zip'}
    )

@sio.event
async def connect(sid, environ):
    await add_system_log('info', f'Client connected: {sid}')


@sio.event
async def disconnect(sid):
    await add_system_log('info', f'Client disconnected: {sid}')


if __name__ == '__main__':
    # Run with: uvicorn server_fastapi:socket_app --reload --port 5000
    import uvicorn
    uvicorn.run('server_fastapi:socket_app', host='127.0.0.1', port=5000, reload=True)
