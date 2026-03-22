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
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
active_tasks = []
# Global executor for real background jobs
executor = ThreadPoolExecutor(max_workers=4)


class UploadAudioResponse(BaseModel):
    success: bool
    filename: str
    title: str


class GenerateChartRequest(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=120)
    artist: str = Field(default="Unknown", min_length=1, max_length=120)
    bpm: int = Field(default=120, ge=40, le=320)
    genre: str = Field(default="electronic", min_length=1, max_length=64)
    difficulty: str = Field(default="oni", min_length=1, max_length=32)
    pattern_style: str = Field(default="balanced", min_length=1, max_length=64)


class PaginatedTasksResponse(BaseModel):
    tasks: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class PaginatedChartsResponse(BaseModel):
    charts: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


class SaveChartRequest(BaseModel):
    id: int
    notes: List[Dict[str, Any]]


class StartTrainingRequest(BaseModel):
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    learning_rate: float = 0.0001
    batch_size: int = 8


class SubmitEvaluationRequest(BaseModel):
    chart_id: int
    fun: int
    musicality: int
    playability: int
    coherence: int
    comments: str = ''


class SubmitComparativeEvaluationRequest(BaseModel):
    chartA_id: int
    chartB_id: int
    preference: str


def _load_persisted_state():
    """Load persisted entities for compatibility with existing callers/tests."""
    system_logs[:] = store.list_logs(limit=200)
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    model_configurations[:] = store.get_json("model_configurations", [])
    audio_features[:] = store.get_json("audio_features", [])
    active_tasks[:] = store.list_tasks(limit=100)


_load_persisted_state()


def _config_path() -> Path:
    return CONFIG_FOLDER / "default.yaml"


def _load_runtime_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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


@app.get('/editor')
async def editor():
    editor_path = BASE_DIR / 'editor.html'
    if editor_path.exists():
        return HTMLResponse(editor_path.read_text())
    return JSONResponse({'error': 'editor.html not found'}, status_code=404)


@app.get('/favicon.ico')
async def favicon():
    fav_path = BASE_DIR / 'static' / 'favicon.ico'
    if fav_path.exists():
        return FileResponse(fav_path)
    return JSONResponse(status_code=204, content="")


@app.get('/static/{filename:path}')
async def static_files(filename: str):
    static_path = BASE_DIR / 'static' / filename
    if not static_path.exists():
        return JSONResponse(status_code=404, content="")
    return FileResponse(static_path, headers={'Cache-Control': 'max-age=3600'})


@app.get('/api/status')
async def api_status():
    generation_active = any(t.get('status') in {'queued', 'running'} for t in active_tasks if t.get('task_type') == 'generation')
    return {'status': 'ready', 'models_loaded': len(active_models), 'training_active': False, 'generation_active': generation_active}


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


@app.get('/api/models')
async def api_models():
    return {'models': active_models}


@app.get('/api/config')
async def api_get_config(_auth: bool = Depends(token_auth)):
    return _load_runtime_config()


@app.post('/api/config')
async def api_update_config(payload: ConfigUpdateRequest, _auth: bool = Depends(token_auth)):
    existing = _load_runtime_config()
    merged = _deep_merge(existing, payload.config)
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    await add_system_log("success", "Configuration updated")
    return {"success": True, "config": merged}


@app.post('/api/upload-audio')
async def api_upload_audio(
    filename: Optional[str] = None,
    content: Optional[bytes] = None,
    file: Optional[UploadFile] = File(default=None),
    _auth: bool = Depends(token_auth),
) -> UploadAudioResponse:
    # Preferred request style: multipart upload via `file`.
    if isinstance(file, UploadFile):
        filename = file.filename
        content = await file.read()

    if not filename or content is None:
        raise HTTPException(status_code=400, detail='No file provided')

    filename = Path(filename).name
    allowed = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
    if Path(filename).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail='Unsupported file type')
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='File too large (max 50MB)')

    dest = UPLOAD_FOLDER / filename
    # Stream to disk
    with open(dest, 'wb') as f:
        f.write(content)

    # Minimal processing: return title
    title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    await add_system_log('success', f'Audio uploaded: {filename}')
    return UploadAudioResponse(success=True, filename=filename, title=title)


def _run_generation_job(task_id: int, title: str, artist: str, bpm: int, genre: str, difficulty: str, pattern_style: str):
    import time
    from web.tasks import run_task, TASKS_REGISTRY

    try:
        # We hook into the existing run_task from tasks.py.
        # But first, we need to register the task in TASKS_REGISTRY.
        # It's better to just delegate to web.tasks entirely so we don't reinvent the wheel.
        # But for now, we'll just run our logic here that interacts with the real store.

        store.update_task(
            task_id,
            status='running',
            progress=0,
            message='Generation started',
            updated_at=datetime.utcnow().isoformat(),
        )

        from web.server import start_chart_generation

        params = {
            'title': title,
            'artist': artist,
            'bpm': bpm,
            'genre': genre,
            'difficulty': difficulty,
            'pattern_style': pattern_style,
        }

        # This function generates the chart and saves the .osu file using real ML code.
        chart = start_chart_generation(params)

        # update the db task state.
        store.update_task(
            task_id,
            status='completed',
            progress=100,
            message='Generation completed',
            result={'chart_id': chart['id']},
            updated_at=datetime.utcnow().isoformat(),
        )

        # sync in-memory lists (mostly for compatibility with existing tests/endpoints)
        active_tasks[:] = store.list_tasks(limit=100)
        generated_charts[:] = store.list_charts()

    except Exception as exc:
        store.update_task(
            task_id,
            status='failed',
            message=str(exc),
            updated_at=datetime.utcnow().isoformat(),
        )
        active_tasks[:] = store.list_tasks(limit=100)


def _run_training_job(task_id: int, model_dump: dict):
    from web.tasks import train_model_with_progress
    try:
        store.update_task(
            task_id,
            status='running',
            progress=0,
            message='Training started',
            updated_at=datetime.utcnow().isoformat(),
        )

        result = train_model_with_progress(str(task_id), model_dump)

        store.update_task(
            task_id,
            status='completed',
            progress=100,
            message='Training completed',
            result=result,
            updated_at=datetime.utcnow().isoformat(),
        )
        active_tasks[:] = store.list_tasks(limit=100)
    except Exception as exc:
        store.update_task(
            task_id,
            status='failed',
            message=str(exc),
            updated_at=datetime.utcnow().isoformat(),
        )
        active_tasks[:] = store.list_tasks(limit=100)


@app.post('/api/generate-chart')
async def api_generate_chart(
    request: GenerateChartRequest = GenerateChartRequest(),
    _auth: bool = Depends(token_auth)
):
    title = request.title
    artist = request.artist
    bpm = request.bpm
    genre = request.genre
    difficulty = request.difficulty
    pattern_style = request.pattern_style
    await add_system_log('info', f'Chart generation started: {title}')
    task = store.create_task(
        task_type='generation',
        payload={'title': title, 'artist': artist, 'difficulty': difficulty, 'bpm': bpm, 'genre': genre},
        created_at=datetime.utcnow().isoformat(),
    )
    active_tasks.insert(0, task)

    # schedule real background job
    executor.submit(_run_generation_job, task['id'], title, artist, bpm, genre, difficulty, pattern_style)
    return {'success': True, 'message': 'Chart generation started', 'task_id': task['id']}


@app.get('/api/tasks')
async def api_tasks(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _auth: bool = Depends(token_auth),
) -> PaginatedTasksResponse:
    active_tasks[:] = store.list_tasks(limit=limit, offset=offset)
    return PaginatedTasksResponse(tasks=active_tasks, total=store.count_tasks(), limit=limit, offset=offset)


@app.get('/api/tasks/{task_id}')
async def api_task(task_id: int, _auth: bool = Depends(token_auth)):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    return task


@app.post('/api/tasks/{task_id}/cancel')
async def api_cancel_task(task_id: int, _auth: bool = Depends(token_auth)):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    if task.get('status') in {'completed', 'failed', 'cancelled'}:
        return {'success': False, 'message': f"Task already {task.get('status')}"}
    store.update_task(
        task_id,
        status='cancelled',
        message='Cancelled by user',
        updated_at=datetime.utcnow().isoformat(),
    )
    active_tasks[:] = store.list_tasks(limit=100)
    await add_system_log('warning', f'Task cancelled: {task_id}')
    return {'success': True, 'task_id': task_id, 'status': 'cancelled'}


@app.get('/api/charts')
async def api_charts(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _auth: bool = Depends(token_auth),
) -> PaginatedChartsResponse:
    generated_charts[:] = store.list_charts(limit=limit, offset=offset)
    return PaginatedChartsResponse(charts=generated_charts, total=store.count_charts(), limit=limit, offset=offset)


@app.get('/api/get-chart-for-evaluation')
async def api_get_chart_for_evaluation():
    chart = store.get_unrated_chart()
    if chart:
        return chart
    raise HTTPException(status_code=404, detail='No charts available for evaluation')


@app.get('/api/chart-data')
async def api_chart_data(id: int = Query(...)):
    chart = store.get_chart(id)
    if not chart or 'filename' not in chart:
        raise HTTPException(status_code=404, detail='Chart not found')

    filepath = CHART_OUTPUT_FOLDER / chart['filename']
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    # Helper to parse .osu file for editor
    def parse_osu_file(filepath):
        notes = []
        in_hit_objects = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == '[HitObjects]':
                        in_hit_objects = True
                        continue
                    if line.startswith('['):
                        in_hit_objects = False
                        continue
                    if in_hit_objects and line:
                        parts = line.split(',')
                        if len(parts) >= 5:
                            time = int(parts[2])
                            obj_type = int(parts[3])
                            hit_sound = int(parts[4])
                            note = {'time': time}
                            is_finisher = obj_type & 4
                            if hit_sound == 0:
                                note['type'] = 'big_don' if is_finisher else 'don'
                            elif hit_sound == 8:
                                note['type'] = 'big_ka' if is_finisher else 'ka'
                            elif hit_sound == 4:
                                note['type'] = 'big_don'
                            elif hit_sound == 2:
                                note['type'] = 'big_ka'
                            else:
                                continue
                            notes.append(note)
        except Exception:
            return None
        return notes

    notes = parse_osu_file(filepath)
    if notes is None:
        raise HTTPException(status_code=500, detail='Failed to parse chart file')

    return {'notes': notes, 'metadata': chart}


@app.get('/api/download-chart')
async def download_chart(id: int = Query(...), _auth: bool = Depends(token_auth)):
    chart = store.get_chart(id)
    if not chart or 'filename' not in chart:
        raise HTTPException(status_code=404, detail='Chart not found')

    filename = Path(chart['filename']).name
    filepath = CHART_OUTPUT_FOLDER / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    return FileResponse(filepath, filename=filename)


@app.post('/api/submit-evaluation')
async def api_submit_evaluation(request: SubmitEvaluationRequest, _auth: bool = Depends(token_auth)):
    ok = store.submit_evaluation(
        chart_id=request.chart_id,
        fun=request.fun,
        musicality=request.musicality,
        playability=request.playability,
        coherence=request.coherence,
        comments=request.comments,
        created_at=datetime.utcnow().isoformat(),
    )
    if not ok:
        raise HTTPException(status_code=404, detail='Chart not found')
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    await add_system_log('success', f'Evaluation submitted for chart {request.chart_id}')
    return {'success': True}




@app.get('/api/research/experiments')
async def api_research_experiments():
    """Return detailed experiment tracking for research analysis"""
    import numpy as np

    # Simple placeholder logic replicating Flask version
    return {
        'experiments': {},
        'model_comparisons': {
            'transformer_vs_legacy': {
                'accuracy': [m.get('accuracy', 0) for m in active_models.values()],
                'pattern_diversity': [np.random.rand() for _ in active_models] if active_models else [],
            }
        },
        'pattern_evolution': {
            'timestamps': [c.get('created_at', '') for c in generated_charts],
            'diversity': [np.random.rand() for _ in generated_charts] if generated_charts else []
        },
        'human_feedback_stats': {
            'total_comparisons': len(human_evaluations),
            'preference_distribution': {
                'A': sum(1 for f in human_evaluations if f.get('preference') == 'A'),
                'B': sum(1 for f in human_evaluations if f.get('preference') == 'B'),
                'tie': sum(1 for f in human_evaluations if f.get('preference') == 'tie'),
            }
        }
    }


@app.post('/api/save-chart')
async def api_save_chart(request: SaveChartRequest, _auth: bool = Depends(token_auth)):
    chart_id = request.id
    chart = store.get_chart(chart_id)
    if not chart or 'filename' not in chart:
        raise HTTPException(status_code=404, detail='Chart not found')

    filepath = CHART_OUTPUT_FOLDER / chart['filename']
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        try:
            start_index = lines.index('[HitObjects]\n') + 1
        except ValueError:
            lines.append('\n[HitObjects]\n')
            start_index = len(lines)

        end_index = start_index
        while end_index < len(lines) and not lines[end_index].startswith('['):
            end_index += 1

        new_hit_objects = []
        for note in request.notes:
            time = note.get('time', 0)
            note_type = note.get('type', 'don')
            x, y, obj_type, hit_sound, extras = 256, 192, 1, 0, '0:0:0:0:'

            if note_type == 'don':
                hit_sound = 0
            elif note_type == 'ka':
                hit_sound = 8
            elif note_type == 'big_don':
                obj_type = 5
                hit_sound = 4
            elif note_type == 'big_ka':
                obj_type = 5
                hit_sound = 2

            new_hit_objects.append(f"{x},{y},{time},{obj_type},{hit_sound},{extras}\n")

        new_lines = lines[:start_index] + new_hit_objects + lines[end_index:]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        await add_system_log('success', f"Chart '{chart.get('title', 'Unknown')}' updated from editor.")
        return {'success': True, 'message': 'Chart saved successfully.'}
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to save chart file')


@app.post('/api/start-training')
async def api_start_training(request: StartTrainingRequest, _auth: bool = Depends(token_auth)):
    task = store.create_task(
        task_type='training',
        payload=request.model_dump(),
        created_at=datetime.utcnow().isoformat(),
    )
    active_tasks.insert(0, task)

    # schedule real training job
    executor.submit(_run_training_job, task['id'], request.model_dump())
    return {'success': True, 'task_id': task['id'], 'message': 'Training started.'}


@app.post('/api/stop-training')
async def api_stop_training(_auth: bool = Depends(token_auth)):
    await add_system_log('info', 'Training stopped by user')
    return {'success': True}


@app.post('/api/submit-comparative-evaluation')
async def api_submit_comparative_evaluation(request: SubmitComparativeEvaluationRequest, _auth: bool = Depends(token_auth)):
    chart_a = store.get_chart(request.chartA_id)
    chart_b = store.get_chart(request.chartB_id)

    if not chart_a or not chart_b:
        raise HTTPException(status_code=400, detail='Invalid chart IDs')

    eval_record = {
        'chart_pair': [request.chartA_id, request.chartB_id],
        'preference': request.preference,
        'timestamp': datetime.utcnow().isoformat()
    }
    human_evaluations.append(eval_record)
    await add_system_log('success', 'Comparative evaluation submitted successfully')
    return {'success': True}


@app.post('/api/submit_audio_classification')
async def api_submit_audio_classification(request: Request):
    data = await request.json()
    # Placeholder to accept classifications from MediaPipe client
    return {
        'status': 'success',
        'message': 'Audio classification received',
        'sections': [],
        'percussion_ratio': data.get('analysis', {}).get('avg_percussion_ratio', 0),
        'melodic_ratio': data.get('analysis', {}).get('avg_melodic_ratio', 0)
    }


@app.post('/api/generate_chart_with_audio_analysis')
async def api_generate_chart_with_audio_analysis(request: Request, _auth: bool = Depends(token_auth)):
    data = await request.json()
    # Placeholder for the enhanced ML generation API
    return {
        'status': 'success',
        'chart': [],
        'used_classification': True
    }


@app.get('/api/research/export-dataset')
async def api_research_export_dataset():
    """Export the current research dataset as a zip archive."""
    # Ensure we use fresh data from the store for the export
    current_charts = store.list_charts()
    current_evals = store.list_evaluations()
    current_model_configs = store.get_json("model_configurations", [])
    current_audio_features = store.get_json("audio_features", [])

    # Sync global variables for compatibility
    generated_charts[:] = current_charts
    human_evaluations[:] = current_evals
    model_configurations[:] = current_model_configs
    audio_features[:] = current_audio_features

    export_payload = {
        'generated_charts': current_charts,
        'human_evaluations': current_evals,
        'model_configurations': current_model_configs,
        'audio_features': current_audio_features,
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
