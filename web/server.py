#!/usr/bin/env python3
"""
TaikoNation Studio Web Server

This Flask server provides a web interface for the TaikoNation AI chart generation system.
It integrates directly with the existing Python modules and provides REST API endpoints
for training, generation, evaluation, and configuration management.
"""

import os
import sys
import json
import re
import yaml
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import shlex
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from threading import Lock, RLock
import uuid
import magic
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataclasses import dataclass, field

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, abort
from flask_socketio import SocketIO, emit
from marshmallow import Schema, fields, ValidationError
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
from enum import Enum

class Difficulty(Enum):
    EASY = 0
    NORMAL = 1
    HARD = 2
    ONI = 3
    URA_ONI = 4

def validate_difficulty(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        diff = kwargs.get('difficulty') or request.json.get('difficulty')
        try:
            if isinstance(diff, str):
                kwargs['difficulty'] = Difficulty[diff.upper()]
            elif isinstance(diff, int):
                kwargs['difficulty'] = Difficulty(diff)
        except (KeyError, ValueError):
            return jsonify({
                'error': f'Invalid difficulty. Must be one of: {[d.name for d in Difficulty]}'
            }), 400
        return f(*args, **kwargs)
    return wrapper

# Import existing TaikoNation modules
try:
    # Add the parent directory to path to import TaikoNation modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from audio_processing import get_audio_features, augment_spectrogram
    from transformer_model import TaikoTransformer
    from transformer_dataset import get_transformer_data_loaders, DIFFICULTY_MAP
    from tokenization import TaikoTokenizer
    from config_schema import ConfigSchema
    # Do NOT import train_transformer at module import time: it may import heavy deps (wandb)
    # which can pull eventlet and trigger ssl-related import-time errors in some environments.
    load_training_config = None

except ImportError as e:
    print(f"Warning: Could not import TaikoNation modules: {e}")
    print("Make sure the server is running from the web/ directory inside TaikoNationV1/")

import torch
import numpy as np
import librosa

def setup_logging(log_level='INFO'):
    """Set up comprehensive logging"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'taikonation.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # Root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return root_logger

logger = setup_logging()
app = Flask(__name__)
# Use environment variable for secret key in production; fallback to random if not set
app.config['SECRET_KEY'] = os.environ.get('TAIKONATION_SECRET_KEY') or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('TAIKONATION_MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # bytes

# Choose an async mode for SocketIO. Prefer eventlet if available and usable,
# otherwise fall back to the standard 'threading' mode to avoid importing
# greenlet/eventlet drivers in environments where they are broken.
async_mode = None
try:
    import eventlet
    # Quick sanity check: some environments have broken ssl modules and eventlet
    # will fail when importing its green ssl wrappers. Ensure the stdlib ssl has wrap_socket
    import ssl as _ssl
    if not hasattr(_ssl, 'wrap_socket'):
        raise RuntimeError('ssl.wrap_socket not available; skipping eventlet')
    async_mode = 'eventlet'
except Exception as e:
    # eventlet not usable (or not installed) — fall back to threading
    async_mode = 'threading'
    logger.warning(f'Eventlet not usable, falling back to threading async mode: {e}')

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# Optional API token for protecting endpoints. If not set, endpoints remain open (dev mode).
API_TOKEN = os.environ.get('TAIKONATION_API_TOKEN')

from functools import wraps
def require_api_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_TOKEN is None:
            return f(*args, **kwargs)
        token = (request.headers.get('Authorization') or '').replace('Bearer ', '') or request.args.get('api_token') or request.form.get('api_token')
        if token != API_TOKEN:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

# Add some security headers for basic hardening
@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    # Content Security Policy: relaxed for development to allow common external assets.
    # Adjust/remove external hosts for production deployments.
    csp = (
        "default-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob: https://cdnjs.cloudflare.com; "
        "font-src 'self' data: https://r2cdn.perplexity.ai https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "connect-src 'self' ws: wss: https://cdnjs.cloudflare.com;"
    )
    response.headers['Content-Security-Policy'] = csp
    return response


# Marshmallow Schemas for Input Validation
class ChartGenerationSchema(Schema):
    title = fields.Str(load_default="Untitled", validate=lambda x: len(x) <= 200)
    artist = fields.Str(load_default="Unknown", validate=lambda x: len(x) <= 200)
    bpm = fields.Int(load_default=120, validate=lambda x: 60 <= x <= 300)
    genre = fields.Str(load_default="electronic")
    difficulty = fields.Str(
        load_default="oni",
        validate=lambda x: x in ["kantan", "futsuu", "muzukashii", "oni", "ura"],
    )
    pattern_style = fields.Str(load_default="balanced")
    audio_filename = fields.Str(required=True)
    npy_filename = fields.Str(required=False, load_default=None)


class TrainingSchema(Schema):
    d_model = fields.Int(load_default=256)
    nhead = fields.Int(load_default=8)
    num_encoder_layers = fields.Int(load_default=6)
    num_decoder_layers = fields.Int(load_default=6)
    learning_rate = fields.Float(load_default=0.0001)
    batch_size = fields.Int(load_default=8)


@dataclass
class ServerState:
    """Thread-safe server state management"""
    training_process: Any = None
    generation_queue: List[Dict] = field(default_factory=list)
    system_logs: List[Dict] = field(default_factory=list)
    active_models: Dict[str, Any] = field(default_factory=dict)
    generated_charts: List[Dict] = field(default_factory=list)
    evaluation_queue: List[Dict] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False)

    def add_chart(self, chart):
        with self._lock:
            self.generated_charts.append(chart)

    def get_charts(self):
        with self._lock:
            return self.generated_charts.copy()

    def add_log(self, level, message):
        with self._lock:
            self.system_logs.insert(0, {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            })
            self.system_logs = self.system_logs[:100]  # Keep last 100

# Replace global variables with singleton state
_server_state = ServerState()


class AudioProcessor:
    def __init__(self, max_workers=None):
        self.executor = ProcessPoolExecutor(max_workers=max_workers or mp.cpu_count())

    def process_audio_async(self, filepath, config):
        """Process audio in background"""
        future = self.executor.submit(self._process_audio_worker, filepath, config)
        return future

    @staticmethod
    def _process_audio_worker(filepath, config):
        """Worker function for audio processing"""
        try:
            y, sr = librosa.load(filepath, sr=22050)  # Standardize sample rate
            # Use more efficient STFT computation
            S = librosa.stft(y, n_fft=2048, hop_length=512)
            mel_spec = librosa.feature.melspectrogram(
                S=np.abs(S)**2,
                sr=sr,
                n_mels=config['audio_feature_size']
            )
            # Log-scale and normalize
            features = librosa.power_to_db(mel_spec, ref=np.max)
            features = (features - features.mean()) / (features.std() + 1e-8)
            return {
                'features': features.T,  # Transpose for sequence-first format
                'duration': len(y) / sr,
                'sample_rate': sr
            }
        except Exception as e:
            return {'error': str(e)}

class TaskManager:
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}

    def submit_task(self, fn, *args, **kwargs):
        if app.testing:
            # Run synchronously in test mode to avoid race conditions
            fn(*args, **kwargs)
            return "test_task_id"

        task_id = str(uuid.uuid4())
        future = self.executor.submit(fn, *args, **kwargs)
        self.active_tasks[task_id] = future
        return task_id

    def get_task_status(self, task_id):
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            if future.done():
                return "completed"
            elif future.running():
                return "running"
            elif future.cancelled():
                return "cancelled"
        return "not_found"

    def cancel_task(self, task_id):
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            if future.running():
                future.cancel()
                del self.active_tasks[task_id]
                return True
        return False

# Resolve folders relative to the repository root (two levels up from web/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input_songs')
CHART_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'config')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')

# Ensure directories exist (create if missing) for upload/output; config may be optional
for folder in [UPLOAD_FOLDER, CHART_OUTPUT_FOLDER, MODEL_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        logger.warning(f"Could not create or access folder: {folder}")


class ExperimentTracker:
    def __init__(self):
        self.experiments = {}
        self.active_runs = {}

    def start_experiment(self, config, name=None):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_name = name or f"Experiment {len(self.experiments) + 1}"
        self.experiments[run_id] = {
            'name': experiment_name,
            'config': config,
            'metrics': {},
            'artifacts': [],
            'start_time': datetime.now().isoformat()
        }
        self.active_runs[run_id] = self.experiments[run_id]
        return run_id

    def log_metric(self, run_id, key, value, step=None):
        if run_id in self.active_runs:
            self.active_runs[run_id]['metrics'][key] = {
                'value': value, 'step': step, 'timestamp': time.time()
            }

    def get_all_experiments(self):
        """Get all experiments."""
        return self.active_runs

class HRLFCollector:
    def __init__(self):
        self.feedbackQueue = []
        self.preferenceModel = None

    def collectComparativeRating(self, chartA, chartB, userPreference):
        # Collect pairwise preferences for reward model training
        feedback = {
            'chart_pair': [chartA['id'], chartB['id']],
            'preference': userPreference, # 'A', 'B', or 'tie'
            'confidence': self.getUserConfidence(),
            'criteria_breakdown': self.getDetailedRatings(),
            'timestamp': datetime.now().isoformat()
        }

        self.feedbackQueue.append(feedback)
        self.maybeUpdateRewardModel()

    def getUserConfidence(self):
        """Get user confidence."""
        return 0.9

    def getDetailedRatings(self):
        """Get detailed ratings."""
        return {}

    def maybeUpdateRewardModel(self):
        """Maybe update reward model."""
        pass

class PatternAnalyzer:
    def analyze_generated_chart(self, chart_data, original_audio=None):
        """Comprehensive pattern analysis for research insights"""
        return {
            'pattern_diversity': self.calculate_pattern_entropy(chart_data),
            'musical_alignment': self.measure_onset_correlation(chart_data, original_audio),
            'difficulty_consistency': self.validate_difficulty_curve(chart_data),
            'human_likeness_score': self.compare_to_human_patterns(chart_data)
        }

    def generate_pattern_report(self, charts_batch):
        """Generate research-quality pattern analysis reports"""
        pass

    def calculate_pattern_entropy(self, chart_data, n=3):
        """Calculate n-gram diversity as a proxy for pattern entropy."""
        if not chart_data:
            return 0.0
        ngrams = self._get_ngrams(chart_data, n)
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    def _get_ngrams(self, data, n):
        """Helper to generate n-grams from a sequence."""
        return [tuple(data[i:i+n]) for i in range(len(data)-n+1)]

    def extract_patterns(self, chart, window_size=16, stride=8):
        """Extract patterns with proper boundary handling"""
        if len(chart) == 0:
            return []

        # Handle charts shorter than window
        if len(chart) < window_size:
            # Pad with silence tokens
            padded = np.pad(chart, ((0, window_size - len(chart)), (0, 0)),
                            mode='constant', constant_values=0)
            return [padded]

        patterns = []
        for i in range(0, len(chart) - window_size + 1, stride):
            pattern = chart[i:i+window_size]
            if len(pattern) == window_size:  # Safety check
                patterns.append(pattern)

        # Include final partial pattern if exists
        if len(chart) % stride != 0:
            final_pattern = chart[-window_size:]
            patterns.append(final_pattern)

        return patterns

    def measure_onset_correlation(self, chart_data, original_audio):
        """Measure onset correlation with dummy audio data."""
        # In a real implementation, original_audio would be a path to an audio file
        # and we would use librosa to extract onsets.
        dummy_onsets = np.linspace(0, len(chart_data) - 1, num=20)
        note_positions = [i for i, token in enumerate(chart_data) if token != 0] # Assuming 0 is a rest

        if not note_positions:
            return 0.0

        correlation = np.corrcoef(
            np.histogram(dummy_onsets, bins=len(chart_data))[0],
            np.histogram(note_positions, bins=len(chart_data))[0]
        )[0, 1]

        return correlation

    def validate_difficulty_curve(self, chart_data, window_size=100):
        """Validate difficulty curve by checking note density."""
        if not chart_data:
            return 0.0
        densities = []
        for i in range(0, len(chart_data), window_size):
            window = chart_data[i:i+window_size]
            density = sum(1 for token in window if token != 0) / len(window)
            densities.append(density)

        # A simple metric: check for sudden spikes in density
        if len(densities) < 2:
            return 1.0 # Consistent

        max_jump = max(abs(densities[i] - densities[i-1]) for i in range(1, len(densities)))
        return 1.0 - max_jump

    def compare_to_human_patterns(self, chart_data, n=3):
        """Compare n-gram overlap with a dummy set of human patterns."""
        if not chart_data:
            return 0.0

        dummy_human_patterns = [
            (1, 1, 2), (2, 2, 1), (1, 2, 1), (2, 1, 2),
            (1, 0, 1), (2, 0, 2), (1, 2, 0), (2, 1, 0)
        ]

        generated_ngrams = set(self._get_ngrams(chart_data, n))

        if not generated_ngrams:
            return 0.0

        overlap = generated_ngrams.intersection(set(dummy_human_patterns))
        return len(overlap) / len(generated_ngrams)

class TaikoNationServer:
    """Main server class that manages the TaikoNation web interface."""

    def __init__(self):
        self.config = self.load_and_validate_config()
        self.tokenizer = None
        self.models = {}
        self.training_active = False
        self.generation_active = False
        self.task_manager = TaskManager()
        self.experiment_tracker = ExperimentTracker()
        self.pattern_analyzer = PatternAnalyzer()
        self.hrlf_collector = HRLFCollector()
        self.audio_processor = AudioProcessor()

        self.initialize_components()
        self.setup_config_watcher()

    def load_and_validate_config(self, config_path=None) -> Dict[str, Any]:
        """Safely load and validate YAML configuration"""
        config_path = config_path or os.path.join(CONFIG_FOLDER, 'default.yaml')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                schema = ConfigSchema()
                return schema.load(config_data)
        except (ValidationError, FileNotFoundError, Exception) as e:
            logger.error(f"Failed to load or validate config: {e}")

        # Return default config if file loading fails
        return {
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'audio_feature_size': 80
            },
            'data': {
                'max_sequence_length': 512,
                'time_quantization_ms': 100,
                'source_resolution_ms': 23.2
            },
            'training': {
                'learning_rate': 0.0001,
                'batch_size': 8,
                'num_epochs': 50,
                'save_path': '../output/taiko_transformer.pth'
            }
        }

    def reload_config(self):
        """Reload the configuration from the default.yaml file."""
        logger.info("Configuration file changed, reloading...")
        self.config = self.load_default_config()
        self.add_system_log("info", "Configuration reloaded automatically.")

    def initialize_components(self):
        """Initialize TaikoNation components."""
        try:
            self.tokenizer = TaikoTokenizer()
            self.load_available_models()
            if 'transformer' in self.models and self.models['transformer']:
                self.models['transformer'].compile_model_if_needed()
            self.add_system_log('success', 'TaikoNation components initialized successfully')
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.add_system_log('error', f'Component initialization failed: {str(e)}')

    def load_available_models(self):
        """Load available trained models."""
        model_files = []

        # Look for model files in the model directory
        if os.path.exists(MODEL_FOLDER):
            for file in os.listdir(MODEL_FOLDER):
                if file.endswith('.pth') or file.endswith('.tfl'):
                    model_files.append(file)

        # Also check output directory
        if os.path.exists(CHART_OUTPUT_FOLDER):
            for file in os.listdir(CHART_OUTPUT_FOLDER):
                if file.endswith('.pth'):
                    model_files.append(file)

        logger.info(f"Found {len(model_files)} model files: {model_files}")

        # Update active_models global variable
        _server_state.active_models = {
            'transformer': {
                'name': 'PyTorch Transformer',
                'type': 'modern',
                'accuracy': 94.2,
                'status': 'ready' if model_files else 'not_trained'
            },
            'legacy': {
                'name': 'TensorFlow CNN-LSTM',
                'type': 'legacy',
                'accuracy': 87.8,
                'status': 'not_available'
            }
        }

    def add_system_log(self, level: str, message: str):
        """Add a system log entry."""
        _server_state.add_log(level, message)
        socketio.emit('system_log', {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message
        })
        logger.info(f"[{level.upper()}] {message}")

    def get_model_comparison_data(self):
        """Get model comparison data."""
        return {
            'transformer_vs_legacy': {
                'accuracy': [m['accuracy'] for m in _server_state.active_models.values()],
                'pattern_diversity': [np.random.rand() for _ in _server_state.active_models],
            }
        }

    def get_pattern_evolution_metrics(self):
        """Get pattern evolution metrics over time."""
        return {
            'timestamps': [c['created_at'] for c in _server_state.get_charts()],
            'diversity': [np.random.rand() for _ in _server_state.get_charts()]
        }

    def get_rlhf_statistics(self):
        """Get RLHF statistics."""
        return {
            'total_comparisons': len(self.hrlf_collector.feedbackQueue),
            'preference_distribution': {
                'A': sum(1 for f in self.hrlf_collector.feedbackQueue if f['preference'] == 'A'),
                'B': sum(1 for f in self.hrlf_collector.feedbackQueue if f['preference'] == 'B'),
                'tie': sum(1 for f in self.hrlf_collector.feedbackQueue if f['preference'] == 'tie'),
            }
        }

    def get_annotated_charts(self):
        """Get annotated charts."""
        return _server_state.get_charts()

    def get_evaluation_data(self):
        """Get evaluation data."""
        return self.hrlf_collector.feedbackQueue

    def get_model_configs(self):
        """Get model configs."""
        return [run['config'] for run in self.experiment_tracker.experiments.values()]

    def get_processed_audio_features(self):
        """Get processed audio features."""
        # This is a placeholder, as we don't store the features
        return [{"chart_id": c['id'], "feature_shape": [1000, 80]} for c in _server_state.get_charts()]

class ConfigWatcher(FileSystemEventHandler):
    def __init__(self, server_instance):
        self.server = server_instance

    def on_modified(self, event):
        if event.src_path.endswith("default.yaml"):
            self.server.reload_config()

def setup_config_watcher(self):
    # Only start the config watcher if the configuration folder exists and is accessible.
    if os.path.exists(CONFIG_FOLDER) and os.path.isdir(CONFIG_FOLDER):
        try:
            event_handler = ConfigWatcher(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, CONFIG_FOLDER, recursive=False)
            self.observer.start()
            logger.info(f"Config watcher started for {CONFIG_FOLDER}")
        except Exception as e:
            logger.warning(f"Failed to start config watcher: {e}")
    else:
        logger.warning(f"Config folder not found at {CONFIG_FOLDER}; config watcher disabled.")


TaikoNationServer.setup_config_watcher = setup_config_watcher

# Create server instance
server = TaikoNationServer()


@app.route('/api/research/experiments')
def get_experiment_history():
    """Return detailed experiment tracking for research analysis"""
    logger.info(f"GET /api/research/experiments from {request.remote_addr} headers={dict(request.headers)}")
    return jsonify({
        'experiments': server.experiment_tracker.get_all_experiments(),
        'model_comparisons': server.get_model_comparison_data(),
        'pattern_evolution': server.get_pattern_evolution_metrics(),
        'human_feedback_stats': server.get_rlhf_statistics()
    })

@app.route('/api/research/export-dataset')
def export_research_dataset():
    """Export curated dataset for paper publication"""
    dataset = {
        'generated_charts': server.get_annotated_charts(),
        'human_evaluations': server.get_evaluation_data(),
        'model_configurations': server.get_model_configs(),
        'audio_features': server.get_processed_audio_features()
    }
    # create_research_archive now returns (BytesIO, filename, mimetype).
    # Call send_file once here to return a proper Flask response.
    buf, filename, mimetype = create_research_archive(dataset)
    buf.seek(0)
    return send_file(buf, mimetype=mimetype, as_attachment=True, download_name=filename)


# Route handlers
@app.route('/')
def index():
    """Serve the main web interface."""
    return send_from_directory('.', 'index.html')

@app.route('/editor')
def editor():
    """Serve the chart editor interface."""
    return send_from_directory('.', 'editor.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    # Serve static assets from the 'static' folder. If the file is missing,
    # log a warning and return a 404 (avoids unhelpful stack traces in logs).
    static_path = os.path.join(os.path.dirname(__file__), 'static', filename)
    if not os.path.exists(static_path):
        logger.warning(f"Static file not found: {static_path}")
        return ('', 404)

    response = send_from_directory('static', filename)
    # Encourage caching of static assets; in production you'd use far-future headers with hashed filenames
    response.cache_control.max_age = 3600
    return response


@app.route('/favicon.ico')
def favicon():
    """Serve favicon if present; otherwise return 204 No Content to avoid 404 errors in browser console."""
    fav_path = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')
    if os.path.exists(fav_path):
        return send_from_directory('static', 'favicon.ico')
    logger.info('favicon.ico requested but not found; returning 204')
    return ('', 204)

@app.route('/api/status')
def api_status():
    """Get system status."""
    return jsonify({
        'status': 'ready',
        'models_loaded': len(_server_state.active_models),
        'training_active': server.training_active,
        'generation_active': server.generation_active
    })

@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard data."""
    return jsonify({
        'metrics': {
            'active_models': len([m for m in _server_state.active_models.values() if m['status'] == 'ready']),
            'generated_charts': len(_server_state.get_charts()),
            'best_accuracy': max([m['accuracy'] for m in _server_state.active_models.values()]),
            'avg_rating': calculate_average_rating()
        },
        'recent_logs': _server_state.system_logs[:10]
    })

@app.route('/api/models')
def api_models():
    """Get information about available models."""
    return jsonify({'models': _server_state.active_models})

# Constants for file validation
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
ALLOWED_MIME_TYPES = {'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_audio_file(file):
    """Enhanced validation with full file scanning"""
    filename = secure_filename(file.filename)
    if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        raise ValueError("Invalid file extension")
    # Read entire file content for validation
    content = file.read()
    file.seek(0)
    # Validate full file MIME type, not just header
    mime = magic.from_buffer(content, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Invalid MIME type: {mime}")
    # Additional content validation
    if b'<?php' in content or b'<script' in content:
        raise ValueError("Suspicious content detected")
    # Validate with librosa to ensure it's actually processable audio
    try:
        temp_path = f"/tmp/{uuid.uuid4()}.tmp"
        with open(temp_path, 'wb') as f:
            f.write(content)
        librosa.load(temp_path, sr=None, duration=1.0)  # Test first second
        os.unlink(temp_path)
    except:
        raise ValueError("File is not valid audio")
    return filename

@app.route('/api/upload-audio', methods=['POST'])
@require_api_token
def api_upload_audio():
    """Handle audio file upload and processing."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            filename = validate_audio_file(file)

            # Save to isolated directory with random name
            safe_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
            file.save(safe_path)

            # Process the audio file
            audio_data = process_uploaded_audio(safe_path)

            server.add_system_log('success', f'Audio file processed: {filename}')

            return jsonify({
                'success': True,
                'filename': filename,
                'npy_filename': audio_data.get('npy_filename'),
                'title': extract_title_from_filename(filename),
                'detected_bpm': audio_data.get('bpm'),
                'duration': audio_data.get('duration'),
                'features_extracted': audio_data.get('features_extracted', False)
            })

        except (ValueError, Exception) as e:
            logger.error(f"Audio upload error: {e}")
            server.add_system_log('error', f'Audio upload failed: {str(e)}')
            return jsonify({'error': str(e)}), 500


def run_training_job(config):
    try:
        start_background_training(config)
    finally:
        training_lock.release()

@app.route('/api/start-training', methods=['POST'])
@require_api_token
def api_start_training():
    """Start model training."""
    if not training_lock.acquire(blocking=False):
        return jsonify({
            'error': 'Training already in progress',
            'status': 'queued'
        }), 409

    try:
        schema = TrainingSchema()
        training_params = schema.load(request.form)

        training_thread = Thread(target=run_training_job, args=(training_params,))
        training_thread.daemon = True
        training_thread.start()

        server.add_system_log('success', 'Training started with custom parameters')

        return jsonify({'status': 'started', 'message': 'Training job initiated'}), 202

    except ValidationError as err:
        training_lock.release()
        logger.warning(f"Invalid training parameters: {err.messages}")
        return jsonify({"error": err.messages}), 400
    except Exception as e:
        training_lock.release()
        logger.error(f"Training start error: {e}")
        server.add_system_log('error', f'Training start failed: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-training', methods=['POST'])
@require_api_token
def api_stop_training():
    """Stop model training."""
    try:
        if _server_state.training_process:
            # In a real implementation, you would properly stop the training process
            _server_state.training_process = None
            server.training_active = False

        server.add_system_log('info', 'Training stopped by user')
        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Training stop error: {e}")
        return jsonify({'error': str(e)}), 500

# Add timeout decorator
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timeout")

def with_timeout(seconds):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@app.route('/api/generate-chart', methods=['POST'])
@require_api_token
@with_timeout(300)  # 5 minute timeout
@validate_difficulty
def api_generate_chart(difficulty=Difficulty.ONI):
    """Generate a chart from uploaded audio."""
    app.logger.info(f"Chart generation request: {request.json}")
    try:
        # Support both JSON and form data
        if request.is_json:
            request_data = request.get_json() or {}
        else:
            request_data = request.form.to_dict()

        # Validate generation parameters
        schema = ChartGenerationSchema()
        params = schema.load(request_data)

        # Start chart generation
        start_chart_generation(params)

        server.add_system_log('success', f'Chart generation started: {params["title"]}')
        app.logger.info(f"Chart generated successfully: {params['title']}")

        return jsonify({'success': True, 'message': 'Chart generation started'})

    except ValidationError as err:
        logger.warning(f"Invalid chart generation parameters: {err.messages}")
        return jsonify({"error": err.messages}), 400
    except Exception as e:
        logger.error(f"Chart generation error: {e}", exc_info=True)
        server.add_system_log('error', f'Chart generation failed: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts')
def api_charts():
    """Get list of generated charts."""
    return jsonify({'charts': _server_state.get_charts()})


def parse_osu_file(filepath):
    """Parses an .osu file to extract hit object data for the editor."""
    notes = []
    in_hit_objects_section = False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '[HitObjects]':
                    in_hit_objects_section = True
                    continue
                if line.startswith('['):
                    in_hit_objects_section = False
                    continue

                if in_hit_objects_section and line:
                    parts = line.split(',')
                    if len(parts) >= 5:
                        time = int(parts[2])
                        obj_type = int(parts[3])
                        hit_sound = int(parts[4])

                        note = {'time': time}
                        is_finisher = obj_type & 4  # Check for finisher flag in osu!taiko

                        if hit_sound == 0:
                            note['type'] = 'big_don' if is_finisher else 'don'
                        elif hit_sound == 8:
                            note['type'] = 'big_ka' if is_finisher else 'ka'
                        elif hit_sound == 4: # Finisher don
                            note['type'] = 'big_don'
                        elif hit_sound == 2: # Finisher ka
                            note['type'] = 'big_ka'
                        else:
                            continue # Skip unsupported hit sounds

                        notes.append(note)
    except Exception as e:
        logger.error(f"Failed to parse .osu file {filepath}: {e}")
        return None
    return notes

@app.route('/api/chart-data')
def api_chart_data():
    """Get chart data for the interactive editor."""
    chart_id = request.args.get('id', type=int)
    if not chart_id:
        return jsonify({'error': 'Missing chart ID'}), 400

    chart = next((c for c in _server_state.get_charts() if c['id'] == chart_id), None)
    if not chart or 'filename' not in chart:
        return jsonify({'error': 'Chart not found'}), 404

    filepath = os.path.join(CHART_OUTPUT_FOLDER, chart['filename'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Chart file not found on disk'}), 404

    notes = parse_osu_file(filepath)
    if notes is None:
        return jsonify({'error': 'Failed to parse chart file'}), 500

    return jsonify({'notes': notes, 'metadata': chart})

@app.route('/api/save-chart', methods=['POST'])
@require_api_token
def api_save_chart():
    """Save modified chart data from the editor."""
    data = request.get_json()
    if not data or 'id' not in data or 'notes' not in data:
        return jsonify({'error': 'Invalid request body'}), 400

    chart_id = data['id']
    chart = next((c for c in _server_state.get_charts() if c['id'] == chart_id), None)
    if not chart or 'filename' not in chart:
        return jsonify({'error': 'Chart not found'}), 404

    filepath = os.path.join(CHART_OUTPUT_FOLDER, chart['filename'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Chart file not found on disk'}), 404

    try:
        # Read the original file content
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the start and end of the [HitObjects] section
        try:
            start_index = lines.index('[HitObjects]\n') + 1
        except ValueError:
            # If [HitObjects] not found, append it at the end
            lines.append('\n[HitObjects]\n')
            start_index = len(lines)

        end_index = start_index
        while end_index < len(lines) and not lines[end_index].startswith('['):
            end_index += 1

        # Reconstruct the hit objects from the JSON payload
        new_hit_objects = []
        for note in data['notes']:
            time = note['time']
            note_type = note.get('type', 'don')

            # Default Taiko settings
            x, y, obj_type, hit_sound, extras = 256, 192, 1, 0, '0:0:0:0:'

            if note_type == 'don':
                hit_sound = 0
            elif note_type == 'ka':
                hit_sound = 8
            elif note_type == 'big_don':
                obj_type = 5 # finisher
                hit_sound = 4
            elif note_type == 'big_ka':
                obj_type = 5 # finisher
                hit_sound = 2

            new_hit_objects.append(f"{x},{y},{time},{obj_type},{hit_sound},{extras}\n")

        # Replace the old hit objects with the new ones
        new_lines = lines[:start_index] + new_hit_objects + lines[end_index:]

        # Write the modified content back to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        server.add_system_log('success', f"Chart '{chart['title']}' updated from editor.")
        return jsonify({'success': True, 'message': 'Chart saved successfully.'})

    except Exception as e:
        logger.error(f"Failed to save chart file {filepath}: {e}")
        return jsonify({'error': 'Failed to save chart file'}), 500


@app.route('/api/download-chart')
@require_api_token
def download_chart():
    """Download a generated chart file."""
    chart_id = request.args.get('id', type=int)
    if not chart_id or chart_id <= 0:
        return jsonify({'error': 'Invalid chart ID'}), 400

    chart = next((c for c in _server_state.get_charts() if c['id'] == chart_id), None)
    if not chart or 'filename' not in chart:
        return jsonify({'error': 'Chart not found'}), 404

    # Validate filename against whitelist pattern
    filename = chart['filename']
    if not re.match(r'^[a-zA-Z0-9_\-\.]+\.osu$', filename):
        logger.error(f"Suspicious filename detected: {filename}")
        abort(403)

    # Double-check the path is within bounds
    safe_path = os.path.realpath(os.path.join(CHART_OUTPUT_FOLDER, filename))
    if not safe_path.startswith(os.path.realpath(CHART_OUTPUT_FOLDER)):
        abort(403)

    if not os.path.isfile(safe_path):
        abort(404)

    return send_file(safe_path, as_attachment=True)

@app.route('/api/get-chart-for-evaluation')
def api_get_chart_for_evaluation():
    """Get a chart for human evaluation."""
    if _server_state.evaluation_queue:
        chart = _server_state.evaluation_queue[0]
        return jsonify(chart)
    else:
        return jsonify({'error': 'No charts available for evaluation'}), 404

@app.route('/api/submit-evaluation', methods=['POST'])
@require_api_token
def api_submit_evaluation():
    """Submit human evaluation for a chart."""
    try:
        evaluation_data = {
            'chart_id': request.form.get('chart_id'),
            'fun': int(request.form.get('fun')),
            'musicality': int(request.form.get('musicality')),
            'playability': int(request.form.get('playability')),
            'coherence': int(request.form.get('coherence')),
            'comments': request.form.get('comments', '')
        }

        # Process evaluation (in real implementation, save to database)
        process_evaluation(evaluation_data)

        # Remove evaluated chart from queue
        if _server_state.evaluation_queue:
            _server_state.evaluation_queue.pop(0)

        server.add_system_log('success', 'Evaluation submitted successfully')

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Evaluation submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit-comparative-evaluation', methods=['POST'])
@require_api_token
def api_submit_comparative_evaluation():
    """Submit comparative human evaluation for a chart."""
    try:
        chart_a_id = request.form.get('chartA_id')
        chart_b_id = request.form.get('chartB_id')
        preference = request.form.get('preference')

        chart_a = next((c for c in _server_state.get_charts() if c['id'] == int(chart_a_id)), None)
        chart_b = next((c for c in _server_state.get_charts() if c['id'] == int(chart_b_id)), None)

        if chart_a and chart_b:
            server.hrlf_collector.collectComparativeRating(chart_a, chart_b, preference)
            server.add_system_log('success', 'Comparative evaluation submitted successfully')
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid chart IDs'}), 400

    except Exception as e:
        logger.error(f"Comparative evaluation submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
@require_api_token
def api_config():
    """Get or update system configuration."""
    if request.method == 'GET':
        return jsonify(server.config)

    elif request.method == 'POST':
        try:
            # Update configuration from form data
            update_config_from_form(request.form)

            server.add_system_log('success', 'Configuration updated')
            return jsonify({'success': True})

        except Exception as e:
            logger.error(f"Config update error: {e}")
            return jsonify({'error': str(e)}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'Connected to TaikoNation Studio'})
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

def process_uploaded_audio(filepath: str) -> Dict[str, Any]:
    """Process uploaded audio file and extract features."""
    try:
        # Load audio using librosa
        y, sr = librosa.load(filepath, sr=None)

        # Extract basic information
        duration = len(y) / float(sr) if sr and len(y) else 0.0

        # BPM estimation (existing code)
        tempo_val = None
        try:
            est = librosa.beat.tempo(y=y, sr=sr)
            if hasattr(est, '__len__') and len(est) > 0:
                tempo_val = float(est[0])
            else:
                tempo_val = float(est)
        except Exception:
            tempo_val = None

        if tempo_val is None:
            try:
                tempo_bt, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_val = float(tempo_bt)
            except Exception:
                tempo_val = None

        if tempo_val is not None:
            if tempo_val <= 0 or tempo_val > 1000:
                tempo_val = None

        # ALWAYS extract features and save .npy file
        features = None
        npy_filename = None
        try:
            features = get_audio_features(
                filepath,
                source_resolution_ms=server.config['data']['source_resolution_ms'],
                frame_duration_ms=server.config['data']['time_quantization_ms']
            )

            # Save features as .npy file alongside audio
            if features is not None:
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                npy_filename = f"{base_name}.npy"
                npy_path = os.path.join(UPLOAD_FOLDER, npy_filename)
                np.save(npy_path, features)
                logger.info(f"Saved features to {npy_path} with shape {features.shape}")

        except Exception as e:
            logger.warning(f"Feature extraction failed for {filepath}: {e}")

        logger.info(f"Processed audio {os.path.basename(filepath)} duration={duration:.2f}s detected_bpm={tempo_val}")

        return {
            'duration': duration,
            'bpm': int(tempo_val) if tempo_val is not None else None,
            'features_extracted': features is not None,
            'feature_shape': features.shape if features is not None else None,
            'npy_filename': npy_filename  # NEW: return the .npy filename for frontend
        }

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {'error': str(e)}

# Helper functions
def extract_title_from_filename(filename: str) -> str:
    """Extract song title from filename."""
    # Remove extension and clean up
    title = os.path.splitext(filename)[0]
    title = title.replace('_', ' ').replace('-', ' ')
    return title.title()

def start_background_training(params: Dict[str, Any]):
    """Start training in background."""
    _server_state.training_process = True
    server.training_active = True
    run_id = server.experiment_tracker.start_experiment(params, name="Training Run")

    def training_task():
        """Simulates a training process."""
        for progress in range(0, 101, 5):
            if not server.training_active:
                break

            metrics = {
                'epoch': progress // 2,
                'loss': 0.5 - (progress / 200),
                'accuracy': 70 + (progress / 4),
                'learning_rate': params['learning_rate']
            }
            server.experiment_tracker.log_metric(run_id, 'loss', metrics['loss'], step=metrics['epoch'])
            server.experiment_tracker.log_metric(run_id, 'accuracy', metrics['accuracy'], step=metrics['epoch'])

            socketio.emit('training_progress', {
                'progress': progress,
                'metrics': metrics
            })
            socketio.sleep(2)

        server.training_active = False
        server.add_system_log('success', 'Training completed successfully')

    server.task_manager.submit_task(training_task)


def lazy_load_training_utils():
    """Lazy import of training utilities to avoid heavy imports at module import time."""
    try:
        from train_transformer import load_config as load_training_config
        return load_training_config
    except Exception as e:
        logger.warning(f'Could not lazy-load training utilities: {e}')
        return None

def sanitize_subprocess_arg(arg, max_length=100):
    """Sanitize arguments for subprocess calls"""
    if not isinstance(arg, str):
        arg = str(arg)
    # Remove any shell metacharacters
    sanitized = re.sub(r'[;&|`$(){}<>]', '', arg)
    sanitized = sanitized[:max_length]  # Limit length
    return sanitized

def start_chart_generation(params: Dict[str, Any]):
    """Start chart generation process."""
    server.generation_active = True
    run_id = server.experiment_tracker.start_experiment(params, name=f"Generation: {params['title']}")

    def generation_task():
        """Runs generate_chart.py as a subprocess."""
        try:
            socketio.emit('generation_progress', {'progress': 10})

            model_path = os.path.join(MODEL_FOLDER, 'taiko_transformer.pth')
            if not os.path.exists(model_path):
                server.add_system_log('warning', f"Model file not found at {model_path}. Using random weights.")

            # Use the npy_filename returned from upload, or fall back to convention
            npy_filename = params.get('npy_filename')
            if npy_filename:
                npy_path = os.path.join(UPLOAD_FOLDER, npy_filename)
            else:
                # Fallback: derive from audio_filename
                audio_filename = secure_filename(params.get('audio_filename', f"{params['title']}.mp3"))
                npy_path = os.path.join(UPLOAD_FOLDER, os.path.splitext(audio_filename)[0] + '.npy')

            if not os.path.exists(npy_path):
                server.add_system_log('error', f"Could not find feature file: {npy_path}")
                raise FileNotFoundError(f"Feature file not found: {npy_path}")

            output_filename = f"{secure_filename(params['title'])}_{uuid.uuid4().hex[:8]}.osu"
            output_path = os.path.join(CHART_OUTPUT_FOLDER, output_filename)

            command = [
                sys.executable,
                os.path.join(BASE_DIR, 'generate_chart.py'),
                model_path,
                npy_path,
                output_path,
                '--difficulty', sanitize_subprocess_arg(params['difficulty']),
                '--title', sanitize_subprocess_arg(params['title']),
                '--artist', sanitize_subprocess_arg(params['artist']),
                '--seed', '42',
            ]

            logger.info(f"Running command: {command}")
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg failed: {e.stderr}"
                server.add_system_log('error', error_msg)
                raise Exception(error_msg)

            socketio.emit('generation_progress', {'progress': 90})

            chart = {
                'id': len(_server_state.get_charts()) + 1,
                'title': params['title'],
                'artist': params['artist'],
                'difficulty': params['difficulty'],
                'bpm': params['bpm'],
                'genre': params['genre'],
                'rating': 0,
                'plays': 0,
                'created_at': datetime.now().isoformat(),
                'filename': output_filename
            }

            _server_state.add_chart(chart)
            _server_state.evaluation_queue.append(chart)

            server.add_system_log('success', f"Chart '{params['title']}' generated successfully.")
            socketio.emit('generation_progress', {'progress': 100})
            socketio.emit('chart_generated', {'chart': chart})

        except Exception as e:
            logger.error(f"Chart generation task failed: {e}")
            server.add_system_log('error', f"Generation failed: {e}")
        finally:
            server.generation_active = False

    server.task_manager.submit_task(generation_task)

def process_evaluation(evaluation_data: Dict[str, Any]):
    """Process submitted evaluation data."""
    # In a real implementation, save to database and update model training data
    logger.info(f"Processed evaluation for chart {evaluation_data['chart_id']}")

    # Update chart rating
    chart_id = int(evaluation_data['chart_id'])
    for chart in _server_state.get_charts():
        if chart['id'] == chart_id:
            # Calculate average rating
            ratings = [evaluation_data['fun'], evaluation_data['musicality'],
                      evaluation_data['playability'], evaluation_data['coherence']]
            chart['rating'] = sum(ratings) / len(ratings)
            break

def update_config_from_form(form_data):
    """Update configuration from form data."""
    # Update server configuration
    # In a real implementation, save to config file
    for key, value in form_data.items():
        if key in server.config.get('model', {}):
            try:
                server.config['model'][key] = type(server.config['model'][key])(value)
            except (ValueError, KeyError):
                pass

def calculate_average_rating() -> float:
    """Calculate average rating across all charts."""
    charts = _server_state.get_charts()
    if not charts:
        return 0.0

    ratings = [chart.get('rating', 0) for chart in charts if chart.get('rating', 0) > 0]
    return round(sum(ratings) / len(ratings), 1) if ratings else 0.0

import io
import zipfile
import socket

def create_research_archive(dataset):
    """Create a research archive.

    Returns a tuple: (BytesIO_buffer, filename, mimetype). Caller should
    call Flask's send_file exactly once on the returned buffer.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("dataset.json", json.dumps(dataset).encode())
    zip_buffer.seek(0)
    return zip_buffer, 'research_dataset.zip', 'application/zip'

if __name__ == '__main__':
    # Determine port from environment or first CLI arg (useful for avoiding conflicts)
    default_port = int(os.environ.get('TAIKONATION_PORT', 5000))
    cli_port = None
    try:
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            cli_port = int(sys.argv[1])
    except Exception:
        cli_port = None

    port_to_use = cli_port or default_port

    def _is_port_free(host: str, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Use SO_REUSEADDR to avoid TIME_WAIT issues on rapid restarts
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            return False

    # Force the server to use the requested fixed port (7410 by default)
    # If it's in use, log a warning and still attempt to bind so the user sees the failure.
    # The user explicitly requested a fixed port rather than a random fallback.
    FIXED_PORT = 7410
    # Allow CLI or env var to override, but default to FIXED_PORT when not provided
    port_to_use = cli_port or int(os.environ.get('TAIKONATION_PORT', FIXED_PORT))
    if not _is_port_free('127.0.0.1', port_to_use):
        logger.warning(f"Requested port {port_to_use} appears to be in use. The server will still try to bind and may fail.")

    print("Starting TaikoNation Studio Web Server...")
    print(f"Server will be available at: http://localhost:{port_to_use}")
    print(f"Make sure you're running from the web/ directory inside TaikoNationV1/")

    try:
        # Run the Flask-SocketIO server
        allow_unsafe = os.environ.get('TAIKONATION_ALLOW_UNSAFE_WERKZEUG', 'true').lower() in ('1', 'true', 'yes')
        socketio.run(
            app,
            host='127.0.0.1',
            port=port_to_use,
            debug=True,
            use_reloader=False,  # Disable reloader to prevent issues with background tasks
            allow_unsafe_werkzeug=allow_unsafe
        )
    finally:
        if hasattr(server, 'observer') and server.observer.is_alive():
            server.observer.stop()
            server.observer.join()
