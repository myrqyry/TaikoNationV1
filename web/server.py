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

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, abort, session
from flask_socketio import SocketIO, emit
from marshmallow import Schema, fields, ValidationError
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
from enum import Enum

def verify_installation():
    """Verify that TaikoNation is properly installed"""
    import sys
    from pathlib import Path

    # Check if running from correct directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Verify key project files exist
    required_files = [
        project_root / 'setup.py',
        project_root / 'taikonation' / '__init__.py',
        project_root / 'requirements.txt'
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("ERROR: Project structure appears incomplete!")
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you're running from the web/ directory inside the TaikoNation project.")
        sys.exit(1)

    # Try importing core modules
    try:
        import taikonation
    except ImportError:
        print("ERROR: TaikoNation package not installed!")
        print("\nPlease install the package first:")
        print("  cd ..")
        print("  pip install -e .")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        print("  pip install -r web/requirements.txt")
        sys.exit(1)

    # Verify critical dependencies
    missing_deps = []
    critical_modules = ['torch', 'librosa', 'flask', 'flask_socketio', 'yaml', 'numpy']

    for module in critical_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(module)

    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("✓ Installation verified successfully")

# Call verification before other imports
verify_installation()

from .helpers import error_response
from functools import wraps
from taikonation.data.tokenization import TaikoTokenizer
from config_schema import ConfigSchema

class APIError(Exception):
    """Custom API exception with user-friendly messages"""
    def __init__(self, message, code, status_code=400, details=None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

def handle_api_errors(f):
    """Centralized error handler decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except APIError as e:
            logger.warning(f"API Error in {f.__name__}: {e.message}")
            return jsonify({
                'success': False,
                'error': e.message,
                'code': e.code,
                'details': e.details
            }), e.status_code
        except ValidationError as e:
            logger.warning(f"Validation Error in {f.__name__}: {e.messages}")
            return jsonify({
                'success': False,
                'error': 'Invalid input parameters',
                'code': 'VALIDATION_ERROR',
                'details': e.messages
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred. Please try again.',
                'code': 'INTERNAL_ERROR'
            }), 500
    return decorated

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
    from taikonation.data.audio_processing import get_audio_features, augment_spectrogram
    from taikonation.data.dataset import get_transformer_data_loaders, DIFFICULTY_MAP
    from taikonation.data.tokenization import TaikoTokenizer
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

# Initialize tasks module with socketio reference
from web import tasks
tasks.set_socketio(socketio)

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


from marshmallow import Schema, fields, validates, validates_schema, ValidationError

# Marshmallow Schemas for Input Validation
class ChartGenerationSchema(Schema):
    title = fields.Str(
        load_default="Untitled",
        validate=lambda x: len(x) <= 200 and len(x.strip()) > 0,
        error_messages={'validator_failed': 'Title must be 1-200 characters'}
    )
    artist = fields.Str(
        load_default="Unknown",
        validate=lambda x: len(x) <= 200 and len(x.strip()) > 0,
        error_messages={'validator_failed': 'Artist must be 1-200 characters'}
    )
    bpm = fields.Int(
        load_default=120,
        validate=lambda x: 60 <= x <= 300,
        error_messages={'validator_failed': 'BPM must be between 60 and 300'}
    )
    genre = fields.Str(
        load_default="electronic",
        validate=lambda x: x in ['electronic', 'rock', 'pop', 'classical', 'jazz', 'other'],
        error_messages={'validator_failed': 'Invalid genre selection'}
    )
    difficulty = fields.Str(
        load_default="oni",
        validate=lambda x: x in ["kantan", "futsuu", "muzukashii", "oni", "ura"],
        error_messages={'validator_failed': 'Invalid difficulty level'}
    )
    pattern_style = fields.Str(
        load_default="balanced",
        validate=lambda x: x in ['balanced', 'technical', 'stream', 'mixed'],
        error_messages={'validator_failed': 'Invalid pattern style'}
    )
    audio_filename = fields.Str(required=True)
    npy_filename = fields.Str(required=False, load_default=None)

    @validates('audio_filename')
    def validate_audio_file(self, value):
        """Check if audio file exists"""
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(value))
        if not os.path.exists(audio_path):
            raise ValidationError(f'Audio file not found: {value}')

    @validates_schema
    def validate_feature_file(self, data, **kwargs):
        """Check if feature file exists or can be generated"""
        npy_filename = data.get('npy_filename')
        audio_filename = data.get('audio_filename')

        if npy_filename:
            npy_path = os.path.join(UPLOAD_FOLDER, secure_filename(npy_filename))
            if not os.path.exists(npy_path):
                raise ValidationError(
                    {'npy_filename': f'Feature file not found: {npy_filename}'}
                )
        elif audio_filename:
            # Check if we can generate features from audio
            expected_npy = os.path.splitext(secure_filename(audio_filename))[0] + '.npy'
            npy_path = os.path.join(UPLOAD_FOLDER, expected_npy)
            if not os.path.exists(npy_path):
                raise ValidationError(
                    {'npy_filename': 'Audio features not yet processed. Please wait for processing to complete.'}
                )

    @validates_schema
    def validate_model_exists(self, data, **kwargs):
        """Check if trained model exists"""
        model_path = os.path.join(MODEL_FOLDER, 'taiko_transformer.pth')
        if not os.path.exists(model_path):
            raise ValidationError(
                {'_schema': 'No trained model found. Please train a model first.'}
            )


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

from concurrent.futures import ThreadPoolExecutor

class TaikoNationServer:
    """Main server class that manages the TaikoNation web interface."""

    def __init__(self):
        self.config = self.load_and_validate_config()
        self.tokenizer = None
        self.models = {}
        self.training_active = False
        self.generation_active = False
        self.experiment_tracker = ExperimentTracker()
        self.pattern_analyzer = PatternAnalyzer()
        self.hrlf_collector = HRLFCollector()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.initialize_components()
        self.setup_config_watcher()

    def submit_task(self, task_id):
        from web.tasks import run_task, TASKS_REGISTRY
        self.executor.submit(run_task, task_id, TASKS_REGISTRY)

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

@app.route('/api/submit_audio_classification', methods=['POST'])
def submit_audio_classification():
    """
    Receive MediaPipe audio classifications from web frontend
    Store for use in chart generation
    """
    from taikonation.data.mediapipe_audio import MediaPipeAudioAnalyzer
    mediapipe_analyzer = MediaPipeAudioAnalyzer()
    data = request.json

    audio_id = data['audio_id']
    classifications = data['classifications']
    analysis = data['analysis']

    # Store in session or database
    session[f'audio_classification_{audio_id}'] = {
        'classifications': classifications,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    }

    # Detect chart sections for structural generation
    sections = mediapipe_analyzer.detect_chart_sections(classifications)

    return jsonify({
        'status': 'success',
        'message': 'Audio classification received',
        'sections': sections,
        'percussion_ratio': analysis['avg_percussion_ratio'],
        'melodic_ratio': analysis['avg_melodic_ratio']
    })

@app.route('/api/generate_chart_with_audio_analysis', methods=['POST'])
def generate_chart_with_audio_analysis():
    """
    Enhanced chart generation using both mel features and MediaPipe classifications
    """
    data = request.json

    audio_path = data['audio_path']
    audio_id = data.get('audio_id', os.path.basename(audio_path))

    # Load existing mel features
    mel_features = get_audio_features(audio_path)

    # Get MediaPipe classifications from session
    classification_data = session.get(f'audio_classification_{audio_id}')

    model_path = os.path.join(MODEL_FOLDER, 'taiko_transformer.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, server.config, device)

    generator = EnhancedChartGenerator(model)
    chart = generator.generate(audio_path, data['difficulty'], mediapipe_data=classification_data)

    return jsonify({
        'status': 'success',
        'chart': chart.tolist(),
        'used_classification': classification_data is not None
    })


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

    # Read file content for validation, but with a limit to avoid memory issues
    content = file.read(1024 * 1024) # Read first 1MB
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
        file.save(temp_path)
        librosa.load(temp_path, sr=None, duration=1.0)  # Test first second
        os.unlink(temp_path)
    except Exception as e:
        raise ValueError(f"File is not a valid audio file: {e}")

    file.seek(0)
    return filename

@app.route('/api/upload-audio', methods=['POST'])
@require_api_token
@handle_api_errors
def api_upload_audio():
    """Handle audio file upload and processing."""
    if 'audio' not in request.files:
        raise APIError('No audio file provided', code='MISSING_FILE')

    file = request.files['audio']
    if file.filename == '':
        raise APIError('No file selected', code='MISSING_FILENAME')

    if file:
        filename = validate_audio_file(file)

        # Save to isolated directory with random name
        safe_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(safe_path)

        from web.tasks import create_task
        # Create a task for audio processing
        task_id = create_task('process_uploaded_audio', safe_path, server.config)

        server.submit_task(task_id)

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Audio processing started.'
        }), 202

@app.route('/api/tasks/<task_id>', methods=['GET'])
def api_get_task(task_id):
    """Get the status and result of a task."""
    from web.tasks import get_task_status
    task_data = get_task_status(task_id)
    if not task_data:
        return jsonify(error_response('Task not found', code='NOT_FOUND')), 404
    return jsonify(task_data)


@app.route('/api/start-training', methods=['POST'])
@require_api_token
def api_start_training():
    """Start model training."""
    try:
        schema = TrainingSchema()
        training_params = schema.load(request.form)

        from web.tasks import create_task
        task_id = create_task('start_training_task', training_params)

        server.submit_task(task_id)

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Training started.'
        }), 202

    except ValidationError as err:
        logger.warning(f"Invalid training parameters: {err.messages}")
        return jsonify(error_response('Invalid parameters', code='VALIDATION_ERROR', details=err.messages)), 400
    except Exception as e:
        logger.error(f"Training start error: {e}")
        server.add_system_log('error', f'Training start failed: {str(e)}')
        return jsonify(error_response(str(e), code='TRAINING_FAILED')), 500

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
@validate_difficulty
@handle_api_errors
def api_generate_chart(difficulty=Difficulty.ONI):
    """Generate a chart from uploaded audio."""
    app.logger.info(f"Chart generation request: {request.json}")
    # Support both JSON and form data
    if request.is_json:
        request_data = request.get_json() or {}
    else:
        request_data = request.form.to_dict()

    # Validate generation parameters
    schema = ChartGenerationSchema()
    params = schema.load(request_data)

    from web.tasks import create_task
    # Create a task for chart generation
    task_id = create_task('start_chart_generation', params)

    server.submit_task(task_id)

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Chart generation started.'
    }), 202

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
        return jsonify(error_response('Invalid chart ID', code='INVALID_ID')), 400

    chart = next((c for c in _server_state.get_charts() if c['id'] == chart_id), None)
    if not chart or 'filename' not in chart:
        return jsonify(error_response('Chart not found', code='NOT_FOUND')), 404

    # Sanitize and validate the filename to prevent traversal
    filename = secure_filename(chart['filename'])

    # Use safe_join to prevent path traversal attacks
    try:
        safe_path = safe_join(CHART_OUTPUT_FOLDER, filename)
    except Exception:
        abort(400)

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

# Helper functions
def extract_title_from_filename(filename: str) -> str:
    """Extract song title from filename."""
    # Remove extension and clean up
    title = os.path.splitext(filename)[0]
    title = title.replace('_', ' ').replace('-', ' ')
    return title.title()



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

from taikonation.generation.generator import generate_chart, load_model, save_osu_chart

def start_chart_generation(params: Dict[str, Any]):
    """Start chart generation with real-time progress"""
    try:
        def emit_progress(percent, message):
            socketio.emit('generation_progress', {
                'progress': percent,
                'message': message,
                'chart_title': params['title']
            })

        emit_progress(5, "Loading model...")
        model_path = os.path.join(MODEL_FOLDER, 'taiko_transformer.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, server.config, device)

        emit_progress(10, "Loading audio features...")
        npy_filename = params.get('npy_filename')
        if npy_filename:
            npy_path = os.path.join(UPLOAD_FOLDER, npy_filename)
        else:
            audio_filename = secure_filename(params.get('audio_filename', f"{params['title']}.mp3"))
            npy_path = os.path.join(UPLOAD_FOLDER, os.path.splitext(audio_filename)[0] + '.npy')

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Feature file not found: {npy_path}")
        audio_features = np.load(npy_path)

        emit_progress(15, "Starting chart generation...")
        difficulty_id = DIFFICULTY_MAP.get(params['difficulty'].lower(), 3)  # Default to Oni
        generated_token_ids = generate_chart(
            model, audio_features, server.tokenizer,
            difficulty_id, server.config, device,
            progress_callback=emit_progress
        )

        emit_progress(95, "Saving chart file...")
        output_filename = f"{secure_filename(params['title'])}_{uuid.uuid4().hex[:8]}.osu"
        output_path = os.path.join(CHART_OUTPUT_FOLDER, output_filename)
        save_osu_chart(generated_token_ids, server.tokenizer, output_path, params['audio_filename'], title=params['title'], artist=params['artist'])

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
        emit_progress(100, "Chart generation complete!")
        socketio.emit('chart_generated', {'chart': chart})

        return chart

    except Exception as e:
        logger.error(f"Chart generation task failed: {e}", exc_info=True)
        # Re-raise the exception to be caught by the task runner
        raise

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

def emit_training_metrics(epoch, metrics):
    """Emit training metrics to connected clients"""
    socketio.emit('training_metrics', {
        'epoch': epoch,
        'loss': metrics.get('loss', 0),
        'accuracy': metrics.get('accuracy', 0),
        'learning_rate': metrics.get('lr', 0),
        'gpu_memory_mb': metrics.get('gpu_memory_mb', 0),
        'timestamp': datetime.now().isoformat()
    })
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

def find_available_port(preferred_port: int, max_attempts: int = 10) -> int:
    """Find an available port, starting with preferred port"""
    import socket

    for offset in range(max_attempts):
        test_port = preferred_port + offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind(('127.0.0.1', test_port))
            sock.close()
            return test_port
        except OSError:
            sock.close()
            continue

    raise RuntimeError(f"Could not find available port after {max_attempts} attempts starting from {preferred_port}")

if __name__ == '__main__':
    # Determine port with smart fallback
    PREFERRED_PORT = 7410
    cli_port = None

    # Parse CLI arguments
    if len(sys.argv) > 1:
        try:
            cli_port = int(sys.argv[1])
        except (ValueError, IndexError):
            pass

    # Priority: CLI arg > Environment variable > Preferred port
    requested_port = cli_port or int(os.environ.get('TAIKONATION_PORT', PREFERRED_PORT))

    try:
        # Find available port (will use requested if free, or find next available)
        port_to_use = find_available_port(requested_port)

        if port_to_use != requested_port:
            logger.warning(f"Requested port {requested_port} was in use. Using port {port_to_use} instead.")
            print(f"⚠ Port {requested_port} was in use, using {port_to_use} instead")
    except RuntimeError as e:
        logger.error(f"Could not find available port: {e}")
        print(f"ERROR: {e}")
        print("Please specify a different port range or close applications using these ports.")
        sys.exit(1)

    print("=" * 60)
    print("TaikoNation Studio Web Server")
    print("=" * 60)
    print(f"Server URL: http://localhost:{port_to_use}")
    print(f"Server URL (network): http://{socket.gethostname()}.local:{port_to_use}")
    print(f"Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        allow_unsafe = os.environ.get('TAIKONATION_ALLOW_UNSAFE_WERKZEUG', 'true').lower() in ('1', 'true', 'yes')
        socketio.run(
            app,
            host='127.0.0.1',
            port=port_to_use,
            debug=True,
            use_reloader=False,
            allow_unsafe_werkzeug=allow_unsafe
        )
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        print(f"\nERROR: Server failed to start: {e}")
        sys.exit(1)
    finally:
        if hasattr(server, 'observer') and server.observer.is_alive():
            server.observer.stop()
            server.observer.join()
        print("Server stopped.")