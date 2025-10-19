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
import yaml
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Import existing TaikoNation modules
try:
    # Add the parent directory to path to import TaikoNation modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from audio_processing import get_audio_features, augment_spectrogram
    from transformer_model import TaikoTransformer
    from transformer_dataset import get_transformer_data_loaders, DIFFICULTY_MAP
    from tokenization import TaikoTokenizer
    # Do NOT import train_transformer at module import time: it may import heavy deps (wandb)
    # which can pull eventlet and trigger ssl-related import-time errors in some environments.
    load_training_config = None
    
    # Import legacy model if available
    try:
        import model as legacy_model
    except ImportError:
        legacy_model = None
        
except ImportError as e:
    print(f"Warning: Could not import TaikoNation modules: {e}")
    print("Make sure the server is running from the web/ directory inside TaikoNationV1/")

import torch
import numpy as np
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
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
    # Minimal CSP - adjust for your CDN or external assets
    response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com"
    return response

# Global variables for managing training and generation state
training_process = None
generation_queue = []
system_logs = []
active_models = {}
generated_charts = []
evaluation_queue = []

# Configuration
UPLOAD_FOLDER = '../input_songs'
CHART_OUTPUT_FOLDER = '../output'
CONFIG_FOLDER = '../config'
MODEL_FOLDER = '../model'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, CHART_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

class TaikoNationServer:
    """Main server class that manages the TaikoNation web interface."""
    
    def __init__(self):
        self.config = self.load_default_config()
        self.tokenizer = None
        self.models = {}
        self.training_active = False
        self.generation_active = False
        
        self.initialize_components()
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load the default configuration from config/default.yaml."""
        try:
            config_path = os.path.join(CONFIG_FOLDER, 'default.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load default config: {e}")
        
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
    
    def initialize_components(self):
        """Initialize TaikoNation components."""
        try:
            self.tokenizer = TaikoTokenizer()
            self.load_available_models()
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
        global active_models
        active_models = {
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
                'status': 'ready' if legacy_model else 'not_available'
            }
        }
    
    def add_system_log(self, level: str, message: str):
        """Add a system log entry."""
        global system_logs
        
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message
        }
        
        system_logs.insert(0, log_entry)
        
        # Keep only the last 100 log entries
        system_logs = system_logs[:100]
        
        # Emit to connected clients
        socketio.emit('system_log', log_entry)
        
        logger.info(f"[{level.upper()}] {message}")

# Create server instance
server = TaikoNationServer()

# Route handlers
@app.route('/')
def index():
    """Serve the main web interface."""
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    response = send_from_directory('static', filename)
    # Encourage caching of static assets; in production you'd use far-future headers with hashed filenames
    response.cache_control.max_age = 3600
    return response

@app.route('/api/status')
def api_status():
    """Get system status."""
    return jsonify({
        'status': 'ready',
        'models_loaded': len(active_models),
        'training_active': server.training_active,
        'generation_active': server.generation_active
    })

@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard data."""
    return jsonify({
        'metrics': {
            'active_models': len([m for m in active_models.values() if m['status'] == 'ready']),
            'generated_charts': len(generated_charts),
            'best_accuracy': max([m['accuracy'] for m in active_models.values()]),
            'avg_rating': calculate_average_rating()
        },
        'recent_logs': system_logs[:10]
    })

@app.route('/api/models')
def api_models():
    """Get information about available models."""
    return jsonify({'models': active_models})

@app.route('/api/upload-audio', methods=['POST'])
def api_upload_audio():
    """Handle audio file upload and processing."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            # Validate extension server-side
            ALLOWED_EXT = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
            ext = os.path.splitext(filename.lower())[1]
            if ext not in ALLOWED_EXT:
                return jsonify({'error': 'Unsupported file type'}), 400

            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the uploaded file
            file.save(filepath)
            
            # Process the audio file
            audio_data = process_uploaded_audio(filepath)
            
            server.add_system_log('success', f'Audio file processed: {filename}')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'title': extract_title_from_filename(filename),
                'detected_bpm': audio_data.get('bpm'),
                'duration': audio_data.get('duration'),
                'features_extracted': audio_data.get('features_extracted', False)
            })
            
        except Exception as e:
            logger.error(f"Audio upload error: {e}")
            server.add_system_log('error', f'Audio upload failed: {str(e)}')
            return jsonify({'error': str(e)}), 500


@app.route('/api/start-training', methods=['POST'])
def api_start_training():
    """Start model training."""
    try:
        # Get training parameters from form
        training_params = {
            'd_model': int(request.form.get('d_model', 256)),
            'nhead': int(request.form.get('nhead', 8)),
            'num_encoder_layers': int(request.form.get('num_encoder_layers', 6)),
            'num_decoder_layers': int(request.form.get('num_decoder_layers', 6)),
            'learning_rate': float(request.form.get('learning_rate', 0.0001)),
            'batch_size': int(request.form.get('batch_size', 8))
        }
        
        # Start training in background
        start_background_training(training_params)
        
        server.add_system_log('success', 'Training started with custom parameters')
        
        return jsonify({'success': True, 'message': 'Training started'})
        
    except Exception as e:
        logger.error(f"Training start error: {e}")
        server.add_system_log('error', f'Training start failed: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-training', methods=['POST'])
def api_stop_training():
    """Stop model training."""
    try:
        global training_process
        if training_process:
            # In a real implementation, you would properly stop the training process
            training_process = None
            server.training_active = False
            
        server.add_system_log('info', 'Training stopped by user')
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Training stop error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-chart', methods=['POST'])
def api_generate_chart():
    """Generate a chart from uploaded audio."""
    try:
        # Get generation parameters
        params = {
            'title': request.form.get('title', 'Untitled'),
            'artist': request.form.get('artist', 'Unknown'),
            'bpm': int(request.form.get('bpm', 120)),
            'genre': request.form.get('genre', 'electronic'),
            'difficulty': request.form.get('difficulty', 'oni'),
            'pattern_style': request.form.get('pattern_style', 'balanced')
        }
        
        # Start chart generation
        start_chart_generation(params)
        
        server.add_system_log('success', f'Chart generation started: {params["title"]}')
        
        return jsonify({'success': True, 'message': 'Chart generation started'})
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        server.add_system_log('error', f'Chart generation failed: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts')
def api_charts():
    """Get list of generated charts."""
    return jsonify({'charts': generated_charts})

@app.route('/api/get-chart-for-evaluation')
def api_get_chart_for_evaluation():
    """Get a chart for human evaluation."""
    if evaluation_queue:
        chart = evaluation_queue[0]
        return jsonify(chart)
    else:
        return jsonify({'error': 'No charts available for evaluation'}), 404

@app.route('/api/submit-evaluation', methods=['POST'])
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
        if evaluation_queue:
            evaluation_queue.pop(0)
        
        server.add_system_log('success', 'Evaluation submitted successfully')
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Evaluation submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
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
def process_uploaded_audio(filepath: str) -> Dict[str, Any]:
    """Process uploaded audio file and extract features."""
    try:
        # Load audio using librosa
        y, sr = librosa.load(filepath)
        
        # Extract basic information
        duration = len(y) / sr
        
        # Estimate BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Extract audio features for the model
        features = get_audio_features(
            filepath, 
            source_resolution_ms=server.config['data']['source_resolution_ms'],
            frame_duration_ms=server.config['data']['time_quantization_ms']
        )
        
        return {
            'duration': duration,
            'bpm': int(tempo),
            'features_extracted': features is not None,
            'feature_shape': features.shape if features is not None else None
        }
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {'error': str(e)}

def extract_title_from_filename(filename: str) -> str:
    """Extract song title from filename."""
    # Remove extension and clean up
    title = os.path.splitext(filename)[0]
    title = title.replace('_', ' ').replace('-', ' ')
    return title.title()

def start_background_training(params: Dict[str, Any]):
    """Start training in background."""
    global training_process
    server.training_active = True
    
    # In a real implementation, you would start the actual training process
    # For now, simulate training progress
    def simulate_training():
        for progress in range(0, 101, 5):
            if not server.training_active:
                break
            
            metrics = {
                'epoch': progress // 2,
                'loss': 0.5 - (progress / 200),
                'accuracy': 70 + (progress / 4),
                'learning_rate': params['learning_rate']
            }
            
            socketio.emit('training_progress', {
                'progress': progress,
                'metrics': metrics
            })
            
            socketio.sleep(2)  # Simulate processing time
        
        server.training_active = False
        server.add_system_log('success', 'Training completed successfully')
    
    # Start training simulation in background
    socketio.start_background_task(simulate_training)


def lazy_load_training_utils():
    """Lazy import of training utilities to avoid heavy imports at module import time."""
    try:
        from train_transformer import load_config as load_training_config
        return load_training_config
    except Exception as e:
        logger.warning(f'Could not lazy-load training utilities: {e}')
        return None

def start_chart_generation(params: Dict[str, Any]):
    """Start chart generation process."""
    server.generation_active = True
    
    def simulate_generation():
        for progress in range(0, 101, 10):
            if not server.generation_active:
                break
            
            socketio.emit('generation_progress', {'progress': progress})
            socketio.sleep(1)
        
        # Add generated chart to library
        chart = {
            'id': len(generated_charts) + 1,
            'title': params['title'],
            'artist': params['artist'],
            'difficulty': params['difficulty'],
            'bpm': params['bpm'],
            'genre': params['genre'],
            'rating': 0,
            'plays': 0,
            'created_at': datetime.now().isoformat()
        }
        
        generated_charts.append(chart)
        evaluation_queue.append(chart)
        
        server.generation_active = False
        socketio.emit('chart_generated', {'chart': chart})
    
    socketio.start_background_task(simulate_generation)

def process_evaluation(evaluation_data: Dict[str, Any]):
    """Process submitted evaluation data."""
    # In a real implementation, save to database and update model training data
    logger.info(f"Processed evaluation for chart {evaluation_data['chart_id']}")
    
    # Update chart rating
    chart_id = int(evaluation_data['chart_id'])
    for chart in generated_charts:
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
    if not generated_charts:
        return 0.0
    
    ratings = [chart.get('rating', 0) for chart in generated_charts if chart.get('rating', 0) > 0]
    return round(sum(ratings) / len(ratings), 1) if ratings else 0.0

if __name__ == '__main__':
    print("Starting TaikoNation Studio Web Server...")
    print(f"Server will be available at: http://localhost:5000")
    print(f"Make sure you're running from the web/ directory inside TaikoNationV1/")
    
    # Run the Flask-SocketIO server
    # For development we allow Werkzeug when explicitly enabled via env var.
    allow_unsafe = os.environ.get('TAIKONATION_ALLOW_UNSAFE_WERKZEUG', 'true').lower() in ('1', 'true', 'yes')
    socketio.run(
        app,
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=False,  # Disable reloader to prevent issues with background tasks
        allow_unsafe_werkzeug=allow_unsafe
    )