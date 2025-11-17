import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Callable

# Initialize logger for tasks module
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Global task registry
TASKS_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Store socketio reference as module variable
_socketio = None

def set_socketio(socketio_instance):
    """Set SocketIO instance for emitting events"""
    global _socketio
    _socketio = socketio_instance

def create_task(task_type: str, *args, **kwargs) -> str:
    """Create a new task and return its ID"""
    task_id = str(uuid.uuid4())
    TASKS_REGISTRY[task_id] = {
        'id': task_id,
        'type': task_type,
        'status': TaskStatus.PENDING.value,
        'progress': 0,
        'message': 'Task queued',
        'result': None,
        'error': None,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'args': args,
        'kwargs': kwargs
    }
    return task_id

def update_task_status(task_id: str, status: TaskStatus, progress: int = None,
                        message: str = None, result: Any = None, error: str = None):
    """Update task status and emit to clients"""
    if task_id not in TASKS_REGISTRY:
        return

    task = TASKS_REGISTRY[task_id]
    task['status'] = status.value
    task['updated_at'] = datetime.now().isoformat()

    if progress is not None:
        task['progress'] = progress
    if message is not None:
        task['message'] = message
    if result is not None:
        task['result'] = result
    if error is not None:
        task['error'] = error

    # Emit update via WebSocket if available
    if _socketio is not None:
        _socketio.emit('task_update', task)

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get current task status"""
    return TASKS_REGISTRY.get(task_id)

def run_task(task_id: str, registry: Dict):
    """Execute a task based on its type"""
    task = registry.get(task_id)
    if not task:
        return

    task_type = task['type']
    args = task['args']
    kwargs = task['kwargs']

    try:
        update_task_status(task_id, TaskStatus.RUNNING, progress=0,
                           message=f'Starting {task_type}')

        if task_type == 'process_uploaded_audio':
            result = process_audio_with_progress(task_id, *args, **kwargs)
        elif task_type == 'start_chart_generation':
            from web.server import start_chart_generation
            result = start_chart_generation(*args, **kwargs)
        elif task_type == 'start_training_task':
            result = train_model_with_progress(task_id, *args, **kwargs)
        else:
            raise ValueError(f'Unknown task type: {task_type}')

        update_task_status(task_id, TaskStatus.COMPLETED, progress=100,
                          message='Task completed successfully', result=result)

    except Exception as e:
        logger.error(f'Task {task_id} failed: {e}', exc_info=True)
        update_task_status(task_id, TaskStatus.FAILED,
                           message='Task failed', error=str(e))

def process_audio_with_progress(task_id: str, audio_path: str, config: Dict):
    """Process audio file with progress updates"""
    import librosa
    import numpy as np
    import os

    # Safe config access with defaults
    def get_config_value(config, keys, default):
        """Safely get nested config value"""
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    n_mels = get_config_value(config, ['model', 'audio_feature_size'], 80)
    hop_length = get_config_value(config, ['audio', 'hop_length'], 512)

    update_task_status(task_id, TaskStatus.RUNNING, progress=10,
                      message='Loading audio file...')

    try:
        # Validate audio path
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract features with progress
        update_task_status(task_id, TaskStatus.RUNNING, progress=30,
                          message='Extracting audio features...')

        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

        if len(y) == 0:
            raise ValueError("Audio file is empty or corrupted")

        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=2048
        )

        # Convert to log scale for better feature representation
        features = np.log1p(mel_spectrogram).T

        update_task_status(task_id, TaskStatus.RUNNING, progress=70,
                          message='Saving feature file...')

        # Save features with validation
        npy_path = os.path.splitext(audio_path)[0] + '.npy'
        np.save(npy_path, features)

        # Verify saved file
        if not os.path.exists(npy_path):
            raise IOError("Failed to save feature file")

        update_task_status(task_id, TaskStatus.RUNNING, progress=90,
                          message='Finalizing...')

        # Calculate audio metrics
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        result = {
            'audio_filename': os.path.basename(audio_path),
            'npy_filename': os.path.basename(npy_path),
            'feature_shape': list(features.shape),
            'duration_seconds': float(duration),
            'sample_rate': int(sr),
            'estimated_tempo': float(tempo)
        }

        logger.info(f"Successfully processed audio: {result}")
        return result

    except Exception as e:
        logger.error(f'Audio processing failed for {audio_path}: {e}', exc_info=True)
        raise

def train_model_with_progress(task_id: str, training_params: Dict):
    """Train model with progress updates"""
    import torch
    import os
    from pathlib import Path

    update_task_status(task_id, TaskStatus.RUNNING, progress=5,
                      message='Initializing training environment...')

    try:
        # Get paths relative to repository root
        BASE_DIR = Path(__file__).parent.parent
        MODEL_FOLDER = BASE_DIR / 'model'
        CONFIG_FOLDER = BASE_DIR / 'config'

        MODEL_FOLDER.mkdir(exist_ok=True)

        update_task_status(task_id, TaskStatus.RUNNING, progress=10,
                          message='Loading configuration and data...')

        # Import training utilities (lazy to avoid import-time issues)
        try:
            from taikonation.training.trainer import train_transformer_model
            from taikonation.data.dataset import get_transformer_data_loaders
        except ImportError as e:
            logger.error(f"Failed to import training modules: {e}")
            raise ImportError(f"Training modules not available: {e}")

        # Load configuration
        config_path = CONFIG_FOLDER / 'default.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update config with user parameters
        config['training'].update(training_params)

        update_task_status(task_id, TaskStatus.RUNNING, progress=20,
                          message='Loading training dataset...')

        # Get data loaders
        train_loader, val_loader = get_transformer_data_loaders(
            batch_size=config['training']['batch_size'],
            num_workers=2
        )

        # Progress callback for training
        def progress_callback(epoch, total_epochs, metrics):
            progress = 20 + int((epoch / total_epochs) * 70)
            update_task_status(
                task_id, TaskStatus.RUNNING, progress=progress,
                message=f'Training epoch {epoch}/{total_epochs} - Loss: {metrics.get("loss", 0):.4f}'
            )

        update_task_status(task_id, TaskStatus.RUNNING, progress=25,
                          message='Starting model training...')

        # Train the model
        model_path = train_transformer_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            progress_callback=progress_callback
        )

        update_task_status(task_id, TaskStatus.RUNNING, progress=95,
                          message='Finalizing and saving model...')

        result = {
            'model_path': str(model_path),
            'config': config,
            'training_completed': True
        }

        logger.info(f"Training completed successfully. Model saved to: {model_path}")
        return result

    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        raise