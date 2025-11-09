import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Callable

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Global task registry
TASKS_REGISTRY: Dict[str, Dict[str, Any]] = {}

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

    # Emit update via WebSocket
    from web.server import socketio
    socketio.emit('task_update', task)

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
    from taikonation.data.audio_processing import get_audio_features
    import numpy as np
    import os
    from web.server import logger

    update_task_status(task_id, TaskStatus.RUNNING, progress=10,
                      message='Loading audio file...')

    try:
        # Extract features with progress
        update_task_status(task_id, TaskStatus.RUNNING, progress=30,
                          message='Extracting audio features...')

        features = get_audio_features(
            audio_path,
            n_mels=config['model']['audio_feature_size']
        )

        update_task_status(task_id, TaskStatus.RUNNING, progress=70,
                          message='Saving feature file...')

        # Save features
        npy_path = os.path.splitext(audio_path)[0] + '.npy'
        np.save(npy_path, features)

        update_task_status(task_id, TaskStatus.RUNNING, progress=90,
                          message='Finalizing...')

        result = {
            'audio_filename': os.path.basename(audio_path),
            'npy_filename': os.path.basename(npy_path),
            'feature_shape': features.shape,
            'duration_seconds': features.shape[0] * 0.0232  # 23.2ms per frame
        }

        return result

    except Exception as e:
        logger.error(f'Audio processing failed: {e}')
        raise
