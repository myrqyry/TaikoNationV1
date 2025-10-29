# web/tasks.py
import os
import json
import uuid
import time
import traceback
from pathlib import Path
from enum import Enum

TASK_DIR = Path(__file__).parent / "tasks_data"
TASK_DIR.mkdir(exist_ok=True)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"

def create_task(target_fn_name, *args, **kwargs):
    """Creates a new task and saves it to the file system."""
    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "status": TaskStatus.PENDING,
        "target_fn": target_fn_name,
        "args": args,
        "kwargs": kwargs,
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    task_file = TASK_DIR / f"{task_id}.json"
    with open(task_file, "w") as f:
        json.dump(task_data, f, indent=2)
    return task_id

def get_task_status(task_id):
    """Gets the status of a task."""
    task_file = TASK_DIR / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file, "r") as f:
        return json.load(f)

def update_task_status(task_id, status, result=None, error=None):
    """Updates the status of a task."""
    task_file = TASK_DIR / f"{task_id}.json"
    if not task_file.exists():
        return

    with open(task_file, "r+") as f:
        task_data = json.load(f)
        task_data["status"] = status
        if result is not None:
            task_data["result"] = result
        if error is not None:
            task_data["error"] = error
        task_data["updated_at"] = time.time()
        f.seek(0)
        json.dump(task_data, f, indent=2)
        f.truncate()

import importlib

def run_task(task_id, tasks_registry):
    """Runs a task and updates its status."""
    task_data = get_task_status(task_id)
    if not task_data or task_data["status"] != TaskStatus.PENDING:
        return

    update_task_status(task_id, TaskStatus.RUNNING)

    try:
        target_fn_name = task_data["target_fn"]
        if target_fn_name not in tasks_registry:
            raise ValueError(f"Task function '{target_fn_name}' not found in registry.")

        module_path, func_name = tasks_registry[target_fn_name].rsplit('.', 1)
        module = importlib.import_module(module_path)
        target_fn = getattr(module, func_name)

        args = task_data["args"]
        kwargs = task_data["kwargs"]

        result = target_fn(*args, **kwargs)
        update_task_status(task_id, TaskStatus.SUCCESS, result=result)
    except Exception as e:
        error_info = {
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        update_task_status(task_id, TaskStatus.FAILURE, error=error_info)

def start_training_task(params):
    """A placeholder for the training task."""
    print(f"Starting training with params: {params}")
    time.sleep(10)  # Simulate a long-running task
    return {"training_complete": True, "model_path": "model.pth"}

# A registry to map function names to actual functions
TASKS_REGISTRY = {
    "start_chart_generation": "web.server.start_chart_generation",
    "process_uploaded_audio": "web.helpers.process_uploaded_audio",
    "start_training_task": "web.tasks.start_training_task",
}
