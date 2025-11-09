import json
from pathlib import Path
from typing import Dict, Any
import os
import tempfile
import contextlib

def error_response(message: str, code: str = "UNKNOWN", details: Any = None) -> Dict[str, Any]:
    """Create a standardized error response dictionary."""
    return {
        "success": False,
        "error": message,
        "code": code,
        "details": details or {},
    }

@contextlib.contextmanager
def atomic_write(path, mode="w"):
    """Safely write to a file by using a temporary file."""
    temp_dir = os.path.dirname(path)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)
    with os.fdopen(temp_fd, mode) as temp_file:
        yield temp_file
    os.rename(temp_path, path)