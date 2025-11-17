"""Helper functions for TaikoNation web server"""
import logging
import os
import tempfile
import contextlib
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def error_response(message: str, code: str = 'ERROR', details: Optional[Any] = None) -> Dict:
    """
    Create standardized error response dictionary

    Args:
        message: Human-readable error message
        code: Machine-readable error code
        details: Optional additional error details

    Returns:
        Dictionary with error information
    """
    response = {
        'success': False,
        'error': message,
        'code': code
    }

    if details is not None:
        response['details'] = details

    return response

def success_response(data: Any = None, message: Optional[str] = None) -> Dict:
    """
    Create standardized success response dictionary

    Args:
        data: Response data payload
        message: Optional success message

    Returns:
        Dictionary with success information
    """
    response = {
        'success': True
    }

    if data is not None:
        response['data'] = data

    if message is not None:
        response['message'] = message

    return response

@contextlib.contextmanager
def atomic_write(path, mode="w"):
    """Safely write to a file by using a temporary file."""
    temp_dir = os.path.dirname(path)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)
    with os.fdopen(temp_fd, mode) as temp_file:
        yield temp_file
    os.rename(temp_path, path)