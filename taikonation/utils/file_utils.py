"""Helper functions for TaikoNation."""
import os
import tempfile
import contextlib

@contextlib.contextmanager
def atomic_write(path, mode="w"):
    """Safely write to a file by using a temporary file."""
    temp_dir = os.path.dirname(path)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)
    try:
        with os.fdopen(temp_fd, mode) as temp_file:
            yield temp_file
        os.rename(temp_path, path)
    except Exception:
        os.remove(temp_path)
        raise
