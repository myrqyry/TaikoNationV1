from pathlib import Path
import subprocess
import sys


def test_route_audit_generates_status_file():
    result = subprocess.run(
        [sys.executable, "tools/route_audit.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    content = Path("docs/flask-migration-status.md").read_text()
    assert "Flask → FastAPI Migration Status" in content
    assert "Missing in FastAPI (from Flask)" in content
