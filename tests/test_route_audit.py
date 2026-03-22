import subprocess
import sys
from pathlib import Path

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
    assert "migration is complete" in content