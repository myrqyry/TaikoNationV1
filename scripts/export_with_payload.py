#!/usr/bin/env python3
"""Populate the running server module with a payload and call the export endpoint.

This script imports the `web.server` module, injects the supplied dataset into the
server state, then uses Flask's test client to request `/api/research/export-dataset`
and saves the resulting zip to /tmp/research_custom.zip.

Usage: run from repo root with the project venv python.
"""
import os
import sys
import json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server as web_server


payload = {
    "generated_charts": [{
        "id": 1,
        "title": "Pixie Remix 4",
        "artist": "",
        "difficulty": "muzukashii",
        "bpm": 200,
        "genre": "electronic",
        "rating": 0,
        "plays": 0,
        "created_at": "2025-10-19T21:52:11.066556",
        "chart_data": [1,0,2,0,1,2,1,0] * 400
    }],
    "human_evaluations": [],
    "model_configurations": [{
        "title": "Pixie Remix 4",
        "artist": "",
        "bpm": 200,
        "genre": "electronic",
        "difficulty": "muzukashii",
        "pattern_style": "balanced"
    }],
    "audio_features": [{"chart_id": 1, "feature_shape": [1000, 80]}]
}


def inject_payload(p):
    # Replace module-level globals used by the export endpoint
    web_server.generated_charts.clear()
    web_server.generated_charts.extend(p.get('generated_charts', []))

    # Populate experiment tracker so get_model_configs() returns sensible data
    web_server.server.experiment_tracker.experiments = {
        'run_1': {'config': p.get('model_configurations', [])[0] if p.get('model_configurations') else {}}
    }

    # HRLF feedback queue
    web_server.server.hrlf_collector.feedbackQueue = p.get('human_evaluations', [])


def call_export_and_save(path='/tmp/research_custom.zip'):
    with web_server.app.test_client() as client:
        resp = client.get('/api/research/export-dataset')
        if resp.status_code != 200:
            print('Export failed:', resp.status_code, resp.get_data(as_text=True))
            return 1
        with open(path, 'wb') as f:
            f.write(resp.data)
        print('Saved export to', path, 'size=', os.path.getsize(path))
        return 0


if __name__ == '__main__':
    inject_payload(payload)
    sys.exit(call_export_and_save())
