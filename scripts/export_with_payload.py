#!/usr/bin/env python3
"""Populate the FastAPI server module with a payload and call export endpoint."""

import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server_fastapi as web_server


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
        "chart_data": [1, 0, 2, 0, 1, 2, 1, 0] * 400,
    }],
    "human_evaluations": [],
    "model_configurations": [{
        "title": "Pixie Remix 4",
        "artist": "",
        "bpm": 200,
        "genre": "electronic",
        "difficulty": "muzukashii",
        "pattern_style": "balanced",
    }],
    "audio_features": [{"chart_id": 1, "feature_shape": [1000, 80]}],
}


def inject_payload(p):
    web_server.generated_charts[:] = p.get('generated_charts', [])
    web_server.human_evaluations[:] = p.get('human_evaluations', [])
    web_server.model_configurations[:] = p.get('model_configurations', [])
    web_server.audio_features[:] = p.get('audio_features', [])


def call_export_and_save(path='/tmp/research_custom.zip'):
    with TestClient(web_server.app) as client:
        resp = client.get('/api/research/export-dataset')
        if resp.status_code != 200:
            print('Export failed:', resp.status_code, resp.text)
            return 1
        with open(path, 'wb') as f:
            f.write(resp.content)
        print('Saved export to', path, 'size=', os.path.getsize(path))
        return 0


if __name__ == '__main__':
    inject_payload(payload)
    sys.exit(call_export_and_save())
