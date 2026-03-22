import asyncio
import os
import sys
import io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server_fastapi as web_server
from fastapi import UploadFile


def test_status_endpoint():
    payload = asyncio.run(web_server.api_status())
    assert payload['status'] == 'ready'


def test_audio_upload():
    web_server.system_logs.clear()
    web_server.generated_charts.clear()

    content = b'RIFF....WAVEfmt '
    response = asyncio.run(web_server.api_upload_audio(filename='test_song.wav', content=content))

    assert response.success is True
    assert response.filename == 'test_song.wav'
    assert any(log['message'].startswith('Audio uploaded:') for log in web_server.system_logs)


def test_cancel_task_endpoint():
    task = web_server.store.create_task(
        task_type='generation',
        payload={'title': 'cancel me'},
        created_at='2026-03-21T00:00:00'
    )
    response = asyncio.run(web_server.api_cancel_task(task['id']))
    assert response['success'] is True
    task_after = web_server.store.get_task(task['id'])
    assert task_after is not None
    assert task_after['status'] == 'cancelled'


def test_audio_upload_with_multipart_file():
    upload = UploadFile(filename='multipart.wav', file=io.BytesIO(b'RIFF....WAVEfmt '))
    response = asyncio.run(web_server.api_upload_audio(file=upload))
    assert response.success is True
    assert response.filename == 'multipart.wav'


def test_tasks_endpoint_pagination_shape():
    payload = asyncio.run(web_server.api_tasks(limit=5, offset=0))
    assert payload.limit == 5
    assert payload.offset == 0
    assert isinstance(payload.total, int)
    assert isinstance(payload.tasks, list)


def test_charts_endpoint_pagination_shape():
    payload = asyncio.run(web_server.api_charts(limit=3, offset=0))
    assert payload.limit == 3
    assert payload.offset == 0
    assert isinstance(payload.total, int)
    assert isinstance(payload.charts, list)
