import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server_fastapi as web_server


def test_status_endpoint():
    payload = asyncio.run(web_server.api_status())
    assert payload['status'] == 'ready'


def test_audio_upload():
    web_server.system_logs.clear()
    web_server.generated_charts.clear()

    content = b'RIFF....WAVEfmt '
    response = asyncio.run(web_server.api_upload_audio(filename='test_song.wav', content=content))

    assert response['success'] is True
    assert response['filename'] == 'test_song.wav'
    assert any(log['message'].startswith('Audio uploaded:') for log in web_server.system_logs)
