import io
import os
import sys
import zipfile

import pytest

# Ensure project root is on sys.path so we can import the web package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server as web_server


@pytest.fixture
def client():
    web_server.app.config['TESTING'] = True
    with web_server.app.test_client() as client:
        yield client


def test_export_dataset_returns_zip(client):
    resp = client.get('/api/research/export-dataset')
    assert resp.status_code == 200
    assert resp.headers.get('Content-Type') in ('application/zip', 'application/octet-stream')

    data = resp.data
    # Ensure it starts with PK (zip magic)
    assert data[:2] == b'PK'

    # Try to open as zip and inspect dataset.json
    z = zipfile.ZipFile(io.BytesIO(data))
    assert 'dataset.json' in z.namelist()
    content = z.read('dataset.json')
    assert b'generated_charts' in content
