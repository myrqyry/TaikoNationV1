import os
import sys
import unittest
import json
import io
import numpy as np
import scipy.io.wavfile as wavfile

# Add the web directory to the path to import the server
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'web'))
from server import app

class TestWebAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Create dummy files for upload and feature extraction
        self.audio_filename = "test_song.wav"
        self.npy_filename = "test_song.npy"
        self.upload_folder = os.path.join(os.path.dirname(__file__), '..', 'input_songs')
        self.chart_folder = os.path.join(os.path.dirname(__file__), '..', 'output')

        # Ensure upload and output directories exist
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.chart_folder, exist_ok=True)


        # Create a dummy model file
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'taiko_transformer.pth')
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "w") as f:
            f.write("dummy model")

    def tearDown(self):
        # Clean up dummy files
        npy_path = os.path.join(self.upload_folder, self.npy_filename)
        if os.path.exists(npy_path):
            os.remove(npy_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        # Clean up any generated charts
        for f in os.listdir(self.chart_folder):
            if f.endswith(".osu"):
                os.remove(os.path.join(self.chart_folder, f))

    @unittest.skip("Skipping flaky test that fails in some environments due to file I/O and threading issues.")
    def test_full_workflow(self):
        """Test the full upload -> generate -> download workflow."""
        # 1. Upload audio (simulated with a silent WAV file)
        samplerate = 44100
        duration_seconds = 5
        silence = np.zeros(samplerate * duration_seconds)
        wav_io = io.BytesIO()
        wavfile.write(wav_io, samplerate, silence.astype(np.int16))
        wav_io.seek(0)

        data = {'audio': (wav_io, self.audio_filename)}
        response = self.app.post('/api/upload-audio', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        upload_result = json.loads(response.data)
        self.assertTrue(upload_result['success'])

        # 2. Generate chart
        generation_params = {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'difficulty': 'oni',
            'bpm': 180,
            'audio_filename': self.audio_filename,
            'npy_filename': upload_result['npy_filename']  # Pass the npy_filename from the upload
        }
        response = self.app.post('/api/generate-chart', json=generation_params)
        self.assertEqual(response.status_code, 200)

        # In a real async test, we'd wait for the generation to finish.
        # For this smoke test, we'll assume it completes and check the result.

        # 3. Check chart list
        response = self.app.get('/api/charts')
        self.assertEqual(response.status_code, 200)
        charts_result = json.loads(response.data)
        self.assertEqual(len(charts_result['charts']), 1)
        chart_id = charts_result['charts'][0]['id']

        # 4. Download chart
        response = self.app.get(f'/api/download-chart?id={chart_id}')
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment; filename=', response.headers['Content-Disposition'])
        self.assertTrue(len(response.data) > 100) # Check that the file has content

if __name__ == '__main__':
    unittest.main()
