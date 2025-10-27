import os
import subprocess
import sys
import unittest
import numpy as np
import torch
from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer
from transformer_dataset import DIFFICULTY_MAP

class TestCli(unittest.TestCase):
    def setUp(self):
        self.sample_npy_path = "tests/sample_features.npy"
        self.output_osu_path = "tests/output_chart.osu"
        # Create a dummy .npy file
        np.save(self.sample_npy_path, np.random.rand(100, 80))
        # Create a dummy model file
        self.model_path = "tests/dummy_model.pth"
        config = {
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'audio_feature_size': 80,
                'max_sequence_length': 512
            }
        }
        model = TaikoTransformer(
            vocab_size=TaikoTokenizer().vocab_size,
            num_genres=10,
            num_difficulties=len(DIFFICULTY_MAP),
            **config['model']
        )
        torch.save({'model_state_dict': model.state_dict()}, self.model_path)

    def tearDown(self):
        if os.path.exists(self.sample_npy_path):
            os.remove(self.sample_npy_path)
        if os.path.exists(self.output_osu_path):
            os.remove(self.output_osu_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_generate_chart_cli(self):
        """Test that generate_chart.py produces a non-empty .osu file."""
        command = [
            sys.executable,
            "generate_chart.py",
            self.model_path,
            self.sample_npy_path,
            self.output_osu_path,
            "--difficulty", "oni",
            "--seed", "42"
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, f"CLI script failed with output:\\n{result.stderr}")

        self.assertTrue(os.path.exists(self.output_osu_path))

        with open(self.output_osu_path, "r") as f:
            content = f.read()
            self.assertIn("[HitObjects]", content)
            self.assertTrue(len(content.splitlines()) > 10) # Check for a reasonable number of lines

if __name__ == '__main__':
    unittest.main()
