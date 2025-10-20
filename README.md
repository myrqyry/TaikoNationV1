# TaikoNationV1 (myrqyry fork)

_A cutting-edge AI system for human-like Taiko no Tatsujin beatmap generation — now with a full-featured web interface, modern transformer architecture, and integrated human evaluation._

## Overview

TaikoNationV1 is the modern successor to the influential research of ["TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games"](https://arxiv.org/abs/2107.12506) by Emily Halina and Matthew Guzdial. This project extends the original system with advanced transformer models, pattern-aware memory, multi-difficulty control, and a production-ready web/UI layer for both experimentation and real-world use.

***

## Key Features

- **Multi-difficulty chart generation:** Fully supports Easy, Normal, Hard, Oni, and Ura Oni levels
- **Pattern-aware intelligence:** Pattern memory and sliding window loss produce musically natural, human-like chart patterns
- **Modern deep learning architecture:** Transformer with attention, positional encoding, difficulty conditioning, and multi-task heads
- **Integrated web interface:** Accessible dashboard, drag-and-drop audio/chart upload, real-time progress display, and human-in-the-loop chart feedback system
- **Data pipeline for RLHF:** Collect and use human preferences to further improve models
- **Export formats:** Generate .osu and .tja files, compatible with osu!taiko and simulators
- **RESTful API & batch CLI:** For seamless automation and platform integration

***

## Live Demo & Gallery

- Try the web interface locally: Open http://localhost:5000 after setup
- Evaluate and rate charts in-browser, and see pattern visualizations (see screenshots and videos in `/web/README`)

***

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, librosa, Flask
- (Optional: ffmpeg, TensorFlow, CUDA for max speed)

### Project Setup

```bash
git clone https://github.com/myrqyry/TaikoNationV1.git
cd TaikoNationV1
pip install -r requirements.txt
# Optionally: pip install -r web/requirements.txt  # for web interface
```

***

## Usage

### 1. Web Interface

```bash
cd web
python server.py
# Visit http://localhost:5000 to access the dashboard
```

- Upload audio for chart generation, set difficulty/style, view progress, and export charts
- Train models and monitor status live in the browser
- Rate and comment on generated charts to provide feedback for RLHF

### 2. CLI

- **Train a model:**
  ```bash
  python train_transformer.py --config config/default.yaml
  ```
- **Generate a chart:**
  ```bash
  python generate_chart.py model.pth input_songs/song.npy output_chart.osu --difficulty oni
  ```
- **Batch processing:** See CLI docs and scripts.

***

## Configuration

- All hyperparameters are in YAML files (see `config/`). Example:
  ```yaml
  model:
    d_model: 256
    nhead: 8
    num_encoder_layers: 6
    num_decoder_layers: 6
  ```
- Change batch size, pattern memory, or audio settings as needed.

***

## Data Format

- **Input:** Numpy spectrograms (80 bands, 23.2ms frames)
- **Chart tokens:** `[don, ka, big_don, big_ka, roll_start, roll_end, finisher]` (boolean vectors)
- **Configurable tokenization & quantization for research

***

## Evaluation & Research

- **Automated:** Pattern overlap, space coverage, note type distribution, F1 timing score
- **Human:** Fun, musicality, playability, pattern logic, difficulty accuracy (all 1–10 scale)
- **RLHF pipeline:** Human ratings are saved for reward learning

***

## Contributions

We welcome PRs for:
- Model architectures & optimizations
- Audio processing, pattern analysis, and visualization enhancements
- Web/dashboard UX and new export formats
- Data augmentation and chart importers
- RLHF integration and evaluation analysis

***

## Cite Us

If you use our system, please cite:

```bibtex
@inproceedings{halina2021taikonation,
  title={TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games},
  author={Halina, Emily and Guzdial, Matthew},
  booktitle={Proceedings of the Twelfth Workshop on Procedural Content Generation},
  year={2021}
}
@misc{taikonationv1_2025,
  title={TaikoNationV1: Modern Implementation with Multi-Difficulty and Pattern-Aware Generation},
  author={myrqyry},
  year={2025},
  url={https://github.com/myrqyry/TaikoNationV1}
}
```

***

## Acknowledgments

Thanks to Emily Halina, Matthew Guzdial, the osu!taiko community, and all contributors. Powered by PyTorch, Hugging Face, and open-source rhythm game fans worldwide.

***

*"The rhythm is just a click away."* 🥁

***
