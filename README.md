# TaikoNationV1: AI-Powered Taiko Chart Generation

[![License: MIT](https://img.shields.://opensource.org/licenses/MITfor generating human-like Taiko no Tatsujin beatmaps from audio, with multi-difficulty support and pattern-aware intelligence.

## Overview

TaikoNationV1 is a modernized implementation of the groundbreaking research from [**"TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games"**](https://arxiv.org/abs/2107.12506) by Emily Halina and Matthew Guzdial (2021). This project extends the original work with state-of-the-art transformer architecture, multi-difficulty generation, and human-in-the-loop evaluation.

### What Makes This Special

Unlike traditional onset-detection approaches, TaikoNation focuses on **human-like patterning** - creating charts where note placement forms musically coherent patterns that feel natural to play. This implementation takes that core innovation and enhances it with modern AI techniques.

## Key Features

### 🎯 **Multi-Difficulty Generation**
- Generate charts for any difficulty: Easy, Normal, Hard, Oni, Ura Oni
- Difficulty-aware pattern memory learns appropriate patterns for each skill level
- Smooth difficulty progression for educational use

### 🧠 **Pattern-Aware Intelligence**
- **Sliding window loss** ensures local pattern coherence
- **Pattern memory attention** learns and reuses common rhythmic motifs
- **Multi-task learning** optimizes for both note accuracy and difficulty consistency

### 🔄 **Human-in-the-Loop Evaluation**
- Built-in web server for collecting human ratings
- Multi-criteria evaluation: Fun, Musicality, Playability, Pattern Coherence
- Data pipeline for reinforcement learning from human feedback (RLHF)

### 🚀 **Production Ready**
- CLI tool for batch generation
- Export to `.osu` and `.tja` formats
- RESTful API server for web integration
- Comprehensive configuration management

## Research Foundation

This work builds on the original TaikoNation paper's key insights:

> *"Patterning is a key identifier of high quality rhythm game content, seen as a necessary component in human rankings. We establish a new approach for chart generation that produces charts with more congruent, human-like patterning than seen in prior work."*

**Original Contributions (Halina & Guzdial, 2021):**
- Sliding window prediction for pattern continuity
- Curated high-difficulty training data
- Pattern overlap and pattern space coverage metrics

**Our Extensions (2025):**
- Modern transformer architecture with attention mechanisms
- Difficulty-conditioned generation with separate pattern memories
- Real-time human evaluation and preference learning
- Multi-task learning framework
- Production-ready deployment pipeline

## Architecture

```
Audio Features → Encoder → Pattern-Aware → Multi-Task → Chart Tokens
                          Transformer     Heads      
                               ↓
                     Difficulty-Specific
                     Pattern Memory Banks
```

### Model Hierarchy
- **TaikoTransformer**: Base encoder-decoder with positional encoding
- **PatternAwareTransformer**: Adds learned pattern memory with attention
- **MultiTaskTaikoTransformer**: Adds difficulty conditioning and multi-task heads

## Installation

```bash
git clone https://github.com/myrqyry/TaikoNationV1.git
cd TaikoNationV1
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, librosa
- Flask (for evaluation server)
- ffmpeg (for audio processing)

## Quick Start

### 1. Training a Model
```bash
python train_transformer.py --config config/default.yaml
```

### 2. Generating Charts
```bash
python generate_chart.py model.pth input_songs/song.npy output_chart.osu --difficulty oni
```

### 3. Human Evaluation
```bash
python tools/human_eval/server.py
# Open http://localhost:5000 to rate generated charts
```

### 4. API Server
```bash
python server.py
# POST to /generate with audio files
```

## Configuration

All hyperparameters are managed through YAML configuration files:

```yaml
model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6

training:
  batch_size: 16
  learning_rate: 5e-5
  multi_task:
    difficulty_loss_weight: 0.1
    pattern_loss_weight: 0.2

data:
  max_sequence_length: 2048
  time_quantization_ms: 100
```

## Data Format

### Input Audio Features
- Pre-computed spectral features (`.npy` files)
- Frame rate: ~43Hz (23.2ms resolution)
- Feature size: 80 dimensions (mel-scale spectrogram)

### Chart Format
- 7-dimensional vectors: `[don, ka, big_don, big_ka, roll_start, roll_end, finisher]`
- Boolean encoding for simultaneous events
- Tokenized using BEaRT-style discrete vocabulary

## Evaluation Metrics

### Quantitative (Automated)
- **Pattern Overlap**: Percentage of human patterns used
- **Pattern Space Coverage**: Variety of unique patterns
- **Note Type Distribution**: Statistical similarity to human charts
- **Onset Detection**: F1 score for note timing accuracy

### Qualitative (Human)
- **Fun Rating**: Subjective enjoyment (1-10)
- **Musicality**: How well notes match the music (1-10)
- **Playability**: Physical comfort and flow (1-10)
- **Pattern Coherence**: Logical pattern structure (1-10)
- **Difficulty Accuracy**: Appropriate for target skill level (1-10)

## Research Applications

This codebase enables several research directions:

### Implemented
- **Multi-difficulty conditioning** for skill-appropriate generation
- **Pattern-aware attention** for musical coherence
- **Human preference learning** through evaluation pipeline

### Future Directions
- **Reinforcement Learning from Human Feedback (RLHF)**
- **Style transfer** between mappers
- **Cross-game adaptation** (DDR, Guitar Hero, etc.)
- **Real-time generation** for live performances

## Contributing

We welcome contributions in several areas:

- **Model Architecture**: New attention mechanisms, loss functions
- **Data Pipeline**: Audio processing, augmentation techniques  
- **Evaluation**: New metrics, visualization tools
- **Applications**: Web interfaces, game integrations

## Citation

If you use this work in your research, please cite both the original paper and this implementation:

```bibtex
@inproceedings{halina2021taikonation,
  title={TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games},
  author={Halina, Emily and Guzdial, Matthew},
  booktitle={Proceedings of the Twelfth Workshop on Procedural Content Generation},
  year={2021}
}

@misc{taikonationv1_2025,
  title={TaikoNationV1: Modern Implementation with Multi-Difficulty and Pattern-Aware Generation},
  author={[Your Name]},
  year={2025},
  url={https://github.com/myrqyry/TaikoNationV1}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Emily Halina & Matthew Guzdial** for the original TaikoNation research and codebase
- **The osu!taiko community** for inspiration and feedback
- **PyTorch and Hugging Face** teams for excellent ML frameworks

***

*"The rhythm is just a click away."* 🥁
