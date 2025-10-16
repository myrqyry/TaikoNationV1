# 🥁 TaikoNationV1: AI-Powered Taiko Chart Generation 🥁

Welcome to the modernized TaikoNationV1, a sophisticated AI system for generating high-quality Taiko no Tatsujin (太鼓の達人) charts from any audio file. This project leverages a powerful Transformer-based architecture to understand musical patterns and create playable, engaging charts across multiple difficulty levels.

## ✨ Core Features

*   **🧠 Advanced Transformer Model:** At its core, TaikoNationV1 uses a `MultiTaskTaikoTransformer` built with PyTorch. This model learns the relationship between audio features and note patterns.
*   **🎵 Multi-Difficulty Generation:** Generate charts for a specific difficulty level, from `Easy` to `Oni`. The model uses a difficulty-aware pattern memory to produce stylistically appropriate patterns for each level.
*   **💿 Multiple Export Formats:** Export your generated charts to both `.osu` (for osu!) and `.tja` (for Taiko Jiro and other simulators) formats.
*   **🧑‍🔬 Human Evaluation Pipeline:** A built-in web server allows for human feedback on generated charts, creating a powerful loop for model improvement.
*   **🔬 Extensible and Modular:** The codebase is designed to be modular and easy to extend, making it a great platform for research and experimentation in AI and music.

## 🚀 Getting Started

### 1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone https://github.com/myrqyry/TaikoNationV1.git
cd TaikoNationV1
pip install -r requirements.txt
```

### 2. Usage

#### Generating a Chart

The primary way to generate a chart is through the `generate_chart.py` script.

**Arguments:**
-   `model_path`: Path to a trained model checkpoint (`.pth` file).
-   `audio_path`: Path to the input audio file (must be a pre-processed `.npy` file).
-   `output_path`: Path to save the generated chart file.
-   `--difficulty`: The target difficulty (e.g., `easy`, `normal`, `hard`, `oni`).
-   `--format`: The output format (`osu` or `tja`).

**Example:**
```bash
# Generate an "Oni" chart in .osu format
python generate_chart.py output/model.pth "input_songs/your_song.npy" "output/generated_charts/my_chart.osu" --difficulty oni --format osu

# Generate a "Hard" chart in .tja format
python generate_chart.py output/model.pth "input_songs/your_song.npy" "output/generated_charts/my_chart.tja" --difficulty hard --format tja
```

#### Running the Human Evaluation Server

To collect feedback on generated charts, you can use the human evaluation server.

1.  Make sure you have some generated charts in the `output/generated_charts` directory.
2.  Run the server:
    ```bash
    python tools/human_eval/server.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000`.

## 🔬 For Researchers and Developers

This project is designed to be a platform for research into AI-driven content creation. Here are some of the key components you can explore:

*   **`transformer_model.py`**: Contains the `MultiTaskTaikoTransformer` architecture, including the difficulty-aware pattern memory.
*   **`train_transformer.py`**: The main script for training the model.
*   **`tools/analyze_patterns.py`**: A script for analyzing n-gram frequencies in the dataset.
*   **`tools/human_eval/server.py`**: The human evaluation server.

## 🙌 Contributing

Contributions are welcome! Whether you're interested in improving the model, adding new features, or enhancing the documentation, feel free to fork the repository and submit a pull request.

## 📜 License

TaikoNationV1 is released under the [MIT License](LICENSE).