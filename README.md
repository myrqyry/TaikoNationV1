# 🥁 TaikoNation: Your Personal AI Taiko Chart Creator! 🥁

Welcome to TaikoNation, the place where music meets AI to create awesome Taiko no Tatsujin charts! 🎶

Have you ever wanted to play a Taiko chart for your favorite song, but couldn't find one? Or maybe you're a seasoned charter looking for some AI-powered inspiration? Well, you've come to the right place!

TaikoNation is a deep learning-based system that can generate a Taiko chart from any song. And now, thanks to the power of Gemini, you can even guide the AI with your own creative prompts! 🚀

## ✨ Features

*   **🤖 AI-Powered Chart Generation:** TaikoNation uses a deep learning model to automatically generate Taiko charts from your favorite songs.
*   **✍️ Gemini-Powered Prompting:** With the new Gemini integration, you can guide the AI's creativity with your own text prompts. Want a chart with more katsus? Or a super-fast, high-energy chart? Just tell the AI what you want!
*   **🎤 Audio Feature Extraction:** The system automatically extracts audio features from your songs to feed into the model.
*   **📦 `.osz` Package Creation:** TaikoNation automatically packages the generated chart and your song into a `.osz` file, ready to be imported into osu!

## 🚀 Getting Started

Ready to create your own AI-powered Taiko charts? Here's how to get started:

### 1. Clone the Repository

First, you'll need to clone this repository to your local machine. You can do this using the following command:

```bash
git clone https://github.com/your-username/TaikoNation.git
```

### 2. Install Dependencies

Next, you'll need to install the necessary dependencies. TaikoNation uses a few Python libraries to work its magic. You can install them using pip:

```bash
pip install -r requirements.txt
```

**Note:** If you don't have a `requirements.txt` file, you can install the dependencies manually:

```bash
pip install google-generativeai numpy essentia pydub
```

You'll also need to install ffmpeg, which is a dependency for pydub. You can find installation instructions for your operating system on the [ffmpeg website](https://ffmpeg.org/download.html).

### 3. Set Up Your Gemini API Key

To use the Gemini-powered prompting feature, you'll need to get a Gemini API key. You can get one from the [Google AI Studio](https://aistudio.google.com/).

Once you have your API key, you'll need to set it as an environment variable named `GEMINI_API_KEY`. You can do this by adding the following line to your `.bashrc` or `.zshrc` file:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Replace `"YOUR_API_KEY"` with your actual API key.

## 🎶 Usage

Once you've set up the project, you're ready to start generating charts! Here's how to do it:

1.  **Place your song in the `input_songs` directory.** Make sure your song is in `.mp3` format.

2.  **Run the `gemini_output.py` script.** Open a terminal in the root directory of the project and run the following command:

    ```bash
    python output/gemini_output.py "input_songs/your_song.mp3" "Your creative prompt"
    ```

    Replace `"input_songs/your_song.mp3"` with the path to your song and `"Your creative prompt"` with your desired prompt for the AI.

3.  **Find your chart in the `output` directory.** Once the script has finished running, you'll find a new `.osz` file in the `output` directory. This file contains your generated chart and your song, ready to be imported into osu!

    **Note:** The first time you run the script for a new song, it will take some time to process the audio and extract the features. Subsequent runs for the same song will be much faster.

## 🙌 Contributing

We love contributions! If you'd like to contribute to TaikoNation, please feel free to fork the repository and submit a pull request. Here are some ideas for how you can contribute:

*   **Improve the model:** The current model is a good starting point, but it can always be improved. If you have experience with deep learning and music information retrieval, we'd love to see what you can do!
*   **Add new features:** Have an idea for a new feature? We'd love to hear it!
*   **Improve the documentation:** Good documentation is essential for any project. If you see something that could be clearer or more detailed, please let us know.
*   **Report bugs:** If you find a bug, please open an issue on GitHub.

We're excited to see what you come up with!

## 📜 License

TaikoNation is released under the [MIT License](LICENSE).
