# TaikoNation Studio Web Interface

A comprehensive web interface for the TaikoNation AI taiko chart generation system.

## Features

- **Model Training**: Train both modern PyTorch Transformer and legacy TensorFlow CNN-LSTM models
- **Chart Generation**: Generate taiko charts with AI from uploaded audio files  
- **Human Evaluation**: Collect human feedback on generated charts for RLHF
- **Real-time Monitoring**: Live progress tracking and system logs
- **Configuration Management**: Easy parameter tuning and settings management

## File Structure

```
TaikoNationV1/
├── web/                          # 🆕 Add this folder
│   ├── server.py                # Flask backend server
│   ├── index.html               # Main interface
│   ├── requirements.txt         # Python dependencies
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # Dark theme CSS
│   │   └── js/
│   │       └── app.js          # JavaScript application
├── config/                      # Your existing config
├── model/                       # Your existing models
└── [other existing files]       # Keep as-is
```

## Installation

1. **Create the web directory** inside your TaikoNationV1 folder:
   ```bash
   cd TaikoNationV1
   mkdir -p web/static/css web/static/js
   ```

2. **Copy the generated files** into the correct locations:
   - `web-interface-index.html` → `web/index.html`
   - `web-interface-styles.css` → `web/static/css/styles.css`
   - `web-interface-app.js` → `web/static/js/app.js`
   - `web-interface-server.py` → `web/server.py`
   - `web-requirements.txt` → `web/requirements.txt`

3. **Install dependencies**:
   ```bash
   cd web
   pip install -r requirements.txt
   ```

## Usage

1. **Start the server**:
   ```bash
   cd web
   python server.py
   ```

2. **Access the interface**: Open http://localhost:5000 in your browser

3. **Upload audio**: Drag and drop MP3/WAV/OGG/FLAC files to generate charts

4. **Configure training**: Set hyperparameters for transformer or legacy models

5. **Monitor progress**: View real-time training and generation progress

## Integration with TaikoNation

This interface directly integrates with your existing TaikoNation modules:

- `audio_processing.py` - For feature extraction
- `transformer_model.py` - PyTorch transformer architecture  
- `model.py` - Legacy TensorFlow CNN-LSTM model
- `config/default.yaml` - Configuration management
- `tokenization.py` - Chart tokenization system

## API Endpoints

- `GET /api/status` - System status
- `POST /api/upload-audio` - Upload audio files
- `POST /api/start-training` - Start model training
- `POST /api/generate-chart` - Generate charts
- `GET /api/charts` - List generated charts
- `POST /api/submit-evaluation` - Submit human evaluations

## WebSocket Events

Real-time updates via WebSocket:
- `training_progress` - Training progress updates
- `generation_progress` - Chart generation progress  
- `system_log` - System log messages
- `chart_generated` - New chart notifications

## Configuration

The interface uses your existing `config/default.yaml` file and provides a web UI for:

- Model architecture parameters (d_model, attention heads, layers)
- Training settings (learning rate, batch size, epochs)
- Audio processing settings (feature size, quantization)
- Generation parameters (difficulty, pattern style)

## Human Evaluation System

Collect multi-criteria feedback on generated charts:
- **Fun Factor**: Subjective enjoyment rating
- **Musicality**: How well notes match the music
- **Playability**: Physical comfort and flow
- **Pattern Coherence**: Logical pattern structure
- **Difficulty Accuracy**: Appropriate for target skill level

This data can be used for reinforcement learning from human feedback (RLHF) to improve the AI models.

## Dark Theme Features

- Compact, minimal design with reduced padding
- Multiple aesthetic themes (Cyberpunk, Deep Space, Electric)
- Interactive taiko drums with authentic styling
- Professional color scheme optimized for extended use

## Troubleshooting

**Import Errors**: Make sure you're running the server from the `web/` directory inside TaikoNationV1, and that all TaikoNation dependencies are installed.

**Port Issues**: If port 5000 is in use, modify the port in `server.py`.

**File Upload Issues**: Check that the `input_songs/` and `output/` directories exist and are writable.

**Training Issues**: Verify that your training data is properly formatted in the `input_charts_nr/` directory.

## FastAPI migration (optional, recommended)

This repository also includes an experimental FastAPI ASGI server at `web/server_fastapi.py` which provides the same API surface but uses an async Socket.IO server and works well with `uvicorn` (no eventlet required).

Why use FastAPI?
- Async first — works better with asyncio-based workers and modern async libraries.
- Built-in OpenAPI docs (`/docs`) and automatic request validation via Pydantic.
- Easier deployment with `uvicorn` / ASGI hosts.

Quickstart (FastAPI)

1. Create and activate a virtualenv and install web deps:
```bash
cd web
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the FastAPI ASGI app with Uvicorn:
```bash
# Run the ASGI Socket.IO + FastAPI app
uvicorn server_fastapi:socket_app --host 127.0.0.1 --port 5001 --reload
```

3. Open the UI in your browser at http://127.0.0.1:5001

Notes about configuration and environment variables
- `TAIKONATION_SECRET_KEY` — optional Flask/legacy secret; not required for FastAPI but safe to set.
- `TAIKONATION_API_TOKEN` — optional API token. If set, POST endpoints require the token via `Authorization: Bearer <token>` header or a form field named `api_token`.
- For production, run behind a reverse proxy (nginx) and use HTTPS.

Background jobs and workers
- The FastAPI server includes simulated background generation for demo purposes. For real training and chart generation you should run heavy jobs in a separate worker process (Celery, RQ, or a dedicated microservice) and emit progress via Socket.IO or push to a shared database.

Compatibility notes
- The original Flask server `web/server.py` remains in the repo. You can keep it for development or backward-compatibility, but we recommend using the FastAPI server for new deployments.

If you want, I can:
- Add a small `docker-compose` recipe to run the FastAPI server + a worker + Redis for background tasks.
- Scaffold a Celery worker that publishes Socket.IO events back to the FastAPI app.