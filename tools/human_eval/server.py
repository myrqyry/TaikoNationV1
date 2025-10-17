from flask import Flask, render_template_string, request, jsonify
import sqlite3
import os
import json

app = Flask(__name__)
DB_PATH = "output/evaluations.db"
CHART_DIR = "output/generated_charts_for_eval"

# --- Database Setup ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chart_filename TEXT NOT NULL,
            rater_id TEXT,
            fun_rating REAL,
            musicality_rating REAL,
            playability_rating REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Taiko Chart Evaluation</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
        .chart-container { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
        .slider-container { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="range"] { width: 100%; }
        #chart-display { white-space: pre-wrap; font-family: monospace; background: #f4f4f4; padding: 10px; }
    </style>
</head>
<body>
    <h1>Taiko Chart Evaluation</h1>
    <div id="chart-container" class="chart-container">
        <h2>Chart: <span id="chart-name"></span></h2>
        <div id="chart-display">Loading chart...</div>
    </div>
    <form id="eval-form">
        <div class="slider-container">
            <label for="fun">Fun:</label>
            <input type="range" id="fun" name="fun" min="0" max="1" step="0.1" value="0.5">
        </div>
        <div class="slider-container">
            <label for="musicality">Musicality:</label>
            <input type="range" id="musicality" name="musicality" min="0" max="1" step="0.1" value="0.5">
        </div>
        <div class="slider-container">
            <label for="playability">Playability:</label>
            <input type="range" id="playability" name="playability" min="0" max="1" step="0.1" value="0.5">
        </div>
        <input type="hidden" id="chart-filename" name="chart_filename">
        <button type="submit">Submit and Next</button>
    </form>

    <script>
        let charts = [];
        let currentIndex = 0;

        async function fetchCharts() {
            const response = await fetch('/get_charts');
            charts = await response.json();
            loadChart();
        }

        function loadChart() {
            if (currentIndex >= charts.length) {
                document.getElementById('chart-container').innerHTML = '<h2>No more charts to evaluate!</h2>';
                document.getElementById('eval-form').style.display = 'none';
                return;
            }
            const chartFilename = charts[currentIndex];
            document.getElementById('chart-name').innerText = chartFilename;
            document.getElementById('chart-filename').value = chartFilename;

            // For this simple version, we'll just display the filename.
            // A more advanced version would fetch and render the chart content.
            document.getElementById('chart-display').innerText = `Evaluating: ${chartFilename}`;
        }

        document.getElementById('eval-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = {
                chart_filename: formData.get('chart_filename'),
                fun_rating: parseFloat(formData.get('fun')),
                musicality_rating: parseFloat(formData.get('musicality')),
                playability_rating: parseFloat(formData.get('playability')),
            };

            await fetch('/submit_evaluation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            currentIndex++;
            loadChart();
        });

        fetchCharts();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_charts')
def get_charts():
    """Returns a list of chart filenames to be evaluated."""
    os.makedirs(CHART_DIR, exist_ok=True) # Ensure directory exists
    # In a real scenario, you'd filter for charts the user hasn't rated yet.
    charts = [f for f in os.listdir(CHART_DIR) if f.endswith('.tja')]
    return jsonify(charts)

@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO evaluations (chart_filename, fun_rating, musicality_rating, playability_rating) VALUES (?, ?, ?, ?)",
        (data['chart_filename'], data['fun_rating'], data['musicality_rating'], data['playability_rating'])
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    init_db()
    # Create dummy chart files for testing
    os.makedirs(CHART_DIR, exist_ok=True)
    for i in range(5):
        with open(os.path.join(CHART_DIR, f'chart_{i}.tja'), 'w') as f:
            f.write(f"This is a dummy chart file for chart_{i}.tja")

    app.run(debug=True, port=5001)