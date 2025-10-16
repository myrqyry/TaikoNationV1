import os
import sys
import random
import sqlite3
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# --- Configuration ---
GENERATED_CHARTS_DIR = "output/generated_charts"
DB_PATH = "output/evaluations.db"
TABLE_NAME = "evaluations"

app = Flask(__name__)

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    print("Initializing database...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with the recommended schema
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        chart_id TEXT PRIMARY KEY,
        model_version TEXT,
        quantitative_scores TEXT,
        human_ratings TEXT,
        evaluator_metadata TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        session_id TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def get_random_chart():
    """Selects a random chart from the output directory."""
    try:
        charts = [f for f in os.listdir(GENERATED_CHARTS_DIR) if f.endswith('.osu')]
        if not charts:
            return None
        return random.choice(charts)
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    """Displays the main rating page with a random chart."""
    chart_filename = get_random_chart()
    if chart_filename is None:
        return "No generated charts found in 'output/generated_charts'. Please generate some charts first.", 404

    return render_template('index.html', chart_filename=chart_filename)

@app.route('/rate', methods=['POST'])
def rate_chart():
    """Handles the form submission and saves the rating to the database."""
    try:
        chart_id = request.form.get('chart_filename')

        ratings = {
            'fun': request.form.get('fun'),
            'musicality': request.form.get('musicality'),
            'playability': request.form.get('playability'),
            'pattern_coherence': request.form.get('pattern_coherence'),
            'difficulty_accuracy': request.form.get('difficulty_accuracy')
        }

        # For this phase, we'll use placeholders for other data
        model_version = "v1.0-alpha"
        session_id = request.remote_addr # Use IP as a simple session ID

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Insert or replace the rating for this chart
        cursor.execute(f"""
        INSERT OR REPLACE INTO {TABLE_NAME} (chart_id, model_version, human_ratings, session_id)
        VALUES (?, ?, ?, ?)
        """, (chart_id, model_version, json.dumps(ratings), session_id))

        conn.commit()
        conn.close()

        print(f"Successfully saved rating for {chart_id}")

    except Exception as e:
        print(f"Error saving rating to database: {e}")

    # Redirect back to the main page to rate another chart
    return redirect(url_for('index'))

@app.route('/chart_files/<path:filename>')
def serve_chart(filename):
    """Serves the generated chart files for download."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    charts_dir = os.path.join(project_root, GENERATED_CHARTS_DIR)
    return send_from_directory(charts_dir, filename, as_attachment=True)

def main():
    init_db()

    if not os.path.exists(GENERATED_CHARTS_DIR):
        os.makedirs(GENERATED_CHARTS_DIR)
        print(f"Created directory: {GENERATED_CHARTS_DIR}")
        print("Please add some generated .osu charts to this directory to begin evaluation.")

    print(f"Starting human evaluation server...")
    print(f"Visit http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)

if __name__ == '__main__':
    main()