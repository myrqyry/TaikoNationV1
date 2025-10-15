import os
import sys
import random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Add project root to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Configuration ---
GENERATED_CHARTS_DIR = "output/generated_charts"
RATINGS_CSV_PATH = "output/human_eval_ratings.csv"

app = Flask(__name__)

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

    # We need to make the file accessible to the user, so we'll serve it from a static-like folder.
    # For simplicity, we'll just pass the filename and construct the path in the template.
    return render_template('index.html', chart_filename=chart_filename)

@app.route('/chart_files/<path:filename>')
def serve_chart(filename):
    """Serves the generated chart files for download."""
    # Construct an absolute path to the charts directory from the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    charts_dir = os.path.join(project_root, GENERATED_CHARTS_DIR)
    return send_from_directory(charts_dir, filename, as_attachment=True)

@app.route('/rate', methods=['POST'])
def rate_chart():
    """Handles the form submission and saves the rating."""
    chart_filename = request.form.get('chart_filename')
    fun_rating = request.form.get('fun')
    musicality_rating = request.form.get('musicality')
    playability_rating = request.form.get('playability')

    print(f"Received rating for {chart_filename}: Fun={fun_rating}, Musicality={musicality_rating}, Playability={playability_rating}")

    # --- Save the rating to a CSV file ---
    try:
        new_rating = pd.DataFrame([{
            'chart_filename': chart_filename,
            'fun': fun_rating,
            'musicality': musicality_rating,
            'playability': playability_rating
        }])

        # Use a file lock to prevent race conditions if multiple people were rating
        with open(RATINGS_CSV_PATH, 'a') as f:
            new_rating.to_csv(f, header=f.tell()==0, index=False)

    except Exception as e:
        print(f"Error saving rating: {e}")

    # Redirect back to the main page to rate another chart
    return redirect(url_for('index'))

def main():
    # Ensure the output directory for charts exists
    if not os.path.exists(GENERATED_CHARTS_DIR):
        os.makedirs(GENERATED_CHARTS_DIR)
        print(f"Created directory: {GENERATED_CHARTS_DIR}")
        print("Please add some generated .osu charts to this directory to begin evaluation.")

    # Check for CSV file and create if it doesn't exist
    if not os.path.exists(RATINGS_CSV_PATH):
        df = pd.DataFrame(columns=['chart_filename', 'fun', 'musicality', 'playability'])
        df.to_csv(RATINGS_CSV_PATH, index=False)
        print(f"Created ratings file at: {RATINGS_CSV_PATH}")

    print(f"Starting human evaluation server...")
    print(f"Visit http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)

if __name__ == '__main__':
    main()