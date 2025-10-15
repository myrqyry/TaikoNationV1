import os
import numpy as np
import re

# --- Constants ---
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
EXPECTED_SONG_FEATURES = 80
EXPECTED_NOTE_FEATURES = 7
REPORT_FILENAME = "validation_report.txt"

def validate_dataset():
    """
    Scans the dataset for issues like missing files, corrupted data, and
    shape mismatches, then generates a report.
    """
    issues = []

    print("Starting dataset validation...")

    try:
        chart_files = sorted(os.listdir(INPUT_CHART_DIR))
        song_files = os.listdir(INPUT_SONG_DIR)
        song_map = {s.split()[0]: s for s in song_files}
    except FileNotFoundError as e:
        issues.append(f"FATAL: Could not find input directory: {e}")
        _write_report(issues)
        return

    # --- Check 1: Song-Chart Mismatches ---
    print("Checking for song-chart mismatches...")
    chart_ids = {c.split('_')[0] for c in chart_files}
    song_ids = set(song_map.keys())

    missing_songs = chart_ids - song_ids
    for chart_id in missing_songs:
        charts_with_id = [c for c in chart_files if c.startswith(f"{chart_id}_")]
        for chart in charts_with_id:
            issues.append(f"Mismatch: Chart '{chart}' has no corresponding song file.")

    missing_charts = song_ids - chart_ids
    for song_id in missing_charts:
        issues.append(f"Mismatch: Song '{song_map[song_id]}' has no corresponding chart file.")

    # --- Check 2: File Integrity and Shape Validation ---
    print("Validating file integrity and data shapes...")
    for chart_filename in chart_files:
        chart_path = os.path.join(INPUT_CHART_DIR, chart_filename)

        # Validate chart file
        try:
            chart_data = np.load(chart_path)
            if chart_data.ndim != 2 or chart_data.shape[1] != EXPECTED_NOTE_FEATURES:
                issues.append(f"ShapeError: Chart '{chart_filename}' has shape {chart_data.shape}, expected (*, {EXPECTED_NOTE_FEATURES}).")
        except Exception as e:
            issues.append(f"CorruptedFile: Could not load chart '{chart_filename}'. Error: {e}")
            continue # Skip song check if chart is broken

        # Validate corresponding song file
        try:
            chart_id = chart_filename.split("_")[0]
            if chart_id in song_map:
                song_filename = song_map[chart_id]
                song_path = os.path.join(INPUT_SONG_DIR, song_filename)
                try:
                    song_data = np.load(song_path)
                    if song_data.ndim != 2 or song_data.shape[1] != EXPECTED_SONG_FEATURES:
                        issues.append(f"ShapeError: Song '{song_filename}' has shape {song_data.shape}, expected (*, {EXPECTED_SONG_FEATURES}).")
                except Exception as e:
                    issues.append(f"CorruptedFile: Could not load song '{song_filename}'. Error: {e}")
        except IndexError:
            # This case should be caught by the mismatch check, but included for safety
            pass

    # --- Check 3: Difficulty Name Parsing ---
    print("Validating difficulty name parsing...")
    for chart_filename in chart_files:
        difficulty_match = re.search(r'\[(.*?)\]', chart_filename)
        if not difficulty_match:
            issues.append(f"NamingWarning: Chart '{chart_filename}' has no difficulty in its name (e.g., [oni]).")

    # --- Write Report ---
    _write_report(issues)
    print(f"Validation complete. Found {len(issues)} issues. Report saved to '{REPORT_FILENAME}'.")

def _write_report(issues):
    """Writes the list of issues to the report file."""
    with open(REPORT_FILENAME, 'w') as f:
        if not issues:
            f.write("Dataset validation successful! No issues found.\n")
        else:
            f.write("Dataset Validation Report\n")
            f.write("=========================\n\n")
            for issue in issues:
                f.write(f"- {issue}\n")

if __name__ == "__main__":
    validate_dataset()