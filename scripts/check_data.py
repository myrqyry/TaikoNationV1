#!/usr/bin/env python3
"""
Small dataset validation script for TaikoNationV1.
Checks that files in `input_charts_nr/` and `input_songs/` align by leading id, and that
`output/genre_labels.json` exists if referenced.

Usage:
    python scripts/check_data.py

Exit codes:
    0 - OK
    1 - Warnings found
    2 - Errors found
"""
import os
import json
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT_CHART_DIR = os.path.join(ROOT, 'input_charts_nr')
INPUT_SONG_DIR = os.path.join(ROOT, 'input_songs')
GENRE_LABELS = os.path.join(ROOT, 'output', 'genre_labels.json')

errors = []
warnings = []

def list_dir_safe(path):
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return None

def main():
    charts = list_dir_safe(INPUT_CHART_DIR)
    songs = list_dir_safe(INPUT_SONG_DIR)

    if charts is None:
        errors.append(f"Missing directory: {INPUT_CHART_DIR}")
    if songs is None:
        errors.append(f"Missing directory: {INPUT_SONG_DIR}")

    if charts is None or songs is None:
        report()
        return 2

    # Map songs by leading id token (first token before a space)
    song_map = {s.split()[0]: s for s in songs}
    missing_songs = []
    for c in charts:
        if not c:
            continue
        # chart files often start with id_... or id followed by underscore
        id_tok = c.split('_')[0]
        if id_tok not in song_map:
            missing_songs.append((c, id_tok))

    if missing_songs:
        warnings.append(f"{len(missing_songs)} chart(s) do not have a matching song file by id. Example: {missing_songs[:5]}")

    # Check genre labels if present
    if os.path.exists(GENRE_LABELS):
        try:
            with open(GENRE_LABELS, 'r') as f:
                j = json.load(f)
            if not isinstance(j, dict):
                warnings.append(f"{GENRE_LABELS} exists but is not a JSON object mapping filenames to genres.")
        except Exception as e:
            warnings.append(f"Failed to parse {GENRE_LABELS}: {e}")
    else:
        warnings.append(f"Genre labels not found at {GENRE_LABELS}. Genres will default to 'unknown'.")

    report()
    if errors:
        return 2
    if warnings:
        return 1
    return 0


def report():
    if errors:
        print("Errors:")
        for e in errors:
            print("  -", e)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print("  -", w)
    if not errors and not warnings:
        print("OK: dataset directories look reasonable.")

if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
