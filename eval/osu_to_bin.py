'''
This program takes a .osu file and converts it into a binary form
for evaluation.

output legend:
0 - no object / middle of hold & spinner
1 - is a "clickable" object

Author: Emily Halina
'''

import sys
import numpy as np
from collections import deque
import logging
import re

# --- Constants ---
TIME_INCREMENT = 23  # in milliseconds

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """Main function to handle file processing and conversion."""
    if len(sys.argv) < 2:
        logging.error("Usage: osu_to_bin.py osufile.osu")
        return

    filepath = sys.argv[1]
    try:
        with open(filepath, "r", encoding="utf8") as f:
            content = deque(f.read().splitlines())
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    filename, slider_multiplier, author_info = parse_metadata(content)
    timing_points = parse_timing_points(content)
    notes = parse_hit_objects(content, timing_points, slider_multiplier)

    if convert_to_binary(notes, filename, author_info):
        logging.info(f"{filename}: Conversion successful.")
    else:
        logging.error(f"{filename}: Conversion failed.")

def parse_metadata(content):
    """Parses the metadata section of the .osu file."""
    try:
        # Navigate to [Metadata]
        while not content[0].startswith("[Metadata]"):
            content.popleft()

        metadata = {}
        content.popleft() # Skip the [Metadata] tag
        while content and not content[0].startswith("["):
            line = content.popleft()
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        # Navigate to [Difficulty]
        while not content[0].startswith("[Difficulty]"):
            content.popleft()

        content.popleft() # Skip the [Difficulty] tag
        while content and not content[0].startswith("["):
            line = content.popleft()
            if line.startswith("SliderMultiplier:"):
                slider_multiplier = float(line.split(":")[1].strip())
                break
        else:
            slider_multiplier = 1.0 # Default value

        filename = f"{metadata.get('BeatmapSetID', '')}_{metadata.get('Artist', '')}_{metadata.get('Title', '')}_{metadata.get('Version', '')}".replace(" ", "_").replace("/", "_")
        author_info = f"Creator: {metadata.get('Creator', 'Unknown')}, Beatmap URL: https://osu.ppy.sh/s/{metadata.get('BeatmapSetID', '')}\n"

        return filename, slider_multiplier, author_info

    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing metadata: {e}")
        raise

def parse_timing_points(content):
    """Parses the timing points section of the .osu file."""
    timing_points = []
    try:
        while not content[0].startswith("[TimingPoints]"):
            content.popleft()
        content.popleft() # Skip the [TimingPoints] tag

        while content and content[0]:
            line = content.popleft().split(',')
            time = int(float(line[0]))
            value = float(line[1])
            if value >= 0:
                timing_points.append({'time': time, 'beat_length': value, 'sv_multiplier': 1})
            else:
                timing_points.append({'time': time, 'beat_length': None, 'sv_multiplier': -100 / value})
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing timing points: {e}")
    return timing_points

def parse_hit_objects(content, timing_points, slider_multiplier):
    """Parses the hit objects section of the .osu file."""
    notes = deque()
    try:
        while not content[0].startswith("[HitObjects]"):
            content.popleft()
        content.popleft() # Skip the [HitObjects] tag

        timing_point_idx = 0

        while content:
            line = content.popleft()
            if not line: continue

            parts = line.split(',')
            time = int(parts[2])
            object_type = int(parts[3])

            # Update timing point
            while timing_point_idx + 1 < len(timing_points) and time >= timing_points[timing_point_idx + 1]['time']:
                timing_point_idx += 1
            current_timing = timing_points[timing_point_idx]

            end_point = None
            if object_type & 2:  # Slider
                slider_length = float(parts[7])
                beat_length = current_timing['beat_length']
                sv_multiplier = current_timing['sv_multiplier']
                end_point = time + (slider_length / (slider_multiplier * 100) * beat_length)
            elif object_type & 8:  # Spinner
                end_point = int(parts[5])

            notes.append({'time': time, 'end_point': end_point})

    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing hit objects: {e}")
    return notes

def convert_to_binary(notes, filename, author_info):
    """Converts the parsed notes into a binary format and saves to a .npy file."""
    if not notes:
        logging.warning("No notes to convert.")
        return False

    bin_output = []
    time_output = []
    current_time = 0

    try:
        while notes:
            note = notes.popleft()
            time_output.append(note['time'])

            while note['time'] - current_time > TIME_INCREMENT:
                bin_output.append(0)
                current_time += TIME_INCREMENT

            bin_output.append(1)
            current_time += TIME_INCREMENT

            if note['end_point']:
                while note['end_point'] - current_time > TIME_INCREMENT:
                    bin_output.append(0)
                    current_time += TIME_INCREMENT

        np.save(f"{filename}_bin.npy", np.array(bin_output, dtype=np.int32))
        np.save(f"{filename}.npy", np.array(time_output, dtype=np.int32))
        return True

    except Exception as e:
        logging.error(f"An error occurred during binary conversion: {e}")
        return False

if __name__ == "__main__":
    main()
