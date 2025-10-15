import numpy as np
import sys
import logging
import re

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_sm_metadata(lines):
    """Parses the metadata from the .sm file content."""
    metadata = {}
    for line in lines:
        if line.startswith('#'):
            match = re.match(r"#([^:]+):([^;]+);", line)
            if match:
                key, value = match.groups()
                metadata[key.strip()] = value.strip()
    return metadata

def get_bpms(lines):
    """Parses BPM changes from the .sm file content."""
    bpms = {}
    for line in lines:
        if line.startswith('#BPMS'):
            try:
                parts = line.split(':')[1].split(',')
                for part in parts:
                    beat, bpm = part.split('=')
                    bpms[float(beat)] = float(bpm)
            except (ValueError, IndexError) as e:
                logging.error(f"Could not parse BPMs: {e}")
    return bpms

def get_offset(lines):
    """Parses the offset from the .sm file content."""
    for line in lines:
        if line.startswith('#OFFSET'):
            try:
                return float(line.split(':')[1].strip(';'))
            except (ValueError, IndexError) as e:
                logging.error(f"Could not parse offset: {e}")
    return 0.0

def get_notes_section(lines):
    """Extracts the notes section for a specific chart type."""
    in_notes_section = False
    notes_lines = []
    for line in lines:
        if re.match(r'#NOTES:', line):
            # This is just an example, you might need to select a specific chart type
            # e.g., dance-single, challenge, etc.
            in_notes_section = True
            continue
        if in_notes_section:
            if line.startswith(';'):
                break
            notes_lines.append(line)
    return notes_lines

def convert_sm_to_binary(note_lines, bpms, offset):
    """Converts the .sm notes to a binary representation."""
    note_times = []
    current_time = -offset * 1000  # ms
    current_bpm = next(iter(bpms.values())) if bpms else 60.0

    for i, measure in enumerate("".join(note_lines).split(',')):
        measure = measure.strip()
        if not measure:
            continue

        notes_in_measure = len(measure) / 4 # Assuming 4 characters per note
        if notes_in_measure == 0:
            continue

        beat_duration = 60 * 1000 / current_bpm
        note_duration = beat_duration * 4 / notes_in_measure

        for j, note in enumerate(measure):
            if note in '124': # Note or hold head
                note_times.append(current_time)
            current_time += note_duration

    bin_output = []
    out_time = 0
    for note_time in sorted(note_times):
        while abs(note_time - out_time) > 23:
            bin_output.append(0)
            out_time += 23
        bin_output.append(1)
        out_time += 23

    return np.array(bin_output, dtype=np.int32), np.array(sorted(note_times), dtype=np.int32)

def main():
    """Main function to handle .sm file conversion."""
    if len(sys.argv) < 2:
        logging.error("Usage: sm_to_bin.py <file.sm>")
        return

    filepath = sys.argv[1]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return

    metadata = parse_sm_metadata(lines)
    bpms = get_bpms(lines)
    offset = get_offset(lines)
    note_lines = get_notes_section(lines)

    if not note_lines:
        logging.error("No notes found in the file.")
        return

    bin_output, time_output = convert_sm_to_binary(note_lines, bpms, offset)

    bin_filename = f"{filepath}_bin.npy"
    time_filename = f"{filepath}_time.npy"
    np.save(bin_filename, bin_output)
    np.save(time_filename, time_output)
    logging.info(f"Successfully converted {filepath} to {bin_filename} and {time_filename}")

if __name__ == "__main__":
    main()
