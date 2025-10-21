#!/bin/bash
#
# This script provides an example of how to use the generate_chart.py CLI.
# It generates a chart for a sample song from the input_songs/ directory
# using a placeholder model file.

# --- Configuration ---

# The path to your trained model checkpoint.
# IMPORTANT: You must replace this with a real, trained model file (.pth)
# for the script to generate a meaningful chart.
MODEL_PATH="model/placeholder.pth"

# The input audio file (.npy format)
# This example uses one of the smaller files from the dataset.
INPUT_AUDIO="input_songs/1031143 RO-KYU-BU! - SHOOT! (TV Size) Input.npy"

# The desired output path for the generated chart (.osu format)
OUTPUT_CHART="output/generated_example_chart.osu"

# The difficulty level for the chart.
# Options: easy, normal, hard, oni, ura
DIFFICULTY="oni"

# --- Execution ---

echo "Running chart generation example..."
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_CHART"
echo "Difficulty: $DIFFICULTY"
echo ""

# Create the output directory if it doesn't exist
mkdir -p output

# Run the generation script
python generate_chart.py "$MODEL_PATH" "$INPUT_AUDIO" "$OUTPUT_CHART" --difficulty "$DIFFICULTY"

echo ""
echo "Example script finished."
echo "Generated chart (if successful) is at: $OUTPUT_CHART"
