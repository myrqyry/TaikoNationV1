'''
evaluation script for binary representation

checks given files vs random noise, as well as vs human benchmark

press 1 for full folder, press 2 for just one (enter them as program args)
'''

import numpy as np
import sys
import os
import glob

def main():
    """Main function to run the evaluation."""
    # initialize random noise, can put seed here if needed!
    rng = np.random.default_rng(2009000042)
    noise = rng.integers(low=0, high=2, size=1000000)

    mode = input("entire folder or single given input? (1/2)")

    if mode == "1":
        evaluate_folder(noise)
    elif mode == "2":
        if len(sys.argv) < 3:
            print("Usage: python bin_evaluate.py <ai_chart.npy> <human_chart.npy>")
            return
        evaluate_single_file(sys.argv[1], sys.argv[2], noise)
    else:
        print("Invalid mode selected. Please choose 1 or 2.")

def evaluate_folder(noise):
    """Evaluates all .npy files in the current directory."""
    ai_files = sorted(glob.glob("ai_*.npy"))
    human_files = sorted(glob.glob("human_*.npy"))

    if len(ai_files) != len(human_files):
        print("Warning: Mismatch in the number of AI and human chart files.")
        return

    noise_similarity_sum = 0
    human_similarity_sum = 0
    human_similarity_oc_sum = 0
    chart_vs_chart_similarity_sum = 0
    ai_pattern_score_sum = 0
    human_pattern_score_sum = 0
    num_files = len(ai_files)

    for ai_file, human_file in zip(ai_files, human_files):
        try:
            ai_chart = np.load(ai_file)
            human_chart = np.load(human_file)
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            continue

        print(f"{ai_file} versus random noise:")
        noise_similarity_sum += vs_random(ai_chart, noise)

        print(f"{ai_file} versus {human_file}")
        human_similarity_sum += vs_human(ai_chart, human_chart)
        human_similarity_oc_sum += vs_human_sliding_window(ai_chart, human_chart)

        print(f"{ai_file} sliding scale")
        ai_pattern_score_sum += sliding_scale(ai_chart)

        print(f"{human_file} sliding scale")
        human_pattern_score_sum += sliding_scale(human_chart)

        print("chart vs chart")
        chart_vs_chart_similarity_sum += sliding_scale_comparison(ai_chart, human_chart)

    if num_files > 0:
        print("\nRESULTS:")
        print(f"{noise_similarity_sum / num_files:.2f}% average similarity to noise (DC RANDOM)")
        print(f"{human_similarity_sum / num_files:.2f}% average similarity to human (DC HUMAN)")
        print(f"{human_similarity_oc_sum / num_files:.2f}% avg similarity to human with slide scale (OC HUMAN)")
        print(f"{ai_pattern_score_sum / num_files:.2f} ai pattern score, {human_pattern_score_sum / num_files:.2f} human pattern score (OVER P-SPACE)")
        print(f"{chart_vs_chart_similarity_sum / num_files:.2f}% of human patterns ai used (HI P-SPACE")

def evaluate_single_file(ai_file, human_file, noise):
    """Evaluates a single pair of AI and human charts."""
    try:
        ai_chart = np.load(ai_file)
        human_chart = np.load(human_file)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return

    print(f"{ai_file} versus Random Noise:")
    vs_random(ai_chart, noise)

    print(f"{ai_file} versus {human_file}")
    vs_human(ai_chart, human_chart)

def vs_random(chart, noise):
    """Compares a given chart to random noise by similarities vs total."""
    total = len(chart)
    similarity = np.sum(chart == noise[:total])
    result = (similarity / total) * 100
    print(f"{result:.2f}% similar\n")
    return result

def vs_human(chart, chart2):
    """Compares two charts by similarities vs total."""
    limit = min(len(chart), len(chart2))
    start = 0
    while start < len(chart2) and chart2[start] != 1:
        start += 1

    total = limit - start
    if total <= 0:
        return 0

    similarity = np.sum(chart[start:limit] == chart2[start:limit])
    under_chart = np.sum(chart[start:limit] == 0)
    over_chart = total - similarity - under_chart

    result = (similarity / total) * 100
    under = (under_chart / total) * 100
    over = (over_chart / total) * 100
    print(f"{result:.2f}% similar, {under:.2f}% underchart, {over:.2f}% overchart\n")
    return result

def sliding_scale(chart, scale=8):
    """Returns the percent of the possibility space that chart is covering over a sliding window."""
    patterns = set()
    last_ind = len(chart) - scale + 1
    for i in range(last_ind):
        chunk = tuple(chart[i:i+scale])
        patterns.add(chunk)

    p_score = (len(patterns) / (2**scale)) * 100
    print(f"{p_score:.2f} pattern score\n")
    return p_score

def sliding_scale_comparison(chart, chart2, scale=8):
    """Compares the pattern space of two charts."""
    patterns1 = set()
    last_ind1 = len(chart) - scale + 1
    for i in range(last_ind1):
        chunk = tuple(chart[i:i+scale])
        patterns1.add(chunk)

    patterns2 = set()
    last_ind2 = len(chart2) - scale + 1
    for i in range(last_ind2):
        chunk = tuple(chart2[i:i+scale])
        patterns2.add(chunk)

    if not patterns2:
        return 0

    intersection_score = len(patterns1.intersection(patterns2))
    p_score = (intersection_score / len(patterns2)) * 100
    print(f"{p_score:.2f} pattern score\n")
    return p_score

def vs_human_sliding_window(chart, chart2, buffer=1):
    """Compares two charts with a sliding window for hit notes."""
    limit = min(len(chart), len(chart2))
    start = 0
    while start < len(chart2) and chart2[start] != 1:
        start += 1

    total = limit - start
    if total <= 0:
        return 0

    similarity = 0
    for i in range(start, limit):
        if chart[i] == 1:
            for b in range(-buffer, buffer + 1):
                try:
                    if chart2[i + b] == 1:
                        similarity += 1
                        break
                except IndexError:
                    pass  # Ignore out-of-bounds access
        elif chart[i] == 0 and chart[i] == chart2[i]:
            similarity += 1

    result = (similarity / total) * 100
    print(f"{result:.2f}% similar\n")
    return result

if __name__ == "__main__":
    main()
