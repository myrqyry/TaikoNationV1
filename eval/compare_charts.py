import argparse
import json
import os
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taikonation.eval.metrics import compare_metrics, AcademicEvaluator
from taikonation.data.tokenization import TaikoTokenizer

def load_metrics_from_dir(directory):
    """Loads all _metrics.json files from a directory."""
    metrics_list = []
    for filepath in Path(directory).rglob("*_metrics.json"):
        with open(filepath, 'r') as f:
            metrics_list.append(json.load(f))
    return metrics_list

def main():
    parser = argparse.ArgumentParser(description="Compare academic metrics of two sets of generated charts.")
    parser.add_argument("dir_a", help="Directory containing the first set of charts and metrics.")
    parser.add_argument("dir_b", help="Directory containing the second set of charts and metrics.")
    parser.add_argument("--output_csv", default="comparison.csv", help="Path to save the comparison in CSV format.")
    parser.add_argument("--output_latex", default="comparison.tex", help="Path to save the comparison in LaTeX format.")
    args = parser.parse_args()

    metrics_a = load_metrics_from_dir(args.dir_a)
    metrics_b = load_metrics_from_dir(args.dir_b)

    if not metrics_a:
        print(f"Error: No metrics files found in {args.dir_a}")
        return
    if not metrics_b:
        print(f"Error: No metrics files found in {args.dir_b}")
        return

    # To use the export functions, we need an AcademicEvaluator instance
    # The tokenizer is not actually used in the export functions, so we can pass a dummy one
    evaluator = AcademicEvaluator(tokenizer=TaikoTokenizer())

    # Flatten the metrics for comparison and export
    flat_metrics_a = []
    for m in metrics_a:
        row = {}
        for cat, sub_dict in m.items():
            for key, val in sub_dict.items():
                row[f"{cat}_{key}"] = val
        flat_metrics_a.append(row)

    flat_metrics_b = []
    for m in metrics_b:
        row = {}
        for cat, sub_dict in m.items():
            for key, val in sub_dict.items():
                row[f"{cat}_{key}"] = val
        flat_metrics_b.append(row)

    comparison_results = compare_metrics(flat_metrics_a, flat_metrics_b)

    print("--- Comparison Results ---")
    for metric, results in comparison_results.items():
        print(f"\nMetric: {metric}")
        print(f"  Model A: Mean={results['mean_a']:.3f}")
        print(f"  Model B: Mean={results['mean_b']:.3f}")
        print(f"  t-statistic={results['t_statistic']:.3f}, p-value={results['p_value']:.3f}")
        print(f"  Cohen's d={results['cohen_d']:.3f}")

    # Export the raw metrics to CSV
    evaluator.export_to_csv(flat_metrics_a, "metrics_a.csv")
    evaluator.export_to_csv(flat_metrics_b, "metrics_b.csv")
    print(f"\nRaw metrics for Model A saved to metrics_a.csv")
    print(f"Raw metrics for Model B saved to metrics_b.csv")

    # Create a summary for LaTeX export
    summary_a = {key: [m[key] for m in flat_metrics_a] for key in flat_metrics_a[0]}
    evaluator.export_to_latex(summary_a, args.output_latex)
    print(f"LaTeX summary for Model A saved to {args.output_latex}")

if __name__ == "__main__":
    main()
