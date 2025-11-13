import numpy as np
from collections import Counter
from scipy import stats

def compare_metrics(metrics_a, metrics_b):
    """
    Compares two sets of metrics using t-tests and Cohen's d.

    Args:
        metrics_a (list[dict]): A list of metric dictionaries for model A.
        metrics_b (list[dict]): A list of metric dictionaries for model B.

    Returns:
        dict: A dictionary containing the results of the statistical tests.
    """
    results = {}

    # Flatten the metrics into lists of values for each metric
    flat_a = {key: [m[key] for m in metrics_a] for key in metrics_a[0]}
    flat_b = {key: [m[key] for m in metrics_b] for key in metrics_b[0]}

    for metric_name in flat_a:
        a = flat_a[metric_name]
        b = flat_b[metric_name]

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test

        # Calculate Cohen's d for effect size
        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
        pooled_std = np.sqrt(((len(a) - 1) * std_a**2 + (len(b) - 1) * std_b**2) / (len(a) + len(b) - 2))
        cohen_d = (mean_a - mean_b) / pooled_std if pooled_std != 0 else 0

        results[metric_name] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohen_d": cohen_d,
        }
    return results

class AcademicEvaluator:
    """
    Calculates a suite of academic-grade metrics for generated Taiko charts.
    """
    def __init__(self, tokenizer):
        """
        Initializes the evaluator.

        Args:
            tokenizer: An instance of TaikoTokenizer to decode tokens.
        """
        self.tokenizer = tokenizer

    def evaluate_chart(self, token_ids):
        """
        Runs all evaluations on a given sequence of token IDs.

        Args:
            token_ids (list[int]): The list of token IDs representing the chart.

        Returns:
            dict: A dictionary containing all calculated metrics.
        """
        if not token_ids:
            return self._get_empty_metrics()

        metrics = {
            "pattern_metrics": self._calculate_pattern_metrics(token_ids),
            "timing_metrics": self._calculate_timing_metrics(token_ids),
            "difficulty_metrics": self._calculate_difficulty_metrics(token_ids),
            "human_likeness_metrics": self._calculate_human_likeness_metrics(token_ids),
        }
        return metrics

    def _get_empty_metrics(self):
        """Returns a dictionary with default values for an empty chart."""
        return {
            "pattern_metrics": {"n_gram_uniqueness": 0.0, "pattern_entropy": 0.0},
            "timing_metrics": {"token_note_ratio": 0.0},
            "difficulty_metrics": {"estimated_difficulty": 0.0},
            "human_likeness_metrics": {"some_future_metric": 0.0},
        }

    def _calculate_pattern_metrics(self, token_ids):
        """
        Calculates metrics related to musical patterns.
        - N-gram Uniqueness: Measures the diversity of patterns. A higher value suggests more variety.
        - Pattern Entropy: Measures the predictability of the patterns. A higher value suggests less repetition.
        """
        n_grams = [tuple(token_ids[i:i+3]) for i in range(len(token_ids) - 2)]
        if not n_grams:
            return {"n_gram_uniqueness": 0.0, "pattern_entropy": 0.0}

        uniqueness = len(set(n_grams)) / len(n_grams) if n_grams else 0.0

        counts = Counter(n_grams)
        probs = [count / len(n_grams) for count in counts.values()]
        entropy = -np.sum(probs * np.log2(probs)) if probs else 0.0

        return {"n_gram_uniqueness": uniqueness, "pattern_entropy": entropy}

    def _calculate_timing_metrics(self, token_ids):
        """
        Calculates metrics related to rhythm and timing.
        - Token Note Ratio: The ratio of note tokens to the total number of tokens.
        """
        num_notes = sum(1 for token in token_ids if token != self.tokenizer.vocab.get('[EMPTY]', -1))
        ratio = num_notes / len(token_ids) if token_ids else 0.0
        return {"token_note_ratio": ratio}

    def _calculate_difficulty_metrics(self, token_ids):
        """
        Estimates the chart's difficulty. This is a heuristic and not a substitute for a trained difficulty model.
        """
        pattern_metrics = self._calculate_pattern_metrics(token_ids)
        timing_metrics = self._calculate_timing_metrics(token_ids)

        # A more nuanced heuristic that considers entropy
        difficulty_score = (timing_metrics["token_note_ratio"] * 5) + \
                           (pattern_metrics["n_gram_uniqueness"] * 3) + \
                           (pattern_metrics["pattern_entropy"] * 2)
        return {"estimated_difficulty": difficulty_score}

    def _calculate_human_likeness_metrics(self, token_ids):
        """
        Calculates metrics comparing the chart to human-authored charts.
        """
        # Placeholder: This would require a reference dataset of human charts.
        return {"some_future_metric": 0.0}

    def export_to_csv(self, metrics_list, output_path):
        """
        Exports a list of metric dictionaries to a CSV file.
        """
        if not metrics_list:
            return

        import csv

        # Flatten the nested dictionaries
        flat_metrics = []
        for m in metrics_list:
            row = {}
            for key, value in m.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            flat_metrics.append(row)

        header = flat_metrics[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(flat_metrics)

    def export_to_latex(self, metrics_summary, output_path):
        """
        Exports a summary of metrics to a LaTeX table.
        """
        if not metrics_summary:
            return

        latex_string = "\\begin{tabular}{lrr}\n\\toprule\n"
        latex_string += "Metric & Mean & Std Dev \\\\\n\\midrule\n"

        for metric, values in metrics_summary.items():
            mean = np.mean(values)
            std = np.std(values)
            latex_string += f"{metric.replace('_', ' ')} & {mean:.3f} & {std:.3f} \\\\\n"

        latex_string += "\\bottomrule\n\\end{tabular}"

        with open(output_path, "w") as f:
            f.write(latex_string)
