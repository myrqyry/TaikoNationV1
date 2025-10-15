import collections
import numpy as np

class PatternEvaluator:
    def _get_ngrams(self, tokens, n):
        """Helper function to extract n-grams from a sequence of tokens."""
        if len(tokens) < n:
            return collections.Counter()
        return collections.Counter(zip(*[tokens[i:] for i in range(n)]))

    def calculate_pattern_overlap(self, generated_tokens, human_patterns, n=3):
        """
        Measure human-like patterning by calculating the overlap of n-grams
        between the generated sequence and a set of human-created patterns.
        Uses Jaccard similarity.
        """
        if not generated_tokens or not human_patterns:
            return 0.0

        generated_ngrams = self._get_ngrams(generated_tokens, n)
        human_ngrams = collections.Counter()
        for pattern in human_patterns:
            human_ngrams.update(self._get_ngrams(pattern, n))

        if not generated_ngrams or not human_ngrams:
            return 0.0

        intersection = generated_ngrams.keys() & human_ngrams.keys()
        union = generated_ngrams.keys() | human_ngrams.keys()

        return len(intersection) / len(union) if len(union) > 0 else 0.0

    def calculate_pattern_space_coverage(self, token_sequence, n=3):
        """
        Measure the variety of unique patterns used in a sequence by calculating
        the ratio of unique n-grams to the total number of possible n-grams of that length.
        This is a measure of diversity.
        """
        if not token_sequence or len(token_sequence) < n:
            return 0.0

        ngrams = self._get_ngrams(token_sequence, n)
        total_ngrams = sum(ngrams.values())
        unique_ngrams = len(ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def evaluate_denden_sequences(self, tokens, tokenizer):
        """
        Specific evaluation for 'denden' (roll) patterns.
        This implementation identifies sequences of roll tokens.
        """
        roll_start_token = tokenizer.vocab.get("roll_start")
        roll_end_token = tokenizer.vocab.get("roll_end")

        if roll_start_token is None or roll_end_token is None:
            return {"count": 0, "avg_length": 0}

        in_roll = False
        roll_count = 0
        roll_lengths = []
        current_length = 0

        for token in tokens:
            if token == roll_start_token:
                if not in_roll:
                    in_roll = True
                    roll_count += 1
                current_length += 1
            elif token == roll_end_token:
                if in_roll:
                    in_roll = False
                    roll_lengths.append(current_length)
                    current_length = 0
            elif in_roll:
                current_length += 1

        avg_length = np.mean(roll_lengths) if roll_lengths else 0
        return {"count": roll_count, "avg_length": avg_length}