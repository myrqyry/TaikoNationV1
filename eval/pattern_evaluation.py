import collections

class PatternEvaluator:
    def _get_ngrams(self, tokens, n):
        """Helper function to extract n-grams from a sequence of tokens."""
        return collections.Counter(zip(*[tokens[i:] for i in range(n)]))

    def calculate_pattern_overlap(self, generated_tokens, human_patterns, n=3):
        """
        Measure human-like patterning by calculating the overlap of n-grams
        between the generated sequence and a set of human-created patterns.

        Args:
            generated_tokens (list): A list of token IDs from the model.
            human_patterns (list of lists): A list of human-created token sequences.
            n (int): The size of the n-grams to use for comparison.

        Returns:
            float: The Jaccard similarity between the generated and human n-gram sets.
        """
        if not generated_tokens or not human_patterns:
            return 0.0

        generated_ngrams = self._get_ngrams(generated_tokens, n)
        human_ngrams = collections.Counter()
        for pattern in human_patterns:
            human_ngrams.update(self._get_ngrams(pattern, n))

        if not generated_ngrams or not human_ngrams:
            return 0.0

        intersection = generated_ngrams & human_ngrams
        union = generated_ngrams | human_ngrams

        return sum(intersection.values()) / sum(union.values()) if sum(union.values()) > 0 else 0.0

    def calculate_pattern_space_coverage(self, token_sequence, n=3):
        """
        Measure the variety of unique patterns used in a sequence by calculating
        the ratio of unique n-grams to the total number of n-grams.

        Args:
            token_sequence (list): A list of token IDs.
            n (int): The size of the n-grams to use.

        Returns:
            float: The ratio of unique n-grams to total n-grams.
        """
        if not token_sequence or len(token_sequence) < n:
            return 0.0

        ngrams = self._get_ngrams(token_sequence, n)
        total_ngrams = sum(ngrams.values())
        unique_ngrams = len(ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def evaluate_denden_sequences(self, tokens):
        """
        Specific evaluation for 'denden' (roll) patterns. This is a placeholder
        and would need a more detailed implementation based on the paper's
        specific definition of denden patterns.
        """
        # This would require a specific definition of what constitutes a "denden" pattern.
        # For now, we'll just count the occurrences of a hypothetical roll token.
        # In a real implementation, this would involve looking for specific sequences of Don/Ka notes.
        denden_token = -1 # Placeholder for a denden token ID
        return tokens.count(denden_token)