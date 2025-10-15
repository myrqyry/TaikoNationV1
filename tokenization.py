import numpy as np
import itertools

class TaikoTokenizer:
    """
    A tokenizer for Taiko no Tatsujin charts, inspired by BEaRT-style tokenization.

    It converts high-resolution chart data into a sequence of discrete tokens,
    where each token represents a quantized time step (e.g., 100ms) and encodes
    all the note events that occurred within that step.
    """
    def __init__(self, time_quantization=100, source_resolution=23.2):
        """
        Initializes the tokenizer.

        Args:
            time_quantization (int): The target time resolution in milliseconds for each token.
            source_resolution (float): The assumed time resolution in ms of the source .npy data.
                                       This is a critical assumption as the original data format
                                       is not documented. 23.2ms corresponds to a common audio
                                       setting of 22050 Hz / 512 hop length.
        """
        self.time_quantization = time_quantization
        self.source_resolution = source_resolution
        self.steps_per_chunk = int(round(self.time_quantization / self.source_resolution))

        # Define the events based on the 7-dimensional vector from the original data.
        self.event_names = [
            "don", "ka", "big_don", "big_ka", "roll_start", "roll_end", "finisher"
        ]

        # Define special tokens
        self.special_tokens = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[EMPTY]"]

        # Generate the vocabulary
        self.vocab = self._generate_vocabulary()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _generate_vocabulary(self):
        """
        Generates the vocabulary from all possible combinations of events, plus special tokens.
        Each combination of events within a time step becomes a unique token.
        """
        vocab = {}

        # Add special tokens first
        for token in self.special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        # Generate tokens for all 2^7 - 1 possible event combinations (excluding the empty case)
        num_events = len(self.event_names)
        for i in range(1, 2**num_events):
            combination = []
            for j in range(num_events):
                if (i >> j) & 1:
                    combination.append(self.event_names[j])

            token_str = ",".join(combination)
            if token_str not in vocab:
                vocab[token_str] = len(vocab)

        return vocab

    def _vector_to_token_id(self, vector):
        """Converts a multi-hot vector [0,1,0,1,...] into a token ID."""
        if not np.any(vector):
            return self.vocab["[EMPTY]"]

        active_events = [self.event_names[i] for i, is_active in enumerate(vector) if is_active]
        token_str = ",".join(active_events)

        return self.vocab.get(token_str, self.vocab["[PAD]"]) # Default to PAD if combo is invalid

    def tokenize(self, note_data_path):
        """
        Loads a .npy chart file and converts it into a sequence of token IDs.

        Args:
            note_data_path (str): The path to the .npy chart file.

        Returns:
            list[int]: A sequence of token IDs.
        """
        try:
            note_data = np.load(note_data_path)
            # Assuming the vector is already boolean/integer {0,1}
            note_data = note_data.astype(bool)
        except Exception as e:
            print(f"Error loading or processing {note_data_path}: {e}")
            return []

        num_steps = note_data.shape[0]
        token_ids = []

        for i in range(0, num_steps, self.steps_per_chunk):
            chunk = note_data[i:i + self.steps_per_chunk]

            # Combine all events in the chunk using a bitwise OR.
            # This means if an event happens at all in the 100ms window, it's included.
            combined_vector = np.logical_or.reduce(chunk, axis=0)

            token_id = self._vector_to_token_id(combined_vector)
            token_ids.append(token_id)

        return token_ids

    def detokenize(self, token_ids):
        """Converts a sequence of token IDs back into a list of event strings."""
        return [self.reverse_vocab.get(token_id, "[UNK]") for token_id in token_ids]

    @property
    def vocab_size(self):
        return len(self.vocab)


class PatternAwareTaikoTokenizer(TaikoTokenizer):
    """
    Extends the TaikoTokenizer to include special tokens for representing
    higher-level musical patterns, such as drum rolls and repeated sections.
    This allows the model to learn and generate more structured and coherent charts.
    """
    def __init__(self, *args, **kwargs):
        # --- Define Pattern-Specific Tokens ---
        self.pattern_tokens = {
            "[DENDEN_START]": 0, # Start of a "denden" or roll
            "[DENDEN_CONT]": 0,  # Continuation of a roll
            "[PATTERN_REPEAT]": 0, # Indicates a recently played pattern is repeated
            "[STRONG_BEAT]": 0,    # A token representing a rhythmically strong beat
        }
        super().__init__(*args, **kwargs) # This will call _generate_vocabulary

    def _generate_vocabulary(self):
        """
        Overrides the base vocabulary generation to merge base event tokens
        with the new pattern-specific tokens.
        """
        # 1. Generate the base vocabulary from the parent class
        base_vocab = super()._generate_vocabulary()

        # 2. Add the new pattern tokens, ensuring no ID conflicts
        for token in self.pattern_tokens:
            if token not in base_vocab:
                base_vocab[token] = len(base_vocab)

        # Update the pattern_tokens dictionary with their final assigned IDs
        for token in self.pattern_tokens:
            self.pattern_tokens[token] = base_vocab[token]

        return base_vocab

    def find_and_tokenize_patterns(self, note_data):
        """
        A placeholder for the advanced logic that would be needed to identify
        and replace sequences of notes with the new pattern tokens.

        For example, this method could:
        - Detect a sequence of 'don' or 'ka' notes close together and replace
          them with [DENDEN_START], [DENDEN_CONT], ...
        - Use a sequence matching algorithm to find repeated note phrases and
          replace the second occurrence with [PATTERN_REPEAT].

        This is a complex task and would be a significant feature addition.
        For now, the standard tokenization will proceed without this step.
        """
        # This is where pattern detection logic would go.
        # It would return a modified note_data array or a list of tokens
        # with pattern tokens interspersed.
        pass

    def tokenize(self, note_data_path):
        """
        Overrides the base tokenize method to (in the future) include the
        pattern identification step. For now, it functions identically to the parent.
        """
        # In a full implementation, one would first load the note_data,
        # then call self.find_and_tokenize_patterns(note_data),
        # and then proceed with the rest of the tokenization logic.
        return super().tokenize(note_data_path)
