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
