import random
import sys
import os

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DifficultyScaler:
    """
    A class to algorithmically adjust the difficulty of a Taiko chart's token sequence.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.empty_token_id = self.tokenizer.vocab["[EMPTY]"]
        self.don_token_id = self.tokenizer.vocab.get("don")
        self.ka_token_id = self.tokenizer.vocab.get("ka")

    def increase_difficulty(self, token_ids, factor=0.2):
        """
        Increases the difficulty by converting some single notes followed by a rest
        into simple two-note patterns (e.g., don -> don, ka).
        """
        if not self.don_token_id or not self.ka_token_id:
            return token_ids

        new_token_ids = list(token_ids)
        eligible_indices = []
        for i in range(len(new_token_ids) - 1):
            if (new_token_ids[i] == self.don_token_id or new_token_ids[i] == self.ka_token_id) and \
               new_token_ids[i+1] == self.empty_token_id:
                eligible_indices.append(i)

        num_to_modify = int(len(eligible_indices) * factor)
        indices_to_modify = random.sample(eligible_indices, num_to_modify)

        for i in indices_to_modify:
            if new_token_ids[i] == self.don_token_id:
                new_token_ids[i+1] = self.ka_token_id
            else:
                new_token_ids[i+1] = self.don_token_id

        return new_token_ids

    def decrease_difficulty(self, token_ids, factor=0.2):
        """
        Decreases the difficulty by removing a fraction of the notes.
        """
        new_token_ids = list(token_ids)
        note_indices = [i for i, token_id in enumerate(new_token_ids) if self.reverse_vocab[token_id] not in ["[PAD]", "[EMPTY]", "[CLS]"]]

        num_to_remove = int(len(note_indices) * factor)
        indices_to_remove = random.sample(note_indices, num_to_remove)

        for i in indices_to_remove:
            new_token_ids[i] = self.empty_token_id

        return new_token_ids

if __name__ == '__main__':
    from tokenization import TaikoTokenizer

    tokenizer = TaikoTokenizer()
    scaler = DifficultyScaler(tokenizer)

    original_tokens = ["don", "don", "[EMPTY]", "ka", "ka", "[EMPTY]"]
    original_ids = [tokenizer.vocab[token] for token in original_tokens]
    print("Original Chart:", original_tokens)

    easier_ids = scaler.decrease_difficulty(original_ids, factor=0.5)
    print("Easier Chart:  ", tokenizer.detokenize(easier_ids))

    increase_test_tokens = ["don", "[EMPTY]", "ka", "[EMPTY]", "don", "[EMPTY]"]
    increase_test_ids = [tokenizer.vocab[token] for token in increase_test_tokens]
    print("\nOriginal for Increase:", increase_test_tokens)
    harder_ids = scaler.increase_difficulty(increase_test_ids, factor=0.5)
    print("Harder Chart:  ", tokenizer.detokenize(harder_ids))