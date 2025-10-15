import librosa
import numpy as np

def get_audio_features(audio_path, source_resolution_ms=23.2, frame_duration_ms=100):
    """
    Loads pre-computed audio features from a .npy file and frames them to align
    with the tokenization time steps.

    The original project stored audio features as numpy arrays, not raw audio.
    This function adapts to that format.

    Args:
        audio_path (str): Path to the .npy audio feature file.
        source_resolution_ms (float): The time resolution of each row in the source .npy file.
                                      23.2ms corresponds to 22050Hz / 512 hop_length.
        frame_duration_ms (int): The target duration of each output frame, matching the tokenizer.

    Returns:
        np.ndarray: A numpy array of shape (num_frames, feature_size), where each
                    row is a feature vector for a single time step.
        None: If an error occurs during loading or processing.
    """
    try:
        # 1. Load the pre-computed features from the .npy file
        audio_features = np.load(audio_path)
        if audio_features.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got shape {audio_features.shape}")

        # 2. Calculate how many source steps fit into one of our new frames.
        steps_per_frame = int(round(frame_duration_ms / source_resolution_ms))
        if steps_per_frame == 0: steps_per_frame = 1

        num_source_steps = audio_features.shape[0]
        num_frames = num_source_steps // steps_per_frame

        framed_features = []
        for i in range(num_frames):
            start = i * steps_per_frame
            end = start + steps_per_frame
            chunk = audio_features[start:end]

            # Aggregate the features in the chunk. Mean is a reasonable choice.
            frame = np.mean(chunk, axis=0)
            framed_features.append(frame)

        return np.array(framed_features)

    except Exception as e:
        print(f"Error processing audio feature file {audio_path}: {e}")
        return None
