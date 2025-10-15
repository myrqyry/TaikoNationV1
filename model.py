'''
TaikoNation v1.0
By Emily Halina & Matthew Guzdial
'''

import tflearn
import tensorflow as tf
import numpy as np
import os

# --- Constants ---
# Preprocessing
TEST_DATA_INDICES = [2, 5, 9, 82, 28, 22, 81, 43, 96, 97]
SONG_CHUNK_SIZE = 16
NOTE_CHUNK_SIZE = 12
NOTE_VECTOR_SIZE = 7
SONG_VECTOR_SIZE = 80
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"

# Model Architecture
CONV_FILTERS_1 = 16
CONV_FILTER_SIZE_1 = 3
CONV_FILTERS_2 = 32
CONV_FILTER_SIZE_2 = 3
FC_UNITS_1 = 128
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
OUTPUT_UNITS = 28
LEARNING_RATE = 0.000005
BATCH_SIZE = 1
NUM_EPOCHS = 100

def _load_song_data(song_path):
    """Loads and reshapes the song data from a .npy file."""
    try:
        song_mm = np.load(song_path, mmap_mode="r")
        song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
        song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
        return np.reshape(song_data, song_mm.shape)
    except FileNotFoundError:
        print(f"Error: Song file not found at {song_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading song data: {e}")
        return None

def _load_note_data(chart_path, song_data_len):
    """Loads, reshapes, and pads the note data from a .npy file."""
    try:
        note_mm = np.load(chart_path, mmap_mode="r")
        note_data = np.frombuffer(note_mm, dtype=np.int32, count=-1)
        note_data = np.reshape(note_data, [len(note_mm), NOTE_VECTOR_SIZE])

        # Pad the note data
        diff = song_data_len - len(note_data)
        padding = np.zeros((diff + SONG_CHUNK_SIZE, NOTE_VECTOR_SIZE))
        return np.append(note_data, padding, axis=0)
    except FileNotFoundError:
        print(f"Error: Chart file not found at {chart_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading note data: {e}")
        return None

def _package_data(song_data, note_data, training):
    """Packages the song and note data into training and testing sets."""
    trainX, trainY, testX, testY = [], [], [], []

    for j in range(len(song_data)):
        song_input, note_input, output_chunk = [], [], []
        for k in range(SONG_CHUNK_SIZE):
            is_padding = j - k < 0

            # Song input
            if is_padding:
                song_input.append(np.zeros([SONG_VECTOR_SIZE]))
            else:
                song_input.append(song_data[j - k])

            # Note input
            if k < NOTE_CHUNK_SIZE:
                if is_padding:
                    note_input.append(np.zeros([NOTE_VECTOR_SIZE]))
                else:
                    note_input.append(note_data[j - k])
            elif k != 15:
                note_input.append(np.ones([NOTE_VECTOR_SIZE]))

            # Output chunk
            if k > 11:
                if is_padding:
                    output_chunk.append(np.zeros([NOTE_VECTOR_SIZE]))
                else:
                    output_chunk.append(note_data[j - k])

        input_chunk = np.concatenate([np.array(song_input).flatten(), np.array(note_input).flatten()])
        output_chunk = np.reshape(np.concatenate(output_chunk), [4, NOTE_VECTOR_SIZE])

        if training:
            trainX.append(input_chunk)
            trainY.append(output_chunk)
        else:
            testX.append(input_chunk)
            testY.append(output_chunk)

    return trainX, trainY, testX, testY

def preprocess():
    '''
    This function preprocesses the dataset for use in the model.
    '''
    try:
        charts = os.listdir(path=INPUT_CHART_DIR)
        songs = os.listdir(path=INPUT_SONG_DIR)
    except FileNotFoundError as e:
        print(f"Error: Input directory not found - {e}. Please ensure '{INPUT_CHART_DIR}' and '{INPUT_SONG_DIR}' exist.")
        return [], [], [], []

    trainX, trainY, testX, testY = [], [], [], []

    song_map = {s.split()[0]: s for s in songs}

    for i, chart in enumerate(charts):
        is_training = i not in TEST_DATA_INDICES

        try:
            id_number = chart.split("_")[0]
            if id_number not in song_map:
                print(f"Warning: No matching song found for chart {chart}. Skipping.")
                continue
            song_name = song_map[id_number]
        except IndexError:
            print(f"Warning: Could not parse ID from chart file {chart}. Skipping.")
            continue

        print(f"Processing: {song_name}, {chart}")

        song_data = _load_song_data(os.path.join(INPUT_SONG_DIR, song_name))
        if song_data is None or song_data.shape[1] != SONG_VECTOR_SIZE:
            print(f"Warning: Corrupted or invalid shape for song data {song_name}. Skipping.")
            continue

        note_data = _load_note_data(os.path.join(INPUT_CHART_DIR, chart), len(song_data))
        if note_data is None or note_data.shape[1] != NOTE_VECTOR_SIZE:
            print(f"Warning: Corrupted or invalid shape for note data {chart}. Skipping.")
            continue

        if len(song_data) == 0:
            print(f"Warning: Empty song data for {song_name}. Skipping.")
            continue

        packaged_data = _package_data(song_data, note_data, is_training)
        trainX.extend(packaged_data[0])
        trainY.extend(packaged_data[1])
        testX.extend(packaged_data[2])
        testY.extend(packaged_data[3])

    print(f"{len(trainX)} train X, {len(trainY)} train Y")
    print(f"{len(testX)} test X, {len(testY)} test Y")

    if not trainX or not testX:
        print("Error: No data was successfully processed. Please check your input files and directories.")

    return trainX, trainY, testX, testY

def build_model():
    """Builds and returns the TaikoNation model."""
    net = tflearn.input_data([None, 1385])
    song = tf.slice(net, [0,0], [-1, 1280])
    song = tf.reshape(song, [-1, SONG_CHUNK_SIZE, SONG_VECTOR_SIZE])
    prev_notes = tf.slice(net, [0,1280], [-1, 105])
    prev_notes = tf.reshape(prev_notes, [-1, NOTE_VECTOR_SIZE, 15])

    song_encoder = tflearn.conv_1d(song, nb_filter=CONV_FILTERS_1, filter_size=CONV_FILTER_SIZE_1, activation="relu")
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

    song_encoder = tflearn.conv_1d(song_encoder, nb_filter=CONV_FILTERS_2, filter_size=CONV_FILTER_SIZE_2, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

    song_encoder = tflearn.fully_connected(song_encoder, n_units=FC_UNITS_1, activation="relu")
    song_encoder = tf.reshape(song_encoder, [-1,8,16])

    past_chunks = tf.slice(song_encoder, [0,0,0], [-1, 8, 15])
    curr_chunk = tf.slice(song_encoder, [0,0,15], [-1, 8, 1])

    lstm_input = tf.unstack(past_chunks, axis=1)
    lstm_input = tf.math.multiply(lstm_input, prev_notes)
    lstm_input = tf.reshape(lstm_input, [-1])

    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([8, 15]))
    curr_chunk = tf.reshape(curr_chunk, [-1])
    lstm_input = tf.concat([lstm_input, curr_chunk], 0)

    lstm_input = tf.reshape(lstm_input, [-1, 16, 88])

    lstm_input = tflearn.lstm(lstm_input, LSTM_UNITS_1, dropout=0.8, activation="relu")
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])

    lstm_input = tflearn.lstm(lstm_input, LSTM_UNITS_2, dropout=0.8, activation="relu")

    lstm_input = tflearn.fully_connected(lstm_input, n_units=OUTPUT_UNITS, activation="softmax")
    lstm_input = tflearn.reshape(lstm_input, [-1,4,7])

    network = tflearn.regression(lstm_input, optimizer = "adam", loss="categorical_crossentropy", learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    return tflearn.DNN(network, checkpoint_path="model_rt.tfl")

def main():
    '''
    Main function to preprocess data, build, train, and save the model.
    '''
    trainX, trainY, testX, testY = preprocess()

    model = build_model()

    try:
        model.load("model_rt.tfl")
    except Exception as e:
        print(f"No saved model found. Starting training from scratch. Error: {e}")

    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=BATCH_SIZE, n_epoch=NUM_EPOCHS)
    model.save("model_rt.tfl")

if __name__ == "__main__":
    main()
