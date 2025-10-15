import google.generativeai as genai
import numpy as np
import sys
import os
import random
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands
from collections import deque
from zipfile import ZipFile
import csv
import json

from chop_song import process_song

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)


def main():
    '''
    process the given song, analyze it, then create a chart for it
    Usage: gemini_output.py song_file.mp3 "prompt"
    '''
    if len(sys.argv) < 3:
        print("Usage: gemini_output.py song_file.mp3 \"prompt\"")
        return

    dir_name = sys.argv[1] + " chunks"
    prompt = sys.argv[2]

    songfile = sys.argv[1]
    song = sys.argv[1][:len(sys.argv[1]) - 4]
    npy_file = song + " Input.npy"
    outfile = song + " TaikoNation.osu"
    package = song + " TaikoNation.osz"
    dir_list = os.listdir()

    if dir_name not in dir_list:
        print("Chopping up your song..")
        process_song(sys.argv[1])
    else:
        print("Song already chopped, moving on..")

    print("Analyzing your song..")
    analyze_song(npy_file, dir_.name)

    print("Making predictions..")
    make_predictions(npy_file, outfile, prompt)

    print("Packing up your beatmap..")
    create_osz(songfile, outfile, package)
    print("All done!", package, "created in current directory.")
    return

def create_analyzers(fs=44100.0,
                     nhop=1024,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    '''
    create analyzer from DDC, adapted to TaikoNation
    https://arxiv.org/abs/1703.06891
    '''
    analyzers = []
    for nfft in nffts:
        window = Windowing(size=nfft, type='blackmanharris62')
        spectrum = Spectrum(size=nfft)
        mel = MelBands(inputSize=(nfft // 2) + 1,
                       numberBands=mel_nband,
                       lowFrequencyBound=mel_freqlo,
                       highFrequencyBound=mel_freqhi,
                       sampleRate=fs)
        analyzers.append((window, spectrum, mel))
    return analyzers[0][0], analyzers[0][1], analyzers[0][2]

def analyze_song(file_name = None, dir_name = None):
        '''
        write something here
        '''
        file_list = os.listdir()
        for f in file_list:
            if f == file_name:
                print("Song already has been processed, exiting processing..")
                #os.chdir(cwd)
                return

        cwd = os.getcwd()
        if dir_name != None:
            new_dir = cwd + "/" + dir_name
            os.chdir(new_dir)

        file_list = os.listdir()
        window, spectrum, mel = create_analyzers()
        feats_list = []
        i = 0

        for fn in file_list:
            if fn[len(fn) - 1] != 'v':
                continue
            try:
                loader = MonoLoader(filename=fn, sampleRate=44100.0)
                samples = loader()
                feats = window(samples)
                if len(feats) % 2 != 0:
                    feats = np.delete(feats, random.randint(0, len(feats) - 1))
                feats = spectrum(feats)
                feats = mel(feats)
                feats_list.append(feats)
                i+=1
            except Exception as e:
                feats_list.append(np.zeros(80, dtype=np.float32))
                i += 1

        # Apply numerically-stable log-scaling
        feats_list = np.array(feats_list)
        feats_list = np.log(feats_list + 1e-16)
        print(len(feats_list), "length of feats list")
        print(type(feats_list[0][0]))
        if dir_name != None:
            os.chdir(cwd)
        np.save(file_name, feats_list)
        return

def make_predictions(npy_file=None, outfile=None, prompt=None):
    '''
    Makes note predictions using the given song data (npy_file), then calls create_chart to... create the chart!
    '''
    # load the song data from memory map and reshape appropriately
    song_mm = np.load(npy_file, mmap_mode="r")
    song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
    song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
    song_data = np.reshape(song_data, song_mm.shape)

    # Convert song data to a list of lists for JSON serialization
    song_data_list = song_data.tolist()

    # Create the prompt for Gemini
    gemini_prompt = f"""
    You are an expert rhythm game chart creator. Your task is to create a Taiko chart for a given song.
    The song is represented as a series of Mel-frequency cepstral coefficients (MFCCs).
    The user has provided the following prompt to guide the chart generation: '{prompt}'
    Based on the song data and the prompt, generate a sequence of notes for the chart.

    The output should be a JSON object with a single key "notes" which is a list of integers.
    Each integer represents a note type:
    0: No note
    1: Red note (don)
    2: Blue note (kat)
    3: Big red note (DON)
    4: Big blue note (KAT)

    Here is the song data:
    {json.dumps(song_data_list)}
    """

    # Call the Gemini API
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(gemini_prompt)

    # Process the response
    try:
        response_text = response.text.replace("```json", "").replace("```", "")
        note_data = json.loads(response_text)
        note_selections = note_data["notes"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing Gemini response: {e}")
        print(f"Response text: {response.text}")
        return

    create_chart(note_selections, outfile)
    return

def create_chart(note_selections, file_name="outfile.osu"):
    '''
    Create the .osu file based on the note selections
    '''
    # template for beginning of file
    osu_file = """osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: 0
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 0.8
BeatDivisor: 4
GridSize: 32
TimelineZoom: 3.14

[Metadata]
Title:SongTitle
TitleUnicode:SongTitle
Artist:ArtistName
ArtistUnicode:ArtistName
Creator:TaikoNation
Version:TaikoNation v1
Source:
Tags:
BeatmapID:-1
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:2
OverallDifficulty:6
ApproachRate:10
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,368,4,1,0,40,1,0


[HitObjects]
"""
    current_ms = 0
    last_note_active = False
    outfile = open(file_name, "w+")
    # add each note to the string with its corresponding time
    for note in note_selections:
        if note == 1 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,0,0:0:0:0:\n")
            last_note_active = True
        elif note == 2 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,2,0:0:0:0:\n")
            last_note_active = True
        elif note == 3 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,4,0:0:0:0:\n")
            last_note_active = True
        elif note == 4 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,6,0:0:0:0:\n")
            last_note_active = True
        else:
            last_note_active = False
        current_ms += 23
    outfile.write(osu_file)
    return

def create_osz(songfile, outfile, package):
    '''
    Package the song .mp3 (songfile) and .osu chart data (outfile) into a single .osz which can be dragged into osu for instant use (package)
    '''
    # set up
    temp_name = songfile
    os.rename(songfile, "audio.mp3")
    # zip up
    with ZipFile(package, mode="w") as oszf:
        oszf.write("audio.mp3")
        oszf.write(outfile)
    # clean up
    os.rename("audio.mp3", temp_name)
    os.remove(outfile)
    return
main()
