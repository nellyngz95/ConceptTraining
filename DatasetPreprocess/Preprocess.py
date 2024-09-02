import librosa
import torch
import torchaudio
import numpy as np
import os
import sys

# Script to down sample, normalize, and convert the .wav files to .pt files for any dataset.
# The dataset is a folder with .wav files. The script will downsample the files to 16000 Hz, normalize the audio, and save the files as .wav files.

print("Preprocessing audio files... 5 seconds sr=4410, bps=16000, normalize=True, mono=True")

# Path to the folder containing the .wav files
path = "/Users/nellygarcia/Downloads/WhooshSynth"

# Description of the modified files:
bps = 16000   # Bits per second (bitrate for saving the file)
sr = 44100    # Original sampling rate
duration = 5  # Duration of the file in seconds
normalize = True
channels = 1
c = 0  # Counter for the number of files

# Change the current working directory to the path where your audio files are located
os.chdir(path)

# Reading the files in the directory
for file_name in os.listdir(os.getcwd()):
    # Process only .wav or .aiff files
    if file_name.endswith(".wav") or file_name.endswith(".aiff"):
        # Load the audio file
        y, s = librosa.load(file_name, sr=sr)
        
        # Fix the length of the audio to match the specified duration
        target_length = int(duration * sr)
        if len(y) > target_length:
            y = y[:target_length]  # Trim the audio to the target length
        else:
            y = librosa.util.fix_length(y, size=target_length)

        # Normalize the audio
        if normalize:
            y = librosa.util.normalize(y)
        
        # Ensure the audio is single channel
        if y.ndim > 1:
            y = y[0]
        
        # Convert audio to 16-bit PCM format (bit depth)
        y = (y * 32767).astype(np.int16)  # Convert to 16-bit integer format
        c += 1
        print("Original file:", file_name)

            # New name for the file before the .wav
        new_name = "JetSynth-" + str(c) + "-8.wav"

        waveform = torch.tensor(y).unsqueeze(0)  # Add channel dimension for torchaudio
        torchaudio.save(new_name, waveform, sample_rate=s, bits_per_sample=16)
        print("File saved with new name:", new_name)

        # Optionally, delete the original file if it's no longer needed
        os.remove(file_name)
        print("Original file deleted:", file_name)
    

print("All files saved")
