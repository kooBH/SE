import glob,os
import librosa as rs
import soundfile as sf



def convert_16kHz(input_path, output_path):
    audio,rs = rs.load(input_path, sr=None)