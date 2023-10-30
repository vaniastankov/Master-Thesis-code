import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import sys
import os
from pedalboard import Pedalboard, Distortion, Bitcrush, Resample
from pedalboard.io import AudioFile
import pyroomacoustics as pra
from multiprocessing import Pool
import random
import IPython
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import scipy.io.wavfile

sr = 48000
udb = (-120, 10)
uflim = (20, 20000)
# So you found an absolute path...
plot_directory = "/Users/ivanstankov/Documents/Study/thesis_code/imgs/"


def load_audio(path, sr):
    """
    This function is used to read audio file into array.
    It also allows user to read audio data with target samplerate and convert stereo signal to mono.
    """
    # We use a popular audio tool called Librosa.
    data, _ = librosa.load(path, sr=sr, mono=True)
    data = librosa.util.normalize(data) 
    return data


def save_audio(path, data, sr):
    """
    This function is used to save audio data into audio file.
    """
    scipy.io.wavfile.write(path, sr, data)