import os
import pickle


def load_data(input_filepath):
    """Loads all pickled data from the processed data folder"""
    return [pickle.load(open(f'{input_filepath}\\{f}', "rb")) for f in os.listdir(input_filepath) if f.endswith('.p')]
