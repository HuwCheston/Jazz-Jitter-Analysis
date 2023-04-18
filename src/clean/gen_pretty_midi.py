"""Generates MIDI data in Python-compatible form and cleans"""

import pandas as pd
import numpy as np
import pretty_midi
import os
import re
import collections

# Define the modules we can import from this file in others
__all__ = [
    'gen_raw_midi_output', 'gen_pm_output'
]

# Define constants
DURATION = 93   # Duration of the jitter array applied in each performance
START_POS = 7     # Defaults to 7 seconds (rather than 8) to include any notes played just before the count-in ends
STOP_POS = START_POS + DURATION + 1     # Start position plus duration, plus 1


def return_formatted_dic_from_filename(file, lat_pat=r'- (\d+) (\d+) - ', block_pat=fr'block_(\d+)') -> dict:
    """Returns a dictionary containing file metadata, e.g. condition number, jitter, instrument..."""
    latency, jitter = re.search(lat_pat, file).groups()
    return {
            'trial': int(re.search(fr'trial_(\d+)', file).group(1)),
            'block': int(re.search(block_pat, file).group(1)),
            'condition': int(re.search(fr'Condition (\d+)', file).group(1)),
            'latency': int(latency),
            'jitter': float(jitter[:1] + '.' + jitter[1:]),
            'instrument': 'Keys' if 'KEYS' in file.upper() else 'Drums',
            'filename': file,
    }


def return_list_of_files(input_dir) -> list:
    """Iterate through all folders in input directory and append MIDI files to list"""
    fol = f'{input_dir}/midi_bpm_cleaning/'
    return [os.path.join(r, n) for r, d, f in os.walk(fol, topdown=False) for n in f if n.endswith(('.mid', '.MID'))]


def return_list_of_trials(f_list) -> list:
    """Returns list of lists corresponding to each trial"""
    result = collections.defaultdict(list)
    for f in f_list:
        result[f['trial']].append(f)
    return list(result.values())


def return_pm_output(f) -> list:
    """Loads a file into PrettyMIDI, then returns a dataframe ready for cleaning"""
    # Load the file into PrettyMIDI
    pm = pretty_midi.PrettyMIDI(f['filename'])
    # Extract onsets, MIDI mappings, note velocities from PrettyMIDI and return as list of tuples
    return get_data_from_pm_object(pm)


def get_data_from_pm_object(pm) -> list:
    """Returns a flat list of onset start positions, midi pitch numbers, and velocity from a PrettyMIDI object"""
    # Define function to get note onset, pitch, and velocity from midi instrument
    get_data = lambda i: [(note.start, note.pitch, note.velocity) for note in i.notes]
    # Define variable to be assigned later
    data = None
    # We need to treat the midi object differently depending on number of instruments/voices
    if len(pm.instruments) == 1:
        # If there is only one instrument in the PrettyMIDI object, we can access notes easily
        data = get_data(pm.instruments[0])
    elif len(pm.instruments) > 1:
        # If there are multiple instruments in the midi file, we need to get notes for all of them as a list of lists
        biglist = [get_data(instrument) for instrument in pm.instruments]
        # Then we can flatten the list of lists to get all the notes in one list
        data = [item for sublist in biglist for item in sublist]
    return data


def clean_pm_output(i: str, trial: list, dic_name: str = 'midi_bpm') -> list:
    """Clean raw prettymidi output: truncate start and stop times, map midi notes onto musical notes..."""
    # Get midi mappings for each instrument as dictionary
    get_map = lambda s: pd.read_csv(os.path.normpath(f"{i}/{s}_midi_mapping.csv"), header=None, index_col=0).squeeze("columns").to_dict()
    keys_map = get_map('keys')
    drums_map = get_map('drums')
    # Iterate through all conditions and add clean data as key to dictionary
    l1 = []
    for (m, d) in trial:
        o, p, v = zip(*d)
        mapping = keys_map if m['instrument'] == 'Keys' else drums_map
        onsets = tuple(onset for onset in o if START_POS < onset < STOP_POS)
        pitches = tuple(mapping[pitch] for pitch in p)
        m[dic_name] = np.array([(a, b, c) for a, b, c in zip(onsets, pitches, v)], dtype=np.dtype('object'))
        l1.append(m)
    # Sort according to block, condition (natsort used so that true numerical ascending order used, not 10 before 2 etc)
    return sorted(l1, key=lambda k: (k['block'], k['condition'],))


def gen_pm_output(input_dir, **kwargs) -> list:
    """Iterates through MIDI BPM files in input directory and extracts data (onset, pitch, velocity) using PrettyMIDI"""
    midi_mapping_fpath = kwargs.get('midi_mapping_fpath', input_dir)
    # Get all .MIDI BPM files from our input directory
    f_list = return_list_of_files(input_dir)
    # For each .MIDI file, extract metadata from filename - trial, condition, block number, amount of latency...
    d_list = [return_formatted_dic_from_filename(file) for file in f_list]
    # Format d_list into list of lists, each list corresponding to data from each trial
    t_list = return_list_of_trials(d_list)
    # Create a new list of lists containing metadata & pretty_midi output for every condition in all trials
    pm_output = [[(file, return_pm_output(f=file)) for file in trial] for num, trial in enumerate(t_list, 1)]
    # Clean our pretty_midi output list and return
    clean_pm = [clean_pm_output(i=midi_mapping_fpath, trial=t) for t in pm_output]
    return clean_pm


def return_list_of_raw_midi_files(input_dir):
    """Iterate through input directory, return formatted dictionary for every raw midi file"""
    fol = f'{input_dir}/avmanip_output/'
    for r, d, f in os.walk(fol, topdown=False):
        for n in f:
            if n.endswith(('.mid', '.MID')) and 'Warm-Up' not in r and 'Delay' not in n:
                yield return_formatted_dic_from_filename(
                  os.path.join(r, n),
                  lat_pat=r'- (\d+) (\d+)',
                  block_pat=fr'Block (\d+)'
                )


def gen_raw_midi_output(
        input_dir, **kwargs
) -> list:
    """
    Iterates through raw MIDI files in input directory and extracts data (onset, pitch, velocity) using PrettyMIDI
    """

    midi_mapping_fpath = kwargs.get('midi_mapping_fpath', input_dir)
    # Get all .MIDI BPM files from our input directory
    f_list = return_list_of_raw_midi_files(input_dir)
    # Format d_list into list of lists, each list corresponding to data from each trial
    t_list = return_list_of_trials(f_list)
    # Create a new list of lists containing metadata & pretty_midi output for every condition in all trials
    pm_output = [[(file, return_pm_output(f=file)) for file in trial] for num, trial in enumerate(t_list, 1)]
    # Clean our pretty_midi output list and return
    clean_pm = [clean_pm_output(i=midi_mapping_fpath, trial=t, dic_name='midi_raw') for t in pm_output]
    return clean_pm
