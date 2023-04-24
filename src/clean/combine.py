"""Combines all outputs together"""

import numpy as np
import collections
import itertools
import pickle
import pandas as pd

# Define the modules we can import from this file in others
__all__ = [
    'combine_output',
    'save_questionnaire'
]


def scale_array(
        array: np.array, latency: float, jitter: float
) -> np.array:
    """
    Function imported from AV-Manip to scale array to given latency and jitter values for each condition
    """

    # Scale the array by the jitter value
    scale = lambda x: (x * jitter)
    scaled_array = scale(array)
    # Transpose the array onto the latency value
    amount_to_subtract = scaled_array.min() - latency
    transpose = lambda x: (x - amount_to_subtract)
    scaled_array = transpose(scaled_array)
    # Round the array to the nearest integer and return
    return np.rint(scaled_array)


def _format_perceptual_data_with_condition_numbers(
        perceptual_data: list, midi_data: list
) -> list:
    """
    Formats perceptual data with extra information, e.g. condition numbers, that we couldn't add earlier
    """

    # Create an empty list to hold all of our results
    res = []
    # Iterate through the raw MIDI data for each duo
    for trial in midi_data:
        # Get the perceptual data for the corresponding duo
        p_t = perceptual_data[trial[0]['trial'] - 1]
        # Append a new list to hold results for this duo
        res.append([])
        # Iterate through MIDI data for each performance by our current duo
        for perf in trial:
            # This line gets the corresponding perceptual data based on duo, session, latency, and jitter values
            # i.e. it matches perceptual data from one condition with raw MIDI data from the same condition
            match = next(
                (item for item in p_t if
                 item["trial"] == perf['trial'] and
                 item['block'] == perf['block'] and
                 item['latency'] == perf['latency'] and
                 item['jitter'] == perf['jitter']),
                # If we can't find a match (e.g. no perceptual data was gathered for this stimuli), this line creates
                # a new dictionary with an empty list for our answers. This will prevent any errors later on.
                {
                    'trial': perf['trial'],
                    'block': perf['block'],
                    'latency': perf['latency'],
                    'jitter': perf['jitter'],
                    'perceptual_answers': []
                }
            )
            # Now, we set our instrument and condition values for this particular condition
            match['instrument'] = perf['instrument']
            match['condition'] = perf['condition']
            # We append the results to the corresponding list. The .copy() prevents weird issues.
            res[trial[0]['trial'] - 1].append(match.copy())
    return res


def combine_output(
        input_dir: str, output_dir: str, zoom_arr=None, dump_pickle: bool = True, **kwargs
) -> list:
    """
    Combines data streams together into list of dictionaries per trial (one dictionary per condition/performer)
    """

    # If we didn't pass our latency array directly, we try and load it here
    if zoom_arr is None:
        zoom_arr = np.genfromtxt(f'{input_dir}/latency_array.csv')
    # Format our perceptual data by adding values we didn't initially have access to when creating it
    kwargs['perceptual'] = _format_perceptual_data_with_condition_numbers(kwargs['perceptual'], kwargs['midi_raw'])
    # Initialise an empty list to hold our results
    raw_data = []
    # Zip all iterables in kwargs together
    for num, values in enumerate(zip(*[a for a in kwargs.values()]), 1):
        # Merge all data into one dictionary
        d = collections.defaultdict(dict)
        for i in itertools.chain(*values):
            results = {k: v for k, v in i.items() if k not in ('trial', 'block', 'condition', 'instrument')}
            d[(i['trial'], i['block'], i['condition'], i['instrument'])].update(results)
        # Convert our single default dict into big list of multiple dictionaries, one per condition/performer
        li = [dict({'trial': k[0], 'block': k[1], 'condition': k[2], 'instrument': k[3]}, **v) for k, v in d.items()]
        # Append the Zoom array used for each condition to every dictionary
        for d in li:
            d['zoom_array'] = scale_array(zoom_arr, latency=d['latency'], jitter=d['jitter'])
        # Pickle the result
        if dump_pickle:
            pickle.dump(li, open(f"{output_dir}/trial_{num}.p", "wb"))
        raw_data.append(li)
    return raw_data


def save_questionnaire(
        xls_path: str, df_list: list
) -> None:
    """
    Saves the questionnaire to the processed data folder
    """

    with pd.ExcelWriter(xls_path) as writer:
        for (n, df) in enumerate(df_list, 1):
            df.to_excel(writer, f'trial{n}')
