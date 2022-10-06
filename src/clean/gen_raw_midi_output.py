import os

from src.clean.gen_pm_output import return_formatted_dic_from_filename, return_list_of_trials, \
    return_pm_output, clean_pm_output


def return_list_of_raw_midi_files(input_dir):
    """Iterate through input directory, return formatted dictionary for every raw midi file"""
    fol = f'{input_dir}/avmanip_output/'
    for r, d, f in os.walk(fol, topdown=False):
        for n in f:
            if n.endswith(('.mid', '.MID')) and 'Warm-Up' not in r and 'Delay' not in n:
                yield return_formatted_dic_from_filename(os.path.join(r, n),
                                                         lat_pat=r'- (\d+) (\d+)\\',
                                                         block_pat=fr'Block (\d+)')


def gen_raw_midi_output(input_dir) -> list:
    """Iterates through raw MIDI files in input directory and extracts data (onset, pitch, velocity) using PrettyMIDI"""
    # Get all .MIDI BPM files from our input directory
    f_list = return_list_of_raw_midi_files(input_dir)
    # Format d_list into list of lists, each list corresponding to data from each trial
    t_list = return_list_of_trials(f_list)
    # Create a new list of lists containing metadata & pretty_midi output for every condition in all trials
    pm_output = [[(file, return_pm_output(f=file)) for file in trial] for num, trial in enumerate(t_list, 1)]
    # Clean our pretty_midi output list and return
    clean_pm = [clean_pm_output(i=input_dir, trial=t, dic_name='midi_raw') for t in pm_output]
    return clean_pm
