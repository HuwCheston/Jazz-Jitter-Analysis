import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pymer4.models import Lmer
from prepare_data import zip_same_conditions_together, generate_df, append_zoom_array, reg_func, average_bpms
from src.visualise.phase_correction_graphs import create_plots


def delay_event_onset_by_latency(df: pd.DataFrame = object, ) -> pd.DataFrame:
    """
    Delays all event onsets in a performance by the amount of latency applied at that point.
    Replicates the performance as it would have been heard by each participant.
    Returns a dataframe with delayed onset and delayed IOI column appended.
    """
    # Need to suppress this warning as it is a false positive
    pd.options.mode.chained_assignment = None
    # Fill na in latency column with next/previous valid observation
    df['lat'] = df['lat'].bfill().ffill()
    # Add the latency column (divided by 1000 to convert from ms to sec) to the onset column
    df['onset_delayed'] = (df['onset'] + (df['lat'] / 1000))
    return df


def nearest_neighbor(live_arr, delayed_arr, live_i: str = '', delayed_i: str = '', drop: bool = True) -> pd.DataFrame:
    """
    For all the events in array 1, pair the event to the nearest unique event in array 2.
    Two events in array 1 may be paired to the same event in array 1, and can be dropped by setting drop to True
    Returns a dataframe of paired events.
    """
    results = []
    # Iterate through all values in our live array
    for i in range(live_arr.shape[0]):
        # Matrix subtraction
        temp_result = abs(live_arr[i] - delayed_arr)
        # Get the minimum value to get closest element
        min_val = np.amin(temp_result)
        closest_element = delayed_arr[np.where(temp_result == min_val)][0]
        # Append live event and closest element to list
        results.append((live_arr[i], closest_element))

    # If we're dropping duplicate values, we should do this now
    if drop:
        return (pd.DataFrame(results, columns=[live_i, delayed_i])
                .groupby(delayed_i)
                .apply(filter_duplicates_from_grouped_df, live=live_i, delayed=delayed_i))
    else:
        return pd.DataFrame(results, columns=[live_i, delayed_i])


def filter_duplicates_from_grouped_df(grp, live, delayed):
    """
    Removes duplicate values. Dataframe should be grouped by events from delayed performer: if an event is duplicated,
    len(group) > 1. In these cases, the pairing with the closest distance is identified, with all others set to NaN.
    """
    grp.loc[grp.index != (grp[live] - grp[delayed]).idxmin(), delayed] = np.nan
    return grp


def format_df_for_phase_correction_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces a dataframe into the format required for the phrase-correction model, including setting required columns.
    """
    # Create our temporary output dataframe
    output_df = pd.DataFrame()
    output_df['live_prev_onset'] = df.iloc[:, 0]
    output_df['delayed_prev_onset'] = df.iloc[:, 1]
    # Shift onsets
    output_df['live_next_onset'] = output_df['live_prev_onset'].shift(periods=1)
    output_df['delayed_next_onset'] = output_df['delayed_prev_onset'].shift(periods=1)
    # Calculate IOIs
    output_df['live_prev_ioi'] = output_df['live_prev_onset'].diff()
    output_df['delayed_prev_ioi'] = output_df['delayed_prev_onset'].diff()
    # Shift IOIs
    output_df['live_next_ioi'] = output_df['live_prev_ioi'].shift(periods=1)
    output_df['delayed_next_ioi'] = output_df['delayed_prev_ioi'].shift(periods=1)
    # Calculate differences between live and delayed performers
    output_df['live_delayed_ioi'] = output_df['live_prev_ioi'] - output_df['delayed_prev_ioi']
    output_df['live_delayed_onset'] = output_df['live_prev_onset'] - output_df['delayed_prev_onset']
    return output_df


def construct_phase_correction_model(df: pd.DataFrame) -> tuple:
    """
    Construct linear phase correction model by predicting next IOI of live musician from previous IOI of live musician
    and IOI difference between live and delayed musician.
    Returns a tuple containing coefficients for all predictors and rsquared value.
    """
    md_onset = smf.ols('live_next_ioi~live_prev_ioi+live_delayed_onset', data=df).fit()
    return *md_onset.params.iloc[1:].values, md_onset.rsquared,


def mixed_effects_model(df: pd.DataFrame):
    for idx, grp in df.groupby('trial'):
        md = Lmer(
            'correction_partner~latency+jitter+instrument+(latency|block)+(jitter|block)', data=grp
        ).fit(
            factors={
                "latency": ["0", "23", "45", "90", "180"],
                "jitter": ["0.0", "0.5", "1.0", ],
                "instrument": ['Keys', 'Drums'],
            }
        )
        print(md)


def pc_live_ioi_delayed_ioi(raw_data, output_dir):
    """
    Creates a phase correction model for all performances.
    """
    # TODO: sort docstring and comments here
    zipped_data = zip_same_conditions_together(raw_data=raw_data)
    res = []
    for z in zipped_data:
        for c1, c2 in z:
            keys = delay_event_onset_by_latency(
                append_zoom_array(
                    generate_df(c1['midi_bpm'], iqr_range=(0.05, 0.95)),
                    c1['zoom_array']
                )
            )
            drms = delay_event_onset_by_latency(
                append_zoom_array(
                    generate_df(c2['midi_bpm'], iqr_range=(0.05, 0.95)),
                    c2['zoom_array']
                )
            )
            keys_nn = format_df_for_phase_correction_model(
                nearest_neighbor(
                    live_arr=keys['onset'].to_numpy(), delayed_arr=drms['onset_delayed'].to_numpy(),
                    live_i=c1['instrument'], delayed_i=c2['instrument']
                )
            )
            drms_nn = format_df_for_phase_correction_model(
                nearest_neighbor(
                    live_arr=drms['onset'].to_numpy(), delayed_arr=keys['onset_delayed'].to_numpy(),
                    live_i=c2['instrument'], delayed_i=c1['instrument']
                )
            )

            averaged_bpms = average_bpms(generate_df(c1['midi_bpm']), generate_df(c2['midi_bpm']))
            tempo_slope = reg_func(averaged_bpms, ycol='bpm_avg', xcol='elapsed').params.iloc[1:].values[0]
            res.append(
                (c1['trial'], c1['block'], c1['latency'], c1['jitter'], c1['instrument'],
                 *construct_phase_correction_model(keys_nn), tempo_slope,)
            )
            res.append(
                (c2['trial'], c2['block'], c2['latency'], c2['jitter'], c2['instrument'],
                 *construct_phase_correction_model(drms_nn), tempo_slope,)
            )
    df = pd.DataFrame(res, columns=['trial', 'block', 'latency', 'jitter', 'instrument', 'correction_self',
                                    'correction_partner', 'rsquared', 'tempo_slope'])
    create_plots(df=df, output_dir=output_dir)
