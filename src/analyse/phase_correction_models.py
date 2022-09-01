import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from prepare_data import zip_same_conditions_together, generate_df, append_zoom_array, reg_func


def delay_event_onset_by_latency(df: pd.DataFrame) -> pd.DataFrame:
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
    output_df['live_prev_ioi'] = df.iloc[:, 0].diff()
    output_df['live_next_ioi'] = output_df['live_prev_ioi'].shift(periods=1)
    output_df['delayed_prev_ioi'] = df.iloc[:, 1].diff()
    output_df['live_delayed_ioi'] = output_df['delayed_prev_ioi'] - output_df['live_prev_ioi']
    return output_df


def construct_phase_correction_model(df: pd.DataFrame) -> tuple:
    """
    Construct linear phase correction model by predicting next IOI of live musician from previous IOI of live musician
    and IOI difference between live and delayed musician.
    Returns a tuple of coefficients for both predictors and rsquared value.
    """
    md = smf.ols('live_next_ioi~live_prev_ioi+live_delayed_ioi', data=df).fit()
    return *md.params.iloc[1:].values, md.rsquared


def plot_phase_correction(df, output):
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex='all', sharey='all', figsize=(15, 5))
    # We do a double for loop here to get separate x and y values to subset axes object with
    for y, (_, g) in enumerate(df.groupby('trial')):
        # Extremely fucking hacky way of iterating in order of keys-drums, rather than drums-keys (default with groupby)
        for x, (idx, grp) in enumerate(reversed(tuple(g.groupby('instrument')))):
            means = grp.groupby('latency').mean()
            c = "#1f77b4" if idx == 'Keys' else '#ff7f0e'
            sns.barplot(data=means, x=means.index, y='live_delay_diff_coeff', ax=ax[x, y], color=c)
            sns.swarmplot(data=grp, x='latency', y='live_delay_diff_coeff', ax=ax[x, y], s=2, color='#000000')
            ax[x, y].tick_params(axis='both', which='both', bottom=False, left=False,)
            ax[x, y].set_xlabel('')
            ax[x, y].set_ylabel(f'{idx}' if y == 0 else '', rotation=90)
            ax[x, y].set_title(f'Duo {y + 1}' if x == 0 else '')
    fig.supylabel('Phase Correction Coefficient')
    fig.supxlabel('Latency (ms)')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'{output}\\figures\\phase_correction_by_latency.png')


def pc_live_ioi_delayed_ioi(raw_data, output_dir):
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
            res.append((c1['trial'], c1['block'], c1['latency'], c1['jitter'], c1['instrument'],
                        *construct_phase_correction_model(keys_nn), reg_func(df=keys, xcol='bpm', ycol='onset').params.iloc[1:].values[0]))
            res.append((c2['trial'], c2['block'], c2['latency'], c2['jitter'], c2['instrument'],
                        *construct_phase_correction_model(drms_nn), reg_func(df=drms, xcol='bpm', ycol='onset').params.iloc[1:].values[0]))
    df = pd.DataFrame(res, columns=['trial', 'block', 'latency', 'jitter', 'instrument',
                                    'live_prev_ioi_coeff', 'live_delay_diff_coeff', 'rsquared', 'tempo_slope'])

    fig, ax = plt.subplots(ncols=5, nrows=1, sharex='all', sharey='all')
    for num, (idx, grp) in enumerate(df.groupby('trial')):
        md = smf.ols('tempo_slope~live_delay_diff_coeff', data=grp).fit()
        b, m = md.params
        ax[num].scatter(grp['live_delay_diff_coeff'], grp['tempo_slope'])
        ax[num].axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')
    plt.tight_layout()
    plt.show()
