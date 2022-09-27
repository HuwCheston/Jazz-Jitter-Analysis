import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from prepare_data import zip_same_conditions_together, generate_df, append_zoom_array, reg_func, average_bpms
from src.visualise.phase_correction_graphs import make_pairgrid, make_single_condition_phase_correction_plot, make_polar, make_single_condition_slope_animation, make_correction_boxplot_by_variable


def delay_heard_onsets_by_latency(df: pd.DataFrame = object, ) -> pd.DataFrame:
    """
    Delays all event onsets in a performance by the amount of latency applied at that point.
    Replicates the performance as it would have been heard by each participant.
    Returns a dataframe with delayed onset and delayed IOI column appended.
    """
    # TODO: need to add in the constant amount of delay induced by the testbed - 12ms?
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


def format_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
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
    output_df['live_delayed_ioi'] = output_df['delayed_prev_ioi'] - output_df['live_prev_ioi']
    output_df['live_delayed_onset'] = output_df['live_prev_onset'] - output_df['delayed_prev_onset']
    return output_df


def construct_phase_correction_model(df: pd.DataFrame, mod: str = 'live_next_ioi~live_prev_ioi+live_delayed_onset'):
    """
    Construct linear phase correction model by predicting next IOI of live musician from previous IOI of live musician
    and IOI difference between live and delayed musician.
    Returns a tuple containing coefficients for all predictors and rsquared value.
    """
    return smf.ols(mod, data=df).fit()


def predict_from_model(md, df: pd.DataFrame) -> pd.DataFrame:
    """
    From a linear regression model, return a dataframe of actual and predicted onset times and IOIs
    """
    pred = pd.Series(np.insert(np.cumsum(md.predict()) + df['live_prev_onset'].loc[0], 0, df['live_prev_onset'].loc[0]))
    df['predicted_onset'] = pred
    df['predicted_ioi'] = pred.diff()
    df['predicted_bpm'] = 60 / df['predicted_ioi']
    df['live_bpm'] = 60 / df['live_prev_ioi']
    df['elapsed'] = np.floor(df['live_prev_onset'])
    return df


def pc_live_ioi_delayed_ioi(raw_data, output_dir, make_anim: bool = False):
    """
    Creates a phase correction model for all performances.
    """
    zipped_data = zip_same_conditions_together(raw_data=raw_data)
    res = []
    nn = []
    # Iterate through each conditions
    for z in zipped_data:
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            # For each performer, create dataframe, append zoom array, and delay partner's onsets
            f1 = lambda c: delay_heard_onsets_by_latency(append_zoom_array(generate_df(c['midi_bpm']), c['zoom_array']))
            keys = f1(c1)
            drms = f1(c2)
            # For each performer, carry out the nearest neighbor algorithm
            f2 = lambda la, da, li, di: format_df_for_model(nearest_neighbor(la, da, li, di))
            keys_nn = f2(keys['onset'].to_numpy(), drms['onset_delayed'].to_numpy(), c1['instrument'], c2['instrument'])
            drms_nn = f2(drms['onset'].to_numpy(), keys['onset_delayed'].to_numpy(), c2['instrument'], c1['instrument'])
            # For each performer, create the phase correction model
            keys_md = construct_phase_correction_model(keys_nn, mod='live_next_ioi~live_prev_ioi+live_delayed_onset')
            drms_md = construct_phase_correction_model(drms_nn, mod='live_next_ioi~live_prev_ioi+live_delayed_onset')
            # Generate output from single condition
            make_single_condition_phase_correction_plot(keys_df=predict_from_model(keys_md, keys_nn), keys_md=keys_md,
                                                        drms_df=predict_from_model(drms_md, drms_nn), drms_md=drms_md,
                                                        meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                                                        output=output_dir, keys_o=keys, drms_o=drms)
            if make_anim:
                make_single_condition_slope_animation(keys_df=predict_from_model(keys_md, keys_nn),
                                                      drms_df=predict_from_model(drms_md, drms_nn),
                                                      keys_o=keys, drms_o=drms, output_dir=output_dir,
                                                      meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']))
            # Append the results
            tempo_slope = reg_func(
                average_bpms(generate_df(c1['midi_bpm']), generate_df(c2['midi_bpm'])), ycol='bpm_avg', xcol='elapsed'
            ).params.iloc[1:].values[0]
            f3 = lambda c, m1: (c['trial'], c['block'], c['latency'], c['jitter'], c['instrument'], tempo_slope,
                                *m1.params.iloc[1:].values,)
            res.append(f3(c1, keys_md))
            res.append(f3(c2, drms_md))
            nn.append((c1['trial'], c1['block'], c1['latency'], c1['jitter'], keys_nn, drms_nn, tempo_slope))
    # Generate outputs from all conditions
    df = pd.DataFrame(res, columns=['trial', 'block', 'latency', 'jitter', 'instrument', 'tempo_slope',
                                    'correction_self_onset', 'correction_partner_onset'])
    make_pairgrid(df=df, output=output_dir, xvar='correction_partner_onset')
    make_polar(nn_list=nn, output_dir=output_dir)
    make_correction_boxplot_by_variable(df=df, output_dir=output_dir)
