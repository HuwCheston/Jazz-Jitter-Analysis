import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from datetime import timedelta
from prepare_data import zip_same_conditions_together, generate_df, append_zoom_array, reg_func, average_bpms
from src.analyse.granger_causality import test_stationary
from src.visualise.phase_correction_graphs import make_pairgrid, make_single_condition_phase_correction_plot, \
    make_polar, make_single_condition_slope_animation, make_correction_boxplot_by_variable


PC_MOD = 'live_next_ioi~live_prev_ioi+live_delayed_onset'


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


def construct_static_phase_correction_model(df: pd.DataFrame, mod: str = PC_MOD):
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


def construct_rolling_phase_correction_model(nn: pd.DataFrame, orig: pd.DataFrame, mod: str = PC_MOD,
                                             win_size: timedelta = timedelta(seconds=8)) -> pd.DataFrame:
    """
    Construct a rolling phase correction model
    """
    # Create the rolling phase correction function
    def process(x):
        try:
            reg = smf.ols(mod, data=x).fit()
        # Catch errors if we only have nans and return a NaN coefficient
        except (IndexError, ValueError) as _:
            return np.nan
        else:
            return reg.params['live_delayed_onset']

    # Merge the original dataframe with the nearest neighbour model
    merge = pd.merge(nn, orig.rename(columns={'onset': 'live_prev_onset'}), on='live_prev_onset')
    # Create the timestamp column for rolling on
    merge['td'] = pd.to_timedelta([timedelta(seconds=val) for val in merge.live_prev_onset.values])
    # Merge the dataframe of coefficients back onto the original and subset for required columns
    df = merge.join(
        pd.DataFrame([process(x) for x in merge.rolling(window=win_size, on='td')], columns=['correction_partner'])
    )
    # Roll latency column to get standard deviation
    df['lat'] = df.rolling(window=win_size, on='td')['lat'].std()
    # Test correction to partner for stationarity (we know latency variation is always stationary so no need to test)
    df['correction_partner'] = test_stationary(df['correction_partner'])
    # TODO: subset this data, to only include values for after window is full (e.g. from 16 seconds on?)
    return df[['td', 'live_prev_onset', 'lat', 'correction_partner']]


def construct_correction_jitter_model(df: pd.DataFrame, max_lag: int = 4):
    """
    Construct a linear regression model for predicting variation in phase correction from variation in latency.
    Maximum lag for autoregression defaults to four seconds, which was found to create the best-fitting models in
    testing, however this can be overridden by setting max_lag argument.
    """
    # Define the function for shifting values by a maximum number of seconds
    shifter = lambda s: pd.concat([df[s].shift(i).rename(f'{s}_{i}') for i in range(1, max_lag+1)], axis=1).set_index(
        df.index)
    # Create the data for use in the regression
    train = pd.concat([df, shifter('lat'), shifter('correction_partner'), ], axis=1)
    md = 'correction_partner~' + '+'.join(
        [f'{col}' for col in train.columns if col not in ['td', 'live_prev_onset', 'correction_partner']])
    res = smf.ols(md, train).fit()
    return res


def pc_live_ioi_delayed_ioi(raw_data, output_dir, make_anim: bool = False):
    """
    Creates a phase correction model for all performances.
    """
    output_dir += '\\figures\\phase_correction_plots'
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
            # For each performer, create a static phase correction model (using all the data)
            keys_md = construct_static_phase_correction_model(keys_nn,)
            drms_md = construct_static_phase_correction_model(drms_nn,)
            if c1['jitter'] != 0:
                # For each performer, create a moving phase correction model (using a rolling window, defaults to 8sec)
                keys_roll = construct_rolling_phase_correction_model(keys_nn, keys)
                drms_roll = construct_rolling_phase_correction_model(drms_nn, drms)

            # Generate output from single condition
            make_single_condition_phase_correction_plot(keys_df=predict_from_model(keys_md, keys_nn), keys_md=keys_md,
                                                        drms_df=predict_from_model(drms_md, drms_nn), drms_md=drms_md,
                                                        meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                                                        keys_o=keys, drms_o=drms,
                                                        output_dir=output_dir + '\\individual')
            if make_anim:
                make_single_condition_slope_animation(keys_df=predict_from_model(keys_md, keys_nn),
                                                      drms_df=predict_from_model(drms_md, drms_nn),
                                                      keys_o=keys, drms_o=drms,
                                                      meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                                                      output_dir=output_dir + '\\animations\\tempo_slope')
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
    make_pairgrid(df=df, xvar='correction_partner_onset', output_dir=output_dir + '\\grouped')
    make_polar(nn_list=nn, output_dir=output_dir + '\\grouped')
    make_correction_boxplot_by_variable(df=df, output_dir=output_dir + '\\grouped')
