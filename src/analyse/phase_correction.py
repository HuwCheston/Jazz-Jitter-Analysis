import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from datetime import timedelta
from itertools import chain
import warnings

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
from src.visualise.phase_correction_graphs import make_pairgrid, make_single_condition_phase_correction_plot, \
    make_single_condition_slope_animation, make_correction_boxplot_by_variable, make_lagged_pointplot, make_pairgrid_iqr

PC_MOD = 'live_next_ioi~live_prev_ioi+live_delayed_onset'
WINDOW_SIZE = 6
ROLLING_LAG = 4


def delay_heard_onsets_by_latency(
        df: pd.DataFrame = object
) -> pd.DataFrame:
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


def nearest_neighbor(
        live_arr, delayed_arr, live_i: str = '', delayed_i: str = '', drop: bool = True
) -> pd.DataFrame:
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


def filter_duplicates_from_grouped_df(
        grp, live, delayed
) -> pd.DataFrame.groupby:
    """
    Removes duplicate values. Dataframe should be grouped by events from delayed performer: if an event is duplicated,
    len(group) > 1. In these cases, the pairing with the closest distance is identified, with all others set to NaN.
    """

    grp.loc[grp.index != (grp[live] - grp[delayed]).idxmin(), delayed] = np.nan
    return grp


def format_df_for_model(
        df: pd.DataFrame
) -> pd.DataFrame:
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


def construct_static_phase_correction_model(
        df: pd.DataFrame, mod: str = PC_MOD
):
    """
    Construct linear phase correction model by predicting next IOI of live musician from previous IOI of live musician
    and IOI difference between live and delayed musician.
    Returns a tuple containing coefficients for all predictors and rsquared value.
    """

    return smf.ols(mod, data=df).fit()


def predict_from_model(
        md: smf.ols, df: pd.DataFrame
) -> pd.DataFrame:
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


def construct_rolling_phase_correction_model(
        nn: pd.DataFrame, orig: pd.DataFrame, mod: str = PC_MOD, sub: bool = True,
        win_size: timedelta = timedelta(seconds=8)
) -> pd.DataFrame:
    """
    Construct a rolling phase correction model
    """

    # Create the rolling phase correction function
    def process(x):
        try:
            warnings.filterwarnings("ignore")
            reg = smf.ols(mod, data=x).fit()
            return reg.params['live_delayed_onset'], reg.aic, reg.rsquared
        # Catch errors if we only have nans and return a NaN coefficient
        except (IndexError, ValueError) as _:
            return np.nan, np.nan, np.nan

    # Merge the original dataframe with the nearest neighbour model
    merge = pd.merge(nn, orig.rename(columns={'onset': 'live_prev_onset'}), on='live_prev_onset')
    # Create the timestamp column for rolling on
    merge['td'] = pd.to_timedelta([timedelta(seconds=val) for val in merge.live_prev_onset.values])
    # Merge the dataframe of coefficients back onto the original and subset for required columns
    df = merge.join(
        pd.DataFrame([process(x) for x in merge.rolling(window=win_size, on='td')],
                     columns=['correction_partner', 'aic', 'r2'])
    )
    # Roll latency column to get standard deviation
    df['lat'] = df.rolling(window=win_size, on='td')['lat'].std()
    df['ioi'] = df.rolling(window=win_size, on='td')['live_prev_ioi'].std() * 100
    # Test correction to partner for stationarity (we know latency variation is always stationary so no need to test)
    df['correction_partner'] = df['correction_partner']
    # TODO: subset this data using arg 'sub', to only include values for after window is full (e.g. from 16 seconds on?)
    return df[['td', 'live_prev_onset', 'lat', 'correction_partner', 'ioi', 'aic', 'r2']]


def construct_correction_jitter_model(
        df: pd.DataFrame, max_lag: int = ROLLING_LAG, var: str = 'correction_partner'
):
    """
    Construct a linear regression model for predicting variation in phase correction from variation in latency.
    Maximum lag for autoregression defaults to four seconds, which was found to create the best-fitting models in
    testing, however this can be overridden by setting max_lag argument.
    """

    # Define the function for shifting values by a maximum number of seconds
    shifter = lambda s: pd.concat([df[s].shift(i).rename(f'{s}__{i}') for i in range(1, max_lag+1)], axis=1).set_index(
        df.index)
    # Create the data for use in the regression
    train = pd.concat([df, shifter('lat'), shifter(var)], axis=1)
    md = f'{var}~' + '+'.join(
        [f'{col}' for col in train.columns if col[-3:-1] == '__'])
    res = smf.ols(md, train).fit()
    return res.params.iloc[1: max_lag + 1].values


def phase_correction_pre_processing(
        c1: dict, c2: dict
) -> tuple:
    """
    Carries out preprocessing necessary for creating a phase correction model
    """

    # Create dataframe, append zoom array, and delay partner's onsets
    f1 = lambda c: delay_heard_onsets_by_latency(
        autils.append_zoom_array(autils.generate_df(c['midi_bpm']), c['zoom_array'])
    )
    keys, drms = f1(c1), f1(c2)
    # Carry out the nearest neighbor algorithm
    f2 = lambda la, da, li, di: format_df_for_model(nearest_neighbor(la, da, li, di))
    keys_nn = f2(keys['onset'].to_numpy(), drms['onset_delayed'].to_numpy(), c1['instrument'], c2['instrument'])
    drms_nn = f2(drms['onset'].to_numpy(), keys['onset_delayed'].to_numpy(), c2['instrument'], c1['instrument'])
    # Extract tempo slope
    tempo_slope = autils.reg_func(
        autils.average_bpms(
            autils.generate_df(c1['midi_bpm']), autils.generate_df(c2['midi_bpm'])
        ), ycol='bpm_avg', xcol='elapsed'
    ).params.iloc[1:].values[0]
    return keys, drms, keys_nn, drms_nn, tempo_slope


def gen_static_phase_correction_models(
    raw_data: list, output_dir: str, make_anim: bool = False, make_single_plot: bool = False,
) -> list:
    """
    Generates a static phase correction model, using all of the data in a performance
    """

    figures_output_dir = output_dir + '\\figures\\static_phase_correction_plots'
    zipped_data = autils.zip_same_conditions_together(raw_data=raw_data)
    static_mds = []

    # Iterate through each conditions
    for z in zipped_data:
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            # Pre-processing
            keys, drms, keys_nn, drms_nn, tempo_slope = phase_correction_pre_processing(c1, c2)
            # Create models (can't be done in a loop, as we need to refer to them later)
            keys_static = construct_static_phase_correction_model(keys_nn)
            drms_static = construct_static_phase_correction_model(drms_nn)
            static_mds.append((c1['trial'], c1['block'], c1['latency'], c1['jitter'], c1['instrument'], tempo_slope,
                               *keys_static.params.iloc[1:].values,
                               *chain.from_iterable(keys_static.conf_int(cols=None).iloc[-1:].values.tolist())))
            static_mds.append((c2['trial'], c2['block'], c2['latency'], c2['jitter'], c2['instrument'], tempo_slope,
                               *drms_static.params.iloc[1:].values,
                               *chain.from_iterable(drms_static.conf_int(cols=None).iloc[-1:].values.tolist())))
            # Generate output from single condition
            if make_single_plot:
                make_single_condition_phase_correction_plot(
                    keys_df=predict_from_model(keys_static, keys_nn), keys_md=keys_static, keys_o=keys,
                    drms_df=predict_from_model(drms_static, drms_nn), drms_md=drms_static, drms_o=drms,
                    meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                    output_dir=figures_output_dir + '\\individual_plots'
                )
            # Generate animation if specified
            if make_anim:
                make_single_condition_slope_animation(
                    keys_df=predict_from_model(keys_static, keys_nn), drms_df=predict_from_model(drms_static, drms_nn),
                    keys_o=keys, drms_o=drms, meta=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                    output_dir=figures_output_dir + '\\animations\\tempo_slope'
                )
    return static_mds


def gen_static_model_outputs(
        static_mds: list[tuple], output_dir: str
) -> None:
    """
    Generate output from rolling phase correction models
    """

    figures_output_dir = output_dir + '\\figures\\static_phase_correction_plots'
    static_df = pd.DataFrame(static_mds, columns=['trial', 'block', 'latency', 'jitter', 'instrument', 'tempo_slope',
                                                  'correction_self_ioi', 'correction_partner',
                                                  'correction_partner_ub', 'correction_partner_lb'])
    vutils.output_regression_table(
            mds=autils.create_model_list(df=static_df, md=f'correction_partner~C(latency)+C(jitter)+C(instrument)',
                                         avg_groupers=['latency', 'jitter', 'instrument']),
            output_dir=figures_output_dir, verbose_footer=False
        )
    make_correction_boxplot_by_variable(df=static_df, output_dir=figures_output_dir,
                                        yvar='correction_partner')
    make_pairgrid(df=static_df, xvar='correction_partner', output_dir=figures_output_dir)
    # Outputs using confidence intervals
    make_pairgrid_iqr(df=static_df, value_vars=[col for col in static_df if 'correction_partner' in col],
                      value_name='correction_to_partner', output_dir=figures_output_dir, xvar='correction_to_partner')


def gen_rolling_phase_correction_models(
        raw_data: list, jitter_vs_correction: bool = True,
) -> list[list]:
    """
    Generates a rolling phase correction model
    """
    zipped_data = autils.zip_same_conditions_together(raw_data=raw_data)
    rolling_mds = []
    # Iterate through each condition
    for z in zipped_data:
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            keys, drms, keys_nn, drms_nn, tempo_slope = phase_correction_pre_processing(c1, c2)
            # Iterate through data for each performer separately
            for nn, orig, meta in [(keys_nn, keys, c1), (drms_nn, drms, c2)]:
                # Create models
                correct = construct_rolling_phase_correction_model(nn, orig, win_size=timedelta(seconds=WINDOW_SIZE))
                # Append metadata and correction upper/lower bounds, median
                t = [meta['trial'], meta['block'], meta['latency'], meta['jitter'], meta['instrument'], tempo_slope,
                     correct['correction_partner'].quantile(0.25), correct['correction_partner'].median(),
                     correct['correction_partner'].quantile(0.75)]
                # If we're extracting the effects of jitter on correction, and jitter was actually applied
                if jitter_vs_correction is True and meta['jitter'] != 0:
                    t.extend(construct_correction_jitter_model(correct, var='ioi', max_lag=ROLLING_LAG))
                # Else, add NaNs to make it easier when creating the dataframe later
                else:
                    t.extend([np.nan * ROLLING_LAG])
                # Append all the results to our list
                rolling_mds.append(t)
    return rolling_mds


def gen_rolling_model_outputs(
        rolling_mds: list[list], output_dir: str
) -> None:
    """
    Generate output from rolling phase correction models
    """

    figures_output_dir = output_dir + '\\figures\\rolling_phase_correction_plots'
    # Create the dataframe
    roll_df = pd.DataFrame(
        rolling_mds, columns=['trial', 'block', 'latency', 'jitter', 'instrument', 'tempo_slope', 'lb',
                              'correction_partner', 'ub', *[f'lag_{s}' for s in range(1, ROLLING_LAG + 1)]]
    )

    def rolling_jitter_correction_df() -> pd.DataFrame:
        return (
            pd.melt(
                roll_df, id_vars=['trial', 'block', 'latency', 'jitter', 'instrument'], var_name='lag',
                value_vars=[col for col in roll_df.columns if 'lag_' in col], value_name='coef',
            ).sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument', 'lag'])
        )

    # Outputs using median correction only
    vutils.output_regression_table(
        mds=autils.create_model_list(df=roll_df, md=f'correction_partner~C(latency)+C(jitter)+C(instrument)',
                                     avg_groupers=['latency', 'jitter', 'instrument']),
        output_dir=figures_output_dir, verbose_footer=False
    )
    make_pairgrid(df=roll_df, output_dir=figures_output_dir, xvar='correction_partner')
    make_correction_boxplot_by_variable(df=roll_df,  yvar='correction_partner', output_dir=figures_output_dir,)
    # Outputs using correction upper/lower bounds
    make_pairgrid_iqr(df=roll_df, value_vars=['lb', 'correction_partner', 'ub'], value_name='correction_partner',
                      output_dir=figures_output_dir)
    # Outputs using rolling jitter dataframe
    make_lagged_pointplot(rolling_jitter_correction_df(), output_dir=output_dir)
