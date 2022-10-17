import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle
from datetime import timedelta
import warnings

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
from src.visualise.phase_correction_graphs import PairGrid, SingleConditionPlot, BoxPlot, RegPlotAbsCorrection, \
    RegPlotRSquared, PointPlotLaggedLatency, SingleConditionAnimation, NumberLine, BarPlot
from src.visualise.tempo_stability_graphs import regplot_ioi_std_vs_tempo_slope
from src.visualise.questionnaire_graphs import ScatterPlotQuestionnaire

PC_MOD = 'my_next_ioi~my_prev_ioi+asynchrony'
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


def extract_rolling_ioi_std(
        df: pd.DataFrame, win_size: timedelta = timedelta(seconds=WINDOW_SIZE)
) -> pd.Series:
    """
    Runs a sliding window along the dataset and extracts the standard deviation of IOI values
    """
    df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['onset']])
    return df.rolling(window=win_size, on='td')['ioi'].std()


# TODO: consider making this its own package?
def extract_npvi(
        s: pd.Series,
) -> float:
    """
    Extracts the normalised pairwise variability index (nPVI) from a column of IOIs
    """
    # Drop nan values and convert to list
    li = s.dropna().tolist()
    # Extract constant term (left side of equation)
    m = 100 / (len(li) - 1)
    # Calculate the right side of the equation
    s = sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(li, li[1:])])
    # Return nPVI
    return s * m


def nearest_neighbor(
        live_arr, delayed_arr, live_i: str = '', delayed_i: str = '', drop: bool = True
) -> pd.DataFrame:
    """
    For all the events in array 1, pair the event to the nearest unique event in array 2.
    Two events in array 2 may be paired to the same event in array 1, and can be dropped by setting drop to True
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
    # Extract onsets
    output_df['my_onset'] = df.iloc[:, 0]
    output_df['their_onset'] = df.iloc[:, 1]
    # Extract inter-onset intervals
    output_df['my_next_ioi'] = output_df['my_onset'].diff().shift(-1)
    output_df['my_prev_ioi'] = output_df['my_onset'].diff()
    # Extract asynchronous timings between onsets
    output_df['asynchrony'] = output_df['their_onset'] - output_df['my_onset']
    return output_df


def extract_pairwise_asynchrony(
    keys_nn: pd.DataFrame, drms_nn: pd.DataFrame
) -> float:
    """
    Rasch (2015) defines pairwise asynchrony as as the root-mean-square of the standard deviations of the onset time
    differences for all pairs of voice parts. We can calculate this for each condition, using the nearest-neighbour
    model for both the keyboard and drummer.
    """

    std = lambda i: (i.asynchrony * 1000).std()
    return np.sqrt(np.mean(np.square([std(keys_nn), std(drms_nn)])))


def construct_phase_correction_model(
        df: pd.DataFrame, mod: str = PC_MOD
) -> sm.regression.linear_model.RegressionResults:
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

    pred = pd.Series(np.insert(np.cumsum(md.predict()) + df['my_onset'].loc[0], 0, df['my_onset'].loc[0]))
    df['predicted_onset'] = pred
    df['predicted_ioi'] = pred.diff()
    df['predicted_bpm'] = 60 / df['predicted_ioi']
    df['live_bpm'] = 60 / df['my_prev_ioi']
    df['elapsed'] = np.floor(df['my_onset'])
    return df


def construct_rolling_correction_model(
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
            return reg.params['asynchrony'], reg.aic, reg.rsquared
        # Catch errors if we only have nans and return a NaN coefficient
        except (IndexError, ValueError) as _:
            return np.nan, np.nan, np.nan

    # Merge the original dataframe with the nearest neighbour model
    merge = pd.merge(nn, orig.rename(columns={'onset': 'my_onset'}), on='my_onset')
    # Create the timestamp column for rolling on
    merge['td'] = pd.to_timedelta([timedelta(seconds=val) for val in merge.my_onset.values])
    # Merge the dataframe of coefficients back onto the original and subset for required columns
    df = merge.join(
        pd.DataFrame([process(x) for x in merge.rolling(window=win_size, on='td')],
                     columns=['correction_partner', 'aic', 'r2'])
    )
    # Roll latency column to get standard deviation
    df['lat'] = df.rolling(window=win_size, on='td')['lat'].std()
    df['ioi'] = df.rolling(window=win_size, on='td')['my_prev_ioi'].std() * 100
    # Test if correction to partner is stationary (we know latency variation is always stationary so no need to test)
    # TODO: actually implement this
    df['correction_partner'] = df['correction_partner']
    # TODO: subset this data using arg 'sub', to only include values for after window is full (e.g. from 16 seconds on?)
    return df[['td', 'my_onset', 'lat', 'correction_partner', 'ioi', 'aic', 'r2']]


def regress_rolling_correction_vs_jitter(
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
    # Extract pairwise asymmetry
    pw_asym = extract_pairwise_asynchrony(keys_nn, drms_nn)
    return keys, drms, keys_nn, drms_nn, tempo_slope, pw_asym


def gen_phase_correction_models(
        raw_data: list, output_dir: str, logger=None, make_anim: bool = False, make_single_plot: bool = False,
        force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Generates phase correction models for each instrument in all conditions, using all of the data in a performance.
    """

    # Try and load the models from the disk to save time, unless we're forcing them to rebuild anyway
    if not force_rebuild:
        try:
            mds = pickle.load(open(f"{output_dir}\\output_mds.p", "rb"))
        # If we haven't generated the models in the first place, go ahead and generate them now
        except FileNotFoundError:
            pass
        else:
            if logger is not None:
                logger.info(f'...skipping, loaded models from disc!')
            return mds

    # Create the function used to format the results as a dictionary
    def dic_create(c, md, jitter_md, std, npvi):
        d = {
            'trial': c['trial'], 'block': c['block'], 'latency': c['latency'], 'jitter': c['jitter'],
            'instrument': c['instrument'], 'tempo_slope': tempo_slope, 'ioi_std': std, 'ioi_npvi': npvi,
            'pw_asym': pw_asym, 'rsquared': md.rsquared, 'correction_self': md.params.iloc[1],
            'correction_partner': md.params.iloc[2], 'success': c['success'], 'interaction': c['interaction'],
            'coordination': c['coordination']
        }
        d.update({f'lag_{n+1}': jitter_md[n] for n in range(ROLLING_LAG)})
        return d

    # Create variables we need
    figures_output_dir = output_dir + '\\figures\\phase_correction_plots'
    mds = []

    # Iterate through each conditions
    for z in autils.zip_same_conditions_together(raw_data=raw_data):
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            # If we've passed a logger, start logging information to keep track of the analysis
            if logger is not None:
                logger.info(f'Analysing: trial {c1["trial"]}, block {c1["block"]}, condition {c1["condition"]}')
            # Pre-processing
            keys, drms, keys_nn, drms_nn, tempo_slope, pw_asym = phase_correction_pre_processing(c1, c2)
            # Extract tempo stability
            keys_std = extract_rolling_ioi_std(keys).median()
            drms_std = extract_rolling_ioi_std(drms).median()
            # Extract IOI normalised pairwise variability index
            keys_npvi = extract_npvi(keys['ioi'])
            drms_npvi = extract_npvi(drms['ioi'])
            # Create models (can't be done in a loop, as we need to refer to them later)
            keys_md = construct_phase_correction_model(keys_nn)
            drms_md = construct_phase_correction_model(drms_nn)
            # Extract coefficients from a regression of rolling phase correction vs lagged jitter variation
            keys_j = regress_rolling_correction_vs_jitter(df=construct_rolling_correction_model(nn=keys_nn, orig=keys))
            drms_j = regress_rolling_correction_vs_jitter(df=construct_rolling_correction_model(nn=drms_nn, orig=drms))
            # Append the models and the parameters to the list as a dictionary
            mds.append(dic_create(c1, keys_md, keys_j, keys_std, keys_npvi))
            mds.append(dic_create(c2, drms_md, drms_j, drms_std, drms_npvi))
            # Generate output from single condition
            if make_single_plot:
                single_plot = SingleConditionPlot(
                    keys_df=predict_from_model(keys_md, keys_nn), keys_md=keys_md, keys_o=keys,
                    drms_df=predict_from_model(drms_md, drms_nn), drms_md=drms_md, drms_o=drms,
                    metadata=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                    output_dir=figures_output_dir + '\\individual_plots'
                )
                single_plot.create_plot()
            # Generate animation if specified
            if make_anim:
                anim = SingleConditionAnimation(
                    keys_df=predict_from_model(keys_md, keys_nn), drms_df=predict_from_model(drms_md, drms_nn),
                    keys_o=keys, drms_o=drms, metadata=(c1['trial'], c1['block'], c1['latency'], c1['jitter']),
                    output_dir=figures_output_dir + '\\animations'
                )
                anim.create_animation()
    # Convert our list of dictionaries into a dataframe
    df = pd.DataFrame(mds)
    # Pickle the results so we don't need to create them again
    # TODO: should be saved in the root//models directory!
    pickle.dump(df, open(f"{output_dir}\\output_mds.p", "wb"))
    if logger is not None:
        logger.info(f'Saved models to {output_dir}\\output_mds.p')
    return df


def gen_phase_correction_model_outputs(
        df: pd.DataFrame, output_dir: str
) -> None:
    """
    Generate output from phase correction models
    """

    figures_output_dir = output_dir + '\\figures\\phase_correction_plots'
    # Create regression table
    vutils.output_regression_table(
            mds=autils.create_model_list(df=df, md=f'correction_partner~C(latency)+C(jitter)+C(instrument)',
                                         avg_groupers=['latency', 'jitter', 'instrument']),
            output_dir=figures_output_dir, verbose_footer=False
        )
    # Create boxplots
    bp = BoxPlot(df=df, output_dir=figures_output_dir, yvar='correction_partner', ylim=(-0.2, 1))
    bp.create_plot()
    # Create pairgrid
    pg = PairGrid(
        df=df, xvar='correction_partner', output_dir=figures_output_dir, xlim=(-0.5, 1.5), xlabel='Coupling constant'
    )
    pg.create_plot()
    # Create regplots
    rp1 = RegPlotAbsCorrection(df=df, output_dir=figures_output_dir)
    rp1.create_plot()
    rp2 = RegPlotAbsCorrection(df=df, output_dir=figures_output_dir, yvar='pw_asym', ylabel='Pairwise asynchrony (ms)')
    rp2.create_plot()
    rp3 = RegPlotRSquared(df=df, output_dir=figures_output_dir, xvar='ioi_std', yvar='rsquared', ylabel='R-Squared (%)',
                          xlabel='Median IOI standard deviation, 8-second window (ms)')
    rp3.create_plot()
    # Create pointplot
    pp = PointPlotLaggedLatency(df=df, output_dir=figures_output_dir)
    pp.create_plot()

    # TODO: corpus should be saved in the root//references directory!
    nl = NumberLine(df=df, output_dir=figures_output_dir, corpus_filepath=f'{output_dir}\\pw_asymmetry_corpus.xlsx')
    nl.create_plot()
    bar = BarPlot(df=df, output_dir=figures_output_dir)
    bar.create_plot()


def gen_tempo_slope_outputs(
        df: pd.DataFrame, output_dir: str
) -> None:
    """
    Generates outputs relating to tempo slope
    """

    figures_output_dir = output_dir + '\\figures\\tempo_slopes_plots'
    df = df[df['instrument'] == 'Keys']
    # TODO: generate the heatmap?
    # gen_tempo_slope_heatmap(df=df, output_dir=figures_output_dir)
    mds = autils.create_model_list(df=df, avg_groupers=['latency', 'jitter'], md='tempo_slope~C(latency)+C(jitter)')
    vutils.output_regression_table(mds=mds, output_dir=figures_output_dir)


def gen_tempo_stability_outputs(
        df: pd.DataFrame, output_dir: str,
) -> None:
    """
    Generates outputs from a dataframe including a tempo stability column
    """
    figures_output_dir = output_dir + '\\figures\\tempo_stability_plots'
    # IOI STANDARD DEVIATION
    pg_sd = PairGrid(
        df=df, output_dir=figures_output_dir, xvar='ioi_std',
        xlim=(0, df['ioi_std'].max() + (df['ioi_std'].max() / 10)),
        xlabel='Median IOI standard deviation, 8-second window (ms)'
    )
    pg_sd.create_plot()
    # TODO: use the classes defined in phase_correction_graphs here
    regplot_ioi_std_vs_tempo_slope(df, output_dir=figures_output_dir, xvar='ioi_std')
    mds = autils.create_model_list(df=df, avg_groupers=['latency', 'jitter', 'instrument'],
                                   md="ioi_std~C(latency)+C(jitter)+C(instrument)")
    vutils.output_regression_table(mds=mds, output_dir=figures_output_dir)

    # NORMALISED PAIRWISE VARIABILITY INDEX
    pg_npvi = PairGrid(
        df=df, output_dir=figures_output_dir, xvar='ioi_npvi',
        xlabel='IOI normalised pairwise variability index (nPVI)',
        xlim=(0, df['ioi_npvi'].max() + (df['ioi_npvi'].max() / 10))
    )
    pg_npvi.create_plot()
    regplot_ioi_std_vs_tempo_slope(df, output_dir=figures_output_dir, xvar='ioi_npvi')
    mds = autils.create_model_list(df=df, avg_groupers=['latency', 'jitter', 'instrument'],
                                   md="ioi_npvi~C(latency)+C(jitter)+C(instrument)")
    vutils.output_regression_table(mds=mds, output_dir=figures_output_dir)


def gen_questionnaire_outputs(
        df: pd.DataFrame, output_dir: str
) -> None:
    figures_output_dir = output_dir + '\\figures\\questionnaire_plots'
    sp1 = ScatterPlotQuestionnaire(df=df, output_dir=figures_output_dir, ax_var='block', marker_var='instrument')
    sp1.create_plot()

    sp2 = ScatterPlotQuestionnaire(df=df, output_dir=figures_output_dir, marker_var='block', ax_var='instrument',
                                   one_reg=True)
    sp2.create_plot()
