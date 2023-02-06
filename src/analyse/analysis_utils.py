import os
import pickle
import pandas as pd
import numpy as np
import numba as nb
from datetime import timedelta
import statsmodels.api as sm
from statsmodels.formula import api as smf
from statsmodels.tsa.stattools import adfuller
import itertools
import collections
import warnings

WINDOW_SIZE = 6
NUM_SIMULATIONS = 100
CONSTANT_RESID_NOISE = 0.005


def test_stationary(
        array: pd.Series
) -> pd.Series:
    """
    Tests if data is stationary, if not returns data with first difference calculated
    """
    # Keep taking first difference while data is non-stationary
    while adfuller(array.dropna(), autolag='AIC')[1] > 0.05:
        array = array.diff()
    # Return the array once data is stationary
    return array


def load_data(
        input_filepath: str
) -> list:
    """
    Loads all pickled data from the processed data folder
    """
    return [pickle.load(open(f'{input_filepath}\\{f}', "rb")) for f in os.listdir(input_filepath) if f.endswith('.p')]


def generate_df(
        data: np.array, iqr_range: tuple = (0.05, 0.95), threshold: float = 0, keep_pitch_vel: bool = False
) -> pd.DataFrame:
    """
    Create dataframe from MIDI performance data, either cleaned (just crotchet beats) or raw.
    Optional keyword arguments:
    iqr_range: Upper and lower quartile to clean IOI values by.
    threshold: Value to remove IOI timings below
    keep_pitch_vel: Keep pitch and velocity columns
    """
    # Construct the dataframe
    df = pd.DataFrame(data, columns=['onset', 'pitch', 'velocity']).sort_values('onset')
    # Drop pitch and velocity columns if not needed (default)
    if not keep_pitch_vel:
        df = df.drop(['pitch', 'velocity'], axis=1)
    # Extract IOI values
    df['ioi'] = df['onset'].diff().astype(float)
    # Clean to remove any spurious onsets
    df['ioi'] = iqr_filter('ioi', df, iqr_range,)
    # Calculate BPM for each crotchet (60/ioi)
    df['bpm'] = 60 / df['ioi']
    # Multiply IOI by 1000 to convert to milliseconds
    df['ioi'] = df['ioi'] * 1000
    # Remove iois below threshold (by default, won't remove anything!)
    df['ioi'] = df['ioi'].mask(df['ioi'] <= threshold)
    # Calculate floor of onset time
    df['elapsed'] = pd.to_timedelta(np.floor(df['onset']), unit='s').dt.total_seconds()
    return df.reset_index(drop=True)


def iqr_filter(
        col: str, df: pd.DataFrame, iqr_range: tuple = (0.05, 0.95)
) -> pd.Series:
    """
    Filter duration values below a certain quartile to remove extraneous midi notes not cleaned in Reaper
    """
    # Get upper/lower quartiles and inter-quartile range
    q1, q3 = df[col].quantile(iqr_range)
    iqr = q3 - q1
    # Filter values below Q1-1.5IQR
    fil = df.query(f'(@q1 - 1.5 * @iqr) <= {col}')
    return fil[col]


def reg_func(
        df: pd.DataFrame, xcol: str, ycol: str
) -> sm.regression.linear_model.RegressionResults:
    """
    Calculates linear regression between two given columns, returns results table
    """
    # We can't have NA values in our regression
    df = df.dropna()
    # Get required columns from dataframe, as float dtype
    x, y = df[xcol].astype(float), df[ycol].astype(float)
    # Add constant and create the model
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    # Use HAC standard errors w/ 8 seconds max lag to account for any autocorrelation
    # TODO: check this!
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 8})
    return results


def return_coeff_from_sm_output(
        results: sm.regression.linear_model.RegressionResults
) -> int:
    """
    Formats the table returned by statsmodel to return only the regression coefficient as an integer
    """
    return pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0].iloc[1, 0]


def return_average_coeffs(
        coeffs: list
) -> list[tuple]:
    """
    Returns list of tuples containing average coefficient for keys/drums performance in a single trial
    Tuples take the form of those in generate_tempo_slopes, i.e. (trial, block, latency, jitter, avg. slope coefficient)
    """
    # Define the grouper function
    func = lambda x: (x[0], x[1], x[2], x[3], x[4])
    # Groupby trial, block, latency and jitter, then average coefficients and return tuple in same form
    return [(*idx, float(sum(d[-1] for d in li)) / 2) for idx, li in itertools.groupby(coeffs, key=func)]


def extract_event_density(
        bpm: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Appends a column to performance dataframe showing number of actual notes per extracted crotchet
    """
    # Function to subset dataframe column and convert to array
    fn = lambda a: a['onset'].to_numpy()
    # Iterate through successive values in bpm array and calculate number of notes in raw array between these
    density = [len([i for i in fn(raw) if first <= i <= second]) for first, second in zip(fn(bpm), fn(bpm)[1:])]
    # Insert nan as the first element of density list so that it aligns with pre-existing columns
    density.insert(0, np.nan)
    # Append event density list as new column in performance dataframe
    bpm['density'] = density
    return bpm


def append_zoom_array(
        perf_df: pd.DataFrame, zoom_arr: np.array, onset_col: str = 'onset'
) -> pd.DataFrame:
    """
    Appends a column to a dataframe showing the approx amount of latency by AV-Manip for each event in a performance
    """

    # Create a new array with our Zoom latency times and the approx timestamp they were applied to the performance
    # 8 = performance start time (after count-in, 0.75 = refresh rate
    z = np.c_[zoom_arr, np.linspace(8, 8 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
    # Initialise an empty column in our performance dataframe
    perf_df['lat'] = np.nan
    # Loop through successive timestamps in our zoom dataframe
    for first, second in zip(z, z[1:]):
        # Set any rows in the performance dataframe between our timestamps to the respective latency time
        perf_df.loc[perf_df[onset_col].between(first[1], second[1]), 'lat'] = first[0]
    return perf_df


def generate_tempo_slopes(
        raw_data: list
) -> list[tuple]:
    """
    Returns average tempo slope coefficients for all performances as list of tuples in the form
    (trial, block, latency, jitter, avg. slope coefficient)
    """
    cs = []
    # Iterate through all trials
    for trial in raw_data:
        # Iterate through data for each condition in a trial
        for con in trial:
            # Generate the data frame from the midi bpm array
            df = generate_df(data=con['midi_bpm'])
            # Calculate the regression of elapsed time vs ioi
            res = reg_func(df, xcol='elapsed', ycol='ioi')
            # Construct the tuple and append to list
            cs.append((con['trial'], con['block'], con['condition'], con['latency'], con['jitter'], con['instrument'],
                       return_coeff_from_sm_output(res)))
    # Average coefficients for both performers in a single condition and return as list of tuples
    return return_average_coeffs(cs)


def zip_same_conditions_together(
        raw_data: list
) -> list[zip]:
    """
    Iterates through raw data and zips keys/drums data from the same performance together
    Returns a list of zip objects, each element of which is a tuple containing
    """
    all_trials = []
    for trial in raw_data:
        # Create the empty default dict
        d = collections.defaultdict(lambda: [])
        for x in trial:
            # Set condition number to 100 in the frozen set as number=1 causes some issues
            s = frozenset((x["block"], 100 if x['condition'] == 1 else x['condition']))
            d[s].append(x)
        for k, v in d.items():
            # Sort the list so we always end up with keys first, drums last
            v = sorted(v, key=lambda i: i['instrument'], reverse=True)
            all_trials.append(zip(v, v[1:]))
    return all_trials


def average_bpms(
        df1: pd.DataFrame, df2: pd.DataFrame, window_size: int = 8, elap: str = 'elapsed', bpm: str = 'bpm'
) -> pd.DataFrame:
    """
    Returns a list of averaged BPMs from two performance.
    Data is grouped by every second in a performance.
    """
    # Merge dataframes from both performers together
    bigdf = df1.merge(df2, how='inner', on=elap)
    # Set function
    fn = lambda g: g.drop_duplicates().dropna().tolist()
    # Average tempo of beat onsets created by both musicians within one second
    with warnings.catch_warnings():     # Catch errors when we don't have any values for one particular second
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_tempo = [(idx, np.nanmean([*fn(grp[f'{bpm}_x']), *fn(grp[f'{bpm}_y'])]))
                     for idx, grp in bigdf.groupby(elap)]
    # Create dataframe
    df = pd.DataFrame(avg_tempo, columns=['elapsed', 'bpm_avg'])
    # Reindex in case any rows are missing!
    df.elapsed = df.elapsed.astype(int)
    new_index = pd.Index(
        [num for num in range(df.elapsed.min(), df.elapsed.max()+1)],
        name="elapsed"
    )
    df = df.set_index('elapsed').reindex(new_index).reset_index()
    # Roll average BPM column
    df['bpm_rolling'] = df['bpm_avg'].rolling(window=window_size, min_periods=1).mean()
    return df


def create_model_list(
        df, avg_groupers: list, md='correction_partner_onset~C(latency)+C(jitter)+C(instrument)'
) -> list:
    """
    Subset a dataframe of per-condition results and return a list of statsmodels regression outputs for use in a table.
    By default, the regression will average results from the same condition across multiple measures. This can be
    overridden by setting the averaging argument to False.
    """

    # Create the list of models
    mds = []
    # Group the dataframe by trial and iterate
    for idx, grp in df.groupby('trial'):
        # Average the results obtained for each condition across measures, if required
        grp = grp.groupby(by=avg_groupers).mean().reset_index(drop=False)
        # Create the model and append to the list
        mds.append(smf.ols(md, data=grp).fit())
    return mds


def extract_interpolated_beats(
        c: np.array
) -> tuple[int, int]:
    """
    Extracts the number of beats in the performance that required interpolation in REAPER. This was usually due to a
    performer 'pushing' ahead a crotchet beat by a swung quaver, or due to an implied metric modulation.
    """
    # Create our zoom array
    za = c['zoom_array']
    zarr = np.c_[za, np.linspace(8, 8 + (len(za) * 0.75), num=len(za), endpoint=False)]
    # Extract the timestamp marking the end of a performance
    endpoint = zarr[:, 1][-1] + 0.75
    # Define the sorter function
    sorter = lambda s: sorted([i[0] for i in c[s] if i[0] <= endpoint])
    # Generate an array showing if a note from the cleaned array is in the raw array and occurred before the end
    arr = np.isin(sorter('midi_bpm'), sorter('midi_raw'))
    # Extract the number of notes from the cleaned array that are also in the raw array
    in_raw = len(np.where(arr)[0])
    # Extract the total length of the cleaned array
    total = len(arr)
    # Calculate the number of interpolated beats by subtracting in_raw from the total length of the cleaned array
    num_interpolated = total - in_raw
    # Return as a tuple containing the total number of beats and the number of interpolated beats
    return total, num_interpolated


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


def resample(
        perf: pd.DataFrame, func=np.nanmean,  col: str = 'my_onset',
        resample_window: str = '1s', interpolate: bool = True
) -> pd.DataFrame:
    """
    Resamples an individual performance dataframe to get mean of every second.
    """
    # Create the timedelta index
    idx = pd.to_timedelta([timedelta(seconds=val) for val in perf[col]])
    # Create the offset: 8 - first onset time
    offset = timedelta(seconds=8 - perf.iloc[0][col])
    # Set the index, resample to every second, and take the mean
    if interpolate:
        return perf.set_index(idx).resample(resample_window, offset=offset).apply(func).interpolate(limit_direction='backward')
    else:
        return perf.set_index(idx).resample(resample_window, offset=offset).apply(func)


def load_from_disc(
        output_dir: str, filename: str = 'phase_correction_mds.p'
) -> list:
    """
    Try and load models from disc
    """
    try:
        mds = pickle.load(open(f"{output_dir}\\{filename}", "rb"))
    # If we haven't generated the models in the first place, return None
    except FileNotFoundError:
        return None
    # Else, return the models
    else:
        return mds


# noinspection PyBroadException
@nb.jit(nopython=True)
def create_one_simulation(
        keys_data: nb.typed.Dict, drms_data: nb.typed.Dict,
        keys_params: nb.typed.Dict, drms_params: nb.typed.Dict,
        keys_noise, drms_noise,
        lat: np.ndarray, beats: int
) -> tuple:
    """
    Create data for one simulation, using numba optimisations. This function is defined outside of the Simulation
    class to enable instances of the Simulation class to be pickled.
    """

    def get_latency_at_onset(
            my_onset: float
    ) -> float:
        """
        Get the current amount of latency applied to our partner when we played our onset
        """
        # If our first onset was before the 8 second count-in, return the first amount of delay applied
        if my_onset < lat[0, :][1]:
            return lat[:, 0][0]
        # Else, return the correct amount of latency
        else:
            return lat[lat[:, 1] == lat[:, 1][lat[:, 1] <= my_onset].max()][:, 0][0]

    def calculate_async(
            my_onset: float, their_onset: float
    ) -> float:
        """
        Calculates the current asynchrony with our partner at our onset. Note the order of positional arguments!!
        """
        return (their_onset + get_latency_at_onset(my_onset)) - my_onset

    def predict_next_ioi_diff(
            prev_diff: float, asynchrony: float, params: nb.typed.Dict, noise: np.ndarray
    ) -> float:
        """
        Predict the difference between previous and next IOIs using inputted data and model parameters
        """
        a = prev_diff * params['correction_self']
        b = asynchrony * params['correction_partner']
        c = params['intercept']
        n = np.random.choice(noise)
        return a + b + c + n

    # Calculate our first two async values. We don't do this when initialising the empty data, because it
    # both requires the get_latency_at_onset function and access to both keys and drums data.
    for i in range(0, 2):
        keys_data['asynchrony'][i] = calculate_async(keys_data['my_onset'][i], drms_data['my_onset'][i])
        drms_data['asynchrony'][i] = calculate_async(drms_data['my_onset'][i], keys_data['my_onset'][i])
    # We don't use the full range of beats, given that we've already added some in when initialising our data
    for i in range(2, beats):
        # Shift difference
        keys_data['my_prev_ioi_diff'][i] = keys_data['my_next_ioi_diff'][i - 1]
        drms_data['my_prev_ioi_diff'][i] = drms_data['my_next_ioi_diff'][i - 1]
        # Shift IOI
        keys_data['my_prev_ioi'][i] = keys_data['my_next_ioi'][i - 1]
        drms_data['my_prev_ioi'][i] = drms_data['my_next_ioi'][i - 1]
        # Get next onset by adding previous onset to predicted IOI
        keys_data['my_onset'][i] = keys_data['my_onset'][i - 1] + keys_data['my_prev_ioi'][i]
        drms_data['my_onset'][i] = drms_data['my_onset'][i - 1] + drms_data['my_prev_ioi'][i]
        # Get async value by subtracting partner's onset (plus latency) from ours
        try:
            keys_data['asynchrony'][i] = calculate_async(keys_data['my_onset'][i], drms_data['my_onset'][i])
            drms_data['asynchrony'][i] = calculate_async(drms_data['my_onset'][i], keys_data['my_onset'][i])
        # If there's an issue here, break out of the simulation
        except:
            break
        # Predict difference between previous IOI and next IOI
        keys_data['my_next_ioi_diff'][i] = predict_next_ioi_diff(
            keys_data['my_prev_ioi_diff'][i], keys_data['asynchrony'][i], keys_params, keys_noise
        )
        drms_data['my_next_ioi_diff'][i] = predict_next_ioi_diff(
            drms_data['my_prev_ioi_diff'][i], drms_data['asynchrony'][i], drms_params, drms_noise
        )
        # Use predicted difference between IOIs to get next actual IOI
        keys_data['my_next_ioi'][i] = keys_data['my_next_ioi_diff'][i] + keys_data['my_prev_ioi'][i]
        drms_data['my_next_ioi'][i] = drms_data['my_next_ioi_diff'][i] + drms_data['my_prev_ioi'][i]
        # If we've accelerated to a ridiculous extent (due to noise), we need to break.
        if keys_data['my_next_ioi'][i] < 0 or drms_data['my_next_ioi'][i] < 0:
            break
        # TODO: should be the same length as original performance?
        # If we've exceeded our endpoint, break
        if keys_data['my_onset'][i] > 100 or drms_data['my_onset'][i] > 100:
            break
    return keys_data, drms_data
