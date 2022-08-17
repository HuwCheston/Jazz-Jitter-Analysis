import numpy as np
import pandas as pd
import operator
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import seaborn as sns
import matplotlib.pyplot as plt
from data_preparation import generate_df

pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def extract_event_density(bpm: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Appends a column to performance dataframe showing number of actual notes per extracted crotchet"""
    # Function to subset dataframe column and convert to array
    fn = lambda a: a['onset'].to_numpy()
    # Iterate through successive values in bpm array and calculate number of notes in raw array between these
    density = [len([i for i in fn(raw) if first <= i <= second]) for first, second in zip(fn(bpm), fn(bpm)[1:])]
    # Insert nan as the first element of density list so that it aligns with pre-existing columns
    density.insert(0, np.nan)
    # Append event density list as new column in performance dataframe
    bpm['density'] = density
    return bpm


def append_zoom_array(perf_df: pd.DataFrame, zoom_arr: np.array) -> pd.DataFrame:
    """Appends a column to a dataframe showing the approx amount of latency applied at each moment in a performance"""
    # Create a new array with our Zoom latency times and the approx timestamp they were applied to the performance
    # 8 = performance start time (after count-in, 0.75 = refresh rate
    z = np.c_[zoom_arr, np.linspace(8, 8 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
    # Initialise an empty column in our performance dataframe
    perf_df['lat'] = np.nan
    # Loop through successive timestamps in our zoom dataframe
    for first, second in zip(z, z[1:]):
        # Set any rows in the performance dataframe between our timestamps to the respective latency time
        perf_df.loc[perf_df['onset'].between(first[1], second[1]), 'lat'] = first[0]
    # Create a new column with rolling latency standard deviation
    perf_df['lat_std'] = perf_df['lat'].rolling(4).std()
    return perf_df


def test_stationary(array: pd.Series) -> pd.Series:
    """Tests if data is stationary, if not returns data with first difference calculated"""
    # Carry out augmented Dickey-Fuller root test and get the p-value
    p = adfuller(array.dropna(), autolag='AIC')[1]
    # If the result is significant, the data is stationary, so we don't need to transform it
    if p < 0.05:
        return array
    # If the result is not significant, the data is non-stationary, so  transform it by taking the first difference
    elif p > 0.05:
        # Increase the counter by one if it's performance data that isn't stationary
        return array.diff()


def estimate_max_lag(df: pd.DataFrame, maxlag: int = 15) -> int:
    """Estimate the maximum lag to use in the Granger causality model"""
    results = []
    # Create the model
    model = VAR(df)
    # Iterate through the range of maximum lag terms
    for i in range(1, maxlag):
        result = model.fit(i)
        results.append(result.aic)
    # Return the index of the smallest aic value to use as the maximum lag term
    return np.argmin(results) + 1


def granger_test(df: pd.DataFrame, t: str = 'ssr_ftest', xcol: str = 'density', ycol: str = 'lat_std'):
    """
    Estimates granger causality between specified columns of dataframe.
    Calculates maximum lag by taking smallest AIC value of VAR model.
    Returns list of tuples with test statistic & p-value for specified hypothesis test at each given lag.
    """
    # Create the test dataframe by testing if test columns are stationary and returning w/ first difference if needed
    test_df = pd.DataFrame()
    test_df['x'] = test_stationary(df[xcol])
    test_df['y'] = test_stationary(df[ycol])
    # Estimate the maximum lag to use in the causality model
    max_lag = estimate_max_lag(test_df[['x', 'y']].dropna().reset_index())
    # Estimate granger causality using the maximum lag
    test_result = grangercausalitytests(test_df[['x', 'y']].dropna(), maxlag=max_lag, verbose=False)
    # Format the results to select only the given hypothesis test (ssr_ftest by default)
    formatted = [(test_result[i + 1][0][t][0], test_result[i + 1][0][t][1], i + 1) for i in range(max_lag)]
    return formatted


def return_largest_grang(li: list[tuple]) -> tuple:
    """Return tuple from list of tuples in form (test stat, p-value) that has the largest test stat while p<0.05"""
    # Try and return the tuple with the largest test statistic that is significant
    try:
        return max(filter(lambda a: a[1] <= 0.05, li), key=operator.itemgetter(0))
    # Otherwise, return the tuple with the largest test statistic (even if not significant)
    except ValueError:
        return max(li, key=operator.itemgetter(0))


def generate_granger_causality_df(gc_list: list[tuple]) -> pd.DataFrame:
    """"
    Takes in a list of tuples in form: (trial, block, instrument, latency, jitter, test stat, p-value, lag term)
    One tuple provided per condition. Returns a dataframe of all trials.
    """
    return (pd.DataFrame(gc_list, columns=['trial', 'block', 'instrument', 'latency', 'jitter', 'f', 'p', 'lag'])
              .sort_values(by=['trial', 'block', 'latency', 'jitter'])
              .set_index('trial'))


def gc_event_density_vs_latency_var(raw_data: list, output_dir: str):
    """
    Estimates granger causality between musical event density and latency variation.
    """
    # Iterate through all trials
    for num, trial in enumerate(raw_data, 1):
        # Iterate through data for each condition in a trial
        for con in trial:
            # Generate dataframes for both cleaned MIDI bpms and raw MIDI and extract event density
            df = extract_event_density(
                bpm=generate_df(con['midi_bpm']),
                # For the raw data, make sure to remove values shorter than 64th note at 120BPM (drum rolls etc)
                raw=generate_df(con['midi_raw'], threshold=31, ),
            )
            # Append the latency timings used in the performance
            df = append_zoom_array(df, zoom_arr=con['zoom_array'])
            # Only estimate Granger causality if there actually is variation (jitter) in the latency
            if con['jitter'] != 0:
                res = granger_test(df)
                mx = return_largest_grang(li=res)


def gc_ioi_var_vs_latency_var(raw_data: list, output_dir: str):
    """
    Estimates granger causality between beat duration and latency variation.
    """
    # Iterate through all trials
    all_trials = []
    for num, trial in enumerate(raw_data, 1):
        # Iterate through data for each condition in a trial
        for con in trial:
            # Generate our MIDI bpm dataframe
            df = generate_df(con['midi_bpm'])
            # Append the array of latency timings used in the performance to the dataframe
            df = append_zoom_array(df, con['zoom_array'])
            df['ioi_std'] = df['ioi'].rolling(window=4).std()
            # Estimate granger causality
            if con['jitter'] != 0:
                res = granger_test(df, xcol='ioi_std')
                mx = return_largest_grang(li=res)
                all_trials.append((con['trial'], con['block'], con['instrument'], con['latency'], con['jitter'], *mx))
    bigdf = generate_granger_causality_df(gc_list=all_trials)
    bigdf.to_csv(f'{output_dir}\\gc_bpm.csv', sep=';')
