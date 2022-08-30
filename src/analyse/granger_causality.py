import numpy as np
import pandas as pd
import operator
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
from prepare_data import generate_df, append_zoom_array, extract_event_density, zip_same_conditions_together, ioi_nearest_neighbours_one

pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


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


def pearson_r_ioi_var_vs_latency_var(raw_data: list):
    # Zip same conditions for each trial together
    all_trials = zip_same_conditions_together(raw_data)

    for i in all_trials:
        for c1, c2 in i:
            # Extract MIDI bpm array
            a1 = c1['midi_bpm'][:, 0]
            a2 = c2['midi_bpm'][:, 0]
            keys_nn = ioi_nearest_neighbours_one(a1, a2)
            df = pd.DataFrame(keys_nn, columns=[c1['instrument'], c2['instrument']]).sort_values(by='Keys').reset_index(drop=True)
            df['keys_ioi'] = df['Keys'].diff()
            df['drums_ioi'] = df['Drums'].diff()
            df['keys_next_ioi'] = df['keys_ioi'].shift(-1)
            df['keys_drums_ioi'] = df['drums_ioi'] - df['keys_ioi']
            md = smf.ols('keys_ioi~keys_next_ioi+keys_drums_ioi', data=df).fit()
            # print(md.summary())
            # print(c1['latency'], c1['jitter'])
            # print(df)
            # df['offset'] = df.iloc[:, 1] - df.iloc[:, 0]
            if c1['jitter'] == 1:
                plt.plot(df['keys_drums_ioi'])
                plt.title(f'{c1["latency"]}, {c1["jitter"]}')
                plt.show()
