import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools


def generate_df(data: np.array, iqr_range: tuple = (0.05, 0.95), threshold: float = 0) -> pd.DataFrame:
    """Create dataframe of note onset times and IOIs"""
    # Construct the dataframe
    df = pd.DataFrame(data, columns=['onset', 'pitch', 'velocity']).drop(['pitch', 'velocity'], axis=1).sort_values('onset')
    # Remove crotchets with duration below threshold (by default, won't remove anything!)
    df = df[~(df['onset'].diff() <= threshold)]
    # Extract IOI values
    df['ioi'] = df['onset'].diff().astype(float)
    # Clean to remove any spurious onsets
    df['ioi'] = iqr_filter('ioi', df, iqr_range,)
    # Calculate BPM for each crotchet (60/ioi)
    df['bpm'] = 60 / df['ioi']
    # Multiply IOI by 1000 to convert to milliseconds
    df['ioi'] = df['ioi'] * 1000
    # Calculate floor of onset time
    df['elapsed'] = pd.to_timedelta(np.floor(df['onset']), unit='s').dt.total_seconds()
    return df


def iqr_filter(col: str, df: pd.DataFrame, iqr_range,) -> pd.Series:
    """Filter duration values below a certain quartile to remove extraneous midi notes not cleaned in Reaper"""
    # Get upper/lower quartiles and inter-quartile range
    q1, q3 = df[col].quantile(iqr_range)
    iqr = q3 - q1
    # Filter values below Q1-1.5IQR
    fil = df.query(f'(@q1 - 1.5 * @iqr) <= {col}')
    return fil[col]


def reg_func(df: pd.DataFrame, xcol: str, ycol: str) -> sm.regression.linear_model.RegressionResults:
    """Calculates linear regression between elapsed time and given column, returns coefficient"""
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


def return_coeff(results: sm.regression.linear_model.RegressionResults) -> int:
    """Formats the table returned by statsmodel to return only the regression coefficient as an integer"""
    return pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0].iloc[1, 0]


def return_average_coeffs(coeffs: list) -> list:
    """Returns list of tuples containing average coefficient for keys/drums performance in a single trial"""
    # Define the grouper function
    func = lambda x: (x[0], x[1], x[2], x[3], x[4])
    # Groupby trial, block, latency and jitter, then average coefficients and return tuple in same form
    return [(*idx, float(sum(d[-1] for d in li)) / 2) for idx, li in itertools.groupby(coeffs, key=func)]


def generate_tempo_slopes(raw_data: list) -> list:
    """Takes in raw data, returns list of tuples in form (trial, block, latency, jitter, avg. slope coefficient)"""
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
            cs.append((con['trial'], con['block'], con['condition'], con['latency'], con['jitter'], con['instrument'], return_coeff(res)))
    # Average coefficients for both performers in a single condition and return as list of tuples
    return return_average_coeffs(cs)
