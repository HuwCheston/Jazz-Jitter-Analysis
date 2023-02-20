import pandas as pd
import numpy as np
import numba as nb
import json
import os
import pickle
import warnings
import statsmodels.formula.api as smf
from sklearn.covariance import EllipticEnvelope
from pingouin import partial_corr
from datetime import timedelta

import src.analyse.analysis_utils as autils

# Define the objects we can import from this file into others
__all__ = [
    'generate_phase_correction_models',
    'PhaseCorrectionModel'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
np.set_printoptions(suppress=True)


# noinspection PyBroadException
class PhaseCorrectionModel:
    """
    A linear phase correction model for a single performance (keys and drums).
    """
    def __init__(
            self, c1_keys: list, c2_drms: list, **kwargs
    ):
        # Parameters
        self._iqr_filter_range: tuple[float] = kwargs.get('iqr_filter', (0.1, 0.9))  # Filter IOIs above and below
        self._model: str = kwargs.get('model', 'my_next_ioi_diff~my_prev_ioi_diff+asynchrony')  # regression model
        self._rolling_window_size: str = kwargs.get('rolling_window_size', '2s')   # 2 seconds = 2 beats at 120BPM
        self._rolling_min_periods: int = kwargs.get('rolling_min_periods', 2)
        self._maximum_lag: int = kwargs.get('maximum_lag', 8)   # 8 seconds = 2 bars at 120BPM
        # The cleaned data, before dataframe conversion
        self.keys_raw: dict = c1_keys
        self.drms_raw: dict = c2_drms
        # Number of raw onsets below IOI threshold that were removed
        # The raw data converted into dataframe format
        self.keys_df, keys_onsets_below_thresh = self._generate_df(c1_keys)
        self.drms_df, drms_onsets_below_thresh = self._generate_df(c2_drms)
        # The nearest neighbour models, matching live and delayed onsets together
        self._contamination = self._get_contamination_value_from_json(default=kwargs.get('contamination', None))
        self.keys_nn, keys_ioi_filter_nans = self._match_onsets(
            live_arr=self.keys_df['my_onset'].to_numpy(dtype=np.float64),
            delayed_arr=self.drms_df['my_onset'].to_numpy(dtype=np.float64),
            zoom_arr=self.keys_raw['zoom_array'],
        )
        self.drms_nn, drms_ioi_filter_nans = self._match_onsets(
            live_arr=self.drms_df['my_onset'].to_numpy(dtype=np.float64),
            delayed_arr=self.keys_df['my_onset'].to_numpy(dtype=np.float64),
            zoom_arr=self.drms_raw['zoom_array'],
        )
        # Calculate total number of filtered onsets:
        self.keys_filtered_onsets = keys_onsets_below_thresh + keys_ioi_filter_nans
        self.drms_filtered_onsets = drms_onsets_below_thresh + drms_ioi_filter_nans
        # The phase correction models
        self.keys_md = self._create_phase_correction_model(self.keys_nn)
        self.drms_md = self._create_phase_correction_model(self.drms_nn)
        # The summary dictionaries, which can be appended to a dataframe of all performances
        self.keys_dic = self._create_summary_dictionary(c1_keys, self.keys_md, self.keys_nn, self.keys_filtered_onsets)
        self.drms_dic = self._create_summary_dictionary(c2_drms, self.drms_md, self.drms_nn, self.drms_filtered_onsets)

    def _get_contamination_value_from_json(
            self, default: float = None
    ) -> float:
        """

        """
        if default is not None:
            return default
        # TODO: investigate proper pathing here
        cwd = os.path.dirname(os.path.abspath(__file__)) + r'\contamination_params.json'
        js = json.load(open(cwd))
        return [
            i['contamination'] for i in js
            if i['trial'] == self.keys_raw['trial'] and i['block'] == self.keys_raw['block']
            and i['latency'] == self.keys_raw['latency'] and i['jitter'] == self.keys_raw['jitter']
        ][0]

    @staticmethod
    def _append_zoom_array(
            perf_df: pd.DataFrame, zoom_arr: np.array, onset_col: str = 'my_onset'
    ) -> pd.DataFrame:
        """
        Appends a column to a dataframe showing the approx amount of latency by AV-Manip for each event in a performance
        """
        # TODO: need to add in the constant amount of delay induced by the testbed - 12ms?
        # Create a new array with our Zoom latency times and the approx timestamp they were applied to the performance
        # 8 = performance start time (after count-in, 0.75 = refresh rate
        zarr = np.c_[zoom_arr, np.linspace(8, 8 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
        # Initialise an empty column in our performance dataframe
        perf_df['latency'] = np.nan
        # Loop through successive timestamps in our zoom dataframe
        for first, second in zip(zarr, zarr[1:]):
            # Set any rows in the performance dataframe between our timestamps to the respective latency time
            perf_df.loc[perf_df[onset_col].between(first[1], second[1]), 'latency'] = first[0]
        # Convert latency into milliseconds now
        perf_df['latency'] /= 1000
        return perf_df

    def _generate_df(
            self, data: list, threshold: float = 0.25
    ) -> tuple[pd.DataFrame, int]:
        """
        Create dataframe, append zoom array, and add a column with our delayed onsets.
        This latter column replicates the performance as it would have been heard by our partner.
        """
        df = pd.DataFrame(data['midi_bpm'], columns=['my_onset', 'pitch', 'velocity']).sort_values('my_onset')
        df = self._append_zoom_array(df,  data['zoom_array'])
        # Drop pitch and velocity columns if not needed (default)
        df['my_prev_ioi'] = df['my_onset'].diff().astype(float)
        # Remove ioi values below threshold
        temp = df.dropna()
        temp = temp[temp['my_prev_ioi'] < threshold]
        df = df[~df.index.isin(temp.index)]
        # Recalculate IOI column after removing those below threshold
        df['my_prev_ioi'] = df['my_onset'].diff().astype(float)
        # Fill na in latency column with next/previous valid observation
        df['latency'] = df['latency'].bfill().ffill()
        # Add the latency column (divided by 1000 to convert from ms to sec) to the onset column
        df['my_onset_delayed'] = (df['my_onset'] + df['latency'])
        # Return the dataframe, as well as the number of IOI values we've dropped that are below our threshold
        return df, len(temp)

    def _extract_tempo_slope(
            self
    ):
        """
        Extracts tempo slope, in the form of a float representing BPM change per second.

        Method:
        ---
        - Resample both dataframes to get the mean BPM value every second.
        - Concatenate these dataframes together, and get the mean BPM by both performers each second
        - Compute a regression of this mean BPM against the overall elapsed time and extract the coefficient
        """
        # Resample both raw dataframes
        resampled = [autils.resample(data) for data in [self.keys_df.copy(), self.drms_df.copy()]]
        # Create the BPM column
        for d in resampled:
            d['bpm'] = 60 / d['my_prev_ioi']
        # Concatenate the resampled dataframes together and get the row-wise mean
        conc = pd.DataFrame(pd.concat([df['bpm'] for df in resampled], axis=1).mean(axis=1), columns=['bpm'])
        # Get the elapsed time column as an integer
        conc['my_onset'] = conc.index.total_seconds()
        # Create the regression model and extract the tempo slope coefficient
        return smf.ols('bpm~my_onset', data=conc.dropna()).fit()

    @staticmethod
    def _extract_pairwise_asynchrony(
            nn, asynchrony_col: str = 'asynchrony'
    ):
        """
        Extract pairwise asynchrony as a float (in milliseconds, as is standard for this unit in the literature)

        Method:
        ---
        -	Carry out the nearest-neighbour matching, and get a series of asynchrony values for both musicians
            (I.e. keys -> drums with delay, drums -> keys with delay).
        -	Square all of these values;
        -	Get the overall mean (here we collapse both arrays down to a single value);
        -	Take the square root of this mean.
        """
        # Square all the asynchrony values, take the mean, then the square root, then convert to milliseconds
        return np.sqrt(np.nanmean(np.square(nn[asynchrony_col].to_numpy()))) * 1000

    def _extract_pairwise_asynchrony_with_standard_deviations(
            self, async_col: str = 'asynchrony'
    ):
        """
        Extract pairwise asynchrony using the standard deviation of the asynchrony

        Method:
        ---
        -	Join both nearest-neighbour dataframes in order to match asynchrony values together;
        -	Square all of these values;
        -	Get the overall mean (here we collapse both arrays down to a single value);
        -	Take the square root of this mean.
        -   Repeat the join process with the other dataframe as left join, get the pairwise asynchrony again
        -   Take the mean of both (this is to prevent issues with the dataframe join process)
        """
        means = []
        # Iterate through both combinations of nearest neighbour dataframes
        for le, r in zip([self.keys_nn, self.drms_nn], [self.drms_nn, self.keys_nn]):
            # Copy the dataframes so we don't mess anything up internally with them
            le = le.copy(deep=True)
            r = r.copy(deep=True)
            # Merge the dataframes together on their shared columns
            con = le.rename(columns={'my_onset': 'o'}).merge(r.rename(columns={'their_onset': 'o'}), how='left', on='o')
            # Append the pairwise asynchrony value
            means.append(
                np.sqrt(np.nanmean(np.square(con[[f'{async_col}_x', f'{async_col}_y']].std(axis=1)))) * 1000
            )
        # Calculate the mean of both values
        return np.mean(means)

    def _extract_asynchrony_third_person(
            self, async_col: str = 'asynchrony_third_person'
    ) -> float:
        """
        Extracts asynchrony experienced by an imagined third person joined to the Zoom call
        """
        conc = np.concatenate([self.keys_nn[async_col].to_numpy(), self.drms_nn[async_col].to_numpy()])
        return np.sqrt(np.nanmean(np.square(conc))) * 1000

    def _match_onsets(
            self, live_arr: np.ndarray, delayed_arr: np.ndarray, zoom_arr: np.ndarray
    ) -> pd.DataFrame:
        """
        For a single performer, matches each of their live onsets with the closest delayed onset from their partner.
        """
        # Carry out the nearest neighbour matching
        empty_arr = np.zeros(shape=(live_arr.shape[0], 2), dtype=np.float64)
        results = self._nearest_neighbour(live_arr=live_arr, delayed_arr=delayed_arr, empty_arr=empty_arr)
        # Calculate asynchrony column from nearest neighbour output
        nn_np = np.concatenate([results, (results[:, 1] - results[:, 0]).reshape(-1, 1)], axis=1)
        # Apply specialised cleaning if 180ms of latency has been applied
        if self.keys_raw['latency'] == 180:
            nn_np = self._cleaning_for_180ms(delayed_arr, nn_np)
        # If we're using elliptic envelope cleaning, apply this now
        if 0 < self._contamination < 0.5:
            nn_np = self._apply_elliptic_envelope(delayed_arr, nn_np)
        # Remove duplicate matches from the nearest neighbour output
        nn_np = self._remove_duplicate_matches(nn_np)
        # Convert nearest neighbour output to a dataframe
        nn_df = pd.DataFrame(nn_np, columns=['my_onset', 'their_onset', 'asynchrony'])
        # Append the zoom array onto our dataframe
        nn_df = self._append_zoom_array(nn_df, zoom_arr=zoom_arr)
        # Add the correct amount of latency onto our partner's onset time
        nn_df['their_onset_delayed'] = nn_df['their_onset'] + nn_df['latency']
        # Add the correct amount of latency onto our onset time (for asynchrony calculation)
        nn_df['my_onset_delayed'] = nn_df['my_onset'] + nn_df['latency']
        # Recalculate our asynchrony column with the now delayed partner onsets
        nn_df['asynchrony'] = nn_df['their_onset_delayed'] - nn_df['my_onset']
        # Raw asynchrony - without latency added to either performer
        nn_df['asynchrony_raw'] = nn_df['their_onset'] - nn_df['my_onset']
        # Third person asynchrony - with latency added to both performers
        nn_df['asynchrony_third_person'] = nn_df['their_onset_delayed'] - nn_df['my_onset_delayed']
        # Format the now matched dataframe and return
        return self._format_df_for_model(nn_df)

    @staticmethod
    @nb.njit
    def _nearest_neighbour(
            live_arr, delayed_arr, empty_arr
    ):
        """
        Carry out the nearest-neighbour matching. Optimised with numba.
        """
        # Iterate through every value played by our live performer
        for i in range(live_arr.shape[0]):
            # Matrix subtraction
            temp_result = np.absolute(delayed_arr - live_arr[i])
            # Get the closest match
            match = np.min(temp_result)
            # Get the onset time for this closest match
            closest_element = delayed_arr[np.where(temp_result == match)][0]
            # Append the results to our array in the required index
            empty_arr[i][0] = live_arr[i]
            empty_arr[i][1] = closest_element
        return empty_arr

    @staticmethod
    @nb.njit
    def _cleaning_for_180ms(
            delayed_arr: np.ndarray, nn_np: np.ndarray
    ) -> np.ndarray:
        """
        Applies specialised cleaning using bins to performances with 180ms of latency. Optimised with numba.
        """
        # Calculate the bins
        hist, edges = np.histogram(nn_np[:, 2], bins=2)
        # Extract the histogram midpoint
        midpoint = edges[1]
        # Extract the histogram edges
        edges = np.array([np.min(edges), np.max(edges)])
        # Get the edges of the smallest and largest bin by density
        smallest_bin, largest_bin = edges[np.argmin(hist)], edges[np.argmax(hist)]
        # Subset the input array for outliers depending on which bin has the largest density
        if smallest_bin < largest_bin:
            outliers = nn_np[np.where((nn_np[:, 2] <= midpoint) & (nn_np[:, 2] >= smallest_bin))]
        else:
            outliers = nn_np[np.where((nn_np[:, 2] >= midpoint) & (nn_np[:, 2] <= smallest_bin))]
        # Iterate through all outlying values
        for row in outliers:
            # Get current index within input array
            idx = np.where(nn_np[:, 0] == row[0])[0][0]
            # Matrix subtraction
            matrix_subtract = delayed_arr - row[0]
            # If the midpoint of our smaller bin is smaller than the midpoint of our larger bin
            if smallest_bin < largest_bin:
                # Find the next closest onset that occurred before the incorrect match
                matrix_subtract[matrix_subtract < 0] = np.inf
                # If we have an all-infinity dataset, set the values to nan and break
                if np.isinf(matrix_subtract).all():
                    nn_np[idx, :] = np.array([nn_np[idx, 0], np.nan, np.nan])
                    continue
                nxt = np.argmin(matrix_subtract)
            # If the midpoint of our smaller bin is larger than the midpoint of our larger bin
            else:
                # Find the next closest onset that occurred after the incorrect match
                matrix_subtract[matrix_subtract > 0] = -np.inf
                # If we have an all-infinity dataset, set the values to nan and break
                if np.isneginf(matrix_subtract).all():
                    nn_np[idx, :] = np.array([nn_np[idx, 0], np.nan, np.nan])
                    continue
                nxt = np.argmax(matrix_subtract)
            try:
                nxt_nearest = delayed_arr[nxt]
            except:
                continue
            else:
                # If we've matched with the same onset again, set the value as missing
                if nxt_nearest == row[1]:
                    nn_np[idx, :] = np.array([nn_np[idx, 0], np.nan, np.nan])
                # Else, set the required values to our new match
                else:
                    nn_np[idx, :] = np.array([nn_np[idx, 0], nxt_nearest, nxt_nearest - nn_np[idx, 0]])
        return nn_np

    def _apply_elliptic_envelope(
            self, delayed_arr: np.ndarray, nn_np: np.ndarray
    ) -> pd.DataFrame:
        """
        Applies an EllipticEnvelope filter to data to extract outliers and rematch or set them to missing. Numba isn't
        used here as it isn't supported by EllipticEnvelope and sklearn.
        """
        # Create the elliptic envelope with required contamination parameter
        ee = EllipticEnvelope(contamination=self._contamination, random_state=1)
        # Fit the elliptic envelope to our async column after removing NaNs (which cannot be handled in EllipticEnvelope
        to_fit = nn_np[~np.isnan(nn_np[:, 2])][:, 2].reshape(-1, 1)
        ee.fit(to_fit)
        # Iterate through all rows in our array
        for row in nn_np:
            # Try and predict whether the asynchrony value is an outlier
            try:
                predict = ee.predict(np.float64(row[2]).reshape(-1, 1))[0]
            # If we try and predict on a NaN value, we'll raise a ValueError, so skip these rows
            except ValueError:
                continue
            # If the value is an outlier, apply cleaning procedure
            if predict == -1:
                # Get current index within input array
                idx = np.where(nn_np[:, 0] == row[0])[0][0]
                # Get our absolute subtracted matrix
                matrix_subtract = abs(delayed_arr - row[0])
                # Get the second closest value from the subtracted matrix
                nxt = np.where(matrix_subtract == np.partition(matrix_subtract, 2)[2])
                # Get the onset corresponding with this
                nxt_nearest = delayed_arr[nxt]
                # Calculate the actual async, not in absolute terms
                predicted_async = nxt_nearest - row[0]
                # Calculate whether new asynchrony is still an outlier
                is_outlier = ee.predict(np.float64(predicted_async).reshape(-1, 1))[0]
                # If new asynchrony is no longer an outlier, match with it
                if is_outlier == 1:
                    nn_np[idx, :] = np.array([nn_np[idx, 0], nxt_nearest[0], nxt_nearest[0] - nn_np[idx, 0], ])
                # Else, set the data to missing
                else:
                    nn_np[idx, :] = np.array([nn_np[idx, 0], np.nan, np.nan, ])
        return nn_np

    @staticmethod
    def _remove_duplicate_matches(
            nn_np: np.ndarray
    ) -> np.ndarray:
        """
        Filters onsets for duplicate matches, then keeps whichever match is closest to median asynchrony time.
        """
        # Get median asynchrony time
        median = np.median(nn_np[:, 2])
        # Get unique matches
        # TODO: investigate another function here, as return_counts argument isn't supported by numba
        unique, counts = np.unique(nn_np[:, 1], return_counts=True)
        # Get duplicate matches
        duplicates = nn_np[np.isin(nn_np[:, 1], unique[counts > 1])]
        # Subset original array to just get duplicates
        split = np.split(duplicates, np.where(np.diff(duplicates[:, 1]))[0] + 1)
        # If we actually have duplicates
        if split[0].shape[0] != 0:
            # Iterate through each duplicate
            for arr in split:
                # Get index of largest asynchrony value from median
                idx = np.argmax(np.absolute(arr[:, 2] - median))
                # Get row from index
                row = arr[idx]
                idx_orig = np.where((nn_np[:, 0] == row[0]) & (nn_np[:, 1] == row[1]) & (nn_np[:, 2] == row[2]))[0]
                # Set largest asynchrony value to NaN
                nn_np[idx_orig, 1] = np.nan
        return nn_np

    def _iqr_filter(
            self, df: pd.DataFrame, col: str
    ) -> pd.DataFrame:
        """
        Applies an inter-quartile range filter to set outlying values for a particular column to missing.
        """
        # Get lower quartile
        q1 = df[col].quantile(self._iqr_filter_range[0])
        # Get upper quartile
        q3 = df[col].quantile(self._iqr_filter_range[1])
        # Get interquartile range
        iqr = q3 - q1
        # Multiply by 1.5
        s = 1.5 * iqr
        # Get lower bound for filtering
        lb = q1 - s
        # Get upper bound for filtering
        ub = q3 + s
        # Set values below lower bound = nan
        df.loc[df[col] > ub, col] = np.nan
        # Set values above upper bound = nan
        df.loc[df[col] < lb, col] = np.nan
        return df

    def _format_df_for_model(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerces a dataframe into the format required for the phrase-correction model, including setting required columns
        """
        # Extract inter-onset intervals
        df['my_prev_ioi'] = df['my_onset'].diff()
        df['their_prev_ioi'] = df['their_onset'].diff()
        df = self._iqr_filter(df, 'my_prev_ioi')
        # Add the number of onsets our IQR filter has discarded to the total
        discarded_onsets = df['my_prev_ioi'][1:].isna().sum()
        # Shift to get next IOI
        df['my_next_ioi'] = df['my_prev_ioi'].shift(-1)
        # Extract difference of IOIs
        df['my_next_ioi_diff'] = df['my_next_ioi'].diff()
        df['my_prev_ioi_diff'] = df['my_prev_ioi'].diff()
        return df, discarded_onsets

    def _get_rolling_coefficients(
            self, nn_df: pd.DataFrame, func=None, ind_var: str = 'latency', dep_var: str = 'my_prev_ioi',
            cov: str = 'their_prev_ioi'
    ) -> list[float | int]:
        """
        Centralised function for calculating the relationship between IOI and latency variancy. Takes in a single
        independent and dependent variable and covariants, as well as a function to apply to these.

        Method:
        ---
        -	Get rolling standard deviation values for all passed variables.
        -	Lag these values according to the maximum lag attribute passed when creating the class instance.
        -   Apply a function (defaults to regression) onto the lagged and non-lagged variables and return the results.
        """
        if func is None:
            func = self._regress_shifted_rolling_variables
        roll = self._get_rolling_standard_deviation_values(nn_df, cols=(dep_var, ind_var, cov))
        train = self._lag_rolling_values(roll, cols=(dep_var + '_std', ind_var + '_std', cov + '_std'))
        return func(train, dep_var=dep_var + '_std', ind_var=ind_var + '_std', cov_var=cov + '_std')

    def _get_rolling_standard_deviation_values(
            self, nn_df: pd.DataFrame, cols: tuple[str] = ('my_prev_ioi', 'their_prev_ioi', 'latency')
    ) -> pd.DataFrame:
        """
        Extracts the rolling standard deviation of values within a given window size, then resample to get mean value
        for every second.
        """
        # Create the temporary dataframe
        temp = nn_df.copy(deep=True)
        # Create the timedelta index
        idx = pd.to_timedelta([timedelta(seconds=val) for val in temp['my_onset']])
        # Create the offset -- we'll use this later
        offset = timedelta(seconds=8 - temp.iloc[0]['my_onset'])
        # Set the index to the timedelta
        temp = temp.set_index(idx)
        # Filter warnings thrown repeatedly by df.rolling for no reason
        warnings.filterwarnings('ignore')
        # Create the rolling window with the desired window size
        roll = temp.rolling(
            window=self._rolling_window_size, min_periods=self._rolling_min_periods, closed='both', on=temp.index
        )
        # Iterate through the required columns
        for col in cols:
            # Extract the standard deviation and convert into milliseconds
            temp[f'{col}_std'] = roll[col].std(skipna=True) * 1000
        # Resample to the desired frequency, get the mean value, and interpolate from previous values to fill NaNs
        return temp.resample('1s', offset=offset).apply(np.nanmean).interpolate(limit_direction='backward')

    def _lag_rolling_values(
            self, roll: pd.DataFrame, cols: tuple[str] = ('my_prev_ioi_std', 'their_prev_ioi_std', 'latency_std')
    ) -> pd.DataFrame:
        """
        Shifts rolling values by a given number of seconds and concatenates together with the original dataframe.
        """
        # Define the function for shifting values by a maximum number of seconds
        shifter = lambda s: pd.concat(
            [roll[s].shift(i).rename(f'{s}_l{i}') for i in range(0, self._maximum_lag + 1)], axis=1
        )
        # Create the shifted dataframe with all required columns and return it
        return pd.concat([roll, *(shifter(col) for col in cols)], axis=1)

    def _partial_corr_shifted_rolling_variables(
            self, lagged: pd.DataFrame, dep_var: str = 'my_prev_ioi_std',
            ind_var: str = 'latency_std', cov_var: str = 'their_prev_ioi_std'
    ) -> list[float | int]:
        """
        Gets the partial correlation between dep_var and ind_var, controlling for covariate cov_var
        """
        # If we don't have jitter, we'll get errors when calculating partial correlation, because our latency
        # standard deviation columns will all equal zero. So, we need to manually return a list of zeros
        if self.keys_raw['jitter'] == 0:
            return [0 for _ in range(0, self._maximum_lag + 1)]
        corrs = []
        # Iterate through all the lag terms we want
        for i in range(0, self._maximum_lag + 1):
            # Subset the columns for the correct lags, and then drop NaN values to preserve as much as possible
            data = lagged[[dep_var, ind_var,  f'{dep_var}_l{i}', f'{ind_var}_l{i}']].dropna()
            # Get the partial correlation between dep_var and ind_var, controlling for lagged dep_ and cov_ variables
            c = partial_corr(data=data, x=dep_var, y=f'{ind_var}_l{i}', covar=[f'{dep_var}_l{i}'])
            # Extract the coefficient
            corrs.append(c.r.values[0])
        # Return list of results
        return corrs

    @staticmethod
    def _regress_shifted_rolling_variables(
            lagged: pd.DataFrame, dep_var: str = 'my_prev_ioi_std',
            ind_var: str = 'latency_std', cov_var: str = 'their_prev_ioi_std'
    ) -> list[float]:
        """
        Creates a regression model of lagged variables vs non-lagged variables and extracts coefficients.
        """
        # Create the regression model
        md = f'{dep_var}~' + '+'.join([f'{col}' for col in lagged.columns if col[-3:-1] == '_l'])
        # Subset the dataframe for the required columns
        # This will prevent us from dropping values e.g. where asynchrony or other nearest-neighbour column is NaN
        lagged = lagged[[col for col in lagged.columns if dep_var in col or ind_var in col]]
        # Fit the regression model
        res = smf.ols(md, lagged.dropna()).fit()
        # Return the regression coefficients as a generator of floats
        return res.params.filter(like=ind_var, axis=0).to_list()

    def _create_phase_correction_model(
            self, df: pd.DataFrame
    ):
        """
        Create the linear phase correction model
        """
        return smf.ols(self._model, data=df.dropna(), missing='drop').fit()

    def _create_summary_dictionary(
            self, c, md, nn, rn,
    ):
        """
        Creates a dictionary of summary statistics, used when analysing all models.
        """
        return {
            # Metadata for the condition
            'trial': c['trial'],
            'block': c['block'],
            'latency': c['latency'],
            'jitter': c['jitter'],
            'instrument': c['instrument'],
            # Raw data, including latency and MIDI arrays, used for creating tempo slope graphs
            'raw_beats': [c['midi_bpm']],
            'zoom_arr': c['zoom_array'],
            # Cleaning metadata
            'total_beats': autils.extract_interpolated_beats(c)[0],     # Raw number of beats from the performance
            'interpolated_beats': autils.extract_interpolated_beats(c)[1],  # Beats that required interpolation
            'repeat_notes': rn,     # Number of IOIs below our threshold that we removed
            'asynchrony_na': nn['asynchrony'].isna().sum(),     # Number of removed nearest neighbour matches
            # Summary variables: tempo slope, ioi variability, asynchrony, self-reported success
            'tempo_slope': self._extract_tempo_slope().params.loc['my_onset'],
            'ioi_std': self._get_rolling_standard_deviation_values(nn_df=nn)['my_prev_ioi_std'].median(),
            'pw_asym': self._extract_asynchrony_third_person(),    # both performers delayed - imagined third person
            'success': c['success'],
            # Tempo-slope related additional variables
            'tempo_intercept': self._extract_tempo_slope().params.loc['Intercept'],
            # IOI variability related additional variables
            'my_prev_ioi_diff_mean': nn['my_prev_ioi_diff'].mean(),
            'my_next_ioi_diff_mean': nn['my_next_ioi_diff'].mean(),
            # 'ioi_std_vs_jitter_coefficients': self._get_rolling_coefficients(nn_df=nn),
            'ioi_std_vs_jitter_partial_correlation': self._get_rolling_coefficients(
                nn_df=nn, func=self._partial_corr_shifted_rolling_variables
            ),
            # Asynchrony related additional variables
            'asynchrony_mean': nn['asynchrony'].mean(),
            'pw_asym_indiv': self._extract_pairwise_asynchrony(nn),    # Me live, them delayed
            'pw_asym_raw': self._extract_pairwise_asynchrony(nn, asynchrony_col='asynchrony_raw'),  # Both live
            'pw_asym_std': self._extract_pairwise_asynchrony_with_standard_deviations(),
            'pw_asym_raw_std': self._extract_pairwise_asynchrony_with_standard_deviations(async_col='asynchrony_raw'),
            # Success related additional variables
            'interaction': c['interaction'],
            'coordination': c['coordination'],
            # Phase correction model parameters
            'intercept': md.params.loc['Intercept'] if 'Intercept' in md.params.index else 0,
            'correction_self': md.params.loc['my_prev_ioi_diff'],
            'correction_partner': md.params.loc['asynchrony'],
            # Phase correction model extra variables
            'contamination': self._contamination,
            'resid_std': np.std(md.resid),
            'resid_len': len(md.resid),
            'rsquared': md.rsquared,
            'aic': md.aic,
            'bic': md.bic,
            'log-likelihood': md.llf,
            # Model comparison variables
        }


def generate_phase_correction_models(
        raw_data: list, output_dir: str, logger=None, force_rebuild: bool = False
) -> list[PhaseCorrectionModel]:
    """
    Generates all phase correction models
    """
    # Try and load the models from the disk to save time, unless we're forcing them to rebuild anyway
    if not force_rebuild:
        mds = autils.load_from_disc(output_dir, filename='phase_correction_mds.p')
        # If we've successfully loaded models, return these straight away
        if mds is not None:
            return mds
    # Create an empty list to store our models
    res = []
    # Iterate through each conditions
    for z in autils.zip_same_conditions_together(raw_data=raw_data):
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            # If we've passed a logger, start logging information to keep track of the analysis
            if logger is not None:
                logger.info(f'... trial {c1["trial"]}, block {c1["block"]}, '
                            f'latency {c1["latency"]}, jitter {c1["jitter"]}')
            # Create the model
            pcm = PhaseCorrectionModel(c1, c2, model='my_next_ioi_diff~my_prev_ioi_diff+asynchrony', centre=False)
            # Append the raw phase correction model to our list
            res.append(pcm)
    # Pickle the results so we don't need to create them again
    pickle.dump(res, open(f"{output_dir}\\phase_correction_mds.p", "wb"))
    return res


if __name__ == '__main__':
    # Default location for processed raw data
    raw = autils.load_data(r"C:\Python Projects\jazz-jitter-analysis\data\processed")
    # Default location to save output models
    output = r"C:\Python Projects\jazz-jitter-analysis\models"
    # Generate models and pickle
    mds = generate_phase_correction_models(raw_data=raw, output_dir=output, force_rebuild=True)
