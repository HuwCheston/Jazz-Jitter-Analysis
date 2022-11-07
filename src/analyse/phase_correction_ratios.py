import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import timedelta

import src.analyse.analysis_utils as autils


class PhaseCorrectionModel:
    def __init__(
            self, c1_keys: list, c2_drms: list, **kwargs
    ):
        """

        """
        # Parameters
        self._nn_tolerance: float = 0.1  # The amount of tolerance to use when matching onsets
        self._iqr_filter_range: tuple[float] = (0.05, 0.95)  # Filter IOIs above and below this
        self._model = kwargs.get('model', 'my_next_ioi_diff~my_prev_ioi_diff+asynchrony')  # regression model
        # The raw data converted into dataframe format
        self.keys_raw: pd.DataFrame = self._generate_df(c1_keys)
        self.drms_raw: pd.DataFrame = self._generate_df(c2_drms)
        # The nearest neighbour models, matching live and delayed onsets together
        self.keys_nn: pd.DataFrame = self._nearest_neighbour(
            live_arr=self.keys_raw['onset'].to_numpy(), delayed_arr=self.drms_raw['onset_delayed'].to_numpy()
        )
        self.drms_nn: pd.DataFrame = self._nearest_neighbour(
            live_arr=self.drms_raw['onset'].to_numpy(), delayed_arr=self.keys_raw['onset_delayed'].to_numpy()
        )
        # The phase correction models
        self.keys_md = self._create_phase_correction_model(self.keys_nn)
        self.drms_md = self._create_phase_correction_model(self.drms_nn)
        # Additional summary statistics
        # TODO: need to have functions here to extract tempo stability, etc
        self.tempo_slope: float = self._extract_tempo_slope()
        self.pairwise_asynchrony: float = self._extract_pairwise_asynchrony()
        # The summary dictionaries, which can be appended to a dataframe of all performances
        self.keys_dic = self._create_summary_dictionary(c1_keys, self.keys_md)
        self.drms_dic = self._create_summary_dictionary(c2_drms, self.drms_md)

    @staticmethod
    def _generate_df(
            data: list
    ) -> pd.DataFrame:
        """
        Create dataframe, append zoom array, and add a column with our delayed onsets.
        This latter column replicates the performance as it would have been heard by our partner.
        """
        # TODO: need to add in the constant amount of delay induced by the testbed - 12ms?
        df = autils.append_zoom_array(autils.generate_df(data['midi_bpm']), data['zoom_array'])
        # Fill na in latency column with next/previous valid observation
        df['lat'] = df['lat'].bfill().ffill()
        # Add the latency column (divided by 1000 to convert from ms to sec) to the onset column
        df['onset_delayed'] = (df['onset'] + (df['lat'] / 1000))
        return df

    def _extract_tempo_slope(
            self
    ) -> float:
        """
        Extracts tempo slope, in the form of a float representing BPM change per second.

        Method:
        ---
        - Resample both dataframes to get the mean BPM value every second.
        - Concatenate these dataframes together, and get the mean BPM by both performers each second
        - Compute a regression of this mean BPM against the overall elapsed time and extract the coefficient
        """

        def resample(perf: pd.DataFrame):
            # Create the timedelta index
            idx = pd.to_timedelta([timedelta(seconds=val) for val in perf['onset']])
            # Create the offset = 8 - first onset time
            offset = timedelta(seconds=8 - perf.iloc[0]['onset'])
            # Set the index, resample to every second, and take the mean
            return perf.set_index(idx).resample('1s', offset=offset).mean()

        # Resample both raw dataframes
        resampled = [resample(data) for data in [self.keys_raw.copy(), self.drms_raw.copy()]]
        # Concatenate the resampled dataframes together and get the row-wise mean
        conc = pd.DataFrame(pd.concat([df['bpm'] for df in resampled], axis=1).mean(axis=1), columns=['bpm'])
        # Get the elapsed time column as an integer
        conc['my_onset'] = conc.index.total_seconds()
        # Create the regression model and extract the tempo slope coefficient
        return smf.ols('bpm~my_onset', data=conc.dropna()).fit().params.iloc[1:].values[0]

    def _extract_pairwise_asynchrony(
            self
    ):
        """
        Extract pairwise asynchrony as a float (in milliseconds, as standard for this unit)

        Outside the control conditions -- where both participants received the same feedback! -- this
        value probably doesn't mean that much, because participants heard different things from
        each others performance due to the latency. What this value does give, however, is a sense
        of whether both performers tried to match the delayed feedback they heard.

        Method:
        ---
        - Take the standard deviation of the asynchrony column from both nearest neighbour dataframes
        - Square both standard deviations
        - Calculate the mean of these squared standard deviation
        - Take the square root of the mean (RMS = Root-Mean-Square)
        - Multiply the result by 1000 to convert to milliseconds
        """
        return np.sqrt(np.mean(np.square([self.keys_nn.asynchrony.std(), self.drms_nn.asynchrony.std()]))) * 1000

    def _nearest_neighbour(
            self, live_arr: np.ndarray, delayed_arr: np.ndarray, match_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        For a single performer, matches each of their live onsets with the closest delayed onset from their partner.

        Method:
        ---
        NB. 'our' refers to the live performer, 'partner' refers to the delayed performer in the following description.

        - For every onset we played:
        - First, subtract this from the entire array of onsets our partner played
        - The smallest *positive* value in this matrix is the closest onset *ahead* of where we currently are
        - The largest *negative* value in this matrix is the closest onset *behind* where we currently are
        - In general, we assume that, with latency, our partner will be playing *ahead* of us
        - As such, if the next onset ahead of where we are is closer to us than the one behind, match our onset with it
        - However, if the onset behind where we are is closer, but the difference between the two onsets is less than
         our tolerance threshold, we should still match with the onset ahead of us.
        - Finally, if the onset behind where we are is closer, and the difference is greater than our tolerance
        threshold, only then should we match with the onset behind us.
        - If match_duplicates is true, try and match and duplicated partner onsets with another partner onset that
        wasn't matched to anything, if this is reasonable.
        - For any of our onsets where we matched with the same partner onset twice, set the duplicate appearance to NaN
        """
        results = []
        # Iterate through all values in our live array
        for i in range(live_arr.shape[0]):
            # Matrix subtraction
            temp_result = delayed_arr - live_arr[i]
            # Get the closest onset ahead of where I currently am
            closest_onset_played_ahead_of_me = np.min(np.where(temp_result > 0, temp_result, np.inf))
            # Get the closest onset played before where I currently am
            closest_onset_played_behind_me = np.max(np.where(temp_result < 0, temp_result, -np.inf))
            # In general, we can assume that we're going to be playing behind our delayed partner
            # If the next onset ahead of me is closer than the one behind me (with tolerance), match with it
            if abs(closest_onset_played_ahead_of_me - self._nn_tolerance) < abs(closest_onset_played_behind_me):
                match = closest_onset_played_ahead_of_me
            # However, if the onset behind me is closer to where I currently am:
            else:
                match = closest_onset_played_behind_me
            # Get the index of our match from our partner's array of onsets
            nearest_neighbour = delayed_arr[np.where(temp_result == match)][0]
            # Append my onset and our nearest neighbour match to our results
            results.append((live_arr[i], nearest_neighbour))
        # Create the dataframe
        nn_df = pd.DataFrame(results, columns=['my_onset', 'their_onset'])
        # Attempt to pair remaining unmatched onsets by our partner with duplicate matches
        if match_duplicates:
            self._attempt_to_match_duplicates_with_remainders(nn_df, delayed_arr)
        # Set any remaining duplicate values to NaN
        nn_df.loc[nn_df.duplicated(subset='their_onset'), 'their_onset'] = np.nan
        # Format the dataframe for our model
        return self._format_df_for_model(nn_df)

    @staticmethod
    def _attempt_to_match_duplicates_with_remainders(
            nn_with_dup: pd.DataFrame, remaining_unmatched_partner_onsets: np.ndarray
    ):
        """
        Attempts to pair duplicate matches from the nearest neighbour algorithm with unmatched onsets from our partner

        Method:
        ---
        - Iterate through all of our onsets that were matched to the same onset played by our partner.
        - For each duplicate match, if we can find an onset played by our partner not yet matched with one of ours,
        and this is between our previous and next match, automatically match with it, regardless of distance.

        """
        # Find duplicate matches
        duplicated = nn_with_dup[nn_with_dup.duplicated(subset='their_onset')]
        # Find onsets from our partner that have not been matched yet
        unmatched = np.in1d(remaining_unmatched_partner_onsets, nn_with_dup['their_onset'].to_numpy())
        remainders = remaining_unmatched_partner_onsets[unmatched]
        # If we have unmatched onsets
        if len(remainders) > 0:
            # Iterate through our duplicate matches
            for idx, row in duplicated.iterrows():
                try:
                    # Find the closest unmatched partner onset to our onset
                    temp = abs(remainders - row['my_onset'])
                    match = remainders[np.where(temp == np.min(temp))][0]
                    # If the match is between our previous matched onset and our next matched onset, use it instead
                    if nn_with_dup.iloc[idx - 1]['my_onset'] < match < nn_with_dup.iloc[idx + 1]['my_onset']:
                        nn_with_dup.iloc[idx]['their_onset'] = match
                except IndexError:
                    pass

    def _iqr_filter(
            self, df: pd.DataFrame, col: str
    ):
        """

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
        # Filter inter-onset intervals
        df = self._iqr_filter(df, 'my_prev_ioi')
        # Shift to get next IOI
        df['my_next_ioi'] = df['my_prev_ioi'].shift(-1)
        # Extract asynchrony between onsets
        df['asynchrony'] = df['their_onset'] - df['my_onset']
        # Extract difference of IOIs
        df['my_next_ioi_diff'] = df['my_next_ioi'].diff()
        df['my_prev_ioi_diff'] = df['my_prev_ioi'].diff()
        return df

    def _create_phase_correction_model(
            self, df: pd.DataFrame
    ):
        return smf.ols(self._model, data=df.dropna(), missing='drop').fit()

    def _create_summary_dictionary(
            self, c, md,
    ):
        """
        Creates a dictionary of summary statistics, used when analysing all models.
        """
        return {
            'trial': c['trial'],
            'block': c['block'],
            'latency': c['latency'],
            'jitter': c['jitter'],
            'instrument': c['instrument'],
            'raw_beats': [c['midi_bpm']],
            'total_beats': autils.extract_interpolated_beats(c)[0],
            'interpolated_beats': autils.extract_interpolated_beats(c)[1],
            'tempo_slope': self.tempo_slope,
            # 'ioi_std': std,
            # 'ioi_npvi': npvi,
            'pw_asym': self.pairwise_asynchrony,
            'intercept': md.params.iloc[0],
            'intercept_stderr': md.bse.iloc[0],
            'correction_self': md.params.iloc[1],
            'correction_self_stderr': md.bse.iloc[1],
            'correction_partner': md.params.iloc[2],
            'correction_partner_stderr': md.bse.iloc[2],
            'resid_std': np.std(md.resid),
            'resid_len': len(md.resid),
            'rsquared': md.rsquared,
            'zoom_arr': c['zoom_array'],
            'success': c['success'],
            'interaction': c['interaction'],
            'coordination': c['coordination']
        }

#
# if __name__ == '__main__':
#     raw_data = autils.load_data(input_filepath)
#     res = []
#     for z in autils.zip_same_conditions_together(raw_data=raw_data):
#         # Iterate through keys and drums performances in a condition together
#         for c1, c2 in z:
#             pcm = PhaseCorrectionModel(c1, c2)
#             res.append(pcm.keys_dic)
#             res.append(pcm.drms_dic)
#
#     d = pd.DataFrame(res)

