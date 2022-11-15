import pandas as pd
import numpy as np
import json
import os
import statsmodels.formula.api as smf
from datetime import timedelta
from sklearn.covariance import EllipticEnvelope

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
from src.analyse.simulations_ratio import Simulation


class PhaseCorrectionModel:
    def __init__(
            self, c1_keys: list, c2_drms: list, **kwargs
    ):
        """

        """
        # Parameters
        self._nn_tolerance: float = 0.1  # The amount of tolerance to use when matching onsets
        self._iqr_filter_range: tuple[float] = (0.25, 0.75)  # Filter IOIs above and below this
        self._model = kwargs.get('model', 'my_next_ioi_diff~my_prev_ioi_diff+asynchrony')  # regression model
        # The cleaned data, before dataframe conversion
        self.keys_raw: dict = c1_keys
        self.drms_raw: dict = c2_drms
        # The raw data converted into dataframe format
        self.keys_df: pd.DataFrame = self._generate_df(c1_keys)
        self.drms_df: pd.DataFrame = self._generate_df(c2_drms)
        # The nearest neighbour models, matching live and delayed onsets together
        self._contamination = self._get_contamination_value_from_json()
        self.keys_nn: pd.DataFrame = self._nearest_neighbour(
             live_arr=self.keys_df['my_onset'].to_numpy(), delayed_arr=self.drms_df['my_onset'].to_numpy(),
             zoom_arr=self.keys_raw['zoom_array'], instr='Keys'
        )
        self.drms_nn: pd.DataFrame = self._nearest_neighbour(
            live_arr=self.drms_df['my_onset'].to_numpy(), delayed_arr=self.keys_df['my_onset'].to_numpy(),
            zoom_arr=self.drms_raw['zoom_array'], instr='Drums'
        )
        # The phase correction models
        self.keys_md = self._create_phase_correction_model(self.keys_nn)
        self.drms_md = self._create_phase_correction_model(self.drms_nn)
        # Additional summary statistics
        # TODO: need to have functions here to extract tempo stability, etc
        self.tempo_slope = self._extract_tempo_slope()
        # self.pairwise_asynchrony: float = self._extract_pairwise_asynchrony()
        # The summary dictionaries, which can be appended to a dataframe of all performances
        self.keys_dic = self._create_summary_dictionary(c1_keys, self.keys_md, self.keys_nn)
        self.drms_dic = self._create_summary_dictionary(c2_drms, self.drms_md, self.drms_nn)

    def _get_contamination_value_from_json(
            self
    ) -> float:
        """

        """
        cwd = os.path.dirname(os.path.abspath(__file__)) + '\contamination_params.json'
        js = json.load(open(cwd))
        return [
            i['contamination'] for i in js
            if i['trial'] == self.keys_raw['trial'] and i['block'] == self.keys_raw['block']
            and i['latency'] == self.keys_raw['latency'] and i['jitter'] == self.keys_raw['jitter']
        ][0]

    def _append_zoom_array(
            self, perf_df: pd.DataFrame, zoom_arr: np.array, onset_col: str = 'my_onset'
    ) -> pd.DataFrame:
        """
        Appends a column to a dataframe showing the approx amount of latency by AV-Manip for each event in a performance
        """
        # TODO: need to add in the constant amount of delay induced by the testbed - 12ms?
        # Create a new array with our Zoom latency times and the approx timestamp they were applied to the performance
        # 8 = performance start time (after count-in, 0.75 = refresh rate
        z = np.c_[zoom_arr, np.linspace(8, 8 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
        # Initialise an empty column in our performance dataframe
        perf_df['latency'] = np.nan
        # Loop through successive timestamps in our zoom dataframe
        for first, second in zip(z, z[1:]):
            # Set any rows in the performance dataframe between our timestamps to the respective latency time
            perf_df.loc[perf_df[onset_col].between(first[1], second[1]), 'latency'] = first[0]
        return perf_df

    def _generate_df(
            self, data: list, threshold: float = 0.2
    ) -> pd.DataFrame:
        """
        Create dataframe, append zoom array, and add a column with our delayed onsets.
        This latter column replicates the performance as it would have been heard by our partner.
        """
        df = pd.DataFrame(data['midi_bpm'], columns=['my_onset', 'pitch', 'velocity']).sort_values('my_onset')
        df = self._append_zoom_array(df,  data['zoom_array'])
        # Drop pitch and velocity columns if not needed (default)
        df['my_prev_ioi'] = df['my_onset'].diff().astype(float)

        # Clean to remove any spurious onsets
        # df['ioi'] = iqr_filter('ioi', df, iqr_range, )

        # Remove iois below threshold
        temp = df.dropna()
        temp = temp[temp['my_prev_ioi'] < threshold]
        df = df[~df.index.isin(temp.index)]
        # Recalculate IOI column after removing those below threshold
        df['my_prev_ioi'] = df['my_onset'].diff().astype(float)
        # Fill na in latency column with next/previous valid observation
        df['latency'] = df['latency'].bfill().ffill()
        # Add the latency column (divided by 1000 to convert from ms to sec) to the onset column
        df['my_onset_delayed'] = (df['my_onset'] + (df['latency'] / 1000))

        return df

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

        def resample(perf: pd.DataFrame):
            # Create the timedelta index
            idx = pd.to_timedelta([timedelta(seconds=val) for val in perf['my_onset']])
            perf['bpm'] = 60 / perf['my_prev_ioi']
            # Create the offset = 8 - first onset time
            offset = timedelta(seconds=8 - perf.iloc[0]['my_onset'])
            # Set the index, resample to every second, and take the mean
            return perf.set_index(idx).resample('1s', offset=offset).mean()

        # Resample both raw dataframes
        resampled = [resample(data) for data in [self.keys_df.copy(), self.drms_df.copy()]]
        # Concatenate the resampled dataframes together and get the row-wise mean
        conc = pd.DataFrame(pd.concat([df['bpm'] for df in resampled], axis=1).mean(axis=1), columns=['bpm'])
        # Get the elapsed time column as an integer
        conc['my_onset'] = conc.index.total_seconds()
        # Create the regression model and extract the tempo slope coefficient
        return smf.ols('bpm~my_onset', data=conc.dropna()).fit()

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
            self, live_arr: np.ndarray, delayed_arr: np.ndarray, zoom_arr: np.ndarray, instr: str
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
        for i in range(live_arr.shape[0]):
            # Matrix subtraction
            temp_result = abs(delayed_arr - live_arr[i])
            # Get the closest onset without latency
            match = np.min(temp_result)
            closest_element = delayed_arr[np.where(temp_result == match)][0]
            results.append((live_arr[i], closest_element))

        nn_df = pd.DataFrame(results, columns=['my_onset', 'their_onset'])
        nn_df['asynchrony'] = nn_df['their_onset'] - nn_df['my_onset']

        # 180ms CLEANING ONLY
        if self.keys_raw['latency'] == 180:
            # TRIALS 1/5 -- HISTOGRAM CLEANING
            if self.keys_raw['trial'] == 1 or self.keys_raw['trial'] == 5:
                cut = pd.cut(nn_df['asynchrony'], bins=2)
                smallest_bin = cut.value_counts().reset_index().iloc[1].rename({'index': 'bound'}).bound.mid
                largest_bin = cut.value_counts().reset_index().iloc[0].rename({'index': 'bound'}).bound.mid
                nn_df['category'] = cut
                for idx, row in nn_df.iterrows():
                    if smallest_bin > largest_bin:
                        matrix_subtract = delayed_arr - row['my_onset']
                        nxt = np.max(np.where(matrix_subtract < 0, matrix_subtract, -np.inf))
                        try:
                            nxt_nearest = delayed_arr[np.where(matrix_subtract == nxt)][0]
                        except IndexError:
                            continue
                        else:
                            if nxt_nearest != row['my_onset']:
                                nn_df.at[idx, 'their_onset'] = nxt_nearest
                            else:
                                nn_df.at[idx, 'their_onset'] = np.nan
                    elif smallest_bin < largest_bin:
                        matrix_subtract = delayed_arr - row['my_onset']
                        nxt = np.min(np.where(matrix_subtract > 0, matrix_subtract, np.inf))
                        try:
                            nxt_nearest = delayed_arr[np.where(matrix_subtract == nxt)][0]
                        except IndexError:
                            continue
                        else:
                            if nxt_nearest != row['my_onset']:
                                nn_df.at[idx, 'their_onset'] = nxt_nearest
                            else:
                                nn_df.at[idx, 'their_onset'] = np.nan
                nn_df['asynchrony'] = nn_df['their_onset'] - nn_df['my_onset']

        if self._contamination != 0:
            ee = EllipticEnvelope(contamination=self._contamination, random_state=1)
            ee.fit(nn_df['asynchrony'].to_numpy().reshape(-1, 1))
            nn_df['outliers'] = ee.predict(nn_df['asynchrony'].to_numpy().reshape(-1, 1))

            for idx, row in nn_df[nn_df['outliers'] == -1].iterrows():
                matrix_subtract = abs(delayed_arr - row['my_onset'])
                i = np.where(matrix_subtract == np.partition(matrix_subtract, 2)[2])
                next_nearest = delayed_arr[i]
                predicted_async = next_nearest - row['my_onset']
                if ee.predict(np.float64(predicted_async).reshape(-1, 1))[0] == 1:
                    nn_df.at[idx, 'their_onset'] = next_nearest.astype(np.float64)
                else:
                    nn_df.at[idx, 'their_onset'] = np.nan

        nn_df['asynchrony'] = nn_df['their_onset'] - nn_df['my_onset']
        median = nn_df['asynchrony'].median()
        for idx, grp in nn_df.groupby('their_onset'):
            if len(grp) > 1:
                idxmax = abs(grp['asynchrony'] - median).idxmax()
                nn_df.at[idxmax, 'their_onset'] = np.nan

        nn_df = self._append_zoom_array(nn_df, zoom_arr=zoom_arr)
        nn_df['their_onset'] += (nn_df['latency'] / 1000)
        nn_df['asynchrony'] = nn_df['their_onset'] - nn_df['my_onset']
        nn_df = nn_df.drop(['latency'], axis=1)

        return self._format_df_for_model(nn_df)

    def _iqr_filter(
            self, df: pd.DataFrame, col: str, filter: tuple = (0.25, 0.75)
    ):
        """

        """
        # Get lower quartile
        q1 = df[col].quantile(filter[0])
        # Get upper quartile
        q3 = df[col].quantile(filter[1])
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
        df = self._iqr_filter(df, 'my_prev_ioi', filter=(0.1, 0.9))
        # Shift to get next IOI
        df['my_next_ioi'] = df['my_prev_ioi'].shift(-1)
        # Extract difference of IOIs
        df['my_next_ioi_diff'] = df['my_next_ioi'].diff()
        df['my_prev_ioi_diff'] = df['my_prev_ioi'].diff()
        return df

    def _create_phase_correction_model(
            self, df: pd.DataFrame
    ):
        return smf.ols(self._model, data=df.dropna(), missing='drop').fit()

    def _create_summary_dictionary(
            self, c, md, nn,
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
            'tempo_slope': self.tempo_slope.params.iloc[1],
            'tempo_intercept': self.tempo_slope.params.iloc[0],
            'contamination': self._contamination,
            'asynchrony_na': nn['asynchrony'].isna().sum(),
            # 'ioi_std': std,
            # 'ioi_npvi': npvi,
            # 'pw_asym': self.pairwise_asynchrony,
            'intercept': md.params.iloc[0],
            'correction_self': md.params.iloc[1],
            'correction_partner': md.params.iloc[2],
            'resid_std': np.std(md.resid),
            'resid_len': len(md.resid),
            'rsquared': md.rsquared,
            'zoom_arr': c['zoom_array'],
            'success': c['success'],
            'interaction': c['interaction'],
            'coordination': c['coordination']
        }


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 500)
    raw_data = autils.load_data(r"C:\Python Projects\jazz-jitter-analysis\data\processed")
    res = []
    for z in autils.zip_same_conditions_together(raw_data=raw_data):
        # Iterate through keys and drums performances in a condition together
        for c1, c2 in z:
            print(c1['trial'], c1['block'], c1['latency'], c1['jitter'])
            pcm = PhaseCorrectionModel(c1, c2)
            res.append(pcm.keys_dic)
            res.append(pcm.drms_dic)

    import seaborn as sns
    import matplotlib.pyplot as plt
    from src.visualise.phase_correction_graphs import PairGrid
    df = pd.DataFrame(res)
    g = sns.FacetGrid(data=df, col='trial', hue='instrument')
    g.map(sns.histplot, 'correction_partner')
    plt.show()

    pg = PairGrid(
        df=df, xvar='correction_partner', output_dir=r"C:\Python Projects\jazz-jitter-analysis\reports\test", xlim=(-0.5, 1.5), xlabel='Coupling constant'
    )
    pg.create_plot()

    coef = []
    for idx, grp in df.groupby(by=['trial', 'block', 'latency', 'jitter']):
        ke = grp[grp['instrument'] == 'Keys'].reset_index(drop=False)
        dr = grp[grp['instrument'] != 'Keys'].reset_index(drop=False)
        z = grp.reset_index().iloc[0]['zoom_arr']
        sim = Simulation(ke, dr, z, num_simulations=500)
        sim.create_all_simulations()
        ts = sim.get_average_tempo_slope()
        print(idx, ke['tempo_slope'].iloc[0], ts)
        coef.append((*idx, ke['tempo_slope'].iloc[0], ts))

        for keys, drms in zip(sim.keys_simulations, sim.drms_simulations):
            avg = sim._get_grand_average_tempo([keys, drms])
            avg = avg[(avg.index.seconds > 8) & (avg.index.seconds < 93)]
            plt.plot(avg.index.seconds, (60 / avg.my_next_ioi).rolling(window='8s').mean(), alpha=0.01,
                     color=vutils.BLACK)
        grand_avg = sim._get_grand_average_tempo(
            [sim._get_grand_average_tempo([k, d]) for k, d in zip(sim.keys_simulations, sim.drms_simulations)]
        )
        grand_avg = grand_avg[(grand_avg.index.seconds > 8) & (grand_avg.index.seconds < 93)]
        plt.plot(grand_avg.index.seconds, (60 / grand_avg.my_next_ioi).rolling(window='8s').mean(), alpha=1,
                 color=vutils.BLACK)
        plt.title(f'Duo {idx[0]}, block {idx[1]}, latency {idx[2]}, jitter {idx[3]}')
        plt.ylim(30, 160)
        plt.show()

    coef_df = pd.DataFrame(coef, columns=['trial', 'block', 'latency', 'jitter', 'actual', 'simulated'])
    fig, ax = plt.subplots(1, 1)
    g = sns.scatterplot(data=coef_df, x='simulated', y='actual', hue='latency', palette='tab10')
    # g = sns.regplot(data=coef_df, x='simulated', y='actual', scatter=False, ci=None)
    g.set(ylim=(-0.8, 0.8), xlim=(-0.8, 0.8), xlabel='Simulated slope (BPM/s)', ylabel='Actual slope (BPM/s)')
    g.plot([0, 1], [0, 1], transform=ax.transAxes, c='#000000')
    plt.show()


    # coef = []
    # for idx, grp in d.groupby(by=['trial', 'block', 'latency', 'jitter', 'contamination']):
    #     ke = grp[grp['instrument'] == 'Keys'].reset_index(drop=False)
    #     dr = grp[grp['instrument'] != 'Keys'].reset_index(drop=False)
    #     z = grp.reset_index().iloc[0]['zoom_arr']
    #     try:
    #         sim = Simulation(ke, dr, z, num_simulations=1000)
    #         sim.create_all_simulations()
    #         ts = sim.get_average_tempo_slope()
    #         dic = {
    #             'trial': idx[0],
    #             'block': idx[1],
    #             'latency': idx[2],
    #             'jitter': idx[3],
    #             'contamination': idx[4],
    #             'actual_slope': ke['tempo_slope'].iloc[0],
    #             'keys_asynchrony_na': ke['asynchrony_na'].iloc[0],
    #             'drms_asynchrony_na': dr['asynchrony_na'].iloc[0],
    #             'keys_self_coupling': ke['correction_self'].iloc[0],
    #             'drms_self_coupling': dr['correction_self'].iloc[0],
    #             'keys_partner_coupling': ke['correction_partner'].iloc[0],
    #             'drms_partner_coupling': dr['correction_partner'].iloc[0],
    #             'keys_intercept': ke['intercept'].iloc[0],
    #             'drms_intercept': dr['intercept'].iloc[0],
    #             'predicted_slope': ts
    #         }
    #         print(dic)
    #         coef.append(dic)
    #     except:
    #         dic = {
    #             'trial': idx[0],
    #             'block': idx[1],
    #             'latency': idx[2],
    #             'jitter': idx[3],
    #             'contamination': idx[4],
    #             'actual_slope': ke['tempo_slope'].iloc[0],
    #             'keys_asynchrony_na': ke['asynchrony_na'].iloc[0],
    #             'drms_asynchrony_na': dr['asynchrony_na'].iloc[0],
    #             'keys_self_coupling': ke['correction_self'].iloc[0],
    #             'drms_self_coupling': dr['correction_self'].iloc[0],
    #             'keys_partner_coupling': ke['correction_partner'].iloc[0],
    #             'drms_partner_coupling': dr['correction_partner'].iloc[0],
    #             'keys_intercept': ke['intercept'].iloc[0],
    #             'drms_intercept': dr['intercept'].iloc[0],
    #             'predicted_slope': np.nan
    #         }
    #         print(dic)
    #         coef.append(dic)
    # df = pd.DataFrame(coef)
    # print(df)
    # df.to_clipboard(sep=',')
    # df.to_csv(r'C:\Python Projects\jazz-jitter-analysis\reports\test\df_zeroed.csv', sep=',')
