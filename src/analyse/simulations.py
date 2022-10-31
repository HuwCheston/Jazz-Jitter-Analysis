import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from scipy import stats
import statsmodels.formula.api as smf
import warnings

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


class Simulation:
    """

    """

    def __init__(
            self, keys_data, drms_data, parameter, latency, **kwargs
    ):
        # Default parameters
        self._initial_onset: float = 8.0  # The initial onset of a performance, after 8 second count-in
        self._initial_ioi: float = 0.5  # The presumed initial length of the first IOI = crotchet at 120BPM
        self._end: float = 101.5  # Stop generating data after we exceed this onset value
        self._max_iter: int = 500  # If we generate more than this number of IOIs, we're probably stuck in a loop
        self._resample_interval: timedelta = timedelta(seconds=1)  # Get mean of IOIs within this window
        self._rolling_period: timedelta = timedelta(seconds=4)  # Apply a rolling window of this size to the data
        self._debug_params: dict = {  # Used when debugging: will generate the same simulation every time
            'correction_self': 0.5,
            'correction_partner': 0.5,
            'intercept': 0.5,
            'noise': np.array([0.02])
        }

        # Check and generate our input data
        self.keys_data: pd.DataFrame = self._check_input_data(keys_data)
        self.drms_data: pd.DataFrame = self._check_input_data(drms_data)
        self.leader: str = kwargs.get('leader', None)
        self.parameter: str = self._check_simulation_parameter(parameter)
        self.latency: np.array = self._append_timestamps_to_latency_array(latency)
        self.num_simulations: int = kwargs.get('num_simulations', 500)

        self.keys_params: dict = self._get_simulation_params(input_data=self.keys_data)
        self.drms_params: dict = self._get_simulation_params(input_data=self.drms_data)

        self.keys_simulations: list[pd.DataFrame] = []
        self.drms_simulations: list[pd.DataFrame] = []

    @staticmethod
    def _check_input_data(
            input_data: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Checks to make sure that input dataframe is in correct format and has all necessary columns.
        Raises ValueError if any checks are failed
        """
        required_cols: list[str] = [
            'instrument',
            'total_beats',
            'raw_beats_1',
            'raw_beats_2',
            'correction_self',
            'correction_self_stderr',
            'correction_partner',
            'correction_partner_stderr',
            'intercept',
            'intercept_stderr',
            'resid_std',
            'resid_len',
        ]
        # If we didn't pass a dataframe
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError('Data input was either not provided or is of invalid type')
        # If we don't have all the necessary columns to create the simulation
        if len([col for col in required_cols if col not in input_data.columns]) != 0:
            raise ValueError(
                f'Some required columns in data input were missing '
                f'(missing: {", ".join(str(x) for x in required_cols if x not in input_data.columns)})')
        # If we passed data from multiple performances
        if len(input_data) != 1:
            raise ValueError(
                f'Input data should be from one performance only (passed data: {len(input_data)} performances)')
        # If all checks passed, return data
        else:
            return input_data[required_cols]

    def _check_simulation_parameter(
            self, input_parameter: str
    ) -> str | None:
        """
        Checks if the simulation parameter given by the user is acceptable , and raises value error if not
        """
        acceptable_parameters: list[str] = [
            'original',  # Use the original coefficients we've passed in
            'original_variable',  # Use the original coefficients, but add random noise according to std err
            'democracy',  # Coefficients for both performers set to the mean
            'dictatorship',
            'anarchy',
            'debug'
        ]
        # If we haven't passed an acceptable parameter
        if input_parameter not in acceptable_parameters:
            raise ValueError(
                f'{input_parameter} not in acceptable simulation parameters '
                f'(options: {", ".join(str(x) for x in acceptable_parameters)})')
        # Additional checks if we're setting the leadership parameter
        if input_parameter == 'leadership':
            # We haven't passed a leadership parameter, or it's an incorrect type
            if not isinstance(self.leader, str):
                raise ValueError(
                    'If setting leadership simulation parameter, must also pass leader keyword argument as string')
            # We've passed an incompatible leadership parameter
            elif self.leader.lower() not in ['keys', 'drums']:
                raise ValueError(f'Leader keyword argument {self.leader} is invalid: must be either keys or drums')
        # If all checks passed, return the input parameter
        return input_parameter

    @staticmethod
    def _append_timestamps_to_latency_array(
            latency_array, offset: int = 8, resample_rate: float = 0.75
    ) -> np.array:
        """
        Appends timestampts showing the onset time for each value in the latency array applied to a performance
        """
        # Define the start and endpoint for the linear space
        start = offset
        end = offset + (len(latency_array) * resample_rate)
        # Create the linear space
        lin = np.linspace(start, end, num=len(latency_array), endpoint=False)
        # Append the two arrays together
        return np.c_[latency_array / 1000, lin]

    def _get_simulation_params(
            self, input_data: pd.DataFrame
    ) -> dict:
        """
        Returns the simulation parameters from the given input parameter.
        """
        # Averaging function: returns average of keys and drums value for a given variable
        mean = lambda s: np.mean([self.keys_data[s], self.drms_data[s]])
        rand = lambda s: np.random.normal(loc=input_data[s], scale=input_data[f'{s}_stderr'],
                                          size=int(input_data['total_beats']))
        # Define the initial dictionary, with the noise term and the latency array (with timestamps)
        d: dict = {
            'noise': np.random.normal(loc=0, scale=input_data['resid_std'], size=int(input_data['resid_len'])),
            'latency': self.latency,
        }
        # Original coupling: uses coefficients, intercept from the model itself
        if self.parameter == 'original':
            d.update({
                'correction_self': input_data['correction_self'].iloc[0],
                'correction_partner': input_data['correction_partner'].iloc[0],
                'intercept': input_data['intercept'].iloc[0],
            })
        elif self.parameter == 'original_variable':
            d.update({
                'correction_self': rand('correction_self'),
                'correction_partner': rand('correction_partner'),
                'intercept': rand('intercept'),
            })
        # Democracy: uses mean coefficients and intercepts from across the duo
        elif self.parameter == 'democracy':
            d.update({
                'correction_self': mean('correction_self'),
                'correction_partner': mean('correction_partner'),
                'intercept': mean('intercept'),
            })
        # Leadership:
        elif self.parameter == 'dictatorship':
            # The instrument we're passing is the leader
            if input_data['instrument'].iloc[0].lower() == self.leader.lower():
                d.update({
                    'correction_self': -abs(mean('correction_self')),
                    'correction_partner': 0,
                    'intercept': np.mean(mean('intercept'))
                })
            # The instrument we're passing is the follower
            else:
                d.update({
                    'correction_self': -abs(mean('correction_self')),
                    'correction_partner': mean('correction_partner'),
                    'intercept': mean('intercept')
                })
        # No coupling: all coefficients and intercepts set to 0, with only random noise effecting the data
        elif self.parameter == 'anarchy':
            d.update({
                'correction_self': mean('correction_self'),
                'correction_partner': 0,
                'intercept': mean('intercept'),
            })
        # Debugging: all terms are made constant by setting to those defined in the debug_params.
        # The noise term is no longer chosen randomly, resulting in the same output for every simulation.
        elif self.parameter == 'debugging':
            d = self._debug_params
        return d

    def _initialise_empty_data(
            self, first_latency_value: float = 0
    ) -> dict:
        """
        Initialises an empty dictionary that we will fill with data as the simulation runs
        """
        return {
            'my_actual_onset': [self._initial_onset],
            'their_actual_onset': [self._initial_onset],
            'their_heard_onset': [self._initial_onset + first_latency_value],
            'latency_now': [first_latency_value],
            'my_next_ioi': [self._initial_ioi],
            'my_prev_ioi': [np.nan],
            'asynchrony': [first_latency_value]
        }

    def run_simulations(
            self
    ) -> None:
        """
        Run the simulations and create a list of dataframes for each individual performer
        """
        # Clear both lists
        self.keys_simulations.clear()
        self.drms_simulations.clear()
        # Run the number of simulations which we provided
        for _ in range(0, self.num_simulations):
            # Generate the data and append to the required lists
            keys, drms = self._generate_data()
            self.keys_simulations.append(keys)
            self.drms_simulations.append(drms)
        # Remove simulations that failed to complete
        self.keys_simulations = self._remove_broken_simulations(self.keys_simulations)
        self.drms_simulations = self._remove_broken_simulations(self.drms_simulations)

    @staticmethod
    def _remove_broken_simulations(data: list[pd.DataFrame | None]):
        """
        Remove simulations that have failed to complete
        """
        return [x for x in data if x is not None]

    def _generate_data(
            self
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Generates data for one simulation
        """

        def predict_next_ioi(sim_data: dict, sim_params: dict) -> float:
            get = lambda s: np.random.choice(sim_params[s]) if isinstance(sim_params[s], np.ndarray) else sim_params[s]
            # Multiply previous IOI by coupling to self coefficient
            correction_self: float = sim_data['my_prev_ioi'][-1] * get('correction_self')
            # Multiply async by coupling to partner coefficient
            correction_partner: float = sim_data['asynchrony'][-1] * get('correction_partner')
            # Get the intercept (this is a constant)
            intercept: float = get('intercept')
            # Take a random sample from the noise array
            noise: float = get('noise')
            # Add all the terms together and return as our predicted next IOI
            return correction_self + correction_partner + intercept + noise

        def get_latency_at_onset(sim_data: dict, sim_params: dict) -> float | None:
            try:
                # Subset our latency array for just the onset times times
                latency_onsets: np.array = sim_params['latency'][:, 1]
                # Get the last onset our partner played
                partner_onset: float = sim_data['their_actual_onset'][-1]
                # Get the closest minimum latency onset time to our partner's onset
                matched_latency_onset = latency_onsets[latency_onsets < partner_onset].max()
                # Get the latency value associated with that latency onset time and return
                latency_value: float = float(sim_params['latency'][latency_onsets == matched_latency_onset][:, 0])
            except ValueError:
                return None
            else:
                return latency_value

        # Generate our starter data
        keys_data = self._initialise_empty_data(self.latency[0][0])
        drms_data = self._initialise_empty_data(self.latency[0][0])
        # Start the simulation
        iter_count: int = 0
        while keys_data['my_actual_onset'][-1] < self._end or drms_data['my_actual_onset'][-1] < self._end:
            # Shift the predicted IOI by one beat
            keys_data['my_prev_ioi'].append(keys_data['my_next_ioi'][-1])
            drms_data['my_prev_ioi'].append(drms_data['my_next_ioi'][-1])
            # Get our next onset time by adding our predicted next IOI to our last onset
            keys_data['my_actual_onset'].append(keys_data['my_actual_onset'][-1] + keys_data['my_next_ioi'][-1])
            drms_data['my_actual_onset'].append(drms_data['my_actual_onset'][-1] + drms_data['my_next_ioi'][-1])
            # Get our partners last actual onset time (from their dataframe)
            keys_data['their_actual_onset'].append(drms_data['my_actual_onset'][-1])
            drms_data['their_actual_onset'].append(keys_data['my_actual_onset'][-1])
            # Calculate the amount of latency applied when our partner played their last onset
            keys_latency_now = get_latency_at_onset(keys_data, self.keys_params)
            drms_latency_now = get_latency_at_onset(drms_data, self.drms_params)
            # If we've received a ValueError at the previous stage, break out of the loop
            # We'll discard this simulation
            if keys_latency_now is None or drms_latency_now is None:
                return None, None
            # Else continue
            else:
                keys_data['latency_now'].append(get_latency_at_onset(keys_data, self.keys_params))
                drms_data['latency_now'].append(get_latency_at_onset(drms_data, self.drms_params))
            # Add on the latency time to our partner's onset, to find out when we actually heard them
            keys_data['their_heard_onset'].append(keys_data['latency_now'][-1] + keys_data['their_actual_onset'][-1])
            drms_data['their_heard_onset'].append(drms_data['latency_now'][-1] + drms_data['their_actual_onset'][-1])
            # Calculates the async between our partner's delayed onset and our actual onset
            keys_data['asynchrony'].append(keys_data['their_heard_onset'][-1] - keys_data['my_actual_onset'][-1])
            drms_data['asynchrony'].append(drms_data['their_heard_onset'][-1] - drms_data['my_actual_onset'][-1])
            # Predict our next IOI
            keys_data['my_next_ioi'].append(predict_next_ioi(keys_data, self.keys_params))
            drms_data['my_next_ioi'].append(predict_next_ioi(drms_data, self.drms_params))
            # Increase the iteration counter by one and raise a warning
            iter_count += 1
            if iter_count >= self._max_iter:
                warnings.warn(f'Maximum number of iterations {self._max_iter} exceeded')
                return None, None

        # Once the simulation has finished, return the data as a tuple
        return self._format_simulated_data(keys_data), self._format_simulated_data(drms_data)

    def _format_simulated_data(
            self, data: dict
    ) -> pd.DataFrame:
        """
        Formats data from one simulation by creating a dataframe, adding in the timedelta column, and resampling
        to get the mean IOI (defaults to every second)
        """
        # Create the dataframe from the dictionary
        df = pd.DataFrame(data)
        # Convert my onset column to a timedelta
        df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['my_actual_onset']])
        # Set the index to our timedelta column, resample (default every second), and get mean
        return df.set_index('td').resample(rule=self._resample_interval).mean()

    @staticmethod
    def _get_grand_average_tempo(
            all_perf: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Concatenate all simulations together and get the row-wise average (i.e. avg IOI every second)
        """
        return pd.DataFrame(pd.concat([df['my_prev_ioi'] for df in all_perf], axis=1).mean(axis=1),
                            columns=['my_prev_ioi'])

    @staticmethod
    def _get_grand_average_stdev(all_perf):
        return pd.DataFrame(pd.concat([df['my_prev_ioi'] for df in all_perf], axis=1).std(axis=1),
                            columns=['my_prev_ioi'])

    def get_avg_tempo_slope(self,):
        # TODO: this is gross
        # Raise an error if we haven't generated our simulations
        if len(self.keys_simulations) == 0 or len(self.drms_simulations) == 0:
            raise ValueError('Generate simulations first!')
        res = []
        for keys, drms in zip(self.keys_simulations, self.drms_simulations):
            ga = self._get_grand_average_tempo([keys, drms])
            ga['bpm'] = (60 / ga['my_prev_ioi']).rolling(self._rolling_period, min_periods=1).mean().astype(float)
            ga['my_actual_onset'] = ga.index.total_seconds()
            reg = smf.ols('bpm~my_actual_onset', data=ga.dropna()).fit().params.iloc[1:].values[0]
            res.append(reg)
        return np.mean(res)

    def _plot_original_performance(self):
        # TODO: gross
        for num, (k, d) in enumerate(zip([self.keys_data['raw_beats_1'][0], self.keys_data['raw_beats_2'][0]],
                                         [self.drms_data['raw_beats_1'][0], self.drms_data['raw_beats_2'][0]])):
            avg = autils.average_bpms(autils.generate_df(k), autils.generate_df(d), window_size=16)
            plt.plot(
                avg.elapsed, avg.bpm_rolling, alpha=0.5, lw=2, color=vutils.LINE_CMAP[num], label=f'Repeat {num+1}'
            )

    def _roll_bpm(self, s: pd.Series):
        return (60 / s).rolling(window=self._rolling_period, min_periods=1).mean()

    def plot_simulations(
            self
    ) -> None:
        """
        Plot all simulations and add in the grand average tempo
        """
        # Raise an error if we haven't generated our simulations
        if len(self.keys_simulations) == 0 or len(self.drms_simulations) == 0:
            raise ValueError('Generate simulations first!')
        # Define the roller function for converting IOIs to BPM and rolling
        # Iterate through keys and drums simulations and plot rolled BPM
        for keys, drms in zip(self.keys_simulations, self.drms_simulations):
            avg = self._get_grand_average_tempo([keys, drms])
            plt.plot(avg.index.seconds, self._roll_bpm(avg['my_prev_ioi']), alpha=0.05, lw=1, color=vutils.BLACK)
        # Get our grand average tempo for keys and drums simulations
        keys_avg = self._get_grand_average_tempo(self.keys_simulations)
        drms_avg = self._get_grand_average_tempo(self.drms_simulations)
        # Plot the rolled BPM of our grand average dataframe
        grand_average = self._get_grand_average_tempo([keys_avg, drms_avg])
        plt.plot(
            grand_average.index.seconds, self._roll_bpm(grand_average['my_prev_ioi']),
            alpha=1, lw=2, color=vutils.BLACK, label='Simulation'
        )
        sd_plus1 = self._get_grand_average_tempo([keys_avg + self._get_grand_average_stdev(self.keys_simulations),
                                                  drms_avg + self._get_grand_average_stdev(self.drms_simulations)])
        sd_neg1 = self._get_grand_average_tempo([keys_avg - self._get_grand_average_stdev(self.keys_simulations),
                                                 drms_avg - self._get_grand_average_stdev(self.drms_simulations)])
        plt.fill_between(
            grand_average.index.seconds, self._roll_bpm(sd_neg1['my_prev_ioi']),
            self._roll_bpm(sd_plus1['my_prev_ioi']), alpha=0.05, color=vutils.BLACK
        )
        self._plot_original_performance()
        # Set y limit
        plt.ylim((30, 160))
        plt.legend()
        plt.show()


class BarPlotSimulationParameters(vutils.BasePlot):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.res = kwargs.get('res', None)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(18.8, 7))

    @staticmethod
    def _normalise_values(df: pd.DataFrame) -> pd.DataFrame:
        df['norm'] = 1 - df['original']
        for s in [col for col in df.columns.to_list() if col != 'norm' or 'original']:
            df[s] += df['norm']
        df['original'] = 1
        return df

    def _format_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.res).pivot_table(values='tempo_slope', index='trial', columns='type')
        df = self._normalise_values(df)
        df.to_clipboard(sep=',')
        return (
            df.reset_index(drop=False)
              .drop(columns='norm')
              .rename(columns={'dictatorshipkeys': 'Leadership - Keys', 'dictatorshipdrums': 'Leadership - Drums'})
              .rename(columns={col: col.title() for col in df.columns.to_list()})
              .melt(id_vars='trial',
                    value_vars=['Original', 'Leadership - Keys', 'Leadership - Drums', 'Democracy', 'Anarchy'])
        )

    @vutils.plot_decorator
    def create_plot(self):
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_simulation_by_parameter.png'
        return self.fig, fname

    def _create_plot(self):
        return sns.barplot(
            data=self.df, x='type', y='value', color=vutils.LINE_CMAP[0], ax=self.ax,
            errcolor=vutils.BLACK, errwidth=2, edgecolor=vutils.BLACK, lw=2,
            capsize=0.03
        )

    @staticmethod
    def _get_significance_asterisks(p: float) -> str:
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    def _add_parameter_graphic(self):
        x = 0.26
        for i in range(1, 5):
            self.g.add_patch(plt.Rectangle((x + 0.005, -0.24), width=0.023, height=0.08, clip_on=False, linewidth=0,
                                           transform=self.ax.transAxes, facecolor=vutils.INSTR_CMAP[0]))
            self.g.text(s='K', x=x + 0.007, y=-0.225, transform=self.ax.transAxes, fontsize=24)
            if i == 2 or i == 3:
                self.ax.arrow(x=x + 0.033, y=-0.18, dx=0.025, dy=0, clip_on=False, transform=self.ax.transAxes,
                              linewidth=5, color=vutils.INSTR_CMAP[0], head_width=0.015, head_length=0.005)

            self.g.add_patch(plt.Rectangle((x + 0.07, -0.24), width=0.023, height=0.08, clip_on=False, linewidth=0,
                                           transform=self.ax.transAxes, facecolor=vutils.INSTR_CMAP[1]))
            self.g.text(s='D', x=x + 0.072, y=-0.225, transform=self.ax.transAxes, fontsize=24)
            if i == 1 or i == 3:
                self.ax.arrow(x=x + 0.063, y=-0.22, dx=-0.025, dy=0, clip_on=False, transform=self.ax.transAxes,
                              linewidth=5, color=vutils.INSTR_CMAP[1], head_width=0.015, head_length=0.005)

            self.g.add_patch(
                plt.Rectangle((x, -0.26), width=0.1, height=0.12, facecolor='silver', clip_on=False, linewidth=0,
                              transform=self.ax.transAxes, alpha=vutils.ALPHA))
            x += 0.19

    def _add_significance_asterisks_to_plot(self):
        df = self.df.pivot_table(values='value', index='trial', columns='type')
        z = zip(self.ax.patches[1:], ['Leadership - Keys', 'Leadership - Drums', 'Democracy', 'Anarchy'])
        orig_x = self.ax.patches[0].xy[0] + (self.ax.patches[0].get_width() / 2)

        for n, (p, s) in enumerate(z):
            r, sig = stats.ttest_ind(df['Original'], df[s], nan_policy='omit')
            x = p.xy[0] + p.get_width() - (self.ax.patches[0].get_width() / 2)
            self.ax.text(s=self._get_significance_asterisks(sig), x=x / 2, y=1.27 + n / 14, fontsize=24)
            self.ax.arrow(x=orig_x, y=1.29 + n / 14, dx=x, dy=0, )

    def _format_ax(self):
        self.ax.axhline(y=1, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2)
        self.g.set(xlabel='', ylabel='', ylim=(0, 1.6),
                   xticklabels=[col.replace(' - ', '\n') for col in self.df['type'].unique()])
        self.ax.tick_params(width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        new_value = 0.3
        for patch in self.ax.patches:
            diff = patch.get_width() - new_value
            patch.set_width(new_value)
            patch.set_x(patch.get_x() + diff * .5)
        self._add_significance_asterisks_to_plot()
        self._add_parameter_graphic()

    def _format_fig(self):
        self.fig.supylabel('Normalised tempo slope\n(BPM/s, 1 = original slope)', y=0.6, x=0.01)
        self.fig.supxlabel('Parameter', y=0.03)
        self.fig.subplots_adjust(bottom=0.27, top=0.95, left=0.08, right=0.97)


def _get_weighted_average(
        grped_df: pd.DataFrame
) -> pd.DataFrame:
    """

    """
    cols = [
        'correction_self',
        'correction_self_stderr',
        'correction_partner',
        'correction_partner_stderr',
        'intercept',
        'intercept_stderr',
        'resid_std',
        'resid_len',
        'total_beats'
    ]
    dic = {'instrument': grped_df['instrument'].iloc[0]}
    for st in cols:
        dic[st] = np.average(grped_df[st], weights=grped_df['weights'])
    return pd.DataFrame(pd.Series(dic)).transpose()


def _apply_weighted_average_columns(
        grp: pd.DataFrame, latency: int, jitter: float, weights: tuple = (3, 2, 2, 1)
) -> np.array:
    """

    """
    # Create weighted average conditions
    conditions = [
        (grp['latency'].eq(latency) & (grp['jitter'].eq(jitter))),  # Matches latency and jitter = 3
        (grp['latency'].eq(latency) & (grp['jitter'].ne(jitter))),  # Matches latency, not jitter = 2
        (grp['latency'].ne(latency) & (grp['jitter'].eq(jitter))),  # Matches jitter, not latency = 2
        (grp['latency'].ne(latency) & (grp['jitter'].ne(jitter))),  # Matches neither jitter or latency = 1
    ]
    # Apply weights and conditions and create array
    return np.select(conditions, weights)


def create_simulations_by_condition(
        md_df: pd.DataFrame, latency: int, jitter: float, parameter: str = 'original'
) -> None:
    """

    """
    # TODO: make this a function that does not rely on getting from the model data: this should allow for any
    #  arbitrary latency and jitter value to be provided!

    # Index to get our Zoom array
    zoom_arr = md_df[(md_df['latency'] == latency) & (md_df['jitter'] == jitter)]['zoom_arr'].iloc[0].copy()
    # Iterate through each trial
    for idx, grp in md_df.groupby('trial'):
        # Apply the weighted average column
        grp['weights'] = _apply_weighted_average_columns(grp, latency, jitter)
        # Get the weighted average and create a new dataframe
        keys = _get_weighted_average(grp[grp['instrument'] == 'Keys'])
        drms = _get_weighted_average(grp[grp['instrument'] == 'Drums'])
        # Create the simulation class
        md = Simulation(keys, drms, parameter, latency=zoom_arr, num_simulations=100)
        # Run the simulations and plot
        md.run_simulations()
        md.plot_simulations()


#
# def dic_create(md, i):
#     dic = {
#         'trial': i,
#         'type': md.parameter + md.leader if md.leader is not None else md.parameter,
#         'keys_correction_partner': md.keys_params['correction_partner'],
#         'drms_correction_partner': md.drms_params['correction_partner'],
#         'keys_correction_self': md.keys_params['correction_self'],
#         'drms_correction_self': md.drms_params['correction_self'],
#         'keys_intercept': md.keys_params['intercept'],
#         'drms_intercept': md.drms_params['intercept'],
#     }
#     try:
#         md.run_simulations()
#         dic.update({'tempo_slope': md.get_avg_tempo_slope()})
#     except ValueError:
#         dic.update({'tempo_slope': np.nan})
#     return dic
#
# res = []
# for idx, grp in mds.groupby(by=['trial', 'block', 'latency', 'jitter']):
#     zoom_arr = grp.iloc[0]['zoom_arr']
#     k = grp[grp['instrument'] == 'Keys']
#     d = grp[grp['instrument'] != 'Keys']
#
#
#     orig = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='original', latency=zoom_arr,
#                       num_simulations=100)
#     res.append(dic_create(orig, idx))
#
#     demo = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='democracy', latency=zoom_arr,
#                       num_simulations=100)
#     res.append(dic_create(demo, idx))
#
#     dic_k = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='dictatorship', leader='keys',
#                        latency=zoom_arr,
#                        num_simulations=100)
#     res.append(dic_create(dic_k, idx))
#
#     dic_d = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='dictatorship', leader='drums',
#                        latency=zoom_arr,
#                        num_simulations=100)
#     res.append(dic_create(dic_d, idx))
#
#     anarc = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='anarchy', latency=zoom_arr,
#                        num_simulations=100)
#     res.append(dic_create(anarc, idx))





# zoom_arr = mds[(mds['latency'] == 45) & (mds['jitter'] == 1)]['zoom_arr'].iloc[0].copy()
# for idx, grp in mds.groupby('trial'):
#     keys = pd.DataFrame(grp[(grp['instrument'] == 'Keys') & (grp['latency'] == 45) & (grp['jitter'] == 1)][
#                             ['correction_self', 'correction_partner', 'intercept', 'resid_std',
#                              'resid_len']].mean()).transpose()
#     keys['instrument'] = 'Keys'
#     drms = pd.DataFrame(grp[(grp['instrument'] != 'Keys') & (grp['latency'] == 45) & (grp['jitter'] == 1)][
#                             ['correction_self', 'correction_partner', 'intercept', 'resid_std',
#                              'resid_len']].mean()).transpose()
#     drms['instrument'] = 'Drums'
#     md = PhaseCorrectionSimulation(keys, drms, 'original', latency=zoom_arr, num_simulations=100)
#     md.run_simulations()
#     md.plot_simulations()

#
# keys = mds[mds['instrument'] == 'Keys']
# drms = mds[mds['instrument'] != 'Keys']
# lin = lambda d, s: np.linspace(min(d[s]), max(d[s]), 1000)
# keys_cp = lin(keys, 'correction_partner')
# drms_cp = lin(drms, 'correction_partner')
# keys_cs = lin(keys, 'correction_self')
# drms_cs = lin(drms, 'correction_self')
# keys_intercept = np.random.normal(0.5, np.std(keys['intercept']), 1000)
# drms_intercept = np.random.normal(0.5, np.std(drms['intercept']), 1000)
# zoom_arr = mds[(mds['latency'] == 45) & (mds['jitter'] == 0)]['zoom_arr'].iloc[0].copy()
# for i in range(1, 1000):
#     k = {
#         'instrument': 'Keys',
#         'total_beats': 180,
#         'raw_beats_1': [1],
#         'raw_beats_2' : [1],
#         'correction_self': np.random.choice(keys_cs),
#         'correction_self_stderr': 0,
#         'correction_partner': np.random.choice(keys_cp),
#         'correction_partner_stderr': 0,
#         'intercept': np.random.choice(keys_intercept),
#         'intercept_stderr': 0,
#         'resid_std': 0.03,
#         'resid_len': 180
#     }
#     d = {
#         'instrument': 'Drums',
#         'total_beats': 180,
#         'raw_beats_1': [1],
#         'raw_beats_2' : [1],
#         'correction_self': np.random.choice(drms_cs),
#         'correction_self_stderr': 0,
#         'correction_partner': np.random.choice(drms_cp),
#         'correction_partner_stderr': 0,
#         'intercept': np.random.choice(drms_intercept),
#         'intercept_stderr': 0,
#         'resid_std': 0.03,
#         'resid_len': 180
#     }
#     md = Simulation(pd.DataFrame(k), pd.DataFrame(d), parameter='original', latency=zoom_arr, num_simulations=100)
#     md.run_simulations()
#     dic = {
#         'tempo_slope': md.get_avg_tempo_slope(),
#         'keys_correction_partner': k['correction_partner'],
#         'drms_correction_partner': d['correction_partner'],
#         'keys_correction_self': k['correction_self'],
#         'drms_correction_self': d['correction_self'],
#         'keys_intercept': k['intercept'],
#         'drms_intercept': d['intercept'],
#     }
#     res.append(dic)
#     print(i)
