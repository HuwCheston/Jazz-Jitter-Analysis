import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import src.visualise.visualise_utils as vutils


class PhaseCorrectionSimulation:
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
                f'(required: {", ".join(str(x) for x in required_cols)})')
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

    def _generate_data(
            self
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        def get_latency_at_onset(sim_data: dict, sim_params: dict) -> float:
            # Subset our latency array for just the onset times times
            latency_onsets: np.array = sim_params['latency'][:, 1]
            # Get the last onset our partner played
            partner_onset: float = sim_data['their_actual_onset'][-1]
            # Get the closest minimum latency onset time to our partner's onset
            matched_latency_onset = latency_onsets[latency_onsets < partner_onset].max()
            # Get the latency value associated with that latency onset time and return
            latency_value: float = float(sim_params['latency'][latency_onsets == matched_latency_onset][:, 0])
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
            # Increase the iteration counter by one and raise an exception if we're stuck in a likely infinite loop
            iter_count += 1
            if iter_count >= self._max_iter:
                raise RuntimeError(f'Maximum number of iterations {self._max_iter} exceeded')

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

    def plot_simulations(
            self
    ) -> None:
        """
        Plot all simulations and add in the grand average tempo
        """
        # Raise an error if we haven't generated our simulations
        if len(self.keys_simulations) == 0 or len(self.drms_simulations) == 0:
            raise ValueError('Generate simulations first before plotting')
        # Define the roller function for converting IOIs to BPM and rolling
        roller = lambda s: (60 / s).rolling(window=self._rolling_period, min_periods=1).mean()
        # Iterate through keys and drums simulations and plot rolled BPM
        for keys, drms in zip(self.keys_simulations, self.drms_simulations):
            avg = self._get_grand_average_tempo([keys, drms])
            plt.plot(avg.index.seconds, roller(avg['my_prev_ioi']), alpha=0.05, lw=1, color=vutils.BLACK)
        # Get our grand average tempo for keys and drums simulations
        keys_avg = self._get_grand_average_tempo(self.keys_simulations)
        drms_avg = self._get_grand_average_tempo(self.drms_simulations)
        # Plot the rolled BPM of our grand average dataframe
        grand_average = self._get_grand_average_tempo([keys_avg, drms_avg])
        plt.plot(grand_average.index.seconds, roller(grand_average['my_prev_ioi']), alpha=1, lw=2, color=vutils.BLACK)
        sd_plus1 = self._get_grand_average_tempo([keys_avg + self._get_grand_average_stdev(self.keys_simulations),
                                                  drms_avg + self._get_grand_average_stdev(self.drms_simulations)])
        sd_neg1 = self._get_grand_average_tempo([keys_avg - self._get_grand_average_stdev(self.keys_simulations),
                                                 drms_avg - self._get_grand_average_stdev(self.drms_simulations)])
        plt.fill_between(grand_average.index.seconds, roller(sd_neg1['my_prev_ioi']), roller(sd_plus1['my_prev_ioi']),
                         alpha=0.1, color=vutils.BLACK)
        # Set y limit
        plt.ylim((30, 160))
        plt.show()

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
