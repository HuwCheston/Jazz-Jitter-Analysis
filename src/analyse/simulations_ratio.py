import numpy as np
import pandas as pd
import numba as nb
from datetime import timedelta
import statsmodels.formula.api as smf
import pickle

from src.analyse.phase_correction_ratios import PhaseCorrectionModel


class Simulation:
    def __init__(self, keys_params, drms_params, latency_array, num_simulations: int = 1000):
        # Default parameters
        self._initial_onset: float = 8.0  # The initial onset of a performance, after 8 second count-in
        self._initial_ioi: float = 0.5  # The presumed initial length of the first IOI = crotchet at 120BPM
        self._resample_interval: timedelta = timedelta(seconds=1)  # Get mean of IOIs within this window
        self._rolling_period: timedelta = timedelta(seconds=4)  # Apply a rolling window of this size to the data

        # Check and generate our input data
        self.total_beats: int = self._get_number_of_beats_for_simulation(keys_params, drms_params)
        self.latency: np.array = self._append_timestamps_to_latency_array(latency_array)
        self.num_simulations: int = num_simulations

        # Simulation parameters, in the form of numba dictionaries to be used inside a numba function
        self.keys_params: nb.typed.Dict = self._get_simulation_params(init=keys_params)
        self.drms_params: nb.typed.Dict = self._get_simulation_params(init=drms_params)

        # Empty lists to store our keys and drums simulations in
        self.keys_simulations: list[pd.DataFrame] = []
        self.drms_simulations: list[pd.DataFrame] = []

    @staticmethod
    def _get_number_of_beats_for_simulation(
            kp, dp
    ) -> int:
        """
        Averages the total number of beats across both keys and drums, then gets the upper ceiling.
        """
        return int(np.ceil(np.mean([kp['total_beats'].iloc[0], dp['total_beats'].iloc[0]])))

    @staticmethod
    def _append_timestamps_to_latency_array(
            latency_array, offset: int = 8, resample_rate: float = 0.75
    ) -> np.array:
        """
        Appends timestamps showing the onset time for each value in the latency array applied to a performance
        """
        # Define the endpoint for the linear space
        end = offset + (len(latency_array) * resample_rate)
        # Create the linear space
        lin = np.linspace(offset, end, num=len(latency_array), endpoint=False)
        # Append the two arrays together
        return np.c_[latency_array / 1000, lin]

    @staticmethod
    def _get_simulation_params(
            init: pd.DataFrame
    ) -> nb.typed.Dict:
        """
        Converts simulation parameters from a pandas DataFrame to a numba dictionary
        """
        init = init.reset_index(drop=True)
        # Make dictionary
        nb_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float64,)
        # Fill the dictionary
        for s in ['correction_self', 'correction_partner', 'intercept', 'resid_std']:
            nb_dict[s] = init[s].iloc[0].astype(np.float64)
        return nb_dict

    def _initialise_empty_data(
            self
    ) -> nb.typed.Dict:
        """
        Initialise an empty numba dictionary of string-array pairs, for storing data from one simulation in.
        """
        # Make dictionary with strings as keys and arrays as values
        nb_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float64[:],
        )
        # Fill the dictionary with arrays (pre-allocated in order to make running the simulation easier)
        for s in ['my_onset', 'asynchrony', 'my_next_ioi', 'my_next_diff']:
            nb_dict[s] = np.zeros(shape=self.total_beats)
        # Fill the dictionary arrays with our starting values
        nb_dict['my_onset'][0] = self._initial_onset
        nb_dict['my_onset'][1] = self._initial_onset + self._initial_ioi
        nb_dict['asynchrony'][0] = self.latency[0][0]
        nb_dict['asynchrony'][1] = self.latency[0][0]
        nb_dict['my_next_ioi'][0] = self._initial_ioi
        nb_dict['my_next_ioi'][1] = self._initial_ioi
        nb_dict['my_next_diff'][0] = np.nan
        nb_dict['my_next_diff'][1] = 0
        return nb_dict

    def create_simulations(
            self
    ):
        """
        Run the simulations and create a list of dataframes for each individual performer
        """
        for i in range(0, self.num_simulations):
            # Create the simulation: Numba doesn't work as a class method, so we need to pass arguments in seperately
            keys, drms = self._create_simulation(
                self._initialise_empty_data(), self._initialise_empty_data(),  # Initialise empty data to store in
                self.keys_params, self.drms_params,     # Parameters for the simulation, e.g. coefficients
                self.latency, self.total_beats  # Latency array and total number of beats
            )
            # Format the simulated data and append to our list
            self.keys_simulations.append(self._format_simulated_data(data=keys))
            self.drms_simulations.append(self._format_simulated_data(data=drms))

    def _format_simulated_data(
            self, data: nb.typed.Dict
    ) -> pd.DataFrame:
        """
        Formats data from one simulation by creating a dataframe, adding in the timedelta column, and resampling
        to get the mean IOI (defaults to every second)
        """
        # Create dataframe from the numba dictionary by first converting it to a python dictionary then to a dataframe
        df = pd.DataFrame(dict(data))
        # Convert my onset column to a timedelta
        df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['my_onset']])
        # Set the index to our timedelta column, resample (default every second), and get mean
        return df.set_index('td').resample(rule=self._resample_interval).mean()

    @staticmethod
    @nb.njit
    def _create_simulation(
            keys_data: nb.typed.Dict, drms_data: nb.typed.Dict, keys_params: nb.typed.Dict,
            drms_params: nb.typed.Dict, lat: np.ndarray, beats: int
    ) -> tuple:
        """
        Create data for one simulation, using numba optimisations
        """
        def get_lat(
                partner_onset: float
        ) -> float:
            """
            Get the current amount of latency applied when our partner played their onset
            """
            return lat[lat[:, 1] == lat[:, 1][lat[:, 1] < partner_onset].max()][:, 0][0]

        def predict(
                next_diff: float, asynchrony: float, params: nb.typed.Dict
        ) -> float:
            """
            Predict the difference between previous and next IOIs using inputted data and model parameters
            """
            a = next_diff * params['correction_self']
            b = asynchrony * params['correction_partner']
            c = params['intercept']
            n = np.random.choice(np.random.normal(0, params['resid_std'], 1000))
            return a + b + c + n

        # We don't use the full range of beats, given that we've already added some in when creating our starter data
        for i in range(2, beats):
            # Get next onset by adding previous onset to predicted IOI
            keys_data['my_onset'][i] = keys_data['my_onset'][i - 1] + keys_data['my_next_ioi'][i - 1]
            drms_data['my_onset'][i] = drms_data['my_onset'][i - 1] + drms_data['my_next_ioi'][i - 1]
            # Get asynchrony value by subtracting partner's onset (plus latency) from ours
            keys_data['asynchrony'][i] = (
                    drms_data['my_onset'][i] + get_lat(drms_data['my_onset'][i]) - keys_data['my_onset'][i]
            )
            drms_data['asynchrony'][i] = (
                    keys_data['my_onset'][i] + get_lat(keys_data['my_onset'][i]) - drms_data['my_onset'][i]
            )
            # Predict difference between previous IOI and next IOI
            keys_data['my_next_diff'][i] = predict(
                keys_data['my_next_diff'][i - 1], keys_data['asynchrony'][i], keys_params
            )
            drms_data['my_next_diff'][i] = predict(
                drms_data['my_next_diff'][i - 1], drms_data['asynchrony'][i], drms_params
            )
            # Use predicted difference between IOIs to get next actual IOI
            keys_data['my_next_ioi'][i] = keys_data['my_next_diff'][i] + keys_data['my_next_ioi'][i - 1]
            drms_data['my_next_ioi'][i] = drms_data['my_next_diff'][i] + drms_data['my_next_ioi'][i - 1]
        return keys_data, drms_data

    @staticmethod
    def _get_grand_average_tempo(
            all_perf: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Concatenate all simulations together and get the row-wise average (i.e. avg IOI every second)
        """
        return pd.DataFrame(
            pd.concat([df['my_prev_ioi'] for df in all_perf], axis=1).mean(axis=1), columns=['my_prev_ioi']
        )

    def get_average_tempo_slope(
            self
    ) -> float | None:
        """
        Returns the average tempo slope for all simulations.

        Method:
        ---
        - For every simulation, zip the corresponding keys and drums performance together.
        - Then, get the average IOI for every second across both keys and drums.
            - This is straightforward, because we resampled to average IOI per second in _format_simulated_data
        - Convert average IOI to average BPM by dividing by 60, then regress against elapsed seconds
        - Extract the slope coefficient, average across all simulations, and return
        """
        # If we haven't generated our simulations, do this first
        if len(self.keys_simulations) == 0 or len(self.drms_simulations) == 0:
            self.create_simulations()
        coeffs = []
        # Iterate through every simulation individually
        for keys, drms in zip(self.keys_simulations, self.drms_simulations):
            # Concatenate keyboard and drums performance and get average IOI every second
            avg = pd.DataFrame(
                pd.concat([d['my_next_ioi'] for d in [keys, drms]], axis=1).mean(axis=1), columns=['my_next_ioi']
            )
            # Get elapsed number of seconds
            avg['elapsed_seconds'] = avg.index.seconds
            # Convert IOIs to BPM for tempo slope regression
            avg['my_next_bpm'] = 60 / avg['my_next_ioi']
            # Conduct and fit the regression model
            md = smf.ols('my_next_bpm~elapsed_seconds', data=avg.dropna()).fit()
            # Extract the tempo slope coefficient and append to list
            coeffs.append(md.params[1])
        # Calculate the mean tempo slope coefficient from all simulations and return
        return np.mean(coeffs)


if __name__ == '__main__':
    mds = pickle.load(open(r"C:\Python Projects\jazz-jitter-analysis\reports\output_models_ratioed.p", "rb"))
    k = mds[(mds['trial'] == 4) & (mds['block'] == 1) & (mds['latency'] == 45) & (mds['jitter'] == 1) & (mds['instrument'] == 'Keys')]
    d = mds[(mds['trial'] == 4) & (mds['block'] == 1) & (mds['latency'] == 45) & (mds['jitter'] == 1) & (mds['instrument'] != 'Keys')]
    print(k['tempo_slope'])
    z = k['zoom_arr'].iloc[0]
    sim = Simulation(k, d, z, num_simulations=10000)
    sim.create_simulations()
    print(sim.get_average_tempo_slope())

    # c = []
    # for idx, grp in mds.groupby(by=['trial', 'block', 'latency', 'jitter']):
    #     print(idx)
    #     k = grp[(grp['instrument'] == 'Keys')].copy()
    #     d = grp[(grp['instrument'] != 'Keys')].copy()
    #     z = k['zoom_arr'].iloc[0]
    #     sim = Simulation(k, d, 'original', z, num_simulations=1000)
    #     sim.run_simulations()
    #     c.append((*idx, sim.get_average_tempo_slope()))
    # res = pd.DataFrame(c)
    # pickle.dump(res, open(r"C:\Python Projects\jazz-jitter-analysis\reports\simulations_ratioed.p", "wb"))




#
#
#
#
# # keys_params = d[(d['trial'] == 3) & (d['block'] == 1) & (d['latency'] == 23) & (d['jitter'] == 0) & (
# #         d['instrument'] == 'Keys')].reset_index(drop=True)
# # drms_params = d[(d['trial'] == 3) & (d['block'] == 1) & (d['latency'] == 23) & (d['jitter'] == 0) & (
# #         d['instrument'] != 'Keys')].reset_index(drop=True)
# #
# # sim = Simulation(keys_params, drms_params)
# # print(sim.keys_params)
#
# # #
# # # keys_noise = np.random.normal(0, keys_params['resid_std'], keys_params['resid_len'])
# # # drms_noise = np.random.normal(0, drms_params['resid_std'], drms_params['resid_len'])
# # #
# # # latency = 0.023
# # # total_beats = int(np.ceil(np.mean([keys_params['total_beats'].iloc[0], drms_params['total_beats'].iloc[0]])))
# # #
# # # KEYS_COUPLING_SELF = keys_params['correction_self'].iloc[0]
# # # DRMS_COUPLING_SELF = drms_params['correction_self'].iloc[0]
# # # KEYS_COUPLING_PARTNER = keys_params['correction_partner'].iloc[0]
# # # DRMS_COUPLING_PARTNER = drms_params['correction_partner'].iloc[0]
# # # KEYS_INTERCEPT = keys_params['intercept'].iloc[0]
# # # DRMS_INTERCEPT = drms_params['intercept'].iloc[0]
# # #
# # #
# # #
# # #
# # # @jit(nopython=True)
# # # def simulate():
# # #     KEYS_my_onset = np.zeros(shape=(total_beats))
# # #     KEYS_asynchrony = np.zeros(shape=(total_beats))
# # #     KEYS_my_next_ioi = np.zeros(shape=(total_beats))
# # #     KEYS_my_next_ioi_diff = np.zeros(shape=(total_beats))
# # #
# # #     DRMS_my_onset = np.zeros(shape=(total_beats))
# # #     DRMS_asynchrony = np.zeros(shape=(total_beats))
# # #     DRMS_my_next_ioi = np.zeros(shape=(total_beats))
# # #     DRMS_my_next_ioi_diff = np.zeros(shape=(total_beats))
# # #
# # #     KEYS_my_onset[0] = 8
# # #     KEYS_my_onset[1] = 8.5
# # #     KEYS_asynchrony[0] = 0.023
# # #     KEYS_asynchrony[1] = 0.023
# # #     KEYS_my_next_ioi[0] = 0.5
# # #     KEYS_my_next_ioi[1] = 0.5
# # #     KEYS_my_next_ioi_diff[0] = np.nan
# # #     KEYS_my_next_ioi_diff[1] = 0
# # #
# # #     DRMS_my_onset[0] = 8
# # #     DRMS_my_onset[1] = 8.5
# # #     DRMS_asynchrony[0] = 0.023
# # #     DRMS_asynchrony[1] = 0.023
# # #     DRMS_my_next_ioi[0] = 0.5
# # #     DRMS_my_next_ioi[1] = 0.5
# # #     DRMS_my_next_ioi_diff[0] = np.nan
# # #     DRMS_my_next_ioi_diff[1] = 0
# # #
# # #     for i in range(2, total_beats):
# # #         KEYS_my_onset[i] = KEYS_my_onset[i - 1] + KEYS_my_next_ioi[i - 1]
# # #         DRMS_my_onset[i] = DRMS_my_onset[i - 1] + DRMS_my_next_ioi[i - 1]
# # #
# # #         KEYS_asynchrony[i] = DRMS_my_onset[i] + latency - KEYS_my_onset[i]
# # #         DRMS_asynchrony[i] = KEYS_my_onset[i] + latency - DRMS_my_onset[i]
# # #
# # #         KEYS_my_next_ioi_diff[i] = (KEYS_my_next_ioi_diff[i - 1] * KEYS_COUPLING_SELF) + (
# # #                 KEYS_asynchrony[i] * KEYS_COUPLING_PARTNER) + KEYS_INTERCEPT + 0
# # #         DRMS_my_next_ioi_diff[i] = (DRMS_my_next_ioi_diff[i - 1] * DRMS_COUPLING_SELF) + (
# # #                 DRMS_asynchrony[i] * DRMS_COUPLING_PARTNER) + DRMS_INTERCEPT + 0
# # #
# # #         KEYS_my_next_ioi[i] = KEYS_my_next_ioi_diff[i] + KEYS_my_next_ioi[i - 1]
# # #         DRMS_my_next_ioi[i] = DRMS_my_next_ioi_diff[i] + DRMS_my_next_ioi[i - 1]
# # #     return KEYS_my_next_ioi
# # #
# # #
# # # for i in range(1, 1000):
# # #     print(simulate())
#
#
#
#
#
#
# class Simulation:
#     """
#     Creates a series of simulated musical performances according to the parameters of two provided regression models,
#     one for the pianist and one for the drummer.
#
#     Available parameter:
#     - original
#     - original_variable
#     - anarchy
#     - democracy
#     - dictatorship
#     """
#
#     def __init__(
#             self, keys_data, drms_data, parameter, latency, **kwargs
#     ):
#         # Default parameters
#         self._initial_onset: float = 8.0  # The initial onset of a performance, after 8 second count-in
#         self._initial_ioi: float = 0.5  # The presumed initial length of the first IOI = crotchet at 120BPM
#         self._end: float = 101.5  # Stop generating data after we exceed this onset value
#         self._max_iter: int = 300  # If we generate more than this number of IOIs, we're probably stuck in a loop
#         self._num_broken: int = 0   # The number of simulations that failed to complete
#         self._resample_interval: timedelta = timedelta(seconds=1)  # Get mean of IOIs within this window
#         self._rolling_period: timedelta = timedelta(seconds=4)  # Apply a rolling window of this size to the data
#         self._debug_params: dict = {  # Used when debugging: will generate the same simulation every time
#             'correction_self': 0.5,
#             'correction_partner': 0.5,
#             'intercept': 0.5,
#             'noise': np.array([0.02])
#         }
#
#         # Check and generate our input data
#         self.keys_data: pd.DataFrame = self._check_input_data(keys_data)
#         self.drms_data: pd.DataFrame = self._check_input_data(drms_data)
#         self.leader: str = kwargs.get('leader', None)
#         self.parameter: str = self._check_simulation_parameter(parameter)
#         self.latency: np.array = self._append_timestamps_to_latency_array(latency)
#         self.num_simulations: int = kwargs.get('num_simulations', 500)
#
#         self.keys_params: dict = self._get_simulation_params(input_data=self.keys_data)
#         self.drms_params: dict = self._get_simulation_params(input_data=self.drms_data)
#
#         self.keys_simulations: list[pd.DataFrame] = []
#         self.drms_simulations: list[pd.DataFrame] = []
#
#     @staticmethod
#     def _check_input_data(
#             input_data: pd.DataFrame
#     ) -> pd.DataFrame | None:
#         """
#         Checks to make sure that input dataframe is in correct format and has all necessary columns.
#         Raises ValueError if any checks are failed
#         """
#         required_cols: list[str] = [
#             'instrument',
#             'correction_self',
#             'correction_partner',
#             'intercept',
#             'resid_std',
#             'resid_len',
#         ]
#         # If we didn't pass a dataframe
#         if not isinstance(input_data, pd.DataFrame):
#             raise ValueError('Data input was either not provided or is of invalid type')
#         # If we don't have all the necessary columns to create the simulation
#         if len([col for col in required_cols if col not in input_data.columns]) != 0:
#             raise ValueError(
#                 f'Some required columns in data input were missing '
#                 f'(missing: {", ".join(str(x) for x in required_cols if x not in input_data.columns)})')
#         # If we passed data from multiple performances
#         if len(input_data) != 1:
#             raise ValueError(
#                 f'Input data should be from one performance only (passed data: {len(input_data)} performances)')
#         # If all checks passed, return data
#         else:
#             return input_data[required_cols]
#
#     def _check_simulation_parameter(
#             self, input_parameter: str
#     ) -> str | None:
#         """
#         Checks if the simulation parameter given by the user is acceptable , and raises value error if not
#         """
#         acceptable_parameters: list[str] = [
#             'original',  # Use the original coefficients we've passed in
#             'original_variable',  # Use the original coefficients, but add random noise according to std err
#             'democracy',  # Coefficients for both performers set to the mean
#             'dictatorship',
#             'anarchy',
#             'debug'
#         ]
#         # If we haven't passed an acceptable parameter
#         if input_parameter not in acceptable_parameters:
#             raise ValueError(
#                 f'{input_parameter} not in acceptable simulation parameters '
#                 f'(options: {", ".join(str(x) for x in acceptable_parameters)})')
#         # Additional checks if we're setting the leadership parameter
#         if input_parameter == 'leadership':
#             # We haven't passed a leadership parameter, or it's an incorrect type
#             if not isinstance(self.leader, str):
#                 raise ValueError(
#                     'If setting leadership simulation parameter, must also pass leader keyword argument as string')
#             # We've passed an incompatible leadership parameter
#             elif self.leader.lower() not in ['keys', 'drums']:
#                 raise ValueError(f'Leader keyword argument {self.leader} is invalid: must be either keys or drums')
#         # If all checks passed, return the input parameter
#         return input_parameter
#
#     @staticmethod
#     def _append_timestamps_to_latency_array(
#             latency_array, offset: int = 8, resample_rate: float = 0.75
#     ) -> np.array:
#         """
#         Appends timestampts showing the onset time for each value in the latency array applied to a performance
#         """
#         # Define the start and endpoint for the linear space
#         start = offset
#         end = offset + (len(latency_array) * resample_rate)
#         # Create the linear space
#         lin = np.linspace(start, end, num=len(latency_array), endpoint=False)
#         # Append the two arrays together
#         return np.c_[latency_array / 1000, lin]
#
#     def _get_simulation_params(
#             self, input_data: pd.DataFrame
#     ) -> dict:
#         """
#         Returns the simulation parameters from the given input parameter.
#         """
#         # Averaging function: returns average of keys and drums value for a given variable
#         mean = lambda s: np.mean([self.keys_data[s], self.drms_data[s]])
#         rand = lambda s: np.random.normal(loc=input_data[s], scale=input_data[f'{s}_stderr'],
#                                           size=int(input_data['total_beats']))
#         # Define the initial dictionary, with the noise term and the latency array (with timestamps)
#         d: dict = {
#             'noise': np.random.normal(loc=0, scale=input_data['resid_std'], size=int(input_data['resid_len'])),
#             'latency': self.latency,
#         }
#         # Original coupling: uses coefficients, intercept from the model itself
#         if self.parameter == 'original':
#             d.update({
#                 'correction_self': input_data['correction_self'].iloc[0],
#                 'correction_partner': input_data['correction_partner'].iloc[0],
#                 'intercept': input_data['intercept'].iloc[0],
#             })
#         return d
#
#     def _initialise_empty_data(
#             self, first_latency_value: float = 0
#     ) -> dict:
#         """
#         Initialises an empty dictionary that we will fill with data as the simulation runs
#         """
#         return {
#             'my_onset': [self._initial_onset, self._initial_ioi + self._initial_onset],
#             'their_heard_onset': [self._initial_onset + first_latency_value,
#                                   self._initial_ioi + self._initial_onset + first_latency_value],
#             'my_prev_ioi': [np.nan, self._initial_ioi],
#             'my_prev_ioi_ratio': [np.nan, np.nan, ],
#             'asynchrony': [first_latency_value, first_latency_value],
#             'asynchrony_ratio': [np.nan, first_latency_value / self._initial_ioi],
#             'my_next_ioi_ratio': [np.nan, 1],
#             'my_next_ioi': [self._initial_ioi, self._initial_ioi],
#         }
#
#
#     def run_simulations(
#             self
#     ) -> None:
#         """
#         Run the simulations and create a list of dataframes for each individual performer
#         """
#         # Clear both lists
#         self.keys_simulations.clear()
#         self.drms_simulations.clear()
#         # Run the number of simulations which we provided
#         for i in range(0, self.num_simulations):
#             # Generate the data and append to the required lists
#             keys, drms = self._generate_data()
#             self.keys_simulations.append(keys)
#             self.drms_simulations.append(drms)
#             # Remove simulations that failed to complete
#             self._remove_broken_simulations()
#
#     def _remove_broken_simulations(
#             self
#     ) -> None:
#         """
#         Remove simulations from our dataset that have failed to complete
#         """
#         cleaned_keys = []
#         cleaned_drms = []
#         # Iterate through keys and drums performances TOGETHER
#         for keys, drms in zip(self.keys_simulations, self.drms_simulations):
#             # If we broke out of our loop while creating the simulation, remove:
#             if keys is None or drms is None:
#                 self._num_broken += 1
#                 continue
#             # If the performance exceeds the maximum length (plus a bit of tolerance), remove:
#             elif keys.index.seconds.max() >= self._end + 1 or drms.index.seconds.max() >= self._end + 1:
#                 self._num_broken += 1
#                 continue
#             # If the performance exceeds the maximum number of beats, remove:
#             # elif len(keys) >= self._max_iter or len(drms) >= self._max_iter:
#             #     self._num_broken += 1
#             #     continue
#             # Else, performance is ok and can be used.
#             else:
#                 cleaned_keys.append(keys)
#                 cleaned_drms.append(drms)
#         self.keys_simulations = cleaned_keys
#         self.drms_simulations = cleaned_drms
#
#     def _generate_data(
#             self
#     ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
#         """
#         Generates data for one simulation
#         """
#
#         def get_latency_at_onset(
#                 partner_onset: float, latency_array: np.array
#         ) -> float | None:
#             """
#             Get the amount of latency applied to our simulated partner's feed at the particular point in the performance
#             """
#             # Subset our latency array for just the onset times times
#             latency_onsets: np.array = latency_array[:, 1]
#             try:
#                 # Get the closest minimum latency onset time to our partner's onset
#                 matched_latency_onset = latency_onsets[latency_onsets < partner_onset].max()
#                 # Get the latency value associated with that latency onset time and return
#                 latency_value: float = float(latency_array[latency_onsets == matched_latency_onset][:, 0])
#             except ValueError:
#                 return None
#             else:
#                 return latency_value
#
#         # Generate our starter data
#         keys_data = self._initialise_empty_data(self.latency[0][0])
#         drms_data = self._initialise_empty_data(self.latency[0][0])
#         # Start the simulation
#         iter_count: int = 0
#         while keys_data['my_onset'][-1] < self._end or drms_data['my_onset'][-1] < self._end:
#             # Get my next onset
#             keys_data['my_onset'].append(keys_data['my_onset'][-1] + keys_data['my_next_ioi'][-1])
#             drms_data['my_onset'].append(drms_data['my_onset'][-1] + drms_data['my_next_ioi'][-1])
#             # Get their next heard onset, including latency
#             lat_keys = get_latency_at_onset(drms_data['my_onset'][-1], self.latency)
#             lat_drms = get_latency_at_onset(keys_data['my_onset'][-1], self.latency)
#             # If we couldn't get the next latency time
#             if lat_keys is None or lat_drms is None:
#                 return None, None
#             else:
#                 keys_data['their_heard_onset'].append(drms_data['my_onset'][-1] + lat_keys)
#                 drms_data['their_heard_onset'].append(keys_data['my_onset'][-1] + lat_drms)
#             # Get my previous IOI by shifting values
#             keys_data['my_prev_ioi'].append(keys_data['my_next_ioi'][-1])
#             drms_data['my_prev_ioi'].append(drms_data['my_next_ioi'][-1])
#             # Get my previous IOI ratio by shifting values
#             keys_data['my_prev_ioi_ratio'].append(keys_data['my_next_ioi_ratio'][-1])
#             drms_data['my_prev_ioi_ratio'].append(drms_data['my_next_ioi_ratio'][-1])
#             # Get actual asynchrony
#             keys_data['asynchrony'].append(keys_data['their_heard_onset'][-1] - keys_data['my_onset'][-1])
#             drms_data['asynchrony'].append(drms_data['their_heard_onset'][-1] - drms_data['my_onset'][-1])
#             # Express asynchrony in terms of previous IOI
#             keys_data['asynchrony_ratio'].append(keys_data['asynchrony'][-1] / keys_data['my_prev_ioi'][-1])
#             drms_data['asynchrony_ratio'].append(drms_data['asynchrony'][-1] / drms_data['my_prev_ioi'][-1])
#             # Use coefficients to predict next IOI ratio
#             keys_data['my_next_ioi_ratio'].append(
#                 (keys_data['my_prev_ioi_ratio'][-1] * self.keys_params['correction_self']) + (
#                         keys_data['asynchrony_ratio'][-1] * self.keys_params['correction_partner']) + self.keys_params[
#                     'intercept']
#                 + np.random.choice(self.keys_params['noise']))
#             drms_data['my_next_ioi_ratio'].append(
#                 (drms_data['my_prev_ioi_ratio'][-1] * self.drms_params['correction_self']) + (
#                         drms_data['asynchrony_ratio'][-1] * self.drms_params['correction_partner']) + self.drms_params[
#                     'intercept']
#                 + np.random.choice(self.drms_params['noise']))
#             # Convert next IOI ratio into next IOI prediction
#             keys_data['my_next_ioi'].append(keys_data['my_next_ioi'][-1] * keys_data['my_next_ioi_ratio'][-1])
#             drms_data['my_next_ioi'].append(drms_data['my_next_ioi'][-1] * drms_data['my_next_ioi_ratio'][-1])
#             # Increase the iteration counter by one
#             iter_count += 1
#             # Raise a warning if we've exceeded the maximum number of iterations and return None to discard broken data
#             if iter_count >= self._max_iter:
#                 warnings.warn(f'Maximum number of iterations {self._max_iter} exceeded')
#                 break
#         # Once the simulation has finished, return the data as a tuple = keys simulated data, drums simulated data
#         return self._format_simulated_data(keys_data), self._format_simulated_data(drms_data)
#
#     def _format_simulated_data(
#             self, data: dict
#     ) -> pd.DataFrame:
#         """
#         Formats data from one simulation by creating a dataframe, adding in the timedelta column, and resampling
#         to get the mean IOI (defaults to every second)
#         """
#         # Create the dataframe from the dictionary
#         df = pd.DataFrame(data)
#         # Convert my onset column to a timedelta
#         df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['my_onset']])
#         # Set the index to our timedelta column, resample (default every second), and get mean
#         return df.set_index('td').resample(rule=self._resample_interval).mean()
#
#     @staticmethod
#     def _get_grand_average_tempo(
#             all_perf: list[pd.DataFrame]
#     ) -> pd.DataFrame:
#         """
#         Concatenate all simulations together and get the row-wise average (i.e. avg IOI every second)
#         """
#         return pd.DataFrame(
#             pd.concat([df['my_prev_ioi'] for df in all_perf], axis=1).mean(axis=1), columns=['my_prev_ioi']
#         )
#
#     def get_average_tempo_slope(
#             self
#     ) -> float | None:
#         """
#         Returns the average tempo slope for all simulations.
#
#         Method:
#         ---
#         - First, if simulations haven't been generated yet, raise a warning and don't proceed any further.
#         - Otherwise: for every simulation, zip the corresponding keys and drums performance together.
#         - Then, get the average IOI for every second across both keys and drums.
#             - This is straightforward, because we resampled to average IOI per second in _format_simulated_data
#         - Convert average IOI to average BPM by dividing by 60, then regress against elapsed seconds
#         - Extract the slope coefficient, average across all simulations, and return
#         """
#         # Raise an error if we haven't generated our simulations
#         if len(self.keys_simulations) == 0 or len(self.drms_simulations) == 0:
#             warnings.warn('Simulations not generated yet, or none succeeded!')
#             return
#         coeffs = []
#         # Iterate through every simulation individually
#         for keys, drms in zip(self.keys_simulations, self.drms_simulations):
#             # Concatenate keyboard and drums performance and get average IOI every second
#             avg = pd.DataFrame(pd.concat([d['my_prev_ioi'] for d in [keys, drms]], axis=1).mean(axis=1),
#                                columns=['my_prev_ioi'])
#             # Get elapsed number of seconds
#             avg['elapsed_seconds'] = avg.index.seconds
#             # Convert IOIs to BPM for tempo slope regression
#             avg['my_prev_bpm'] = 60 / avg['my_prev_ioi']
#             # Run the regression
#             md = smf.ols('my_prev_bpm~elapsed_seconds', data=avg).fit()
#             # Extract the tempo slope coefficient and append to list
#             coeffs.append(md.params[1])
#         # Calculate the mean tempo slope coefficient from all simulations and return
#         return np.mean(coeffs)

