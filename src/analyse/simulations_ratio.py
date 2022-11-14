import numpy as np
import pandas as pd
import numba as nb
from datetime import timedelta
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

import src.visualise.visualise_utils as vutils

class Simulation:
    def __init__(self, keys_params, drms_params, latency_array, num_simulations: int = 1000):
        """

        """
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

    def create_all_simulations(
            self
    ):
        """
        Run the simulations and create a list of dataframes for each individual performer
        """
        for i in range(0, self.num_simulations):
            # Create the simulation: Numba doesn't work as a class method, so we need to pass arguments in seperately
            keys, drms = self._create_one_simulation(
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
    def _create_one_simulation(
            keys_data: nb.typed.Dict, drms_data: nb.typed.Dict, keys_params: nb.typed.Dict,
            drms_params: nb.typed.Dict, lat: np.ndarray, beats: int
    ) -> tuple:
        """
        Create data for one simulation, using numba optimisations
        """
        keys_noise = np.random.normal(0, keys_params['resid_std'], 1000)
        drms_noise = np.random.normal(0, drms_params['resid_std'], 1000)

        def get_lat(
                partner_onset: float
        ) -> float:
            """
            Get the current amount of latency applied when our partner played their onset
            """
            return lat[lat[:, 1] == lat[:, 1][lat[:, 1] < partner_onset].max()][:, 0][0]

        def predict(
                next_diff: float, asynchrony: float, params: nb.typed.Dict, noise: np.ndarray
        ) -> float:
            """
            Predict the difference between previous and next IOIs using inputted data and model parameters
            """
            a = next_diff * params['correction_self']
            b = asynchrony * params['correction_partner']
            c = params['intercept']
            n = np.random.choice(noise)
            return a + b + c + n

        # We don't use the full range of beats, given that we've already added some in when creating our starter data
        for i in range(2, beats):
            # Get next onset by adding previous onset to predicted IOI
            keys_data['my_onset'][i] = keys_data['my_onset'][i - 1] + keys_data['my_next_ioi'][i - 1]
            drms_data['my_onset'][i] = drms_data['my_onset'][i - 1] + drms_data['my_next_ioi'][i - 1]
            # Get asynchrony value by subtracting partner's onset (plus latency) from ours
            # noinspection PyBroadException
            try:
                keys_data['asynchrony'][i] = (
                        drms_data['my_onset'][i] + get_lat(drms_data['my_onset'][i]) - keys_data['my_onset'][i]
                )
                drms_data['asynchrony'][i] = (
                        keys_data['my_onset'][i] + get_lat(keys_data['my_onset'][i]) - drms_data['my_onset'][i]
                )
            except:
                break
            # Predict difference between previous IOI and next IOI
            keys_data['my_next_diff'][i] = predict(
                keys_data['my_next_diff'][i - 1], keys_data['asynchrony'][i], keys_params, keys_noise
            )
            drms_data['my_next_diff'][i] = predict(
                drms_data['my_next_diff'][i - 1], drms_data['asynchrony'][i], drms_params, drms_noise
            )
            # Use predicted difference between IOIs to get next actual IOI
            keys_data['my_next_ioi'][i] = keys_data['my_next_diff'][i] + keys_data['my_next_ioi'][i - 1]
            drms_data['my_next_ioi'][i] = drms_data['my_next_diff'][i] + drms_data['my_next_ioi'][i - 1]
            if keys_data['my_next_ioi'][i] < 0 or drms_data['my_next_ioi'][i] < 0:
                break
        return keys_data, drms_data

    @staticmethod
    def _get_grand_average_tempo(
            all_perf: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Concatenate all simulations together and get the row-wise average (i.e. avg IOI every second)
        """
        return pd.DataFrame(
            pd.concat([df['my_next_ioi'] for df in all_perf], axis=1).mean(axis=1), columns=['my_next_ioi']
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
        - Extract the slope coefficient, take the median across all simulations, and return
        """
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
        # Calculate the median tempo slope coefficient (robust to outliers!) from all simulations and return
        s = pd.Series(coeffs).replace(-np.Inf, np.nan)
        return np.nanmedian(s)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 500)

    mds = pickle.load(open(r"C:\Python Projects\jazz-jitter-analysis\reports\output_models_ratioed.p", "rb"))
    # print(mds[mds['correction_partner'] < 0])

    k = mds[(mds['trial'] == 3) & (mds['block'] == 1) & (mds['latency'] == 23) & (mds['jitter'] == 1.0) & (mds['instrument'] == 'Keys')]
    d = mds[(mds['trial'] == 3) & (mds['block'] == 1) & (mds['latency'] == 23) & (mds['jitter'] == 1.0) & (mds['instrument'] != 'Keys')]
    z = k['zoom_arr'].iloc[0]

    sim = Simulation(k, d, z, num_simulations=1000)
    sim.create_all_simulations()
    ts = sim.get_average_tempo_slope()

    for keys, drms in zip(sim.keys_simulations, sim.drms_simulations):
        avg = sim._get_grand_average_tempo([keys, drms])
        plt.plot(avg.index.seconds, (60/avg.my_next_ioi).rolling(window='8s').mean(), alpha=0.01, color=vutils.BLACK)
    grand_avg = sim._get_grand_average_tempo([sim._get_grand_average_tempo([k, d]) for k, d in zip(sim.keys_simulations, sim.drms_simulations)])
    pass
    plt.plot(grand_avg.index.seconds, (60 / grand_avg.my_next_ioi).rolling(window='8s').mean(), alpha=1, color=vutils.BLACK)
    plt.ylim(30, 160)
    plt.show()
    print(k['tempo_slope'].iloc[0], ts)


