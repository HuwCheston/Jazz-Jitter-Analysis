import numpy as np
import pandas as pd
import numba as nb
import pickle
from datetime import timedelta
import statsmodels.formula.api as smf

import src.analyse.analysis_utils as autils


# noinspection PyBroadException
class Simulation:
    def __init__(
            self, keys_nn, drms_nn, keys_params, drms_params, latency_array, num_simulations: int = 1000, **kwargs
    ):
        """

        """
        # Default parameters
        self._keys_initial_iois: list[float] = self._get_initial_iois(keys_nn)
        self._drms_initial_iois: list[float] = self._get_initial_iois(drms_nn)
        self._keys_initial_onset: float = self._get_initial_onset(keys_nn)
        self._drms_initial_onset: float = self._get_initial_onset(drms_nn)
        self._resample_interval: timedelta = timedelta(seconds=1)  # Get mean of IOIs within this window

        # Check and generate our input data
        self.total_beats: int = 1000
        self.latency: np.array = self._append_timestamps_to_latency_array(latency_array)
        self.num_simulations: int = num_simulations

        # Simulation parameters, in the form of numba dictionaries to be used inside a numba function
        self.parameter: str = kwargs.get('parameter', 'original')
        self.leader: str = kwargs.get('leader', None)
        self.noise: float = kwargs.get('noise', 0.005)
        self._keys_pcm: pd.DataFrame = keys_params
        self._drms_pcm: pd.DataFrame = drms_params

        # Musician parameters
        self.keys_params_raw: dict = self._get_raw_musician_parameters(init=self._keys_pcm)
        self.drms_params_raw: dict = self._get_raw_musician_parameters(init=self._drms_pcm)
        self.keys_params: nb.typed.Dict = self._convert_musician_parameters_dict_to_numba(
            self._modify_musician_parameters_by_simulation_type(self.keys_params_raw)
        )
        self.drms_params: nb.typed.Dict = self._convert_musician_parameters_dict_to_numba(
            self._modify_musician_parameters_by_simulation_type(self.drms_params_raw)
        )

        # Empty lists to store our keys and drums simulations in
        self._keys_simulations_raw: list[dict] = []
        self._drms_simulations_raw: list[dict] = []
        self.keys_simulations: list[pd.DataFrame] = []
        self.drms_simulations: list[pd.DataFrame] = []

        # Simulations results dictionary
        self.results_dic = self._create_summary_dictionary(include_simulations=True)

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
    def _get_initial_iois(
            nn, default: float = 0.5
    ) -> tuple[float]:
        """
        Gets the first two IOI values from a performance dataset. If any are invalid, return our default of (0.5, 0.5)
        """
        try:
            iois = nn.iloc[:2]['my_next_ioi'].values.tolist()
        except KeyError:
            return default, default
        else:
            if any(np.isnan(iois)):
                return default, default
            else:
                return iois

    @staticmethod
    def _get_initial_onset(
            nn: pd.DataFrame, default: float = 8, threshold: float = 9
    ) -> float:
        """
        Gets the first onset time from a performance dataset. If this is invalid, returns our default of 8
        """
        try:
            onset = nn.iloc[0]['my_onset']
        except KeyError:
            return default
        else:
            # This is used to capture a few cases where one performer didn't come straight in after the count in
            if onset > threshold:
                return default
            else:
                return onset

    @staticmethod
    def _get_raw_musician_parameters(
            init: pd.DataFrame
    ) -> dict:
        """
        Gets necessary simulation parameters from pandas dataframe and converts to a dictionary
        """
        # Need to reset our index so we can use 0 indexing
        init = init.reset_index(drop=True)
        # Variables we need from our input
        cols = [
            'correction_self', 'correction_partner', 'intercept', 'resid_std',
        ]
        # Create the dictionary and return
        dic = {s: init[s].iloc[0].astype(np.float64) for s in cols}
        dic.update({'instrument': init['instrument'].iloc[0]})
        return dic

    def _modify_musician_parameters_by_simulation_type(
            self, input_data
    ):
        """
        Modifies a simulated musician's parameters according to the given simulation type
        """
        # Used to get the mean of a particular coefficient across both musicians
        mean = lambda s: np.mean([self.keys_params_raw[s], self.drms_params_raw[s]])
        output_data = {}
        # Original coupling: uses coefficients, intercept from the model itself
        if self.parameter == 'original':
            output_data = {
                'correction_self': input_data['correction_self'],
                'correction_partner': input_data['correction_partner'],
                'intercept': input_data['intercept'],
            }
        # Democracy: uses mean coefficients and intercepts from across the duo
        elif self.parameter == 'democracy':
            output_data = {
                'correction_self': mean('correction_self'),
                'correction_partner': mean('correction_partner'),
                'intercept': 0,
            }
        # Leadership: simulation specification differs according to whether instrument is leading or following
        elif self.parameter == 'leadership':
            # Leader: no correction to follower, all other parameters set to means
            if input_data['instrument'].lower() == self.leader.lower():
                output_data = {
                    'correction_self': -abs(mean('correction_self')),
                    'correction_partner': 0,
                    'intercept': 0
                }
            # Follower: corrects to leader by mean correction across duo, all other parameters set to 0
            else:
                output_data = {
                    'correction_self': -abs(mean('correction_self')),
                    'correction_partner': mean('correction_partner'),
                    'intercept': 0,
                }
        # Anarchy: neither musician corrects to each other, all other coefficients set to their mean
        elif self.parameter == 'anarchy':
            output_data = {
                'correction_self': -abs(mean('correction_self')),
                'correction_partner': 0,
                'intercept': 0,
            }
        # Update our dictionary with the required amount of noise
        output_data.update({
            'resid_std': self.noise
        })
        return output_data

    @staticmethod
    def _convert_musician_parameters_dict_to_numba(
            python_dict: dict
    ) -> nb.typed.Dict:
        """
        Converts a Python dictionary into a type that can be utilised by Numba
        """
        # Create the empty dictionary
        nb_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float64,)
        # Iterate through our dictionary
        for k, v in python_dict.items():
            # If the type is compatible with numba floats, set it in the numba dictionary
            if type(v) != str:
                nb_dict[k] = v
        return nb_dict

    def _initialise_empty_data(
            self, iois: tuple[float] = (0.5, 0.5), onset: float = 8
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
        for s in ['my_onset', 'asynchrony', 'my_next_ioi', 'my_prev_ioi', 'my_next_ioi_diff', 'my_prev_ioi_diff']:
            nb_dict[s] = np.zeros(shape=self.total_beats)
        # Fill the dictionary arrays with our starting values
        # My onset
        nb_dict['my_onset'][0] = onset
        nb_dict['my_onset'][1] = onset + iois[0]
        # My next ioi
        nb_dict['my_next_ioi'][0] = iois[0]
        nb_dict['my_next_ioi'][1] = iois[1]
        # My previous ioi
        nb_dict['my_prev_ioi'][0] = np.nan
        nb_dict['my_prev_ioi'][1] = iois[0]
        # My next ioi diff
        nb_dict['my_next_ioi_diff'][0] = np.nan
        nb_dict['my_next_ioi_diff'][1] = iois[1] - iois[0]  # This will always be 0
        # My previous ioi diff
        nb_dict['my_prev_ioi_diff'][0] = np.nan
        nb_dict['my_prev_ioi_diff'][1] = np.nan
        return nb_dict

    def create_all_simulations(
            self
    ):
        """
        Run the simulations and create a list of dataframes for each individual performer
        """
        for i in range(0, self.num_simulations):
            keys, drms = self._create_one_simulation(
                # Initialise empty data to store in
                self._initialise_empty_data(), self._initialise_empty_data(),
                self.keys_params, self.drms_params,     # Parameters for the simulation, e.g. coefficients
                np.random.normal(0, self.keys_params['resid_std'], 10000),
                np.random.normal(0, self.drms_params['resid_std'], 10000),
                self.latency, self.total_beats  # Latency array and total number of beats
            )
            # Append raw data for debugging
            self._keys_simulations_raw.append(dict(keys))
            self._drms_simulations_raw.append(dict(drms))
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
        # Drop rows with all zeros in
        df = df.loc[~(df == 0).all(axis=1)]
        # Convert my onset column to a timedelta
        df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['my_onset']])
        offset = timedelta(seconds=8 - df.iloc[0]['my_onset'])
        # Set the index to our timedelta column, resample (default every second), and get mean
        return df.set_index('td').resample(rule=self._resample_interval, offset=offset).mean()

    @staticmethod
    @nb.njit
    def _create_one_simulation(
            keys_data: nb.typed.Dict, drms_data: nb.typed.Dict,
            keys_params: nb.typed.Dict, drms_params: nb.typed.Dict,
            keys_noise, drms_noise,
            lat: np.ndarray, beats: int
    ) -> tuple:
        """
        Create data for one simulation, using numba optimisations
        """

        def get_latency_at_onset(
                my_onset: float
        ) -> float:
            """
            Get the current amount of latency applied to our partner when we played our onset
            """
            # If our first onset was before the 8 second count-in, return the first amount of delay applied
            if my_onset < lat[0, :][1]:
                return lat[:, 0][0]
            # Else, return the correct amount of latency
            else:
                return lat[lat[:, 1] == lat[:, 1][lat[:, 1] <= my_onset].max()][:, 0][0]

        def calculate_async(
                my_onset: float, their_onset: float
        ) -> float:
            """
            Calculates the current asynchrony with our partner at our onset. Note the order of positional arguments!!
            """
            return (their_onset + get_latency_at_onset(my_onset)) - my_onset

        def predict_next_ioi_diff(
                prev_diff: float, asynchrony: float, params: nb.typed.Dict, noise: np.ndarray
        ) -> float:
            """
            Predict the difference between previous and next IOIs using inputted data and model parameters
            """
            a = prev_diff * params['correction_self']
            b = asynchrony * params['correction_partner']
            c = params['intercept']
            n = np.random.choice(noise)
            return a + b + c + n

        # Calculate our first two async values. We don't do this when initialising the empty data, because it
        # both requires the get_latency_at_onset function and access to both keys and drums data.
        for i in range(0, 2):
            keys_data['asynchrony'][i] = calculate_async(keys_data['my_onset'][i], drms_data['my_onset'][i])
            drms_data['asynchrony'][i] = calculate_async(drms_data['my_onset'][i], keys_data['my_onset'][i])
        # We don't use the full range of beats, given that we've already added some in when initialising our data
        for i in range(2, beats):
            # Shift difference
            keys_data['my_prev_ioi_diff'][i] = keys_data['my_next_ioi_diff'][i - 1]
            drms_data['my_prev_ioi_diff'][i] = drms_data['my_next_ioi_diff'][i - 1]
            # Shift IOI
            keys_data['my_prev_ioi'][i] = keys_data['my_next_ioi'][i - 1]
            drms_data['my_prev_ioi'][i] = drms_data['my_next_ioi'][i - 1]
            # Get next onset by adding previous onset to predicted IOI
            keys_data['my_onset'][i] = keys_data['my_onset'][i - 1] + keys_data['my_prev_ioi'][i]
            drms_data['my_onset'][i] = drms_data['my_onset'][i - 1] + drms_data['my_prev_ioi'][i]
            # Get async value by subtracting partner's onset (plus latency) from ours
            try:
                keys_data['asynchrony'][i] = calculate_async(keys_data['my_onset'][i], drms_data['my_onset'][i])
                drms_data['asynchrony'][i] = calculate_async(drms_data['my_onset'][i], keys_data['my_onset'][i])
            # If there's an issue here, break out of the simulation
            except:
                break
            # Predict difference between previous IOI and next IOI
            keys_data['my_next_ioi_diff'][i] = predict_next_ioi_diff(
                keys_data['my_prev_ioi_diff'][i], keys_data['asynchrony'][i], keys_params, keys_noise
            )
            drms_data['my_next_ioi_diff'][i] = predict_next_ioi_diff(
                drms_data['my_prev_ioi_diff'][i], drms_data['asynchrony'][i], drms_params, drms_noise
            )
            # Use predicted difference between IOIs to get next actual IOI
            keys_data['my_next_ioi'][i] = keys_data['my_next_ioi_diff'][i] + keys_data['my_prev_ioi'][i]
            drms_data['my_next_ioi'][i] = drms_data['my_next_ioi_diff'][i] + drms_data['my_prev_ioi'][i]
            # If we've accelerated to a ridiculous extent (due to noise), we need to break.
            if keys_data['my_next_ioi'][i] < 0 or drms_data['my_next_ioi'][i] < 0:
                break
            # TODO: should be the same length as original performance?
            # If we've exceeded our endpoint, break
            if keys_data['my_onset'][i] > 100 or drms_data['my_onset'][i] > 100:
                break
        return keys_data, drms_data

    @staticmethod
    def _get_average_var_for_one_simulation(
            all_perf: list[pd.DataFrame], var: str = 'my_next_ioi'
    ) -> pd.DataFrame:
        """
        Concatenate all simulations together and get the row-wise average (i.e. avg IOI every second)
        """
        # We use absolute values here, as this makes most sense when getting mean async across performers
        # For example, if keys-drums async = -0.5 and drums-keys async = 0.5, mean without absolute values == 0
        # Instead, this should be 0.5.
        return pd.DataFrame(
            pd.concat([df_[var] for df_ in all_perf], axis=1).abs().mean(axis=1), columns=[var]
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
            avg = self._get_average_var_for_one_simulation([keys, drms])
            # Get elapsed number of seconds
            avg['elapsed_seconds'] = avg.index.seconds
            # Convert IOIs to BPM for tempo slope regression
            avg['my_next_bpm'] = 60 / avg['my_next_ioi']
            avg = avg.dropna()
            # Conduct and fit the regression model
            md = smf.ols('my_next_bpm~elapsed_seconds', data=avg.dropna()).fit()
            # Extract the tempo slope coefficient and append to list
            coeffs.append(md.params[1])
        # Calculate the median tempo slope coefficient (robust to outliers!) from all simulations and return
        return np.nanmedian(pd.Series(coeffs).replace(-np.Inf, np.nan))

    def get_average_pairwise_asynchrony(
            self
    ) -> float:
        """
        Gets the average pairwise asynchrony (in milliseconds!) across all simulated performances
        """
        # Define the function for calculating pairwise async
        # This is the root mean square of the standard deviation of the async
        pw_async = lambda k_, d_: np.sqrt(np.mean(np.square([k_['asynchrony'].std(), d_['asynchrony'].std()]))) * 1000
        # Calculate the mean pairwise async across all performances
        return np.mean([pw_async(k__, d__) for k__, d__ in zip(self.keys_simulations, self.drms_simulations)])

    def get_simulation_data_for_plotting(
            self, plot_individual: bool = True, plot_average: bool = True, var: str = 'my_next_ioi',
            timespan: tuple[int] = (7, 101),
    ) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
        """
        Wrangles simulation data into a format that can be plotted and returns.
        """
        # Create simulations if we haven't already done so
        if len(self.keys_simulations) < 1 or len(self.drms_simulations) < 1:
            self.create_all_simulations()
        # If we're plotting individual simulations
        individual_sims = []
        grand_avg = None
        if plot_individual:
            # Iterate through individual keys and drums simulations
            for k_, d_ in zip(self.keys_simulations, self.drms_simulations):
                # Average individual simulation
                avg = self._get_average_var_for_one_simulation([k_, d_], var=var)
                # Subset for required timespan
                avg = avg[(avg.index.seconds >= timespan[0]) & (avg.index.seconds <= timespan[1])]
                individual_sims.append(avg)
        # If we're plotting our average simulation
        if plot_average:
            # Get grand average simulation by averaging our average simulations
            zi = zip(self.keys_simulations, self.drms_simulations)
            grand_avg = self._get_average_var_for_one_simulation(
                [self._get_average_var_for_one_simulation([k_, d_], var=var) for k_, d_ in zi], var=var
            )
            # Subset for required timespan
            grand_avg = grand_avg[(grand_avg.index.seconds >= timespan[0]) & (grand_avg.index.seconds <= timespan[1])]
        return individual_sims, grand_avg

    def _create_summary_dictionary(
            self, include_simulations: bool = True
    ) -> dict:
        """
        Creates a summary dictionary with important simulation parameters
        """
        dic = {
            'trial': self._keys_pcm["trial"].iloc[0],
            'block': self._keys_pcm["block"].iloc[0],
            'latency': self._keys_pcm["latency"].iloc[0],
            'jitter': self._keys_pcm["jitter"].iloc[0],
            'parameter': self.parameter,
            'leader': self.leader,
            'ts': self.get_average_tempo_slope(),
            'asynchrony': self.get_average_pairwise_asynchrony(),
        }
        if include_simulations:
            dic.update({
                'keys_sim': self.keys_simulations,
                'drms_sim': self.drms_simulations,
            })
        return dic


def generate_phase_correction_simulations(
        mds: pd.DataFrame, output_dir: str, logger=None, force_rebuild: bool = False, num_simulations: int = 100
) -> list[dict]:
    """
    Create simulated performances across a range of artificial coupling parameters for every phase correction model
    """
    # Try and load the models from the disk to save time, unless we're forcing them to rebuild anyway
    if not force_rebuild:
        mds = autils.load_from_disc(output_dir, filename='phase_correction_sims.p')
        # If we've successfully loaded models, return these straight away
        if mds is not None:
            return mds
    # Define the parameters and leader combinations we want to create simulations for
    params = [('original', None), ('democracy', None), ('anarchy', None), ('leadership', 'Drums')]
    # Create an empty list to hold simulations in
    all_sims = []
    for pcm in mds:
        # Get the raw data from our phase correction model
        ke = pd.DataFrame([pcm.keys_dic])
        dr = pd.DataFrame([pcm.drms_dic])
        # Get the zoom array used in the performance
        z = ke['zoom_arr'].iloc[0]
        # Iterate through each parameter and leader combo
        for param, leader in params:
            if logger is not None:
                logger.info(f'... trial {ke["trial"].iloc[0]}, block {ke["block"].iloc[0]}, '
                            f'latency {ke["latency"].iloc[0]}, jitter {ke["jitter"].iloc[0]}, '
                            f'parameter {param}, leader {leader}')
            # Initialise the simulation class
            sim = Simulation(
                keys_params=ke, drms_params=dr, latency_array=z, num_simulations=num_simulations,
                parameter=param, leader=leader, keys_nn=pcm.keys_nn, drms_nn=pcm.drms_nn,
            )
            # Create all simulations for this parameter/leader combination
            sim.create_all_simulations()
            # Append the results dictionary (including the raw simulations)
            all_sims.append(sim.results_dic)
    # Pickle the result -- this will be quite large, depending on the number of simulations!
    pickle.dump(all_sims, open(f"{output_dir}\\phase_correction_sims.p", "wb"))
    return all_sims


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_data(r"C:\Python Projects\jazz-jitter-analysis\models\phase_correction_mds.p")
    # Default location to save output simulations
    output = r"C:\Python Projects\jazz-jitter-analysis\models"
    # Generate simulations and pickle
    mds = generate_phase_correction_simulations(mds=raw, output_dir=output)
