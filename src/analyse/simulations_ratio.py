import numpy as np
import pandas as pd
import numba as nb
from datetime import timedelta
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import src.visualise.visualise_utils as vutils

SIGNAL_NOISE = 4

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
        self._include_penalty: str = kwargs.get('include_penalty', None)
        self._debug: bool = kwargs.get('debug', False)

        # Check and generate our input data
        # self.total_beats: int = self._get_number_of_beats_for_simulation(keys_params, drms_params)
        self.total_beats: int = 1000
        self.latency: np.array = self._append_timestamps_to_latency_array(latency_array)
        self.num_simulations: int = num_simulations

        # Simulation parameters, in the form of numba dictionaries to be used inside a numba function
        self.parameter: str = kwargs.get('parameter', 'original')
        self.leader: str = kwargs.get('leader', None)

        # Musician parameters
        self.keys_params_raw: dict = self._get_raw_musician_parameters(init=keys_params)
        self.drms_params_raw: dict = self._get_raw_musician_parameters(init=drms_params)
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
            nn
    ) -> tuple[float]:
        try:
            iois = nn.iloc[:2]['my_next_ioi'].values.tolist()
        except KeyError:
            return 0.5, 0.5
        else:
            return [ioi if not np.isnan(ioi) else 0.5 for ioi in iois]

    @staticmethod
    def _get_initial_onset(
            nn: pd.DataFrame
    ) -> float:
        """

        """
        try:
            onset = nn.iloc[0]['my_onset']
        except KeyError:
            return 8.0
        else:
            # This is used to capture a few cases where one performer didn't come straight in after the count in
            if onset > 9:
                return 8.0
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
        output_data.update({
            # 'resid_std': input_data['resid_std'],
            'resid_std': 0.005
        })
        return output_data

    @staticmethod
    def _convert_musician_parameters_dict_to_numba(
            python_dict: dict
    ) -> nb.typed.Dict:
        """
        Converts a Python dictionary into a type that can be utilised by Numba
        """
        nb_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float64,)
        for k, v in python_dict.items():
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
        nb_dict['my_next_ioi_diff'][1] = iois[1] - iois[0]
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
            if self._debug:
                keys, drms = self._create_one_simulation(
                    # Initialise empty data to store in, WITHOUT using our initial IOIs/onsets
                    self._initialise_empty_data(), self._initialise_empty_data(),
                    # Parameters to use in the simulation, e.g. coefficients
                    self.keys_params, self.drms_params,
                    # Use empty noise arrays
                    np.zeros(10000), np.zeros(10000),
                    # Latency array and total number of beats
                    self.latency, self.total_beats
                )
            else:
                keys, drms = self._create_one_simulation(
                    # Initialise empty data to store in
                    self._initialise_empty_data(iois=self._keys_initial_iois, onset=self._keys_initial_onset),
                    self._initialise_empty_data(iois=self._drms_initial_iois, onset=self._drms_initial_onset),
                    self.keys_params, self.drms_params,     # Parameters for the simulation, e.g. coefficients
                    np.random.normal(0, self.keys_params['resid_std'], 10000),
                    np.random.normal(0, self.drms_params['resid_std'], 10000),
                    self.latency, self.total_beats  # Latency array and total number of beats
                )
            # Append unformatted data for debugging
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

        def get_lat(
                partner_onset: float
        ) -> float:
            """
            Get the current amount of latency applied when our partner played their onset
            """
            try:
                return lat[lat[:, 1] == lat[:, 1][lat[:, 1] <= partner_onset].max()][:, 0][0]
            # If their first onset was before the 8 second count-in, return the first amount of delay applied
            except:
                return lat[:, 0][0]

        def predict(
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

        keys_data['asynchrony'][0] = (
                (drms_data['my_onset'][0] + get_lat(drms_data['my_onset'][0])) - keys_data['my_onset'][0]
        )
        drms_data['asynchrony'][0] = (
                (keys_data['my_onset'][0] + get_lat(keys_data['my_onset'][0])) - drms_data['my_onset'][0]
        )
        keys_data['asynchrony'][1] = (
                (drms_data['my_onset'][1] + get_lat(drms_data['my_onset'][1])) - keys_data['my_onset'][1]
        )
        drms_data['asynchrony'][1] = (
                (keys_data['my_onset'][1] + get_lat(keys_data['my_onset'][1])) - drms_data['my_onset'][1]
        )
        # We don't use the full range of beats, given that we've already added some in when creating our starter data
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
            # Get asynchrony value by subtracting partner's onset (plus latency) from ours
            # noinspection PyBroadException
            try:
                keys_data['asynchrony'][i] = (
                        (drms_data['my_onset'][i] + get_lat(drms_data['my_onset'][i])) - keys_data['my_onset'][i]
                )
                drms_data['asynchrony'][i] = (
                        (keys_data['my_onset'][i] + get_lat(keys_data['my_onset'][i])) - drms_data['my_onset'][i]
                )
            except:
                break
            # Predict difference between previous IOI and next IOI
            keys_data['my_next_ioi_diff'][i] = predict(
                keys_data['my_prev_ioi_diff'][i], keys_data['asynchrony'][i], keys_params, keys_noise
            )
            drms_data['my_next_ioi_diff'][i] = predict(
                drms_data['my_prev_ioi_diff'][i], drms_data['asynchrony'][i], drms_params, drms_noise
            )
            # Use predicted difference between IOIs to get next actual IOI
            keys_data['my_next_ioi'][i] = keys_data['my_next_ioi_diff'][i] + keys_data['my_prev_ioi'][i]
            drms_data['my_next_ioi'][i] = drms_data['my_next_ioi_diff'][i] + drms_data['my_prev_ioi'][i]
            if keys_data['my_next_ioi'][i] < 0 or drms_data['my_next_ioi'][i] < 0:
                break
            # TODO: should be same length as original performance?
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

        """
        pw_async = lambda k_, d_: np.sqrt(np.mean(np.square([k_['asynchrony'].std(), d_['asynchrony'].std()]))) * 1000
        return np.mean([pw_async(k__, d__) for k__, d__ in zip(self.keys_simulations, self.drms_simulations)])

    def plot_simulation(
            self, plot_individual: bool = True, plot_average: bool = True, color: str = 'r', var: str = 'my_next_ioi',
            timespan: tuple[int] = (7, 101), ax: plt.Axes = None, bpm: bool = True, lw: float = 4., ls: str = '-'
    ) -> None:
        """

        """
        # Create simulations if we haven't already done so
        if len(self.keys_simulations) < 1 or len(self.drms_simulations) < 1:
            self.create_all_simulations()
        if ax is None:
            _, ax = plt.subplots(1, 1)
        # If we're plotting individual simulations
        if plot_individual:
            # Iterate through individual keys and drums simulations
            for k_, d_ in zip(self.keys_simulations, self.drms_simulations):
                # Average individual simulation
                avg = self._get_average_var_for_one_simulation([k_, d_], var=var)
                # Subset for required timespan
                avg = avg[(avg.index.seconds >= timespan[0]) & (avg.index.seconds <= timespan[1])]
                # Plot rolling BPM for individual simulation
                if bpm:
                    avg[var] = 60 / avg[var]
                ax.plot(
                    avg.index.seconds, (avg[var]).rolling(window='4s').mean(), alpha=0.01, color=color,
                )
        # If we're plotting our average simulation
        if plot_average:
            # Get average simulation by averaging our average simulations
            zi = zip(self.keys_simulations, self.drms_simulations)
            grand_avg = self._get_average_var_for_one_simulation(
                [self._get_average_var_for_one_simulation([k_, d_], var=var) for k_, d_ in zi],
                var=var
            )
            # Subset for required timespan
            grand_avg = grand_avg[(grand_avg.index.seconds >= timespan[0]) & (grand_avg.index.seconds <= timespan[1])]
            # Plot the average simulation, including the legend label
            if bpm:
                grand_avg[var] = 60 / grand_avg[var]
            ax.plot(
                grand_avg.index.seconds, (grand_avg[var]).rolling(window='4s').mean(), alpha=1, linewidth=lw, ls=ls,
                color=color, label=f'{self.parameter.title()} {self.leader if self.leader is not None else ""}'
            )


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 500)
    plt.close('all')

    mds = pickle.load(open(r"C:\Python Projects\jazz-jitter-analysis\src\analyse\mds_for_simulations.p", "rb"))
    cond = mds[(mds['latency'] == 45) & (mds['jitter'] == 1)].reset_index()
    z = cond['zoom_arr'].iloc[0]
    res = []
    for i in np.array(np.meshgrid(np.linspace(0, 1., 101), np.linspace(0, 1., 101))).T.reshape(-1, 2):
        coeff1 = round(i[0], 2)
        coeff2 = round(i[1], 2)
        ke = pd.DataFrame(
            [{
                'correction_self': cond[cond['instrument'] == 'Keys']['correction_self'].mean(),
                'correction_partner': coeff1,
                'intercept': cond['intercept'].mean(),
                'resid_std': 0.01,
                'instrument': 'Keys'
            }]
        )
        dr = pd.DataFrame(
            [{
                'correction_self': cond[cond['instrument'] != 'Keys']['correction_self'].mean(),
                'correction_partner': coeff2,
                'intercept': cond['intercept'].mean(),
                'resid_std': 0.01,
                'instrument': 'Drums'
            }]
        )
        sim = Simulation(ke, dr, z, num_simulations=10, debug=False)
        sim.create_all_simulations()
        #
        # for keys, drms in zip(sim.keys_simulations, sim.drms_simulations):
        #     avg = sim.get_grand_average_tempo([keys, drms])
        #     avg = avg[(avg.index.seconds >= 7) & (avg.index.seconds <= 101)]
        #     plt.plot(avg.index.seconds, (60 / avg.my_next_ioi).rolling(window='4s').mean(), alpha=0.01,
        #              color='r')
        # grand_avg = sim.get_grand_average_tempo(
        #     [sim.get_grand_average_tempo([k, d]) for k, d in zip(sim.keys_simulations, sim.drms_simulations)]
        # )
        # grand_avg = grand_avg[(grand_avg.index.seconds >= 7) & (grand_avg.index.seconds <= 101)]
        # plt.plot(grand_avg.index.seconds, (60 / grand_avg.my_next_ioi).rolling(window='4s').mean(), alpha=1,
        #          color='r', label='Simulation')
        #
        #
        # plt.title(f'{coeff1}, {coeff2}')
        # plt.ylim(30, 160)
        # plt.show()
    #     ts = sim.get_average_tempo_slope()
    #     res.append((coeff1, coeff2, ts))
    #     print(coeff1, coeff2, ts)
    #
    # df = pd.DataFrame(res, columns=['Keys -> Drums correction', 'Drums -> Keys correction', 'tempo_slope'])
    # df = df.pivot('Keys -> Drums correction', 'Drums -> Keys correction', 'tempo_slope').sort_index(ascending=False)
    # fig, ax = plt.subplots(1, 1)
    # g = sns.heatmap(df, vmin=-1, vmax=0, cmap='rocket', ax=ax)
    # means = [
    #     tuple(grp.groupby('instrument').mean()['correction_partner'].values) for _, grp in
    #     mds[(mds['latency'] == 45) & (mds['jitter'] == 1)].groupby('trial')
    # ]
    # drms_means = np.interp([m[0] for m in means], np.linspace(0., 1., 21), g.get_xticks())
    # keys_means = np.interp([m[1] for m in means], np.linspace(0., 1., 21), np.flip(g.get_yticks()))
    # for d, k, mark, label in zip(drms_means, keys_means, ['o', 's', 'p', '*', 'X'], ['Duo 1', 'Duo 2', 'Duo 3', 'Duo 4', 'Duo 5']):
    #     ax.scatter(d, k, marker=mark, label=label, s=50, )
    # ax.set_xticks(ticks=[t - 0.5 for t in ax.get_xticks()], labels=ax.get_xticklabels(), ha='right')
    # ax.set_yticks(ticks=[t + 0.5 for t in ax.get_yticks()][:-1], labels=ax.get_yticklabels()[:-1], ha='center')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
