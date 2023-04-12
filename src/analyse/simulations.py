import numpy as np
import pandas as pd
import numba as nb
from joblib import Parallel, delayed
import dill as pickle
from datetime import timedelta
import statsmodels.formula.api as smf

import src.analyse.analysis_utils as autils
from src.analyse.phase_correction_models import PhaseCorrectionModel

# Define the objects we can import from this file into others
__all__ = [
    'generate_phase_correction_simulations_for_coupling_parameters',
    'generate_phase_correction_simulations_for_individual_conditions',
    'Simulation'
]


class Simulation:
    """
    Creates X number of simulated performances from two given phase correction models.

    Number of simulations defaults to 500, the same number in Jacoby et al. (2021).
    """
    def __init__(
            self, pcm: PhaseCorrectionModel | pd.DataFrame, num_simulations: int = 500, **kwargs
    ):
        # Default parameters
        self._resample_interval: timedelta = timedelta(seconds=1)  # Get mean of IOIs within this window
        # Check and generate our input data
        self.total_beats: int = 1000
        self.num_simulations: int = num_simulations
        # Rolling window parameters -- should be the same as for the phase correction models
        self._rolling_window_size: str = kwargs.get('rolling_window_size', '2s')   # 2 seconds = 2 beats at 120BPM
        self._rolling_min_periods: int = kwargs.get('rolling_min_periods', 2)
        # Get the raw data from our phase correction model
        if isinstance(pcm, PhaseCorrectionModel):
            self.keys_pcm = pd.DataFrame([pcm.keys_dic])
            self.drms_pcm = pd.DataFrame([pcm.drms_dic])
        elif isinstance(pcm, pd.DataFrame):
            self.keys_pcm = pcm[pcm['instrument'] == 'Keys']
            self.drms_pcm = pcm[pcm['instrument'] != 'Keys']
        # Get the zoom array used in the performance
        self.latency: np.ndarray = self._append_timestamps_to_latency_array(self.keys_pcm['zoom_arr'].iloc[0])
        # Raw simulation parameters, which will be used to create the dictionaries used in the simulations.
        self.parameter: str = kwargs.get('parameter', 'original')
        # Noise parameters for the simulation
        self.noise = kwargs.get('noise', autils.CONSTANT_RESID_NOISE)     # Default noise term
        self.use_original_noise: bool = kwargs.get('use_original_noise', False)    # Whether to use noise from model
        # Musician parameters
        self.keys_params_raw: dict = self._get_raw_musician_parameters(init=self.keys_pcm)
        self.drms_params_raw: dict = self._get_raw_musician_parameters(init=self.drms_pcm)
        self.keys_params: nb.typed.Dict = self._convert_musician_parameters_dict_to_numba(
            self._modify_musician_parameters_by_simulation_type(self.keys_params_raw)
        )
        self.drms_params: nb.typed.Dict = self._convert_musician_parameters_dict_to_numba(
            self._modify_musician_parameters_by_simulation_type(self.drms_params_raw)
        )
        # Empty lists to store our keys and drums simulations in
        self.keys_simulations: list[pd.DataFrame] = []
        self.drms_simulations: list[pd.DataFrame] = []
        # Empty attribute that we will fill with our results dictionary after creating simulations
        self.results_dic: dict = None

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
        return {
            'correction_self': input_data['correction_self'],
            'correction_partner': input_data['correction_partner'],
            'intercept': input_data['intercept'],
            'resid_std': input_data['resid_std'] if self.use_original_noise else self.noise
        }

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
        for s in [
            'my_onset', 'asynchrony', 'asynchrony_third_person',
            'my_next_ioi', 'my_prev_ioi', 'my_next_ioi_diff', 'my_prev_ioi_diff'
        ]:
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
        # Create the simulations with joblib and numba function
        all_sims = Parallel(n_jobs=-1, prefer='threads')(delayed(autils.create_one_simulation)(
            self._initialise_empty_data(),
            self._initialise_empty_data(),
            self.keys_params,
            self.drms_params,
            np.random.normal(0, self.keys_params['resid_std'], 10000),
            np.random.normal(0, self.drms_params['resid_std'], 10000),
            self.latency,
            self.total_beats
        ) for _ in range(self.num_simulations))
        # Format simulated data
        self.keys_simulations = [self._format_simulated_data(d[0]) for d in all_sims]
        self.drms_simulations = [self._format_simulated_data(d[1]) for d in all_sims]
        # After running simulations, clean up by converting simulation parameters from numba back to python dictionary
        # This will allow us to pickle instances of the Simulation class without errors, as numba dictionaries are
        # currently not supported by the pickle package.
        self.keys_params = dict(self.keys_params)
        self.drms_params = dict(self.drms_params)
        # Save our results dictionary to an attribute in the class, so we don't have to do it later (or multiple times!)
        self.results_dic = self._create_summary_dictionary()

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
        idx = pd.to_timedelta([timedelta(seconds=val) for val in df['my_onset']])
        offset = timedelta(seconds=8 - df.iloc[0]['my_onset'])
        # Set the index to our timedelta column
        df = df.set_index(idx)
        # Get our rolling IOI standard deviation values
        df = self._get_rolling_standard_deviation_values(df)
        # Resample to the desired frequency with offset, get the mean values, and interpolate to fill NaNs
        return df.resample('1s', offset=offset).apply(np.nanmean).interpolate(limit_direction='backward')

    def _get_rolling_standard_deviation_values(
            self, df: pd.DataFrame, cols: tuple[str] = ('my_prev_ioi',)
    ) -> pd.DataFrame:
        # Create the rolling window with the desired window size
        roll = df.rolling(
            window=self._rolling_window_size, min_periods=self._rolling_min_periods, closed='both', on=df.index
        )
        # Iterate through the required columns
        for col in cols:
            # Extract the standard deviation and convert into milliseconds
            df[f'{col}_std'] = roll[col].std(skipna=True) * 1000
        return df

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
            self, func=np.nanmean, **kwargs
    ) -> float | None:
        """
        Returns the average tempo slope for all simulations.

        Method:
        ---
        -   For every simulation, zip the corresponding keys and drums performance together.
        -   Then, get the average IOI for every second across both keys and drums.
            -   This is straightforward, because we resampled to average IOI per second in _format_simulated_data
        -   Convert average IOI to average BPM by dividing by 60, then regress against elapsed seconds
        -   Extract the slope coefficient, take the median across all simulations, and return
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
        return func(pd.Series(coeffs).replace(-np.Inf, np.nan), **kwargs)

    def get_average_ioi_variability(
            self, func=np.nanmean, **kwargs
    ) -> float | None:
        """
        Returns the average tempo slope for all simulations.

        Method:
        ---
        -   For every simulation, get the median IOI standard deviation value over the window size
        -   Calculate the mean of all of these values.
        """
        return func([
            [s['my_prev_ioi_std'].median() for s in self.keys_simulations],
            [s['my_prev_ioi_std'].median() for s in self.drms_simulations]
        ], **kwargs)

    def get_average_pairwise_asynchrony(
            self, func=np.nanmean, async_col: str = 'asynchrony', **kwargs
    ) -> float | None:
        """
        Gets the average pairwise asynchrony (in milliseconds!) across all simulated performances
        """
        def pw_async(keys, drms):
            """
            Function used to calculate the pairwise asynchrony for a single simulation, in milliseconds
            """
            # Concatenate the two asynchrony columns together
            conc = np.concatenate((keys[async_col].to_numpy(), drms[async_col].to_numpy()))
            # Square the values, take the mean, then the square root, then convert to miliseconds and return
            return np.sqrt(np.nanmean(np.square(conc))) * 1000

        # Calculate the mean pairwise async across all performances
        return func([pw_async(k_, d_) for k_, d_ in zip(self.keys_simulations, self.drms_simulations)], **kwargs)

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
            self,
    ) -> dict:
        """
        Creates a summary dictionary with important simulation parameters
        """
        return {
            # Condition metadata
            'trial': self.keys_pcm["trial"].iloc[0],
            'block': self.keys_pcm["block"].iloc[0],
            'latency': self.keys_pcm["latency"].iloc[0],
            'jitter': self.keys_pcm["jitter"].iloc[0],
            # Simulation metadata
            'parameter': self.parameter,
            'original_noise': self.use_original_noise,
            'keys_parameters': self.keys_params_raw,
            'drums_parameters': self.drms_params_raw,
            # Metrics from the actual original performance on which this simulation was based
            'tempo_slope_original': np.mean(
                [self.keys_pcm['tempo_slope'].iloc[0], self.drms_pcm['tempo_slope'].iloc[0]]
            ),
            'ioi_variability_original': np.mean(
                [self.keys_pcm['ioi_std'].iloc[0], self.drms_pcm['ioi_std'].iloc[0]]
            ),
            'asynchrony_original': np.mean(
                [self.keys_pcm['pw_asym'].iloc[0], self.drms_pcm['pw_asym'].iloc[0]]
            ),
            # Summary metrics from all the simulated performances
            'tempo_slope_simulated': self.get_average_tempo_slope(func=np.nanmedian),  # average
            'tempo_slope_simulated_std': self.get_average_tempo_slope(func=np.nanstd),  # standard deviation
            'tempo_slope_simulated_ci': self.get_average_tempo_slope(func=np.nanpercentile, q=[2.5, 97.5]),     # 95% ci
            'ioi_variability_simulated': self.get_average_ioi_variability(func=np.nanmedian),
            'ioi_variability_simulated_std': self.get_average_ioi_variability(func=np.nanstd),
            'ioi_variability_simulated_ci': self.get_average_ioi_variability(func=np.nanpercentile, q=[2.5, 97.5]),
            'asynchrony_simulated': self.get_average_pairwise_asynchrony(func=np.nanmedian,
                                                                         async_col='asynchrony_third_person'),
            'asynchrony_simulated_indiv': self.get_average_pairwise_asynchrony(func=np.nanmedian),
            'asynchrony_simulated_std': self.get_average_pairwise_asynchrony(func=np.nanstd),
            'asynchrony_simulated_ci': self.get_average_pairwise_asynchrony(func=np.nanpercentile, q=[2.5, 97.5]),
        }


def generate_phase_correction_simulations_for_individual_conditions(
        mds: list[PhaseCorrectionModel], output_dir: str, logger=None, force_rebuild: bool = False,
        num_simulations: int = autils.NUM_SIMULATIONS
) -> list[Simulation]:
    """
    Create simulated performances using the coupling within every individual performance.
    """
    # Try and load the models from the disk to save time, unless we're forcing them to rebuild anyway
    if not force_rebuild:
        all_sims = autils.load_from_disc(output_dir, filename='phase_correction_sims_orig.p')
        # If we've successfully loaded models, return these straight away
        if all_sims is not None:
            return all_sims
    # Create an empty list to hold simulations in
    all_sims = []
    for pcm in mds:
        # Initialise the simulation class
        sim = Simulation(
            pcm=pcm, num_simulations=num_simulations, parameter='original', leader=None, use_original_noise=True
        )
        autils.log_simulation(sim, logger)
        # Create all simulations for this parameter/leader combination
        sim.create_all_simulations()
        # Append the simulation to our list
        all_sims.append(sim)
    # Pickle the result -- this will be quite large, depending on the number of simulations!
    pickle.dump(all_sims, open(f"{output_dir}\\phase_correction_sims_orig.p", "wb"))
    return all_sims


def generate_phase_correction_simulations_for_coupling_parameters(
        mds: list[PhaseCorrectionModel], output_dir: str, logger=None, force_rebuild: bool = False,
        num_simulations: int = autils.NUM_SIMULATIONS
) -> tuple[list[Simulation], str]:
    """
    Create simulated performances across a range of artificial coupling parameters for every phase correction model
    """
    def grouper(gr):
        return gr.groupby('instrument', as_index=False).agg(
            {
                'trial': 'mean', 'block': 'mean', 'latency': 'mean', 'jitter': 'mean', 'tempo_slope': 'mean',
                'ioi_std': 'mean', 'pw_asym': 'mean', 'zoom_arr': 'first', 'intercept': 'mean',
                'correction_partner': 'mean', 'correction_self': 'mean', 'resid_std': 'mean'
            }
        )

    # Try and load the models from the disk to save time, unless we're forcing them to rebuild anyway
    if not force_rebuild:
        all_sims = autils.load_from_disc(output_dir, filename='phase_correction_sims.p')
        # If we've successfully loaded models, return these straight away
        if all_sims is not None and isinstance(all_sims, list):
            if len(all_sims) != 0:
                return (
                    all_sims,
                    f'... skipping, simulations loaded from {output_dir}\\phase_correction_sims.p'
                )
    # Create the dataframe
    df = pd.concat(
        [pd.concat([pd.DataFrame([pcm.keys_dic]), pd.DataFrame([pcm.drms_dic])]) for pcm in mds]
    ).reset_index(drop=True)
    # Create an empty list to hold our simulations
    all_sims = []
    avg_noise = 0.005
    # Iterate through each condition
    for idx, grp in df.groupby(by=['latency', 'jitter']):
        # Iterate through each duo
        for i, g in grp.groupby('trial'):
            # Create the grouped model, averaging performance of each duo for one condition over both sessions
            pcm_o = grouper(g)
            # Create the simulation object
            sim = Simulation(
                pcm=pcm_o, num_simulations=num_simulations, parameter='original', leader=None, use_original_noise=False,
                noise=avg_noise
            )
            # Log the current duo and condition in our GUI, if we've passed a logger
            autils.log_simulation(sim, logger)
            # Create all simulations and append the simulation object to our list
            sim.create_all_simulations()
            all_sims.append(sim)
        # Create the grouped phase correction model, across all trials
        pcm_a = grouper(grp)
        # Set our trial metadata to 0 (helpful when logging)
        pcm_a['trial'] = 0
        # Create our anarchy model: both coupling coefficients set to 0
        anarchy_md = pcm_a.copy()
        anarchy_md['correction_partner'] = 0
        anarchy_md['intercept'] = 0
        # Create our democracy model: both coupling coefficients set to their means
        democracy_md = pcm_a.copy()
        democracy_md['correction_partner'] = democracy_md['correction_partner'].mean()
        democracy_md['intercept'] = 0
        # Create our leadership model: drums coupling set to 0, keys coupling set to mean
        leadership_md = pcm_a.copy()
        leadership_md['correction_partner'] = np.where(
            leadership_md['instrument'] == 'Drums', 0,
            leadership_md[leadership_md['instrument'] == 'Keys']['correction_partner']
        )
        leadership_md['intercept'] = 0
        # Iterate over all of our paradigm models
        for md, param in zip([anarchy_md, democracy_md, leadership_md],
                             ['anarchy', 'democracy', 'leadership']):
            # Create our simulation
            sim = Simulation(
                pcm=md, num_simulations=num_simulations, parameter=param, leader=None, use_original_noise=False,
                noise=avg_noise
            )
            # Log the current simulation in our GUI
            autils.log_simulation(sim, logger)
            # Create the simulation and append to our list
            sim.create_all_simulations()
            all_sims.append(sim)
    # Pickle the result -- this can be quite large, if we're creating lots of simulations!
    pickle.dump(all_sims, open(f"{output_dir}\\phase_correction_sims.p", "wb"))
    return all_sims, f'...simulations saved in {output_dir}\\phase_correction_sims.p'


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename="phase_correction_mds.p")
    # Default location to save output simulations
    output = r"C:\Python Projects\jazz-jitter-analysis\models"
    # Generate simulations using coupling parameters and pickle
    generate_phase_correction_simulations_for_coupling_parameters(
        mds=raw, output_dir=output, force_rebuild=True
    )
    # Generate simulations for each individual performance and pickle
    generate_phase_correction_simulations_for_individual_conditions(
        mds=raw, output_dir=output, force_rebuild=True
    )
