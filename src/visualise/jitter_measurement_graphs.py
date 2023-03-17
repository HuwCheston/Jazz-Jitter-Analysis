import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Generator

import src.visualise.visualise_utils as vutils


class JitterMeasurementAnalysis:
    """
    Measures inter-beat latency for recordings in Librosa
    """

    def __init__(self, directory: str, **kwargs):
        self.input_dir = directory
        self.last_timestamp = kwargs.get('last_timestamp', 90)
        self.file_ext = kwargs.get('file_ext', ('.mp3', '.wav'))
        self.sr = kwargs.get('sr', 44100)
        self.files = self._get_files()
        self.audio = self._load_audio()
        self.onsets = [self._extract_onsets(y) for y in self.audio]
        self.echoes = [self._extract_iois(y) for y in self.onsets]
        self.arrs = list(self._match_latency_onset_and_ioi())

    def _get_files(
            self
    ) -> list[str]:
        """
        Returns strings of valid filepaths
        """
        return [os.sep.join([self.input_dir, file]) for file in os.listdir(self.input_dir)]

    def _load_audio(
            self
    ) -> list[np.array]:
        """
        Loads audio into librosa
        """
        return [librosa.load(file, sr=self.sr)[0] for file in self.files]

    def _extract_onsets(
            self, y: np.ndarray
    ) -> np.array:
        """
        Extracts onsets from audio file in Librosa
        """
        ons_detec = librosa.onset.onset_detect(y=y, sr=self.sr, units='time')
        return np.array([ons for ons in ons_detec])

    @staticmethod
    def _get_minimum(
            list_of_arrays: list[np.array]
    ) -> float:
        """
        Gets the minimum shape of multiple input arrays, used when subsetting
        """
        return min([a.shape[0] for a in list_of_arrays])

    def _extract_iois(
            self, onset_times: np.ndarray
    ) -> np.array:
        """
        Extracts inter-onset intervals between real and delayed metronome clicks
        """
        times_even = onset_times[::2]
        times_odd = onset_times[1::2]
        mi = self._get_minimum([times_even, times_odd])
        return times_odd[:mi] - times_even[:mi]

    def _match_latency_onset_and_ioi(
            self
    ) -> list[pd.DataFrame]:
        """
        Matches each inter-onset interval together with the initial onset time
        """
        for onset, ioi, in zip(self.onsets, self.echoes,):
            onset = np.array([ons - np.min(onset) for ons in onset[::2]])
            mi = self._get_minimum([onset, ioi])
            yield pd.DataFrame(np.column_stack((onset[:mi], ioi[:mi])), columns=['onset', 'ioi'])


class LinePlotJitterMeasurement(vutils.BasePlot):
    """
    Creates lineplots (with marginal histrograms) for multiple recordings
    """

    def __init__(self, arrays, **kwargs):
        super().__init__(**kwargs)
        self.last_timestamp = kwargs.get('last_timestamp', 90)
        self.titles = kwargs.get('titles', ['Original', '1', '2', '3'])
        self.arrays = list(self._format_array(arrays))
        self.fig, self.ax = plt.subplots(
            ncols=len(self.arrays) * 3, sharey=True, figsize=(18.8, 7),
            gridspec_kw=dict(width_ratios=[3, 1, 0.1] * len(self.arrays))
        )
        self.main_ax = self.ax.flatten()[::3]
        self.marginal_ax = self.ax.flatten()[1::3]
        self.placeholder_ax = self.ax.flatten()[2::3]
        for p_ax in self.placeholder_ax:
            p_ax.axis('off')

    def _format_array(
            self, arrs: list[np.array]
    ) -> Generator:
        """
        Formats array returned from each recording
        """
        for arr in arrs:
            arr = arr.copy(deep=True)
            arr = arr[arr['onset'] <= self.last_timestamp]
            arr['ioi'] = (arr['ioi'] - arr['ioi'].min()) * 1000
            yield arr

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class and generates the plot, then saves in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\lineplot_latency_tests'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the line plot and histogram for each array
        """
        for arr, ax, m_ax in zip(self.arrays, self.main_ax, self.marginal_ax):
            sns.lineplot(data=arr, x='onset', y='ioi', errorbar=None, lw=2, color=vutils.BLACK, ax=ax)
            sns.histplot(data=arr, y='ioi', kde=True, bins=10, color=vutils.BLACK, line_kws={'lw': 3}, ax=m_ax, )

    def _format_marginal_ax(
            self
    ) -> None:
        """
        Formats the marginal histograms
        """
        for m_ax in self.marginal_ax:
            m_ax.set_xlim([0, 100])
            m_ax.spines['bottom'].set_visible(False)
            m_ax.spines['right'].set_visible(False)
            m_ax.xaxis.tick_top()
            m_ax.xaxis.set_label_position('top')
        self.marginal_ax[0].set_xlabel('Density', y=10, fontsize=vutils.FONTSIZE + 3)

    def _format_main_ax(
            self
    ) -> None:
        """
        Formats the main line plots
        """
        for ax, tit in zip(self.main_ax, self.titles):
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.linspace(0, 90, 4))
            ax.set_title(tit, x=0., y=1.15, fontsize=vutils.FONTSIZE + 5, fontweight='bold')
        self.main_ax[0].set_xlabel('Time (s)', fontsize=vutils.FONTSIZE + 3)

    def _format_ax(
            self
    ) -> None:
        """
        Formats all axis
        """
        for ax, in zip(self.ax.flatten()):
            ax.set(ylabel='', xlabel='', yticks=np.linspace(0, 150, 4))
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
        self._format_main_ax()
        self._format_marginal_ax()

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level attributes
        """
        self.fig.supylabel('Normalized latency (ms)', fontsize=vutils.FONTSIZE + 3)
        self.fig.subplots_adjust(left=0.065, bottom=0.125, right=0.975, top=0.8, wspace=0.15)


def generate_latency_measurement_plots(
        input_dir: str, output_dir: str, **kwargs
) -> None:
    jm = JitterMeasurementAnalysis(input_dir, **kwargs)
    lp = LinePlotJitterMeasurement(arrays=jm.arrs, output_dir=output_dir, **kwargs)
    lp.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    input_ = r"C:\Python Projects\jazz-jitter-analysis\data\raw\jitter_measurement_bounces"
    # Default location to save plots
    output_ = r"C:\Python Projects\jazz-jitter-analysis\reports\figures\misc_plots"
    # Create the plots
    generate_latency_measurement_plots(input_, output_, sr=192000)
