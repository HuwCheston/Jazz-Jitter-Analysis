import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import stats

import src.visualise.visualise_utils as vutils
from src.analyse.simulations_ratio import Simulation


class LinePlotAllParameters(vutils.BasePlot):
    """
    For a single performance, creates two line plots showing tempo and asynchrony change for all simulated parameters
    (anarchy, democracy etc.)
    """
    def __init__(
            self, simulations: list, **kwargs
    ):
        super().__init__(**kwargs)
        # Our list of Simulation classes: we need not have created the simulations yet when passing them in
        self.simulations: list = simulations
        # Our original performances
        self.keys_orig, self.drms_orig = kwargs.get('keys_orig', None), kwargs.get('drms_orig', None)
        self.params = kwargs.get('params', None)
        self.fig, self.ax = plt.subplots(2, 1, sharex=True, figsize=(18.8, 15))
        self.ax[1].set_yscale('log')

    def _plot_all_simulations(
            self
    ) -> None:
        """

        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for sim, col in zip(self.simulations, colors):
            if len(sim.keys_simulations) == 0 or len(sim.drms_simulations) == 0:
                sim.create_all_simulations()
            sim.plot_simulation(color=col, ax=self.ax[0], var='my_next_ioi')
            sim.plot_simulation(color=col, ax=self.ax[1], var='asynchrony', bpm=False)

    def _plot_original_performance(
            self
    ) -> None:
        """

        """
        # Resample our two original performance dataframes
        resampled = [vutils.resample(data) for data in [self.keys_orig, self.drms_orig]]
        # Divide ioi by 60 to get bpm
        for d in resampled:
            d['my_next_ioi'] = 60 / d['my_next_ioi']
        for num, s in enumerate(['my_next_ioi', 'asynchrony']):
            # Concatenate the resampled dataframes together and get the row-wise mean
            conc = pd.DataFrame(
                pd.concat([df[s] for df in resampled], axis=1).abs().mean(axis=1), columns=[s]
            )
            # Get the elapsed time column as an integer
            conc['my_onset'] = conc.index.total_seconds()
            # Plot onto the required axis
            self.ax[num].plot(
                conc['my_onset'], conc[s].rolling(window='4s').mean(), alpha=1, color=vutils.BLACK,
                label='Actual performance', linewidth=4
            )

    @vutils.plot_decorator
    def create_plot(self):
        """

        """
        self._plot_all_simulations()
        if self.keys_orig is not None and self.drms_orig is not None:
            self._plot_original_performance()
        self._format_ax()
        self._format_fig()
        self.fig.suptitle(f"Duo {self.params['trial']}, block {self.params['block']}, "
                          f"latency {self.params['latency']}, jitter {self.params['jitter']}")
        fname = f"{self.output_dir}\\lineplot_all_parameters_{self.params['trial']}_" \
                f"{self.params['block']}_{self.params['latency']}_{self.params['jitter']}.png"
        return self.fig, fname

    def _get_min_max_x_val(
            self, func
    ) -> float:
        """

        """
        res = []
        # Iterate through each simulation object
        for s in self.simulations:
            # Iterate through each simulation
            for k, d in zip(s.keys_simulations, s.drms_simulations):
                # Apply our function to keys, drums individually, then together
                res.append(func([func(k.index.seconds), func(d.index.seconds)]))
        # Apply function to the results
        return func(res)

    def _format_ax(self):
        # Format top axes (BPM)
        self.ax[0].set(xlabel='', ylim=(0, 200), xlim=(self._get_min_max_x_val(min), self._get_min_max_x_val(max)))
        self.ax[0].set_ylabel('Tempo (BPM)', fontsize='large')
        self.ax[0].axhline(y=120, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2)
        # Format bottom axes (async)
        ticks = [0.0025, 0.025, 0.25, 2.5, 25]
        self.ax[1].set(yticks=ticks, yticklabels=ticks, ylim=(0.0025, 25))
        self.ax[1].set_ylabel('Asynchrony (s)', fontsize='large')
        self.ax[1].set_xlabel('Performance duration (s)', fontsize='large')
        self.ax[1].axhline(
            y=self.params['latency'] / 1000, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2
        )
        # Iterate through to set tick and axes line widths
        for num, ax in enumerate(self.ax.flatten()):
            ax.tick_params(width=3, which='both', labelsize=20)
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.suptitle(f"Duo {self.params['trial']}, block {self.params['block']}, "
                          f"latency {self.params['latency']}, jitter {self.params['jitter']}")
        handles, labels = self.ax[0].get_legend_handles_labels()
        self.fig.legend(
            handles[:len(self.simulations) + 1], labels[:len(self.simulations) + 1], ncol=len(self.simulations) + 1,
            loc='lower center', title=None, frameon=False, fontsize='large', columnspacing=1, handletextpad=0.3
        )
        self.fig.subplots_adjust(bottom=0.09, top=0.95, left=0.09, right=0.91)


class BarPlotSimulationParameters(vutils.BasePlot):
    """
    Creates a plot showing the simulation results per parameter, designed to look similar to fig 2.(d)
    in Jacoby et al. (2021).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.res = kwargs.get('res', None)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(18.8, 7))

    @staticmethod
    def _normalise_values(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalise values around original parameter = 1
        """
        df['norm'] = 1 - df['original']
        for s in [col for col in df.columns.to_list() if col != 'norm' or 'original']:
            df[s] += df['norm']
        df['original'] = 1
        return df

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Format model results for plotting by pivoting, normalising original values, then melting into one column
        """
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
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_simulation_by_parameter.png'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the barplot in seaborn
        """
        return sns.barplot(
            data=self.df, x='type', y='value', color=vutils.LINE_CMAP[0], ax=self.ax, errcolor=vutils.BLACK,
            errwidth=2, edgecolor=vutils.BLACK, lw=2, capsize=0.03
        )

    def _add_parameter_graphic(
            self
    ) -> None:
        """
        Adds graphics below each parameter in the barplot, showing the directionality of the coupling
        """
        # Define initial x starting point
        x = 0.26
        for i in range(1, 5):
            # Add silver, transparent background
            self.g.add_patch(
                plt.Rectangle((x, -0.26), width=0.1, height=0.12, facecolor='silver', clip_on=False, linewidth=0,
                              transform=self.ax.transAxes, alpha=vutils.ALPHA)
            )
            # Add coloured rectangles for either keys/drums
            self.g.add_patch(
                plt.Rectangle((x + 0.005, -0.24), width=0.023, height=0.08, clip_on=False, linewidth=0,
                              transform=self.ax.transAxes, facecolor=vutils.INSTR_CMAP[0])
            )
            self.g.add_patch(
                plt.Rectangle((x + 0.07, -0.24), width=0.023, height=0.08, clip_on=False, linewidth=0,
                              transform=self.ax.transAxes, facecolor=vutils.INSTR_CMAP[1])
            )
            # Add text to coloured rectangles
            self.g.text(s='K', x=x + 0.007, y=-0.225, transform=self.ax.transAxes, fontsize=24)
            self.g.text(s='D', x=x + 0.072, y=-0.225, transform=self.ax.transAxes, fontsize=24)
            # Add arrows according to parameter number
            if i == 2 or i == 3:
                self.ax.arrow(x=x + 0.033, y=-0.18, dx=0.025, dy=0, clip_on=False, transform=self.ax.transAxes,
                              linewidth=5, color=vutils.INSTR_CMAP[0], head_width=0.015, head_length=0.005)
            if i == 1 or i == 3:
                self.ax.arrow(x=x + 0.063, y=-0.22, dx=-0.025, dy=0, clip_on=False, transform=self.ax.transAxes,
                              linewidth=5, color=vutils.INSTR_CMAP[1], head_width=0.015, head_length=0.005)
            # Increase x statistic for next patch
            x += 0.19

    @staticmethod
    def _get_significance_asterisks(
            p: float, t: float = None
    ) -> str:
        """
        Converts a raw p-value into asterisks, showing significance boundaries. Can also provide raw test statistics
        by setting optional t argument
        """
        # Define function to combine raw test statistic with asterisks
        fn = lambda a: str(round(t, 2)) + a if t is not None else a
        if p < 0.001:
            return fn('***')
        elif p < 0.01:
            return fn('**')
        elif p < 0.05:
            return fn('*')
        else:
            return fn('')

    def _add_significance_asterisks_to_plot(
            self
    ) -> None:
        """
        Adds asterisks and arrows to each column in the plot showing the significance of an independent sample,
        two-tailed t-test compared to the original simulation results
        """
        # Pivot our dataframe in order to more easily conduct the t-test
        df = self.df.pivot_table(values='value', index='trial', columns='type')
        # Zip each bar on the plot (other than the first) together with its column label in the dataframe
        z = zip(self.ax.patches[1:], ['Leadership - Keys', 'Leadership - Drums', 'Democracy', 'Anarchy'])
        # Gets the positioning of the 'original' bar patch
        orig_x = self.ax.patches[0].xy[0] + (self.ax.patches[0].get_width() / 2)
        # Iterate through all of our patches/column labels
        for n, (p, s) in enumerate(z):
            # Conduct the t-test and get the p-value
            r, sig = stats.ttest_ind(df['Original'], df[s], nan_policy='omit')
            # Get the position to stretch the arrow out until
            x = p.xy[0] + p.get_width() - (self.ax.patches[0].get_width() / 2)
            # Add the significance asterisk text
            self.ax.text(s=self._get_significance_asterisks(sig), x=x / 2, y=1.27 + n / 14, fontsize=24)
            # Add the arrow connecting the original bar to the column bar
            self.ax.arrow(x=orig_x, y=1.29 + n / 14, dx=x, dy=0, )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes object
        """
        # Add a horizontal line at y=1
        self.ax.axhline(y=1, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2)
        # Set axes limits, titles, and replace '-' with a line break in x tick labels
        self.g.set(xlabel='', ylabel='', ylim=(0, 1.6),
                   xticklabels=[col.replace(' - ', '\n') for col in self.df['type'].unique()])
        # Set axes and tick width
        self.ax.tick_params(width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Set the width of the bars in the plot - must do this before adding asterisks/parameter graphics!
        new_value = 0.3
        for patch in self.ax.patches:
            diff = patch.get_width() - new_value
            patch.set_width(new_value)
            patch.set_x(patch.get_x() + diff * .5)
        # Add significance asterisks and parameter graphics
        self._add_significance_asterisks_to_plot()
        self._add_parameter_graphic()

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object
        """
        # Set figure-wise axes labels
        self.fig.supylabel('Normalised tempo slope\n(BPM/s, 1 = original slope)', y=0.6, x=0.01)
        self.fig.supxlabel('Parameter', y=0.03)
        # Adjust plot spacing a bit
        self.fig.subplots_adjust(bottom=0.27, top=0.95, left=0.08, right=0.97)
