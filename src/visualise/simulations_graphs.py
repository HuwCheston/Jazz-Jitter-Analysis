import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


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
        Gets data from all simulation objects and plots individual and average simulations for both BPM and asynchrony
        variables.
        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for sim, col in zip(self.simulations, colors):
            # Calling this function will automatically create our simulations if we haven't done so yet
            ioi_individual, ioi_avg = sim.get_simulation_data_for_plotting(var='my_next_ioi')
            async_individual, async_avg = sim.get_simulation_data_for_plotting(var='asynchrony')
            # Plot individual simulations
            for ioi, async_ in zip(ioi_individual, async_individual):
                self.ax[0].plot(
                    ioi.index.seconds, (60 / ioi['my_next_ioi']).rolling(window='4s').mean(), alpha=0.01, color=col,
                )
                self.ax[1].plot(
                    async_.index.seconds, (async_['asynchrony']).rolling(window='4s').mean(), alpha=0.01, color=col,
                )
            # Plot average simulations
            self.ax[0].plot(
                ioi_avg.index.seconds, (60 / ioi_avg['my_next_ioi']).rolling(window='4s').mean(), alpha=1, linewidth=4,
                ls='-', color=col, label=f'{sim.parameter.title()} {sim.leader if sim.leader is not None else ""}'
            )
            self.ax[1].plot(
                async_avg.index.seconds, (async_avg['asynchrony']).rolling(window='4s').mean(), alpha=1, linewidth=4,
                ls='-', color=col, label=f'{sim.parameter.title()} {sim.leader if sim.leader is not None else ""}'
            )

    def _plot_original_performance(
            self
    ) -> None:
        """
        Wrangles data from original performances and plots against simulated data.
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
                label='Actual performance', linewidth=4, ls='--'
            )

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self._plot_all_simulations()
        if self.keys_orig is not None and self.drms_orig is not None:
            self._plot_original_performance()
        self._format_ax()
        self._format_fig()
        self.fig.suptitle(f"Duo {self.params['trial']}, block {self.params['block']}, "
                          f"latency {self.params['latency']}, jitter {self.params['jitter']}")
        fname = f"{self.output_dir}\\lineplot_all_parameters_{self.params['trial']}_" \
                f"{self.params['block']}_{self.params['latency']}_{self.params['jitter']}"
        return self.fig, fname

    def _get_min_max_x_val(
            self, func
    ) -> float:
        """
        Get the minimum and maximum seconds across all simulations and parameters for use in x axes.
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
        """
        Formats axes, setting ticks, labels etc.
        """
        # Calculate x limit
        xlim = (self._get_min_max_x_val(min), self._get_min_max_x_val(max))
        # Format top axes (BPM)
        ticks_ax0 = np.linspace(0, 200, 6)
        self.ax[0].set(xlabel='', ylim=(0, 200), xlim=xlim, yticks=ticks_ax0, yticklabels=ticks_ax0)
        self.ax[0].set_ylabel('Tempo (BPM)', fontsize='large')
        self.ax[0].axhline(y=120, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2)
        # Format bottom axes (async)
        ticks_ax1 = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        self.ax[1].set(yticks=ticks_ax1, yticklabels=ticks_ax1, ylim=(0.0001, 10), xlim=xlim)
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
        """
        Formats figure, setting title, legend etc.
        """
        self.fig.suptitle(f"Duo {self.params['trial']}, block {self.params['block']}, "
                          f"latency {self.params['latency']}, jitter {self.params['jitter']}")
        handles, labels = self.ax[0].get_legend_handles_labels()
        self.fig.legend(
            handles[:len(self.simulations) + 1], labels[:len(self.simulations) + 1], ncol=1,
            loc='right', title=None, frameon=False, fontsize='large', columnspacing=1, handletextpad=0.3
        )
        self.fig.subplots_adjust(bottom=0.07, top=0.93, left=0.09, right=0.78)


class BarPlotSimulationParameters(vutils.BasePlot):
    """
    Creates a plot showing the simulation results per parameter, designed to look similar to fig 2.(d)
    in Jacoby et al. (2021).
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.key: dict = {'Original': 0, 'Democracy': 1, 'Anarchy': 2, 'Leadership': 3}
        self.df = self._format_df(df=df)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(18.8, 5))

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Wrangles dataframe into form required for plotting.
        """
        # Fill our 'None' leader values with empty strings
        df['leader'] = df['leader'].fillna(value='').str.lower()
        # Format our parameter column by combining with leader column, replacing values with title case
        df['parameter'] = df[['parameter', 'leader', ]].astype(str).agg(''.join, axis=1)
        df['parameter'] = df['parameter'].replace({col: col.title() for col in df['parameter'].unique()})
        # Remove values from parameter column that we don't need
        df = df[(df['parameter'] != 'Leadershipkeys')]
        df = df[(df['parameter'] != 'Actual')]
        df = df[(df['original_noise'] == False)]
        df.loc[df['parameter'].str.contains('Leadership'), 'parameter'] = 'Leadership'
        return df

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_simulation_by_parameter'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the stripplot and barplot for both variables, then carries out t-test and adds in asterisks
        """
        # Iterate through both variables which we wish to plot
        for num, var in enumerate(['tempo_slope_simulated', 'ioi_variability_simulated', 'asynchrony_simulated']):
            # Add the scatter plot, showing each individual performance
            sns.stripplot(
                data=self.df, ax=self.ax[num], x='parameter', y=var, dodge=True, s=3, marker='.', jitter=0.1,
                color=vutils.BLACK,
            )
            # Add the bar plot, showing the median values
            sns.barplot(
                data=self.df, x='parameter', y=var, ax=self.ax[num], errorbar=None, facecolor='#6fcbdc',
                estimator=np.mean, edgecolor=vutils.BLACK,
            )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes objects, setting ticks, labels etc.
        """
        # Apply formatting to tempo slope ax
        self.ax[0].set(ylabel='Tempo slope (BPM/s)', xlabel='', ylim=(-0.75, 0.75))
        self.ax[0].axhline(y=0, linestyle='-', color=vutils.BLACK, linewidth=2)
        self.ax[1].set(ylabel='IOI variability (SD, ms)', xlabel='')
        # Apply formatting to async ax
        t = [1, 10, 100, 1000, 10000]
        self.ax[2].set_yscale('log')
        self.ax[2].set(xlabel='', ylim=(1, 10000), yticks=t, yticklabels=t)
        self.ax[2].set_ylabel('Asynchrony (RMS, ms)', labelpad=-5)
        # Apply joint formatting to both axes
        for ax in self.ax:
            # Adjust width of each bar on the barplot
            for patch in ax.patches:
                cw = patch.get_width()
                patch.set_width(0.35)
                patch.set_x(patch.get_x() + (cw - 0.35) * 0.5)
            # Adjust the width of the major and minor ticks and ax border
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.tick_params(width=3, which='major')
            ax.tick_params(width=0, which='minor')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Add a label to the x axis
        self.fig.supxlabel('Simulated coupling parameter')
        # Adjust subplots positioning a bit to fit in the legend we've just created
        self.fig.subplots_adjust(bottom=0.35, top=0.95, left=0.07, right=0.99, wspace=0.27)


class RegPlotSlopeComparisons(vutils.BasePlot):
    """
    Creates a regression jointplot, designed to show similarity in tempo slope between
    actual and simulated performances (with original coupling patterns).
    """
    def __init__(
            self, df: pd.DataFrame, **kwargs
    ):
        super().__init__(**kwargs)
        # Define variables
        self.var = kwargs.get('var', 'tempo_slope')
        self.orig_var, self.sim_var = self.var + '_original', self.var + '_simulated'
        self.df = self._format_df(df)

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts results data from each Simulation class instance and subsets to get original coupling simulations only.
        Results data is initialised when simulations are created, which makes this really fast.
        """
        return df[(df['parameter'] == 'original') & (df['original_noise'] == False)]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self.g = self._create_plot()
        self._add_correlation_results()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\regplot_simulation_slope_comparison'
        return self.g.fig, fname

    def _create_plot(
            self
    ) -> sns.JointGrid:
        """
        Creates scatter plots for both simulated and actual tempo slope values
        """
        # Create the grid, but don't map any plots onto it just yet
        g = sns.JointGrid(
            data=self.df, x=self.sim_var, y=self.orig_var, xlim=(-0.75, 0.75), ylim=(-0.75, 0.75),
            hue='trial', palette=vutils.DUO_CMAP, height=9.4,
        )
        # Map a scatter plot onto the main ax
        sns.scatterplot(
            data=self.df, x=self.sim_var, y=self.orig_var, s=100, hue='trial', palette=vutils.DUO_CMAP,
            edgecolor=vutils.BLACK, style='trial', ax=g.ax_joint
        )
        # Map a regression plot onto the main ax
        sns.regplot(
            data=self.df, x=self.sim_var, y=self.orig_var, scatter=False, color=vutils.BLACK, n_boot=1000,
            ax=g.ax_joint, truncate=False, line_kws={'linewidth': 3}
        )
        # Map KDE plots onto the marginal ax
        g.plot_marginals(
            sns.kdeplot, lw=2, multiple='layer', fill=False, common_grid=True, cut=0, common_norm=True
        )
        return g

    def _add_correlation_results(
            self
    ) -> None:
        """
        Adds the results of a linear correlation onto the plot
        """
        # Calculate the correlation, get r and p values
        r, p = stats.pearsonr(self.df[self.sim_var], self.df[self.orig_var])
        # Format correlation results into a string
        s = f'$r$ = {round(r, 2)}{vutils.get_significance_asterisks(p)}'
        # Add the annotation onto the plot
        self.g.ax_joint.annotate(
            s, (0.05, 0.925), xycoords='axes fraction', fontsize=vutils.FONTSIZE + 3,
            bbox=dict(facecolor='none', edgecolor=vutils.BLACK, pad=10.0)
        )

    def _format_ax(
            self, marg_line: bool = False
    ) -> None:
        """
        Formats axes objects, setting ticks, labels etc.
        """
        # # Get the axes limit from minimum and maximum values across both simulated and original data
        self.g.ax_joint.set(xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), xlabel='', ylabel='')
        # Set the top and right spines of the joint plot to visible
        self.g.ax_joint.spines['top'].set_visible(True)
        self.g.ax_joint.spines['right'].set_visible(True)
        # Add the diagonal line to the joint ax
        self.g.ax_joint.axline(
            xy1=(0, 0), xy2=(1, 1), linewidth=3, transform=self.g.ax_joint.transAxes,
            color=vutils.BLACK, alpha=vutils.ALPHA, ls='--'
        )
        # Add lines to marginal plot at 0 tempo slope, if required
        if marg_line:
            self.g.ax_marg_x.axvline(x=0, linewidth=3, color=vutils.BLACK, alpha=vutils.ALPHA, ls='--')
            self.g.ax_marg_y.axhline(y=0, linewidth=3, color=vutils.BLACK, alpha=vutils.ALPHA, ls='--')
        # Adjust the width of the major and minor ticks and ax border
        for ax in [self.g.ax_joint, self.g.ax_marg_y, self.g.ax_marg_x]:
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Add the axis labels in
        self.g.fig.supxlabel('Simulated tempo slope (BPM/s)', y=0.03, x=0.5)
        self.g.fig.supylabel('Actual tempo slope (BPM/s)', x=0.02, y=0.5)
        # Store our handles and labels
        hand, lab = self.g.ax_joint.get_legend_handles_labels()
        # Remove the legend
        self.g.ax_joint.get_legend().remove()
        # Add the legend back in
        lgnd = self.g.ax_joint.legend(
            hand, lab, title='Duo', frameon=True, ncol=1, loc='lower right', markerscale=1.5,
            fontsize=vutils.FONTSIZE + 3, edgecolor=vutils.BLACK,
        )
        # Set the legend font size
        plt.setp(lgnd.get_title(), fontsize=vutils.FONTSIZE + 3)
        # Set the legend marker size and edge color
        for handle in lgnd.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([100])
        # Adjust subplots positioning a bit to fit in the legend we've just created
        self.g.fig.subplots_adjust(bottom=0.11, top=0.96, left=0.13, right=0.98,)


def generate_simulations_plots(
    sims: list, output_dir: str,
) -> None:
    df = pd.DataFrame([sim.results_dic for sim in sims])
    figures_output_dir = output_dir + '\\figures\\simulations_plots'
    rp = RegPlotSlopeComparisons(df, output_dir=figures_output_dir)
    rp.create_plot()
    bp = BarPlotSimulationParameters(df, output_dir=figures_output_dir)
    bp.create_plot()


if __name__ == '__main__':
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", "phase_correction_sims.p")
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    generate_simulations_plots(raw, output)
