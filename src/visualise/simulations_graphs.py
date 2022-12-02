import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import stats

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
                label='Actual', linewidth=4
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
                f"{self.params['block']}_{self.params['latency']}_{self.params['jitter']}.png"
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
        # Calculate xlim
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key: dict = {'Original': 0, 'Democracy': 1, 'Anarchy': 2, 'Leadership': 3}
        self.df = self._format_df(df=self.df)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(18.8, 5))
        self.ax[1].set_yscale('log')

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
        df['parameter'] = df['parameter'].str.replace('Leadershipdrums', 'Leadership')
        return df

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self._create_plot()
        self._add_parameter_graphics()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_simulation_by_parameter.png'
        return self.fig, fname

    def _ttest_group_against_original(
            self, var: str
    ) -> pd.DataFrame:
        """
        Calculates the result of a independent sample t-test for each parameter/trial combo to original coupling results
        """
        ttest = []
        # Iterate through every trial in our dataframe
        for idx, grp in self.df.groupby('trial'):
            # Get the original coupling parameter -- we compare the other parameters to this
            a = grp[grp['parameter'] == 'Original'][var].values
            # Iterate through all parameters in our group
            for param, small_g in grp.groupby('parameter'):
                # Get the target parameter
                b = small_g[var].values
                # Carry out the t-test and store the p-value
                _, p = stats.ttest_ind(a, b)
                # Convert our p-values into asterisks and append to the list
                ttest.append((idx, param, vutils.get_significance_asterisks(p)))
        # Return our t test results as a dataframe, in the correct order to enable us to plot them properly
        return (
            pd.DataFrame(ttest, columns=['trial', 'parameter', 'sig'])
            .sort_values(by=['trial', 'parameter'], key=lambda x: x.map(self.key))
            .reset_index(drop=True)
        )

    def _create_plot(
            self
    ) -> None:
        """
        Creates the stripplot and barplot for both variables, then carries out t-test and adds in asterisks
        """
        # Iterate through both variables which we wish to plot
        for num, var in enumerate(['tempo_slope', 'asynchrony']):
            # Add the scatter plot, showing each individual performance
            sns.stripplot(
                data=self.df, ax=self.ax[num], x='parameter', y=var, hue='trial', dodge=True, s=4, marker='.',
                jitter=1, color=vutils.BLACK,
            )
            # Add the bar plot, showing the median values
            sns.barplot(
                data=self.df, x='parameter', y=var, hue='trial', ax=self.ax[num], ci=25, errcolor=vutils.BLACK,
                errwidth=2, estimator=np.median, edgecolor=vutils.BLACK, capsize=0.03,
            )
            # Get the t test results, comparing all parameters across all trials for this variable
            ttest = self._ttest_group_against_original(var)
            # Apply our formatted t test asterisks to the bar plot in the correct places
            for (i, trial), con in zip(ttest.groupby('trial'), self.ax[num].containers):
                self.ax[num].bar_label(con, labels=trial['sig'].to_list(), padding=20 if num == 0 else 30, fontsize=12)

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes objects, setting ticks, labels etc.
        """
        # Apply formatting to tempo slope ax
        self.ax[0].set(ylabel='Tempo slope (BPM/s)', xlabel='', ylim=(-1, 1))
        self.ax[0].axhline(y=0, linestyle='--', alpha=vutils.ALPHA, color=vutils.BLACK, linewidth=2)
        # Apply formatting to async ax
        t = [1, 10, 100, 1000, 10000]
        self.ax[1].set(ylabel='RMS of asynchrony (ms)', xlabel='', ylim=(1, 10000), yticks=t, yticklabels=t)
        # Apply joint formatting to both axes
        for ax in self.ax:
            # Set the label, with padding so it doesn't get in the way of our parameter graphics
            ax.set_xlabel('Parameter', y=0.5, labelpad=35)
            # Adjust the width of the major and minor ticks and ax border
            ax.tick_params(width=3, which='major')
            ax.tick_params(width=0, which='minor')
            plt.setp(ax.spines.values(), linewidth=2)

    def _add_parameter_graphics(
            self, ypad: float = -0.03
    ) -> None:
        """
        Adds graphics below each parameter showing the simulated performer couplings.
        """
        # Iterate through both axes on the figure
        # We use an axes transform when placing the graphics, so we don't have to set values individually
        for ax in self.ax:
            # Define initial x starting point
            x = 0.305
            for i in range(1, 4):
                # Add silver, transparent background
                ax.add_patch(
                    plt.Rectangle((x, -0.18 + ypad), width=0.14, height=0.1, facecolor='silver', clip_on=False,
                                  linewidth=0, transform=ax.transAxes, alpha=vutils.ALPHA)
                )
                # Add coloured rectangles for either keys/drums
                col_rect = lambda pad, num: plt.Rectangle(
                    (x + pad, -0.17 + ypad), width=0.03, height=0.08, clip_on=False, linewidth=0,
                    transform=ax.transAxes, facecolor=vutils.INSTR_CMAP[num]
                )
                ax.add_patch(col_rect(0.005, 0))
                ax.add_patch(col_rect(0.105, 1))
                # Add text to coloured rectangles
                ax.text(s='K', x=x + 0.007, y=-0.155 + ypad, transform=ax.transAxes, fontsize=18)
                ax.text(s='D', x=x + 0.107, y=-0.155 + ypad, transform=ax.transAxes, fontsize=18)
                # Add arrows according to parameter number
                arw = lambda padx, dx, y, num: ax.arrow(
                    x=x + padx, y=y + ypad, dx=dx, dy=0, clip_on=False, transform=ax.transAxes, linewidth=5,
                    color=vutils.INSTR_CMAP[num], head_width=0.015, head_length=0.005
                )
                if i == 1:  # For democracy, we need two arrows
                    arw(0.045, 0.04, -0.11, 0)
                    arw(0.09, -0.04, -0.15, 1)
                elif i == 3:    # For leadership, we just need one
                    arw(0.045, 0.04, -0.11, 0)
                # Increase x value for next patch
                x += 0.24

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Empty variables to hold our legend handles and labels
        hand, lab = None, None
        # Iterate through both our axes
        for ax in self.ax:
            # Store our handles and labels
            hand, lab = ax.get_legend_handles_labels()
            # Remove the legend
            ax.get_legend().remove()
        # Add the legend back in, but only keep the values which relate to the bar plot (also add a title, adjust place)
        self.fig.legend(hand[5:], lab[5:], title='Duo', frameon=False, ncol=1, loc='center right')


class RegPlotSimulationComparisons(vutils.BasePlot):
    """
    Creates two regression scatter plots, designed to show similarity in tempo slope and pairwise asynchrony between
    actual and simulated performances (with original coupling patterns).
    """
    def __init__(
            self, sim_list: list, **kwargs
    ):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(18.8, 9))
        self.df = self._format_df(sim_list)

    @staticmethod
    def _format_df(
            sim_list: list
    ) -> pd.DataFrame:
        """
        Extracts results data from each Simulation class instance and subsets to get original coupling simulations only.
        Results data is initialised when simulations are created, which makes this really fast.
        """
        df = pd.DataFrame([sim.results_dic for sim in sim_list])
        return df[df['parameter'] == 'original']

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
        fname = f'{self.output_dir}\\regplot_simulations_comparison.png'
        return self.fig, fname

    def _get_ax_lim(
            self, orig_var: str, sim_var: str, multiplier: float = 1.1
    ) -> tuple[float, float]:
        """
        Extracts minimum and maximum values from simulated and original datasets then multiplies by a constant
        """
        mi = min([self.df[orig_var].min(), self.df[sim_var].min()])
        ma = max([self.df[orig_var].max(), self.df[sim_var].max()])
        return mi / multiplier, ma * multiplier

    def _create_plot(
            self
    ) -> None:
        """
        Creates scatter plots for both simulated and actual tempo slope/async values
        """
        # Define titles for subplots
        titles = ['Tempo slope (BPM/s)', 'RMS of asynchrony (ms)']
        for num, (var, title) in enumerate(zip(['tempo_slope', 'asynchrony'], titles)):
            # Define variables
            orig_var, sim_var = var + '_original', var + '_simulated'
            # Create the plot
            g = sns.scatterplot(
                data=self.df, x=sim_var, y=orig_var, ax=self.ax[num], hue='trial', palette='tab10', s=70, style='trial'
            )
            # Get the axes limit from minimum and maximum values across both simulated and original data
            lim = self._get_ax_lim(sim_var, orig_var, multiplier=1.1)
            # Set the axes parameters
            if num == 0:
                g.set(xlim=lim, ylim=lim, title=title)
            # For the asynchrony plot, minimum axes value must be 0ms
            else:
                g.set(xlim=(0, lim[1]), ylim=(0, lim[1]), title=title)

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes objects, setting ticks, labels etc.
        """
        for ax in self.ax:
            # Set axes labels
            ax.set(xlabel='Simulated performance', ylabel='Actual performance', )
            # Add the diagonal line
            ax.axline(xy1=(0, 0), xy2=(1, 1), linewidth=3, transform=ax.transAxes,
                      color=vutils.BLACK, alpha=vutils.ALPHA, ls='--')
            # Adjust the width of the major and minor ticks and ax border
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Empty variables to hold our legend handles and labels
        hand, lab = None, None
        # Iterate through both our axes
        for ax in self.ax:
            # Store our handles and labels
            hand, lab = ax.get_legend_handles_labels()
            # Remove the legend
            ax.get_legend().remove()
        # Add the legend back in
        lgnd = self.fig.legend(hand, lab, title='Duo', frameon=False, ncol=1, loc='center right')
        # Set the legend marker size
        for handle in lgnd.legendHandles:
            handle.set_sizes([70])
        # Adjust subplots positioning a bit to fit in the legend we've just created
        self.fig.subplots_adjust(bottom=0.1, top=0.9, left=0.07, right=0.93, wspace=0.15)


def generate_simulations_plots(

) -> None:
    pass


if __name__ == '__main__':
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", "phase_correction_sims.p")
    pass
