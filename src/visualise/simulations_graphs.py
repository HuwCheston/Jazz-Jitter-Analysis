"""Code for generating plots from the simulation objects"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils

from src.analyse.simulations import Simulation

# Define the objects we can import from this file into others
__all__ = [
    'generate_plots_for_simulations_with_coupling_parameters',
    'generate_plots_for_individual_performance_simulations'
]


class LinePlotAllParameters(vutils.BasePlot):
    """
    For a single performance, creates two line plots showing tempo and asynchrony change for all simulated parameters
    (anarchy, democracy etc.)
    """
    def __init__(
            self, simulations: list[Simulation], **kwargs
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
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(18.8, 5))

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Wrangles dataframe into form required for plotting.
        """
        # Fill our 'None' leader values with empty strings
        # df['leader'] = df['leader'].fillna(value='').str.lower()
        # Format our parameter column by combining with leader column, replacing values with title case
        # df['parameter'] = df[['parameter', 'leader', ]].astype(str).agg(''.join, axis=1)
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
        for num, var in enumerate(['tempo_slope_simulated', 'asynchrony_simulated']):
            # Add the scatter plot, showing each individual performance
            sns.stripplot(
                data=self.df, ax=self.ax[num], x='parameter', y=var, dodge=True, s=5, marker='.', jitter=0.1,
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
        self.ax[0].set_ylabel('Tempo slope (BPM/s)', fontsize=vutils.FONTSIZE + 3)
        self.ax[0].set(xlabel='', ylim=(-0.75, 0.75))
        self.ax[0].axhline(y=0, linestyle='-', color=vutils.BLACK, linewidth=2)
        # Apply formatting to async ax
        t = [1, 10, 100, 1000, 10000]
        self.ax[1].set_yscale('log')
        self.ax[1].set(xlabel='', ylim=(1, 10000), yticks=t, yticklabels=t)
        self.ax[1].set_ylabel('Asynchrony (RMS, ms)', fontsize=vutils.FONTSIZE + 3, labelpad=-2)
        # Apply joint formatting to both axes
        for ax in self.ax:
            # Adjust width of each bar on the bar plot
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
        self.fig.supxlabel('Coupling paradigm')
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
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)
        self.error_bar: str = kwargs.get('error_bar', 'sd')
        self.percentiles: tuple[float] = kwargs.get('percentile', (2.5, 97.5))
        self.original_noise = kwargs.get('original_noise', False)
        self.orig_var, self.sim_var = self.var + '_original', self.var + '_simulated'
        self.df = self._format_df(df)

    def _format_df(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts results data from each Simulation class instance and subsets to get original coupling simulations only.
        Results data is initialised when simulations are created, which makes this really fast.
        """
        return df[(df['parameter'] == 'original') & (df['original_noise'] == self.original_noise)]

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
        fname = f'{self.output_dir}\\regplot_simulation_slope_comparison_original_noise_{self.original_noise}'
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
            edgecolor=vutils.BLACK, style='trial', ax=g.ax_joint, markers=vutils.DUO_MARKERS
        )
        # Map a regression plot onto the main ax
        self._add_regression(ax=g.ax_joint, grp=self.df)
        # Map KDE plots onto the marginal ax
        g.plot_marginals(
            sns.kdeplot, lw=2, multiple='layer', fill=False, common_grid=True, cut=0, common_norm=True
        )
        return g

    def _add_regression(
            self, ax: plt.Axes, grp: pd.DataFrame,
    ) -> None:
        """
        Adds in a linear regression fit to a single plot with bootstrapped confidence intervals
        """
        def regress(
                g: pd.DataFrame
        ) -> np.ndarray:
            # Coerce x variable into correct form and add constant
            x = sm.add_constant(g[self.sim_var].to_list())
            # Coerce y into correct form
            y = g[self.orig_var].to_list()
            # Fit the model, predicting y from x
            md = sm.OLS(y, x).fit()
            # Return predictions made using our x vector
            return md.predict(x_vec)

        # Create the vector of x values we'll use when making predictions
        x_vec = sm.add_constant(
            np.linspace(*ax.get_xlim(), len(grp[self.sim_var]))
        )
        # Create model predictions for a regression fitted on our actual data
        act_predict = regress(grp)
        # Create bootstrapped regression predictions
        res = [regress(grp.sample(frac=1, replace=True, random_state=n)) for n in range(0, self.n_boot)]
        # Create empty variable to hold concatenated dataframe so we don't get warnings
        conc = None
        # Convert bootstrapped predictions to dataframe and get standard deviation for each x value
        if self.error_bar == 'ci':
            # Convert bootstrapped predictions to dataframe and get required percentiles for each x value
            boot_res = (
                pd.DataFrame(res)
                  .transpose()
                  .apply(np.nanpercentile, axis=1, q=self.percentiles)
                  .apply(pd.Series)
                  .rename(columns={0: 'low', 1: 'high'})
            )
            # Concatenate bootstrapped standard errors with original predictions
            # No need to add error terms to our original data, we just plot the percentiles
            conc = pd.concat([pd.Series(x_vec[:, 1]).rename('x'), pd.Series(act_predict).rename('y'), boot_res], axis=1)
        elif self.error_bar == 'sd':
            # Convert bootstrapped predictions to dataframe and get standard deviation for each x value
            boot_res = pd.DataFrame(res).transpose().std(axis=1).rename('sd')
            # Concatenate bootstrapped standard errors with original predictions
            conc = pd.concat([pd.Series(x_vec[:, 1]).rename('x'), pd.Series(act_predict).rename('y'), boot_res], axis=1)
            # Get confidence intervals by adding and subtracting error to original y values
            conc['high'] = conc['y'] + conc['sd']
            conc['low'] = conc['y'] - conc['sd']
        # Plot the resulting data
        ax.plot(conc['x'], conc['y'], color=vutils.BLACK, lw=3)
        ax.fill_between(conc['x'], conc['high'], conc['low'], color=vutils.BLACK, alpha=0.2)

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


class ArrowPlotParams(vutils.BasePlot):
    """
    Creates a plot showing the strength, direction, and balance of the coupling for each simulation parameter.
    Designed to look somewhat similar to fig 2.(b) in Jacoby et al. (2021)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create the figure and axes for plotting
        self.fig, self.ax = plt.subplots(nrows=4, ncols=1, figsize=(9.4, 9.4), sharex=True)
        # Define the simulation parameters
        self.params = [
            'Original', 'Anarchy', 'Democracy', 'Leadership'
        ]
        # Instruments, with Greek letters for labelling
        self.instruments = [
            'Keys (α)', 'Drums (β)'
        ]
        # Ratios to scale arrow widths by - these are arbitrary!
        self.means = [
            (1, 1), (0.1, 0.1), (2, 2), (0.1, 2)
        ]
        # Coupling coefficients to place above each arrow
        self.coupling = [
            ('$β_i{_,}_j$', '$α_i{_,}_j$'),
            ('0', '0'),
            ('mean($α_i{_,}_j$, $β_i{_,}_j$)', 'mean($α_i{_,}_j$, $β_i{_,}_j$)'),
            ('0', 'mean($α_i{_,}_j$, $β_i{_,}_j$)')
        ]
        # Equations for calculating overall coupling balance for each simulated parameter
        self.balance = [
            'abs($α_i{_,}_j$, $β_i{_,}_j$)', '0', '0', 'mean($α_i{_,}_j$, $β_i{_,}_j$)'
        ]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate the plot and save in plot decorator
        """
        self._create_plot()
        self._format_fig()
        fname = f'{self.output_dir}\\arrowplot_simulation_params'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates each subplot
        """
        # Iterate through each ax and parameter (each row)
        for num, ax, param in zip(range(0, 4), self.ax.flatten(), self.params):
            # Add the title (parameter name) in
            ax.set_title(param)
            # Turn off the axis
            ax.axis('off')
            # Add the text showing the coupling balance equation
            ax.text(0.5, 0.05, f'$Asymmetry$: {self.balance[num]}', ha='center', va='center')
            # Iterate through all the values we need to create plot features for each musician
            for n_, col, col2, col3, x1, x2, y, text, rot, in zip(
                    range(0, 2), vutils.INSTR_CMAP, vutils.INSTR_CMAP[::-1], [vutils.WHITE, vutils.BLACK],
                    [0.025, 0.975], [0.975, 0.025], [0.7, 0.3, ], self.instruments, [90, 270],
            ):
                # Add in the arrow and scale by the width value
                ax.annotate(
                    '', xy=(x1, y), xycoords=ax.transAxes, xytext=(x2, y), textcoords=ax.transAxes,
                    arrowprops=dict(
                        edgecolor=vutils.BLACK, lw=1.5, facecolor=col2, mutation_scale=1,
                        width=self.means[num][n_] * 5, shrink=0.1, headwidth=20
                    )
                )
                # Add in th text showing the coupling coefficient
                ax.text(
                    0.5, y + 0.17, self.coupling[num][n_], ha='center', va='center'
                )
                # Add in the rectangle for the instrument name to go on
                ax.add_patch(
                    plt.Rectangle(
                        (x1 - 0.05, -0.12), width=0.1, height=1.2, clip_on=False, linewidth=3,
                        edgecolor=vutils.BLACK, transform=ax.transAxes, facecolor=col
                    )
                )
                # Add in the text showing the instrument name, with correct color/rotation
                ax.text(
                    x1, 0.49, text, rotation=rot, ha='center', va='center',
                    fontsize=vutils.FONTSIZE + 3, color=col3,
                )

    def _format_fig(
            self
    ) -> None:
        """
        Sets figure-level attributes
        """
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.4, hspace=0.5)


class DistPlotParams(vutils.BasePlot):
    """
    Creates seperate kernel density plots for each duo showing distribution of asynchrony and tempo
    slope values for each parameter. Demonstrates similarity e.g. between duo 1/3 and democracy parameter,
    2/5 and leadership parameter.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=5, figsize=(18.8, 5), sharex=False, sharey=False, gridspec_kw={'height_ratios': [1.5, 1]}
        )
        self.params = ['democracy', 'leadership', 'anarchy', 'original']

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format
        """
        return df[(df['original_noise'] == False)]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate plot and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\distplot_simulation_params'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates plots in matplotlib
        """
        # Iterate through each trial
        for i, g in self.df.groupby('trial'):
            # Group in order to get mean values for one condition and parameter
            g = g.groupby(by=['parameter', 'latency', 'jitter']).mean().reset_index(drop=False)
            # Iterate through parameter, color, ax, and text position
            for param, color, n_, xy, in zip(
                    self.params, [vutils.BLACK, vutils.BLACK, vutils.BLACK, vutils.DUO_CMAP[i - 1]],
                    [1, 1, 0, 1], [(-0.55, 320), (0, 450), (0.35, 2600), (0, 0)]
            ):
                # Create the kde plot for this parameter
                sns.kdeplot(
                    data=g[(g['parameter'] == param)], x='tempo_slope_simulated', y='asynchrony_simulated',
                    ax=self.ax[n_, i - 1], fill=False if param == 'original' else True,
                    alpha=1 if param == 'original' else vutils.ALPHA, levels=5, legend=None,
                    color=color, clip=((-1, 1), (0, 3000)),
                )
                # Add the text in for this parameter
                self.ax[n_, i - 1].text(
                    *xy, param.title() if param != 'original' else '',
                    ha='center', va='center', fontsize=vutils.FONTSIZE - 3
                )

    def _format_ax(
            self
    ) -> None:
        """
        Format axis-level properties
        """
        # Iterate through each ax
        for num in range(0, 5):
            # Break the ax
            vutils.break_axis(ax1=self.ax[0, num], ax2=self.ax[1, num])
            # Iterate through each row, axis y limit, and title
            for n_, lim, tit in zip(
                    range(0, 2), [(2000, 3000), (0, 500)], [f'Duo {num + 1}', ''],
            ):
                # Create x and y ticks
                xticks = np.linspace(-1, 1, 5, dtype=float)
                yticks = np.linspace(lim[0], lim[1], int(((lim[1] - lim[0]) / 250) + 1), dtype=int)
                # Set axis properties
                self.ax[n_, num].set(
                    ylabel='', xlabel='', title=tit, xlim=(-1, 1), xticks=xticks if n_ == 1 else [], ylim=lim,
                    yticks=yticks, xticklabels=xticks if n_ == 1 else [],  yticklabels=yticks if num == 0 else []
                )
                # Set tick and axis width parameters
                self.ax[n_, num].tick_params(width=3, which='major')
                plt.setp(self.ax[n_, num].spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Format figure-level properties
        """
        # Add axis labels
        self.fig.supxlabel('Tempo slope (BPM/s)', y=0.03,)
        self.fig.supylabel('Asynchrony (RMS, ms)', x=0.01,)
        # Adjust subplot positioning slightly. Use hspace to adjust positioning between broken axis
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.175, top=0.9, wspace=0.15, hspace=0.2)


class DistPlotAverage(vutils.BasePlot):
    """
    Plots mean tempo slope and asynchrony of simulations with letters showing the coupling parameters used
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        # Define the conditions we wish to plot data for -- defaults to all 90 ms conditions
        self.conditions = kwargs.get('conditions', [(0, 0), (90, 0), (90, 0.5), (90, 1)])
        # Markers used for plotting anarchy/democracy/leadership parameters
        self.param_markers = ["$A$", "$D$", "$L$"]
        # Initialise subplots -- always two rows, and as many columns as we have conditions to plot
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=len(self.conditions), figsize=(18.8, 6), sharex=False, sharey=False,
            gridspec_kw={'height_ratios': [0.5, 1.5]}
        )

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the plot in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\distplot_simulation_params_average'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Create each plot individually
        """
        for y, (lat, jit) in zip(range(0, 5), self.conditions):

            # Subset to get required condition
            condition = self.df[(self.df['latency'] == lat) & (self.df['jitter'] == jit)]
            # Plot anarchy, democracy, leadership results
            for n_, (i, g) in enumerate(condition[condition['trial'] == 0].groupby('parameter')):
                # Define which row of plots the markers should be placed on
                x = 0 if i == 'anarchy' else 1
                self.ax[x, y].scatter(
                    g['tempo_slope_simulated'], g['asynchrony_simulated'], marker='*',
                    s=150, color=vutils.BLACK, label=None, zorder=10
                )
                if lat != 0:
                    x_pad = 0
                    if i == 'democracy':
                        x_pad = 0.2
                        y_pad = 10
                    elif i == 'anarchy':
                        y_pad = 100
                    else:
                        y_pad = 10
                    self.ax[x, y].annotate(
                        i.title(), xy=(g['tempo_slope_simulated'], g['asynchrony_simulated']),
                        xytext=(g['tempo_slope_simulated'] + x_pad, g['asynchrony_simulated'] + y_pad),
                        arrowprops=dict(arrowstyle="-", color='black', lw=2, alpha=vutils.ALPHA), ha='left', va='bottom'
                    )
                else:
                    if i == 'leadership':
                        y_pad = 20
                        x_pad = 0.1
                    elif i == 'anarchy':
                        y_pad = 75
                        x_pad = 0
                    else:
                        y_pad = 20
                        x_pad = -0.6
                    self.ax[x, y].annotate(
                        i.title(), xy=(g['tempo_slope_simulated'], g['asynchrony_simulated']),
                        xytext=(g['tempo_slope_simulated'] + x_pad, g['asynchrony_simulated'] + y_pad),
                        arrowprops=dict(arrowstyle="-", color='black', lw=2, alpha=vutils.ALPHA), ha='left', va='bottom'
                    )
            # Plot results from each duo
            for i, g in condition[condition['trial'] != 0].groupby('trial'):
                i = int(i)
                if i == 2 and y == 2:
                    g['tempo_slope_simulated'] += 0.05
                elif i == 3 and y == 2:
                    g['tempo_slope_simulated'] -= 0.05
                self.ax[1, y].scatter(
                    g['tempo_slope_simulated'], g['asynchrony_simulated'], marker=vutils.DUO_MARKERS[i - 1],
                    s=150, edgecolor=vutils.BLACK, facecolor=vutils.DUO_CMAP[i - 1],
                    label=i if y == 0 else None, zorder=1
                )

    def _format_ax(
            self
    ) -> None:
        """
        Format axis-level attributes
        """
        # Iterate through each column of plots and latency/jitter combination
        for y, (lat, jit) in zip(range(0, len(self.conditions)), self.conditions):
            # Break the axis with our utility function
            vutils.break_axis(self.ax[0, y], self.ax[1, y])
            # Define the axis title
            tit = f'Latency: {lat}ms\nJitter: {jit}x' if y == 0 else f'{lat}ms\n{jit}x'
            # Set parameters for both rows of plots
            self.ax[0, y].set(
                xlim=(-1, 1), xticks=[], yticks=[1750, 2500, 3250], title=tit, ylim=(1500, 3500)
            )
            self.ax[1, y].set(
                ylim=(0, 200), xlim=(-0.7, 0.7), xticks=np.linspace(-0.5, 0.5, 3), yticks=np.linspace(0, 150, 4)
            )
            # Apply the same formatting to all plots
            for x in range(0, 2):
                # Remove y tick labels on all but the first column of plots
                if y != 0:
                    self.ax[x, y].set(yticklabels=[])
                # Add in a vertical line at 0 tempo slope
                self.ax[x, y].axvline(x=0, linewidth=3, color=vutils.BLACK, alpha=0.1, ls='--')
                # Adjust tick and axis width slightly
                self.ax[x, y].tick_params(width=3, axis='both', pad=7.5, which='major')
                plt.setp(self.ax[x, y].spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Set figure-level attributes
        """
        # Add in axis labels
        self.fig.supxlabel('Tempo slope (BPM/s)')
        self.fig.supylabel('Asynchrony (RMS, ms)', x=0.01)
        # Add in legend
        self.fig.legend(loc='center right', frameon=False, title='Duo')
        # Adjust plot spacing a bit -- hspace adjusts broken axis
        self.fig.subplots_adjust(bottom=0.15, top=0.85, left=0.075, right=0.935, wspace=0.1, hspace=0.2)


class RegPlotSlopeAsynchrony(vutils.BasePlot):
    """
    Creates a regression jointplot, designed to show similarity in tempo slope and asynchrony
    between simulated and original performances.
    """

    def __init__(
            self, df: pd.DataFrame, **kwargs
    ):
        super().__init__(**kwargs)
        # Define variables
        self.var = kwargs.get('var', 'tempo_slope')
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)
        self.error_bar: str = kwargs.get('error_bar', 'sd')
        self.percentiles: tuple[float] = kwargs.get('percentile', (2.5, 97.5))
        self.original_noise = kwargs.get('original_noise', False)
        self.orig_var, self.sim_var = self.var + '_original', self.var + '_simulated'
        self.df = self._format_df(df)
        self.fig = plt.figure(figsize=(18.8, 8))
        self.main_ax, self.marginal_ax = self._init_gridspec_subplots()
        self.handles, self.labels = None, None

    def _init_gridspec_subplots(
            self, widths: tuple[int] = (5, 1, 0.3, 5, 1), heights: tuple[int] = (1, 5)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialise grid of subplots. Returns two numpy arrays corresponding to main and marginal axes respectively.
        """
        # Initialise subplot grid with desired width and height ratios
        grid = self.fig.add_gridspec(nrows=2, ncols=5, width_ratios=widths, height_ratios=heights, wspace=0.1,
                                     hspace=0.1)
        # Create a list of index values for each subplot type
        margins = [0, 3, 6, 9, ]
        mains = [5, 8, ]
        # Create empty lists to hold subplots
        marginal_ax = []
        main_ax = []
        # Iterate through all the subplots we want to create
        for i in range(len(heights) * len(widths)):
            # Create the subplot object
            ax = self.fig.add_subplot(grid[i // len(widths), i % len(widths)])
            # Append the subplot to the desired list
            if i in margins:
                marginal_ax.append(ax)
            elif i in mains:
                main_ax.append(ax)
            # If we don't want to keep the axis, turn it off so it is hidden from the plot
            else:
                ax.axis('off')
        # Return the figure, main subplots (as numpy array), and marginal subplots (as numpy array)
        return np.array(main_ax), np.array(marginal_ax)

    def _format_df(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts results data from each Simulation class instance and subsets to get original coupling simulations only.
        Results data is initialised when simulations are created, which makes this really fast.
        """
        return df[(df['parameter'] == 'original') & (df['original_noise'] == self.original_noise) & (
                    df['tempo_slope_simulated'] <= 4)]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self._create_plot()
        self._format_main_ax()
        self._format_marginal_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\regplot_simulation_slope_comparison_original_noise_{self.original_noise}'
        return self.fig, fname

    def _create_plot(
            self
    ) -> sns.JointGrid:
        """
        Creates scatter plots for both simulated and actual tempo slope values
        """
        # Create the grid, but don't map any plots onto it just yet
        for main_ax, top_margin, right_margin, var in zip(
                self.main_ax.flatten(), self.marginal_ax.flatten()[:2], self.marginal_ax.flatten()[2:],
                ['tempo_slope', 'asynchrony']
        ):
            sns.scatterplot(
                data=self.df, x=var + '_simulated', y=var + '_original', hue='trial', palette=vutils.DUO_CMAP,
                style='trial', markers=vutils.DUO_MARKERS, s=100, edgecolor=vutils.BLACK, ax=main_ax
            )
            sns.regplot(
                data=self.df, x=var + '_simulated', y=var + '_original', ax=main_ax, scatter=False,
                n_boot=vutils.N_BOOT, line_kws=dict(color=vutils.BLACK, alpha=vutils.ALPHA, lw=3)
            )
            self._add_correlation_results(ax=main_ax, var=var)
            for margin, kwargs, in zip([top_margin, right_margin],
                                       [dict(x=var + '_simulated'), dict(y=var + '_original')]):
                sns.kdeplot(
                    data=self.df, hue='trial', palette=vutils.DUO_CMAP, legend=False, lw=2,
                    multiple='stack', fill=True, common_grid=True, cut=0, ax=margin, **kwargs
                )

    def _add_correlation_results(
            self, ax, var
    ) -> None:
        """
        Adds the results of a linear correlation onto the plot
        """
        # Calculate the correlation, get r and p values
        r, p = stats.pearsonr(self.df[var + '_simulated'], self.df[var + '_original'])
        # Format correlation results into a string
        s = f'$r$ = {round(r, 2)}{vutils.get_significance_asterisks(p)}'
        # Add the annotation onto the plot
        ax.annotate(
            s, (0.8, 0.1), xycoords='axes fraction', fontsize=vutils.FONTSIZE + 3,
            bbox=dict(facecolor='none', edgecolor=vutils.BLACK, pad=10.0), ha='center', va='center'
        )

    def _format_main_ax(self):
        for ax, lim in zip(self.main_ax.flatten(), [(-0.75, 0.75), (0, 250)]):
            # Get the axes limit from minimum and maximum values across both simulated and original data
            ax.set(xlim=lim, ylim=lim, xlabel='Simulated', ylabel='Observed')
            # Set the top and right spines of the joint plot to visible
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            # Add the diagonal line to the joint ax
            ax.axline(
                xy1=(0, 0), xy2=(1, 1), linewidth=3, transform=ax.transAxes,
                color=vutils.BLACK, alpha=vutils.ALPHA, ls='--'
            )
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)
            # Store our handles and labels
            self.handles, self.labels = ax.get_legend_handles_labels()
            # Remove the legend
            ax.get_legend().remove()

    def _format_marginal_ax(self):
        for top_margin, right_margin, lim, tit in zip(
                self.marginal_ax.flatten()[:2], self.marginal_ax.flatten()[2:], [(-0.6, 0.6), (0, 275)],
                ['Tempo slope (BPM/s)', 'Asynchrony (RMS, ms)']
        ):
            top_margin.set(xlim=lim, ylabel='', xlabel='', xticklabels=[], yticks=[])
            top_margin.set_title(tit, fontsize=vutils.FONTSIZE + 5, y=1.1)
            top_margin.spines['left'].set_visible(False)
            right_margin.set(ylim=lim, xlabel='', ylabel='', yticklabels=[], xticks=[])
            right_margin.spines['bottom'].set_visible(False)
            for ax in [top_margin, right_margin]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(width=3, which='major')
                plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Add the legend back in
        self.labels = [int(float(lab)) for lab in self.labels]
        lgnd = plt.legend(
            self.handles, self.labels, title='Duo', frameon=False, ncol=1, loc='right', markerscale=1.5,
            fontsize=vutils.FONTSIZE + 3, edgecolor=vutils.BLACK, bbox_to_anchor=(2.25, 0.65),
        )
        # Set the legend font size
        plt.setp(lgnd.get_title(), fontsize=vutils.FONTSIZE + 3)
        # Set the legend marker size and edge color
        for handle in lgnd.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([100])
        # Adjust subplots positioning a bit to fit in the legend we've just created
        self.fig.subplots_adjust(bottom=0.1, top=0.91, left=0.075, right=0.92, )


def generate_plots_for_individual_performance_simulations(
    sims: list[Simulation], output_dir: str,
) -> None:
    """
    Deprecated(?)
    """
    figures_output_dir = output_dir + '\\figures\\simulations_plots'
    df = pd.DataFrame([sim.results_dic for sim in sims])
    dp = DistPlotParams(df, output_dir=figures_output_dir)
    dp.create_plot()
    rp = RegPlotSlopeComparisons(df, output_dir=figures_output_dir, original_noise=True)
    rp.create_plot()
    rp = RegPlotSlopeComparisons(df, output_dir=figures_output_dir, original_noise=False)
    rp.create_plot()
    ap = ArrowPlotParams(output_dir=figures_output_dir)
    ap.create_plot()


def generate_plots_for_simulations_with_coupling_parameters(
    sims_params: list[Simulation], output_dir: str
) -> None:
    """
    Generates all plots in this file, with required arguments and inputs
    """
    figures_output_dir = output_dir + '\\figures\\simulations_plots'
    df_avg = pd.DataFrame([sim.results_dic for sim in sims_params])
    rp = RegPlotSlopeAsynchrony(df=df_avg, output_dir=figures_output_dir)
    rp.create_plot()
    dp = DistPlotAverage(df=df_avg, output_dir=figures_output_dir)
    dp.create_plot()
    bp = BarPlotSimulationParameters(df_avg, output_dir=figures_output_dir)
    bp.create_plot()


if __name__ == '__main__':
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", "phase_correction_sims.p")
    raw_av = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", "phase_correction_sims_average.p")
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate plots for our simulations using our coupling parameters
    generate_plots_for_simulations_with_coupling_parameters(raw_av, output)
    # Generate plots for our simulations from every performance
    # generate_plots_for_individual_performance_simulations(raw, output)
