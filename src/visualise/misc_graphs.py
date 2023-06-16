"""Code for generating plots that don't fit anywhere else"""

import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import Counter
import matplotlib.cm as cm

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils

# Define the objects we can import from this file into others
__all__ = [
    'generate_misc_plots'
]


class BarPlotInterpolatedIOIs(vutils.BasePlot):
    """
    Creates a stacked barplot showing the total number of IOIs per duo and the number of these which were interpolated
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fmt = self._format_df()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4, 5))

    def _format_df(self):
        # Group the dataframe and get the sum
        fmt = self.df.groupby(by=['trial', 'instrument']).sum()
        # Create the percentage columns
        fmt['total_beats'] = fmt['total_beats'] - fmt['asynchrony_na'] - fmt['repeat_notes']
        fmt['total_raw'] = fmt['total_beats'] - fmt['interpolated_beats']
        fmt['percent_interpolated'] = (fmt['interpolated_beats'] / fmt['total_beats']) * 100
        fmt['percent_raw'] = 100 - fmt['percent_interpolated']
        return fmt.reset_index(drop=False)

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        self._create_plot()
        self._add_total_beats_to_plot()
        self._format_ax()
        self._format_plot()
        fname = f'{self.output_dir}\\barplot_total_vs_interpolated_beats'
        return self.fig, fname

    def _format_plot(self):
        self.ax.tick_params(width=3, )
        self.ax.set(ylabel='', xlabel='',)
        self.fig.supxlabel('Duo', y=0.02, x=0.46)
        self.fig.supylabel('Total onsets', x=0.01)
        plt.setp(self.ax.spines.values(), linewidth=2)
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.12, right=0.775)

    def _format_ax(self):
        # Add invisible data to add another legend for instrument
        n1 = [self.ax.bar(0, 0, color=cmap) for i, cmap in zip(range(2), vutils.INSTR_CMAP)]
        l1 = plt.legend(n1, ['Keys', 'Drums'], title='Instrument', bbox_to_anchor=(1, 0.8), ncol=1, frameon=False,)
        self.ax.add_artist(l1)
        # Add invisible data to add another legend for interpolation
        n2 = [self.ax.bar(0, 0, color='gray', hatch=h, alpha=vutils.ALPHA) for i, h in zip(range(2), ['', '//'])]
        l2 = plt.legend(
            n2, ['No', 'Yes'], bbox_to_anchor=(0.915, 0.5), ncol=1, title='       Interpolation', frameon=False,
        )
        self.ax.add_artist(l2)
        # Set ticks
        self.ax.set_xticks([t + 0.09 for t in self.ax.get_xticks()], labels=[f'{n}' for n in range(1, 6)], rotation=0)
        self.ax.set_yticks([int(num) for num in np.linspace(0, 5000, 6)],
                           labels=[int(num) for num in np.linspace(0, 5000, 6)])
        # Add a bit of padding for the plot labels
        self.ax.set_ylim(0, 5750)

    def _format_rect(self, dfall):
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        hand, lab = self.ax.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(hand[i:i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                    rect.set_width(1 / float(n_df + 1))
                    if rect.get_y() > 0:
                        rect.set_hatch('///')  # edited part

    def _add_total_beats_to_plot(self):
        for num, ins in enumerate([self.ax.containers[:2], self.ax.containers[2:]]):
            for c1, c2 in zip(ins[0], ins[1]):
                self.ax.text(c2.get_x() + (c2.get_width() / 2), c2.get_y() + c2.get_height() + 100,
                             int(c2.get_height()), fontsize=14, color=vutils.BLACK, ha='center')

    def _create_plot(self):
        dfall = [self.fmt[self.fmt['instrument'] == 'Keys'][['total_raw', 'interpolated_beats']],
                 self.fmt[self.fmt['instrument'] != 'Keys'][['total_raw', 'interpolated_beats']]]
        for df, cmap in zip(dfall, vutils.INSTR_CMAP):
            df.plot(kind="bar", stacked=True, ax=self.ax, legend=False,
                    grid=False, color=cmap, edgecolor=vutils.BLACK, lw=2)
        self._format_rect(dfall)


class LinePlotZoomCall(vutils.BasePlot):
    """
    Creates a lineplot and histogram showing latency over time for initial zoom call
    """
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerces data into correct format for plotting
        """
        # Get initial zoom array
        sub = (df[(df['latency'] == 180) & (df['jitter'] == 1) & (df['instrument'] == 'Keys') & (df['trial'] == 1) & (
                    df['block'] == 1)])
        zoom_arr = sub['zoom_arr'].values[0]
        # Append timestamps to zoom array
        z = np.c_[zoom_arr, np.linspace(0, 0 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
        sub = pd.DataFrame(z, columns=['latency', 'timestamp'])
        sub['timestamp'] = pd.to_timedelta([timedelta(seconds=val) for val in sub['timestamp']]).seconds
        # Subset zoom array and return
        return sub[sub['timestamp'] <= 90]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot. Generates and saves in plot decorator
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\lineplot_zoom_call'
        return self.g.figure, fname

    def _create_plot(
            self
    ) -> sns.JointGrid:
        """
        Creates the joint grid
        """
        # Create object in seaborn
        g = sns.JointGrid(data=self.df, x='timestamp', y='latency', )
        # Map line plot to center and histogram to marginal
        g.plot_joint(sns.lineplot, errorbar=None, lw=3, color=vutils.BLACK)
        g.plot_marginals(sns.histplot, kde=True, bins=10, color=vutils.BLACK, line_kws={'lw': 3, })
        # Turn off top marginal plot
        g.ax_marg_x.remove()
        return g

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects
        """
        # Set parameters for both joint and marginal plot
        for ax in [self.g.ax_joint, self.g.ax_marg_y]:
            ax.set(ylabel='', xlabel='')
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
        # Set parameters for margin plot only
        self.g.ax_marg_y.tick_params(labelbottom=True, bottom=True)
        self.g.ax_marg_y.spines['bottom'].set_visible(True)
        self.g.ax_marg_y.set_ylabel('Density')
        # Set parameters for joint plot only
        self.g.ax_joint.set_xticks([0, 30, 60, 90])
        self.g.ax_joint.spines['top'].set_visible(True)
        self.g.ax_joint.spines['right'].set_visible(True)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure level attributes
        """
        # Set overall size of figure
        self.g.figure.set_size_inches((9.4, 5))
        # Add text to substitute for marginal plot y axis label
        self.g.figure.text(x=0.86, y=0.03, s='Density', fontsize=vutils.FONTSIZE + 3)
        # Add center plot labels
        self.g.figure.supylabel('Latency (ms)')
        self.g.figure.supxlabel('Call duration (s)')
        # Adjust plot spacing slightly
        self.g.figure.subplots_adjust(bottom=0.15, top=1.1, left=0.12, right=0.98, wspace=0.2)


class LinePlotAllConditions(vutils.BasePlot):
    """
    Creates a line plot for all conditions, with one column showing raw latency and another showing windowed standard
    deviation (aka jitter), defaults to 6-second window (8 * 0.75)
    """
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(5, 2, sharex=True, sharey='col', figsize=(18.8, 12))

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerce data into correct format for plotting
        """
        # Get the data just for one trial and sort by latency and jitter
        sub = df[(df['trial'] == 1) & (df['instrument'] == 'Keys') & (df['block'] == 1)].reset_index().sort_values(
            by=['latency', 'jitter'])
        res = []
        # Iterate through each row
        for i, r in sub.iterrows():
            # Get the zoom array for this condition
            zoom_arr = r['zoom_arr']
            # Append timestamps to zoom array
            z = np.c_[zoom_arr, np.linspace(0, 0 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
            s = pd.DataFrame(z, columns=['latency', 'timestamp'])
            s['timestamp'] = pd.to_timedelta([timedelta(seconds=val) for val in s['timestamp']]).seconds
            # Subset zoom array and return
            res.append((r['latency'], r['jitter'], s))
        # Create a dataframe of all conditions
        return pd.DataFrame(res, columns=['latency', 'jitter', 'data'])

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Create plot and save in plot decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\lineplot_all_conditions'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Map plots onto axes
        """
        # Iterate through latency values
        for num, (i, g) in enumerate(self.df.groupby(['latency'])):
            # Iterate through jitter scales
            for num_, (i_, g_) in enumerate(g.groupby('jitter')):
                # Plotting latency
                # We don't want to add labels for all our plots
                if num == 1:
                    self.ax[num, 0].plot(
                        g_['data'].values[0]['timestamp'], g_['data'].values[0]['latency'], lw=3, label=f'{i_}x',
                        color=vutils.JITTER_CMAP[num_],
                    )
                else:
                    self.ax[num, 0].plot(
                        g_['data'].values[0]['timestamp'], g_['data'].values[0]['latency'], lw=3,
                        color=vutils.JITTER_CMAP[num_],
                    )
                self.ax[num, 0].set(
                    title=f'{i} ms latency condition', ylim=(-25, 300), xticks=[0, 30, 60, 90], yticks=[0, 100, 200, 300]
                )
                # Plotting jitter
                roll = g_['data'].values[0].rolling(window=8, min_periods=1)['latency']
                self.ax[num, 1].plot(
                    g_['data'].values[0]['timestamp'], roll.std(), lw=3, color=vutils.JITTER_CMAP[num_],
                )
                self.ax[num, 1].set(
                    title=f'{i} ms latency condition', ylim=(-5, 60), xticks=[0, 30, 60, 90], yticks=[0, 20, 40, 60]
                )
                # Add y labels onto only the middle row of plots
                if num == 2:
                    self.ax[num, 0].set_ylabel('Latency (ms)', labelpad=10, fontsize=vutils.FONTSIZE + 3)
                    self.ax[num, 1].set_ylabel(
                        'Latency variability (SD, ms)', labelpad=10, fontsize=vutils.FONTSIZE + 3
                    )

    def _format_ax(
            self
    ) -> None:
        """
        Format axes-level attributes
        """
        for a in self.ax.flatten():
            a.tick_params(width=3, )
            plt.setp(a.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Format figure-level attributes
        """
        self.fig.supxlabel('Performance duration (s)', )
        self.fig.legend(loc='center right', ncol=1, frameon=False, title='Jitter\ncondition')
        self.fig.subplots_adjust(top=0.95, bottom=0.09, left=0.07, right=0.9, wspace=0.15, hspace=0.4)


class BarPlotCouplingExperimentalSessions(vutils.BasePlot):
    """
    Creates a barplot showing differences in coupling between experimental sessions. For supplementary material!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Plotting variables
        self.handles, self.labels = None, None
        self.estimator: callable = kwargs.get('estimator', np.mean)
        self.yvar: str = kwargs.get('yvar', 'correction_partner')
        self.ylim: tuple[float] = kwargs.get('ylim', (0., 1))
        self.ylabel: str = kwargs.get('ylabel', 'Coupling coefficient')
        # Dataframe, axis objects
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18.8, 4))

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces data into correct format for plotting
        """
        return (
            self.df.pivot_table(self.yvar, ['trial', 'latency', 'jitter', 'instrument', ], 'block')
                .reset_index(drop=False)
                .rename(columns={1: 'block_1', 2: 'block_2'})
        )

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Axes, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_{self.yvar}_experimental_sessions'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates plot in matplotlib
        """
        for ax_, block in zip(self.ax.flatten(), ['block_1', 'block_2']):
            sns.barplot(
                data=self.df, x='trial', y=block, hue='instrument', ax=ax_, palette=vutils.INSTR_CMAP,
                hue_order=['Keys', 'Drums'], estimator=self.estimator, edgecolor=vutils.BLACK, lw=2, width=0.8,
                errorbar=('ci', 95), errcolor=vutils.BLACK, errwidth=2, n_boot=vutils.N_BOOT, seed=1, capsize=0.1,
            )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axis-level objects
        """
        # Iterate over axis + titles
        for ax_, tit in zip(self.ax.flatten(), ['First', 'Second']):
            # Set axis aesthetics
            ax_.tick_params(width=3, )
            ax_.set(ylabel='', xlabel='', ylim=(0, 1), title=f'{tit} experimental session')
            plt.setp(ax_.spines.values(), linewidth=2)
            # Store legend handles and labels, then remove
            self.handles, self.labels = ax_.get_legend_handles_labels()
            ax_.get_legend().remove()

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level attributes
        """
        # Set figure labels
        self.fig.supylabel('Coupling coefficient', x=0.02, y=0.55)
        self.fig.supxlabel('Duo', y=0.02)
        # Format the legend to remove the handles/labels added automatically by the strip plot
        plt.legend(
            self.handles, self.labels, ncol=1, title='Instrument', frameon=False, bbox_to_anchor=(1, 0.65),
            markerscale=1.6
        )
        # Adjust the figure a bit and return for saving in decorator
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.075, right=0.9, wspace=0.1)


class BarPlotCouplingPieceParts(vutils.BasePlot):
    """
    Creates a barplot showing differences in coupling between two halves of one piece. For supplementary material!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Plotting variables
        self.handles, self.labels = None, None
        self.estimator: callable = kwargs.get('estimator', np.mean)
        self.yvar: str = kwargs.get('yvar', 'correction_partner')
        self.ylim: tuple[float] = kwargs.get('ylim', (0., 1))
        self.ylabel: str = kwargs.get('ylabel', 'Coupling coefficient')
        # Dataframe, axis objects
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18.8, 4))

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Axes, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_{self.yvar}_piece_parts'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates plot in matplotlib
        """
        for ax_, block in zip(self.ax.flatten(), ['coupling_1st', 'coupling_2nd']):
            sns.barplot(
                data=self.df, x='trial', y=block, hue='instrument', ax=ax_, palette=vutils.INSTR_CMAP,
                hue_order=['Keys', 'Drums'], estimator=self.estimator, edgecolor=vutils.BLACK, lw=2, width=0.8,
                errorbar=('ci', 95), errcolor=vutils.BLACK, errwidth=2, n_boot=vutils.N_BOOT, seed=1, capsize=0.1,
            )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axis-level objects
        """
        # Iterate over axis + titles
        for ax_, tit in zip(self.ax.flatten(), ['First', 'Second']):
            # Set axis aesthetics
            ax_.tick_params(width=3, )
            ax_.set(ylabel='', xlabel='', ylim=(0, 1), title=f'{tit} half of the piece')
            plt.setp(ax_.spines.values(), linewidth=2)
            # Store legend handles and labels, then remove
            self.handles, self.labels = ax_.get_legend_handles_labels()
            ax_.get_legend().remove()

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level attributes
        """
        # Set figure labels
        self.fig.supylabel('Coupling coefficient', x=0.02, y=0.55)
        self.fig.supxlabel('Duo', y=0.02)
        # Format the legend to remove the handles/labels added automatically by the strip plot
        plt.legend(
            self.handles, self.labels, ncol=1, title='Instrument', frameon=False, bbox_to_anchor=(1, 0.65),
            markerscale=1.6
        )
        # Adjust the figure a bit and return for saving in decorator
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.075, right=0.9, wspace=0.1)


class BarPlotHigherOrderModelComparison(vutils.BasePlot):
    """
    Creates a barplot comparing model quality and fit for higher order phase correction models
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vars: list[str] = kwargs.get('vars', ['rsquared_adj', 'aic', 'bic', ])
        self.max_order: int = kwargs.get('max_order', 4)
        self.titles: list[str] = kwargs.get('titles', ['Adjusted $R^{2}$',
                                                       'Akaike Information Criterion',
                                                       'Bayesian Information Criterion'])
        self.ylabels: list[str] = kwargs.get('ylabels', ['$R^{2}_{adj}$', 'AIC', 'BIC'])
        self.ylims = kwargs.get('ylims', [(0, 1), (-800, 0), (-800, 0)])
        self.fig, self.ax = plt.subplots(nrows=1, ncols=len(self.vars), sharex=True, sharey=False, figsize=(18.8, 4))
        self.df = self._format_df()

    def _format_df(self):
        dfs = []
        for var in self.vars:
            # Explode all of our coefficients into a dataframe
            all_ = pd.DataFrame(self.df[f'higher_order_{var}'].to_list())
            # Get a list of column names and set them to our new dataframe
            cols = [s + 1 for s in range(1, all_.shape[1] + 1)]
            all_.columns = cols
            all_ = all_[[i for i in range(2, self.max_order + 1)]]
            # Concatenate the dataframe with the original variable
            conc = pd.concat([self.df[var], all_], axis=1).melt(var_name='order', value_name='value')
            conc['order'] = conc['order'].replace(var, '0 (initial)')
            dfs.append(conc)
        return dfs

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\barplot_higher_order_model_comparison'
        return self.fig, fname

    def _create_plot(self):
        for df_, ax_, in zip(self.df, self.ax.flatten()):
            sns.barplot(
                data=df_, x='order', y='value', ax=ax_, estimator=np.mean, errorbar=('ci', 95),
                edgecolor=vutils.BLACK, lw=2, saturation=0.8, alpha=0.8, palette='husl', width=0.8,
                errwidth=2, seed=1, capsize=0.1, errcolor=vutils.BLACK, n_boot=vutils.N_BOOT,
            )

    def _format_ax(self):
        for ax_, tit, ylab, ylim in zip(
                self.ax.flatten(), self.titles, self.ylabels, self.ylims
        ):
            ax_.set(xticklabels=['0 (initial)', '1', '2', '3'])
            ax_.set(title=tit, xlabel='', ylabel='', ylim=ylim)
            ax_.tick_params(width=3, which='major')
            plt.setp(ax_.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.supxlabel('Model order ($M$)')
        self.fig.supylabel('Value')
        self.fig.subplots_adjust(top=0.9, bottom=0.175, right=0.97, left=0.07, wspace=0.2, hspace=0.2)


class BarPlotQuestionnaireCorrelation(vutils.BasePlot):
    """
    Creates two barplots showing both between-performer and between-repeat correlations for each question,
    stratified by duo number.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(18.8, 4))
        self.handles, self.labels = None, None

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerce df into correct format for plotting
        """
        res = []
        # Group first by trial
        for idx, grp in self.df.groupby(['trial']):
            # Iterate through each combination of independent/dependent variable and variable names
            for iv, dv, names in zip(['instrument', 'block'], ['block', 'instrument'], [('Keys', 'Drums'), ('1', '2')]):
                # Pivot the dataframe according to the variables
                g = (
                       grp.pivot(index=['trial', dv, 'latency', 'jitter', ], columns=iv,
                                 values=['interaction', 'coordination', 'success'])
                       .reset_index(drop=False)
                )
                # Reset the column names
                g.columns = [''.join(map(str, col)) for col in g.columns]
                # Iterate through each question individually
                for s in ['success', 'coordination', 'interaction']:
                    # Get scores for each level of the independent variable separately
                    ke, dr = g[f'{s}{names[0]}'].to_numpy(), g[f'{s}{names[1]}'].to_numpy()
                    # Calculate Pearson's r
                    r, p = stats.pearsonr(ke, dr)
                    # Append the results to the list
                    res.append(
                        {'trial': idx, 'variable': iv, 'question': s.title(),
                         'correlation': r, 'significance': vutils.get_significance_asterisks(p)}
                    )
        # Create a dataframe from all results, sort the values, and reset
        return pd.DataFrame(res).sort_values(by=['trial', 'variable']).reset_index(drop=True)

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class and creates the plot, saving in plot_decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_questionnaire_correlation'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Create the plot and apply axis labels
        """
        # Iterate through each ax and variable
        for num, (ax, var) in enumerate(zip(self.ax.flatten(), ['instrument', 'block'])):
            # Subset the data for the required variable
            sub = self.data[self.data['variable'] == var]
            # Create the bar chart
            g = sns.barplot(
                data=sub, x='question', y='correlation', hue='trial', ax=ax, edgecolor=vutils.BLACK, lw=2
            )
            # Add the bar label onto each container
            for (i, trial), con in zip(sub.groupby('trial'), g.containers):
                ax.bar_label(con, labels=trial['significance'].to_list(), padding=5, fontsize=12)

    def _format_ax(
            self
    ) -> None:
        """
        Set axis-level attributes
        """
        # Iterate through each axis and label combination
        for ax, lab in zip(self.ax.flatten(), ['Inter-rater reliability', 'Test-retest reliability']):
            # Store the legend handles for later, then remove it
            self.handles, self.labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            # Set axis parameters
            ax.set(ylim=(-1.1, 1.1), yticks=np.linspace(-1, 1, 5), ylabel='', xlabel='', title=lab)
            # Add in the horizontal line at y=0
            ax.axhline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, lw=2)
            # Adjust tick and spine width
            ax.tick_params(width=3)
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Set figure-level attributes
        """
        # Add labels
        self.fig.supxlabel('Question')
        self.fig.supylabel('Correlation ($r$)', x=0.01, y=0.55)
        # Add the legend back in
        self.fig.legend(self.handles, self.labels, 'center right', ncol=1, title='Duo', frameon=False)
        # Adjust subplot positioning slightly
        self.fig.subplots_adjust(bottom=0.175, top=0.9, left=0.07, right=0.93, wspace=0.12)


class HeatmapNoteChoice(vutils.BasePlot):
    """
    Creates a heatmap representation of a piano and drum kit showing notes that musicians played most frequently
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cmap_obj = kwargs.get('cmap', cm.Reds)
        self.cmap = lambda v: self.cmap_obj(v)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(18.8, 5))
        self.cax = self.fig.add_axes([0.9, 0.1, 0.02, 0.8])
        self.drms_objs = [
            ('Kick', plt.Rectangle, dict(xy=(0.075, 0.425), width=0.3, height=0.125)),
            ('Kick', plt.Rectangle, dict(xy=(0.2, 0.32), width=0.06, height=0.16)),
            ('Kick', plt.Rectangle, dict(xy=(0.075, 0.45), width=0.3, height=0.15)),
            ('Hi-Hat Pedal', plt.Rectangle, dict(xy=(-0.05, 0.235), width=0.16, height=0.06)),
            ('Hi-Hat Shank|Hi-Hat CC', plt.Circle, dict(xy=(-0.1, 0.34), radius=0.09)),
            ('Hi-Hat Shank|Hi-Hat CC', plt.Circle, dict(xy=(-0.1, 0.34), radius=0.01)),
            ('Snare Rimshot|Snare Sidestick', plt.Circle, dict(xy=(0.05, 0.395), radius=0.09)),
            ('Snare Center', plt.Circle, dict(xy=(0.05, 0.395), radius=0.08)),
            ('Rack Tom 1 Center', plt.Circle, dict(xy=(0.125, 0.54), radius=0.09)),
            ('Rack Tom 1 Rimclick', plt.Circle, dict(xy=(0.125, 0.54), radius=0.08)),
            ('Floor Tom 1 Center', plt.Circle, dict(xy=(0.4, 0.36), radius=0.11)),
            ('Rack Tom 2 Rimclick', plt.Circle, dict(xy=(0.4, 0.36), radius=0.1)),
            ('Crash Right', plt.Circle, dict(xy=(0.59, 0.4), radius=0.11)),
            ('Crash Right', plt.Circle, dict(xy=(0.59, 0.4), radius=0.01)),
            ('Ride Bow Tip|Ride Edge', plt.Circle, dict(xy=(0.45, 0.58), radius=0.14)),
            ('Ride Bell', plt.Circle, dict(xy=(0.45, 0.58), radius=0.05)),
            ('Crash Left', plt.Circle, dict(xy=(-0.05, 0.58), radius=0.12)),
            ('Crash Left', plt.Circle, dict(xy=(-0.05, 0.58), radius=0.01)),
        ]

        self.drms_df = self._init_drms_df()
        self.drms_df = self._format_drms_df()
        self.keys_df = self._init_keys_df()
        self.keys_wnotes, self.keys_bnotes = self._format_keys_objs()

    def _init_drms_df(self):
        expl = self.df.explode(column='all_beats')
        expl = expl[expl['instrument'] == 'Drums']
        c = Counter()
        for idx, grp in expl.groupby(by=['trial', 'instrument']):
            for i, g in grp.iterrows():
                res = [ons[1] for ons in g['all_beats']]
                c.update(dict(Counter(res)))
        return (
            pd.DataFrame(pd.Series(dict(c)))
              .reset_index(drop=False)
              .rename(columns={0: 'freq', 'index': 'instrument'})
        )

    def _format_drms_df(self):
        res = []
        for ins_, __, ___ in self.drms_objs:
            sub = self.drms_df[self.drms_df['instrument'].str.contains(ins_)]
            res.append({'instrument': ins_, 'freq': sub['freq'].sum()})
        res = pd.DataFrame(res).drop_duplicates()
        res['freq'] = self._normalise(res['freq'])
        return res

    def _init_keys_df(self):
        expl = self.df.explode(column='all_beats')
        expl = expl[expl['instrument'] == 'Keys']
        c = Counter()
        for idx, grp in expl.groupby(by=['trial', 'instrument']):
            for i, g in grp.iterrows():
                res = []
                for ons in g['all_beats']:
                    if 1 < [int(i) for i in ons[1] if i.isdigit()][0] < 6:
                        res.append(ons[1])
                c.update(dict(Counter(res)))
        c = pd.DataFrame(pd.Series(dict(c))).reset_index(drop=False).rename(columns={0: 'freq', 'index': 'instrument'})
        c['freq'] = self._normalise(c['freq'])
        return c

    @staticmethod
    def _format_keys_objs():
        notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        notes = [[f'{n}{i}' for n in notes] for i in range(2, 6)]
        notes = [i for li in notes for i in li]
        wnotes = [
            (n, dict(xy=(-0.2 + (0.045 * i), 0.2), width=0.045, height=0.5)) for n, i in
            zip(notes, range(len(notes)))
        ]
        bnotes = ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#']
        bnotes = [[f'{n}{i}' for n in bnotes] for i in range(2, 6)]
        bnotes = [i for li in bnotes for i in li]
        bnotes = [(n, dict(xy=(-0.17 + (0.045 * i), 0.35), width=0.03, height=0.35)) for n, i in
                  zip(bnotes, range(len(bnotes)))]
        return wnotes, bnotes

    @staticmethod
    def _normalise(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @vutils.plot_decorator
    def create_plot(self):
        self._create_drms_plot()
        self._create_keys_plot()
        self._add_cbar()
        self._format_plot()
        fname = f'{self.output_dir}\\heatmap_note_choice'
        return self.fig, fname

    def _create_drms_plot(self):
        seen = set()
        self.ax[0].set_aspect('equal', adjustable='datalim')
        txt_kws = dict(xycoords='data', annotation_clip=False, fontsize=vutils.FONTSIZE + 5, va='center')
        for ins_, func_, kwargs_, in self.drms_objs:
            freq = self.drms_df[self.drms_df['instrument'] == ins_]['freq'].iloc[0]
            self.ax[0].add_patch(
                func_(**kwargs_, clip_on=False, linewidth=3,
                      edgecolor=vutils.BLACK, facecolor=self.cmap(freq))
            )
            if 'Crash' in ins_:
                txt = ins_[:7]
            else:
                txt = ins_.split(' ')[0]
            if txt not in seen:
                seen.add(txt)
                xy = kwargs_['xy']
                if 'Ride' in txt:
                    xy = xy[0], xy[1] + 0.07
                if 'Crash' in txt:
                    xy = xy[0], xy[1] + 0.04 if txt == 'Crash L' else xy[1] - 0.04
                if func_ == plt.Rectangle:
                    xy = xy[0] + (kwargs_['width']) / 2, abs(xy[1]) + (kwargs_['height'] / 2)
                if txt == 'Hi-Hat':
                    self.ax[0].annotate('Pedal', xy=xy, ha='center', **txt_kws)
                    self.ax[0].annotate('Hi-Hat', xy=(-0.1, 0.3), ha='center', **txt_kws)
                else:
                    if txt == 'Ride':
                        self.ax[0].annotate('Bell', xy=(xy[0], xy[1] - 0.07), ha='center', **txt_kws)
                    self.ax[0].annotate(txt, xy=xy, ha='left' if txt == 'Kick' else 'center',
                                        color=vutils.WHITE if freq > 0.5 else vutils.BLACK, **txt_kws)

    def _create_keys_plot(self):
        self.ax[1].text(-0.175, 0.85, '$Octaves$', fontsize=vutils.FONTSIZE)
        for ins_, kwargs_ in self.keys_wnotes:
            try:
                sub = self.keys_df[self.keys_df['instrument'] == ins_]['freq'].iloc[0]
            except IndexError:
                sub = 0
            self.ax[1].add_patch(
                plt.Rectangle(**kwargs_, clip_on=False, linewidth=3,
                              edgecolor=vutils.BLACK, facecolor=self.cmap(sub))
            )
            xy = kwargs_['xy'][0] + 0.01, kwargs_['xy'][1] + 0.02
            self.ax[1].annotate(
                ''.join(i for i in ins_ if not i.isdigit()), xy=xy, fontsize=15,
                xycoords='data', annotation_clip=False,
                color=vutils.WHITE if sub > 0.5 else vutils.BLACK, rotation=0
            )
            if 'F' in ins_:
                self.ax[1].annotate(
                    ''.join(i for i in ins_ if i.isdigit()), xy=(xy[0] + 0.01, xy[1] + 0.55),
                    xytext=(xy[0] + 0.01, xy[1] + 0.56), fontsize=15,
                    xycoords='data', annotation_clip=False, color=vutils.BLACK, rotation=0, ha='center', va='center',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=4.9, lengthB=1', lw=2.0)
                )
            if 'C4' in ins_:
                self.ax[1].annotate(
                    'Middle C', xy=xy, xytext=(xy[0] - 0.025, xy[1] - 0.2), fontsize=15, ha='center', va='center',
                    xycoords='data', annotation_clip=False, color=vutils.BLACK, rotation=45,
                    arrowprops=dict(arrowstyle='->', lw=2)
                )
        for ins_, kwargs_ in self.keys_bnotes:
            try:
                sub = self.keys_df[self.keys_df['instrument'] == ins_]['freq'].iloc[0]
            except IndexError:
                sub = 0
            if 'E#' not in ins_ and 'B#' not in ins_:
                self.ax[1].add_patch(
                    plt.Rectangle(**kwargs_, clip_on=False, linewidth=3,
                                  edgecolor=vutils.BLACK, facecolor=self.cmap(sub))
                )
                xy = kwargs_['xy'][0] + 0.005, kwargs_['xy'][1] + 0.03
                self.ax[1].annotate(
                    ''.join(i for i in ins_ if not i.isdigit()), xy=xy, fontsize=12,
                    xycoords='data', annotation_clip=False,
                    color=vutils.WHITE if sub > 0.5 else vutils.BLACK, rotation=90
                )

    def _format_plot(self):
        for ax, tit, xpos in zip(self.ax.flatten(), ['Drums', 'Piano'], [0.2, 0.4]):
            ax.axis('off')
            ax.set_title(tit, fontsize=vutils.FONTSIZE + 5, x=xpos)
        self.fig.subplots_adjust(left=0.125, right=0.85, wspace=-0.05, hspace=0)

    def _add_cbar(self):
        sm = plt.cm.ScalarMappable(cmap=self.cmap_obj, norm=plt.Normalize(vmin=0, vmax=1))
        cb = plt.colorbar(sm, cax=self.cax)
        cb.ax.tick_params(width=3, labelsize=vutils.FONTSIZE)
        cb.set_label('Normalized frequency\n (1 = most played note)', fontsize=vutils.FONTSIZE)
        plt.setp(cb.ax.spines.values(), linewidth=2)


class CountPlotListenerDemographics(vutils.BasePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = (
            pd.DataFrame(self._format_df())
              .drop_duplicates()
              .sort_values(by='id')
              .reset_index(drop=True)
        )
        self._bin_data()
        self.fig, self.ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(18.8, 10))
        self.pal = sns.color_palette('Set2', n_colors=6)
        self.labs = ['Years', 'Gender', 'Region', 'Years', 'Hours', 'Response']
        self.tits = [
            'What is your age?',
            'How do you identify yourself?',
            'What country are you from?',
            'How many years of formal training on\na musical instrument (including voice)\nhave you had during your '
            'lifetime?',
            'On average, how many hours\ndo you listen to music daily?',
            'Do you make money\nfrom playing music?'
        ]
        self.vars = ['age', 'gender', 'country', 'years', 'hours', 'money']

    def _format_df(self):
        for idx, row in self.df.iterrows():
            for i in row['perceptual_answers']:
                yield {
                    'id': i['participant_id'],
                    'age': i['age'],
                    'gender': i['gender'],
                    'country': i['country'],
                    'years_of_formal_training': i['years_of_formal_training'],
                    'hours_of_daily_music_listening': i['hours_of_daily_music_listening'],
                    'money_from_playing_music': i['money_from_playing_music']
                }

    def _bin_data(self):
        # Bin age
        age_bins = [-np.inf, 24, 40, 55, 70, np.inf]
        age_labs = ['<25', '25–40', '41–55', '56–70', '>70']
        self.df['age'] = pd.Categorical(pd.cut(self.df['age'], bins=age_bins, labels=age_labs), categories=age_labs)
        # Bin country
        col = 'country'
        conditions = [
            np.isin(self.df[col], ['GB', 'IE']),
            np.isin(self.df[col], ['US', 'CA']),
            np.isin(self.df[col], ['RO', 'BG', 'CZ', 'HU', 'NO']),
            np.isin(self.df[col], ['CN', 'MY', 'RU', 'NP']),
        ]
        choices = ["UK/Ireland", 'N. America', 'Europe', 'Asia']
        self.df['country'] = pd.Categorical(np.select(conditions, choices, default=np.nan), choices)
        # Bin money
        self.df['money'] = pd.Categorical(self.df['money_from_playing_music'].str.title(),
                                          ['Never', 'Sometimes', 'Frequently'])
        # Bin gender
        self.df['gender'] = self.df['gender'].str.title()
        self.df['gender'] = self.df['gender'].str.replace('_', '-')
        # Bin hours/years
        for col in ['hours_of_daily_music_listening', 'years_of_formal_training']:
            conditions = [
                self.df[col] < 1,
                (self.df[col] <= 3) & (self.df[col] >= 1),
                (self.df[col] <= 6) & (self.df[col] >= 4),
                (self.df[col] <= 9) & (self.df[col] >= 7),
                self.df[col] >= 10
            ]
            choices = ["0", '1–3', '4–6', '7–9', '>9']
            self.df[col.split('_')[0]] = pd.Categorical(np.select(conditions, choices, default=np.nan), choices)

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\countplot_listener_demographics'
        return self.fig, fname

    def _create_plot(self):
        for var, ax, col in zip(self.vars, self.ax.flatten(), self.pal):
            res = self.df.sort_values(by=var)
            sns.countplot(data=res, x=var, ax=ax, lw=2, edgecolor=vutils.BLACK, width=0.5, color=col)

    def _format_ax(self):
        for ax, tit in zip(self.ax.flatten(), self.tits):
            ax.set(ylim=(0, len(self.df)), xlabel='', ylabel='', title=tit)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.supylabel('Count')
        self.fig.supxlabel('Response')
        self.fig.subplots_adjust(bottom=0.078, top=0.95, left=0.07, right=0.975, wspace=0.035, hspace=0.45)


def generate_misc_plots(
    mds: list, output_dir: str,
) -> None:
    """
    Generates all plots in this file, with required arguments and inputs
    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\misc_plots'
    lp_all = LinePlotAllConditions(df=df, output_dir=figures_output_dir)
    lp_all.create_plot()
    cp = CountPlotListenerDemographics(df=df, output_dir=figures_output_dir)
    cp.create_plot()
    hm = HeatmapNoteChoice(df=df, output_dir=figures_output_dir)
    hm.create_plot()
    bp = BarPlotQuestionnaireCorrelation(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp = BarPlotHigherOrderModelComparison(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp = BarPlotCouplingExperimentalSessions(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp = BarPlotCouplingPieceParts(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    stacked_bp = BarPlotInterpolatedIOIs(df=df, output_dir=figures_output_dir)
    stacked_bp.create_plot()
    lp_orig = LinePlotZoomCall(df=df, output_dir=figures_output_dir)
    lp_orig.create_plot()


if __name__ == '__main__':
    import logging
    import os

    # Configure logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # Default location for phase correction models
    logger.info(f"Making graphs from data in {os.path.abspath(r'../../models')}")
    raw = autils.load_from_disc(
        r'..\..\models', filename='phase_correction_mds.p'
    )
    # Default location to save plots
    output = r"..\..\reports"
    generate_misc_plots(mds=raw, output_dir=output)
