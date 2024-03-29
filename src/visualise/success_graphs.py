"""Code for generating plots using the self-reported success metric"""

import pandas as pd
import numpy as np
from matplotlib import patches, pyplot as plt
import seaborn as sns
from random import uniform
from matplotlib.transforms import ScaledTranslation

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils

# Define the objects we can import from this file into others
__all__ = [
    'generate_questionnaire_plots'
]


class ScatterPlotQuestionnaire(vutils.BasePlot):
    """
    Creates a scatterplot for each duo/question combination.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jitter: bool = kwargs.get('jitter', True)
        self.ax_var: str = kwargs.get('ax_var', 'instrument')
        self.marker_var: str = kwargs.get('marker_var', 'block')
        self.one_reg: bool = kwargs.get('one_reg', False)
        # If we've passed our dataframe
        if self.df is not None:
            self.df = self._format_df()
            self.df.columns = self._format_df_columns()
            self.xvar, self.yvar = (col for col in self.df.columns if 'value' in col)
            if self.jitter:
                self._apply_jitter_for_plotting()

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_facetgrid()
        self._map_facetgrid_plots()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\scatterplot_{self.ax_var}_{self.marker_var}'
        return self.g.figure, fname

    def _format_df(self) -> pd.DataFrame:
        """
        Called from within the class to format the dataframe for plotting
        """
        return (
            self.df.replace({'block': {1: 'Block 1', 2: 'Block 2'}})
                .melt(id_vars=['trial', 'block', 'latency', 'jitter', 'instrument'],
                      value_vars=['success', 'coordination', 'interaction'])
                .pivot(index=['trial', self.marker_var, 'latency', 'jitter', 'variable'],
                       columns=self.ax_var, values=['value'])
                .reset_index(drop=False)
        )

    def _format_df_columns(self):
        """
        Flattens the multiindex of columns to one level
        """
        return [''.join(col) for col in self.df.columns.values]

    def _apply_jitter_for_plotting(self):
        """
        Applies jitter to categorical data to increase readability when plotting
        """
        self.df[self.xvar] = self.df[self.xvar].apply(lambda x: x + uniform(0, .5) - .25)
        self.df[self.yvar] = self.df[self.yvar].apply(lambda x: x + uniform(0, .5) - .25)

    def _create_facetgrid(self):
        """
        Creates the facetgrid object for plotting onto and returns
        """
        if self.one_reg:
            return sns.FacetGrid(
                self.df, col='trial', row='variable', sharex=True, sharey=True, height=3, aspect=1.2545
            )
        else:
            return sns.FacetGrid(
                self.df, col='trial', row='variable', hue=self.marker_var, sharex=True, sharey=True, height=3,
                aspect=1.2545
            )

    def _map_facetgrid_plots(self):
        """
        Maps plots onto the facetgrid
        """
        def scatter(x, y, **kwargs):
            if self.one_reg:
                sns.scatterplot(data=self.df, x=x, y=y, **kwargs)
            else:
                sns.scatterplot(data=self.df, x=x, y=y, style=self.marker_var, **kwargs)

        self.g.map(scatter, self.xvar, self.yvar, s=100, )
        self.g.map(sns.regplot, self.xvar, self.yvar, scatter=False, ci=None)

    def _format_ax(self):
        """
        Formats plot by setting axes-level properties
        """
        # Add in the axes diagonal line
        for ax in self.g.axes.flatten():
            ax.axline((0, 0), (10, 10), linewidth=2, color=vutils.BLACK, alpha=vutils.ALPHA)
        # Add titles, labels to each axes
        self.g.set_titles('Duo {col_name} - {row_name}')
        self.g.set(xlim=(0, 10), ylim=(0, 10), xlabel='', ylabel='', xticks=[0, 5, 10], yticks=[0, 5, 10])

    def _format_fig(self):
        """
        Formats plot by setting figure-level properties
        """
        self.g.figure.supxlabel(f'{self.xvar.replace("value", "")} rating', y=0.05)
        self.g.figure.supylabel(f'{self.yvar.replace("value", "")} rating', x=0.01)
        self.g.figure.subplots_adjust(bottom=0.12, top=0.93, wspace=0.15, left=0.05, right=0.93)


class HeatmapQuestionnaire(vutils.BasePlot):
    """
    Creates a heatmap showing pairwise correlations between survey responses for members of each duo
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=5, figsize=(18.8, 5.5), )
        self.ax[1].get_shared_x_axes().join(*[self.ax[n] for n in range(1, 5)])
        self.cbar_ax = self.fig.add_axes([0.915, 0.1, 0.015, 0.8])

    def _format_df(self):
        res = []
        for idx, grp in self.df.groupby('trial'):
            grp = grp.pivot(index=['trial', 'block', 'latency', 'jitter', ], columns='instrument',
                            values=['interaction', 'coordination', 'success']).reset_index(drop=True)
            grp.columns = [''.join(col) for col in grp.columns]
            corr = grp[['successKeys', 'interactionKeys', 'coordinationKeys', 'successDrums', 'interactionDrums',
                        'coordinationDrums']].corr()
            matrix = np.triu(corr)
            res.append([corr, matrix])
        return res

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_fig()
        fname = f'{self.output_dir}\\heatmap_duo_correlations'
        return self.fig, fname

    def _format_ax(self, g, i):
        ti = ['Successful', 'Interaction', 'Coordination', 'Successful', 'Interaction', 'Coordination']
        self._add_lines_to_ax(g)
        if i == 0:
            g.set_yticks(ticks=g.get_yticks(), labels=ti, minor=False)
            self._shift_ax_ticks(tick_labels=g.yaxis.get_majorticklabels(), x=-15 / 72., y=0 / 72.)
        else:
            g.set_xticks(g.get_xticks(), ['' for _ in range(6)], minor=False)
            g.set_yticks(g.get_yticks(), ['' for _ in range(6)], minor=False)
        g.set_xticks(ticks=g.get_xticks(), labels=ti, minor=False)
        g.set_xticks([1.51, 4.51], labels=['Keys', 'Drums'], rotation=0, minor=True)
        g.set_yticks([1, 3.51], labels=['Keys', 'Drums'], rotation=90, minor=True)
        g.set_title(f'Duo {i + 1}')
        g.tick_params(axis='both', which='minor', length=0)
        self._shift_ax_ticks(tick_labels=g.xaxis.get_majorticklabels(), x=0 / 72., y=-15 / 72.)

    @staticmethod
    def _add_lines_to_ax(g):
        g.plot([0, 6, 0, 0], [0, 6, 6, 0], clip_on=False, color='black', lw=2)
        g.plot((0, 3), (3, 3), color='black', lw=2)
        g.plot((3, 3), (3, 6), color='black', lw=2)
        for i in range(3):
            g.add_patch(patches.Rectangle((i, i + 3), 1, 1, linewidth=2, edgecolor=vutils.BLACK, facecolor='none',
                                          alpha=vutils.ALPHA))

    def _shift_ax_ticks(self, tick_labels, x, y):
        for label in tick_labels:
            offset = ScaledTranslation(x, y, self.fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

    def _create_plot(self):
        for i, ax in enumerate(self.ax.flat):
            g = sns.heatmap(
                self.data[i][0], mask=self.data[i][1], ax=ax, cmap=sns.color_palette('vlag', as_cmap=True),
                annot=True, center=0, square=True, cbar=i == 0, vmin=-1, vmax=1, cbar_ax=None if i else self.cbar_ax,
                annot_kws={'size': 10}, fmt='.2f', cbar_kws={'label': 'Correlation ($r$)'})
            self._format_ax(g, i)

    def _format_fig(self):
        self.fig.supylabel('Question, respondant', y=0.5)
        self.fig.supxlabel('Question, respondant', y=0.03)
        plt.subplots_adjust(bottom=0.3, top=1.1, wspace=0.15, left=0.15, right=0.90)


class BarPlotTestRetestReliability(vutils.BasePlot):
    """
    Creates a plot showing test-retest reliability coefficients across measures for each question, instrument, and duo.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.questions = ['interaction', 'coordination', 'success']
        self.data = self._format_df()

    def _format_df(self):
        res = []
        for idx, grp in self.df.groupby(['trial', 'instrument']):
            idx = tuple(idx)
            corr = (
                grp.pivot(index=['trial', 'latency', 'jitter', ], columns='block', values=self.questions)
                .reset_index(drop=True)
                .corr()
            )
            res.append({
                'trial': idx[0], 'instrument': idx[1], 'interaction': corr.iloc[0, 1],
                'coordination': corr.iloc[2, 3], 'success': corr.iloc[4, 5]
            })
        return pd.DataFrame(res).melt(
            id_vars=['trial', 'instrument'], value_vars=self.questions, var_name='question', value_name='correlation'
        )

    @vutils.plot_decorator
    def create_plot(self):
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_test_retest_reliability'
        return self.g.figure, fname

    def _create_plot(self):
        return sns.catplot(
            data=self.data, kind='bar', col='trial', hue='instrument', x='question', y='correlation',
            palette=vutils.INSTR_CMAP, hue_order=['Keys', 'Drums'],
            height=5.5, aspect=0.62, sharex=True, sharey=True, edgecolor=vutils.BLACK, lw=2
        )

    def _format_fig(self):
        self.g.fig.supxlabel('Question', y=0.09)
        self.g.fig.supylabel('Correlation ($r$)', x=0.005, y=0.65)
        # Move the legend
        sns.move_legend(self.g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.01), )
        # Adjust the plot size a bit
        self.g.fig.subplots_adjust(bottom=0.4, top=0.9, left=0.06, right=0.98)

    def _format_ax(self):
        self.g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK)
        self.g.set(ylim=(-1, 1), yticks=[val for val in np.linspace(-1, 1, 5)],
                   xticklabels=['Interaction', 'Coordination', 'Successful'],
                   ylabel='', xlabel='')
        for num, ax in enumerate(self.g.axes.flatten()):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.tick_params(width=3)
            plt.setp(ax.spines.values(), linewidth=2)
            ax.set_title(f'Duo {num + 1}')


class NumberLineSuccess(vutils.BasePlot):
    """
    Creates a numberline showing difference in success between duos this experiment during the control
    condition and a corpus of pairwise asynchrony values from other studies and genres
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4 * 2, 5.3))

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._add_annotations()
        self._format_plot()
        fname = f'{self.output_dir}\\numberline_success'
        return self.g.figure, fname

    def _format_df(self) -> pd.DataFrame:
        """
        Formats the dataframe by concatenating with data from the corpus
        """
        # Read in the data from the experimental dataframe
        trial = self.df[self.df['latency'] == 0]
        trial = (
            trial.groupby(['trial', 'instrument'])
                 .mean()
                 .reset_index(drop=False)
                 .rename(columns={'trial': 'style'})

        )
        trial = trial[['style', 'success', 'instrument']]
        trial['source'] = ''
        # Add a bit of noise to trial 3 and 4, so they don't overlap exactly on the graph
        trial.loc[trial['instrument'] == 'Keys', 'instrument'] = 'Pianist'
        trial.loc[trial['instrument'] == 'Drums', 'instrument'] = 'Drummer'
        np.random.seed(7)
        noise = np.random.normal(0, 0.2, len(trial))
        trial['success_'] = trial['success'] + noise
        trial['style'] = trial['instrument'] + ', Duo ' + trial['style'].astype(str)
        trial['placeholder'] = 0
        # Concatenate trial and corpus data together
        return trial

    def _create_plot(self):
        """
        Creates the facetgrid object
        """
        return sns.stripplot(
            data=self.df, x='success_', y='placeholder', jitter=False, dodge=False, s=15,
            ax=self.ax, orient='h', palette=vutils.INSTR_CMAP, hue='instrument', hue_order=['Pianist', 'Drummer'],
            edgecolor=vutils.BLACK, linewidth=2
        )

    def _add_annotations(self):
        """
        Add the annotations onto the plot
        """
        for k, v in self.df.iterrows():
            x = v['success_']
            if v['instrument'] == 'Pianist':
                self.g.annotate(text=v['style'], xy=(v['success_'], 0), xytext=(x, -1.4), rotation=315)
            else:
                self.g.annotate(text=v['style'] + '\n' + v['source'], xy=(v['success_'], 0), xytext=(x, 0.05),
                                rotation=45)

    def _format_plot(self):
        """
        Formats the plot
        """
        # Set axis position
        self.ax.spines['bottom'].set_position(('data', 0))
        # Adjust tick parameters and width
        self.ax.tick_params(axis="x", direction="in", pad=-25, width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Set ticks and axis label
        self.g.set(xlim=(0.5, 9.5), ylim=(-1, 1), xticks=np.linspace(1, 9, 5), xlabel='', ylabel='')
        plt.yticks([], [])
        self.g.figure.suptitle('Performer-reported success')
        # Add arrows and labels showing the direction of the x variable
        for text_x, arr_x, lab in zip([0.75, 0.15], [0.9, 0.1], ['Successful', 'Unsuccessful']):
            self.g.annotate(
                f"${lab}$", (arr_x, 0.93), xytext=(text_x, 0.91), annotation_clip=False,
                textcoords='figure fraction', xycoords='figure fraction', fontsize=vutils.FONTSIZE + 3,
                arrowprops=dict(arrowstyle='->', color=vutils.BLACK, lw=4)
            )
        # Remove the left and bottom axis
        sns.despine(left=True, bottom=False)
        # Adjust plot position slightly
        plt.subplots_adjust(top=0.63, bottom=0.15, left=0.03, right=0.97)
        # Remove the legend
        plt.legend([], [], frameon=False)


class NumberLineListenerSuccess(vutils.BasePlot):
    """
    Creates a numberline showing difference in success between duos this experiment during the control
    condition and a corpus of pairwise asynchrony values from other studies and genres
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4 * 2, 3))

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._add_annotations()
        self._format_plot()
        fname = f'{self.output_dir}\\numberline_listener_success'
        return self.g.figure, fname

    def _format_df(self) -> pd.DataFrame:
        """
        Formats the dataframe by concatenating with data from the corpus
        """
        # Read in the data from the experimental dataframe
        trial = self.df[self.df['latency'] == 0]
        means = []
        for idx, row in trial[trial['instrument'] == 'Keys'].groupby('trial'):
            means.append(np.nanmean([i['answer'] for i in np.concatenate(row['perceptual_answers'].values)]))
        pd.options.mode.chained_assignment = None
        pd.options.mode.chained_assignment = 'warn'
        trial = (
            trial.groupby(['trial'])
                 .mean()
                 .reset_index(drop=False)
                 .rename(columns={'trial': 'style'})
        )
        trial['success'] = means
        trial = trial[['style', 'success']]
        np.random.seed(8)
        noise = np.random.normal(0, 0.2, len(trial))
        trial['success'] = trial['success'] + noise
        trial['placeholder'] = 0
        return trial

    def _create_plot(self):
        """
        Creates the facetgrid object
        """
        return sns.stripplot(
            data=self.df, x='success', y='placeholder', jitter=False, dodge=False, s=15,
            ax=self.ax, orient='h', edgecolor=vutils.BLACK, linewidth=2
        )

    def _add_annotations(self):
        """
        Add the annotations onto the plot
        """
        for (k, v), rot, y in zip(self.df.iterrows(), [45, 315, 45, 315, 45], [0.15, -1.2, 0.15, -1.2, 0.15]):
            self.g.annotate(text=f'Duo {int(v["style"])}', xy=(v['success'], 0), xytext=(v['success'], y),
                            rotation=rot)

    def _format_plot(self):
        """
        Formats the plot
        """
        # Set axis position
        self.ax.spines['bottom'].set_position(('data', 0))
        # Adjust tick parameters and width
        self.ax.tick_params(axis="x", direction="in", pad=-25, width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Set ticks and axis label
        self.g.set(xlim=(0.5, 9.5), ylim=(-1, 1), xticks=np.linspace(1, 9, 5), xlabel='', ylabel='')
        plt.yticks([], [])
        self.g.figure.suptitle('Listener-reported success')
        # Add arrows and labels showing the direction of the x variable
        for text_x, arr_x, lab in zip([0.75, 0.15], [0.9, 0.1], ['Successful', 'Unsuccessful']):
            self.g.annotate(
                f"${lab}$", (arr_x, 0.9), xytext=(text_x, 0.88), annotation_clip=False,
                textcoords='figure fraction', xycoords='figure fraction', fontsize=vutils.FONTSIZE + 3,
                arrowprops=dict(arrowstyle='->', color=vutils.BLACK, lw=4)
            )
        # Remove the left and bottom axis
        sns.despine(left=True, bottom=False)
        # Adjust plot position slightly
        plt.subplots_adjust(top=0.63, bottom=0.15, left=0.03, right=0.97)
        # Remove the legend
        plt.legend([], [], frameon=False)


class BarPlotListenerEvaluations(vutils.BasePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=6, sharex=False, sharey=True,
                                         gridspec_kw=dict(width_ratios=[1, 0.2, 1, 0.2, 1, 0.2]), figsize=(18.8, 5))
        self.main_ax = self.ax[::2]
        self.marginal_ax = self.ax[1::2]
        self.vars_ = ['trial', 'latency', 'jitter']
        self.labs = ['Duo', 'Latency (ms)', 'Jitter']
        self.pals = [vutils.DUO_CMAP, sns.color_palette('tab10_r', n_colors=5), vutils.JITTER_CMAP]
        self.dfs = list(self._format_df())

    def _format_df(self):
        for var in self.vars_:
            res = []
            for idx, grp in self.df.groupby(var):
                vals = [i['answer'] for i in np.concatenate(grp['perceptual_answers'].values)]
                res.append(pd.DataFrame({
                    'var': np.array([idx for _ in range(len(vals))]),
                    'vals': np.array(vals)
                }))
            yield pd.concat(res)

    @vutils.plot_decorator
    def create_plot(self):
        self._create_main_ax()
        self._format_main_ax()
        self._create_marginal_ax()
        self._format_fig()
        self._format_marginal_ax()
        fname = f'{self.output_dir}\\barplot_listener_evaluations'
        return self.fig, fname

    def _create_main_ax(self):
        for big, ax, pal in zip(self.dfs, self.main_ax.flatten(), self.pals):
            sns.barplot(
                data=big, x='var', y='vals', palette=pal, errorbar=('ci', 95), n_boot=vutils.N_BOOT,
                seed=1, estimator=np.mean, errcolor=vutils.BLACK, errwidth=2, edgecolor=vutils.BLACK, lw=2,
                capsize=0.1, width=0.5, ax=ax,
            )

    def _format_main_ax(self):
        for ax, lab in zip(self.main_ax.flatten(), self.labs):
            for patch in ax.patches:
                patch.set_facecolor((*patch.get_facecolor()[:3], 0.7))
            ax.set(ylim=(1, 9), yticks=np.linspace(1, 9, 5, dtype=int), yticklabels=np.linspace(1, 9, 5, dtype=int),
                   ylabel='', xlabel=lab)
            ax.set_xlabel(lab, fontsize=20)
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _create_marginal_ax(self):
        for big, ax, pal in zip(self.dfs, self.marginal_ax.flatten(), self.pals):
            sns.kdeplot(
                data=big, y='vals', hue='var', ax=ax, palette=pal, legend=False, lw=2,
                multiple='stack', fill=True, common_grid=True, clip=(1, 9), alpha=0.6
            )

    def _format_marginal_ax(self):
        for ax in self.marginal_ax.flatten():
            # Apply formatting to all marginal plots
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
            ax.set(xticklabels=[], xticks=[], xlim=(0, 0.2), xlabel='', ylabel='')
            box = ax.get_position()
            box.x0 -= 0.01
            box.x1 -= 0.01
            ax.set_position(box)

    def _format_fig(self):
        self.fig.supxlabel('Variable')
        self.fig.supylabel('Listener-reported success')
        self.fig.subplots_adjust(bottom=0.225, top=0.9, left=0.065, right=0.995, wspace=0.15)


class PairGridListenerRatings(vutils.BasePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(18.8, 5))
        self.g = None

    @vutils.plot_decorator
    def create_plot(self):
        self._create_facetgrid()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\pairgrid_listener_reported_success'
        return self.fig, fname

    @staticmethod
    def _bootstrap_mean(arr, quant):
        boot = [arr.sample(frac=1, replace=True, random_state=n).mean() for n in range(0, vutils.N_BOOT)]
        return np.quantile(boot, quant)

    def _format_df(self):
        self.df = self.df[self.df['instrument'] == 'Keys'].sort_values(by=['trial', 'latency', 'jitter'])
        means = []
        low = []
        high = []
        for idx, grp in self.df.groupby(['trial', 'latency', 'jitter']):
            conc = np.array([i['answer'] for i in np.concatenate(grp['perceptual_answers'].values)])
            means.append(np.nanmean(conc))
            low.append(self._bootstrap_mean(pd.Series(conc), 0.025))
            high.append(self._bootstrap_mean(pd.Series(conc), 1 - 0.025))
        self.df = self.df.groupby(['trial', 'latency', 'jitter']).mean().reset_index(drop=False)
        self.df['perceptual_mean'] = means
        self.df['perceptual_low'] = low
        self.df['perceptual_high'] = high
        self.df['abbrev'] = self.df['latency'].astype('str') + 'ms/' + round(self.df['jitter'], 1).astype('str') + 'x'
        return self.df.sort_values(by=['latency', 'jitter']).reset_index(drop=True)

    def _create_facetgrid(self):
        """
        Creates facetgrid object and plots stripplot
        """
        for col, mark, ax, (idx, grp) in zip(vutils.DUO_CMAP, vutils.DUO_MARKERS, self.ax.flatten(),
                                             self.df.groupby('trial')):
            sns.stripplot(
                data=grp, x='perceptual_mean', y='abbrev', marker=mark,
                jitter=False, dodge=False, color=col, ax=ax, s=8
            )

    def _format_ax(self):
        for num, ax in enumerate(self.ax.flatten()):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'Duo {num + 1}', fontsize=vutils.FONTSIZE + 3)
            ax.set(ylabel='', xlim=(0, 10), ylim=(12.5, -0.5), xlabel='', xticks=np.linspace(1, 9, 5))
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
            ax.yaxis.grid(lw=2, alpha=vutils.ALPHA)
            grp = self.df[self.df['trial'] == num + 1]
            ax.errorbar(
                grp['perceptual_mean'],
                range(0, 13),
                xerr=[grp['perceptual_mean'] - grp['perceptual_low'], grp['perceptual_high'] - grp['perceptual_mean']],
                ls='none',
                color=vutils.BLACK,
                lw=2,
                capsize=3,
                capthick=2
            )

    def _format_fig(self):
        self.fig.supylabel('Condition (latency/jitter)', x=0.01)
        self.fig.supxlabel('Listener-reported success')
        self.fig.subplots_adjust(bottom=0.14, top=0.9, wspace=0.17, left=0.11, right=0.975)


def generate_questionnaire_plots(
        mds: list, output_dir: str
) -> None:
    """
    Generates all plots in this file, with required arguments and inputs
    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\success_plots'
    pg = PairGridListenerRatings(df=df, output_dir=figures_output_dir)
    pg.create_plot()
    bp = BarPlotListenerEvaluations(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    nl = NumberLineListenerSuccess(df=df, output_dir=figures_output_dir)
    nl.create_plot()
    hm = HeatmapQuestionnaire(df=df, output_dir=figures_output_dir)
    hm.create_plot()
    bp = BarPlotTestRetestReliability(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    nl = NumberLineSuccess(df=df, output_dir=figures_output_dir)
    nl.create_plot()


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
    generate_questionnaire_plots(mds=raw, output_dir=output)
