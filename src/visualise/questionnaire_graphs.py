import pandas as pd
import numpy as np
from matplotlib import patches, pyplot as plt
import seaborn as sns
from random import uniform
import scipy.stats as stats
from matplotlib.transforms import ScaledTranslation

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils

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


class BarPlotQuestionnaireCorrelation(vutils.BasePlot):
    """
    Creates a barplot showing Pearson's R between scores for the same question by drummer and keys player
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

    def _format_df(self):
        """

        """
        res = []
        for idx, grp in self.df.groupby(['trial']):
            g = (
                grp.pivot(index=['trial', 'block', 'latency', 'jitter', ], columns='instrument',
                          values=['interaction', 'coordination', 'success'])
                   .reset_index(drop=False)
            )
            g.columns = [''.join(col) for col in g.columns]
            for s in ['success', 'coordination', 'interaction']:
                ke, dr = g[f'{s}Keys'].to_numpy(), g[f'{s}Drums'].to_numpy()
                r, p = stats.pearsonr(ke, dr)
                res.append({'trial': idx, 'question': s.title(), 'correlation': r,
                            'significance': vutils.get_significance_asterisks(p)})
        return pd.DataFrame(res).sort_values(by='question')

    @vutils.plot_decorator
    def create_plot(self):
        """

        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_questionnaire_correlation'
        return self.fig, fname

    def _create_plot(self):
        """

        """
        return sns.barplot(
            data=self.data, x='trial', y='correlation', hue='question', ax=self.ax, edgecolor=vutils.BLACK, lw=2
        )

    def _format_fig(self):
        """

        """
        self.fig.supxlabel('Duo', y=0.09)
        self.fig.supylabel('Correlation ($r$)', x=0.02, y=0.6)
        sns.move_legend(self.ax, 'lower center', ncol=3, title=None, frameon=False, bbox_to_anchor=(0.45, -0.33),
                        markerscale=1.6, handletextpad=0.3, )
        self.fig.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95)

    def _format_ax(self):
        """

        """
        self.g.set(ylim=(-1, 1), ylabel='', xlabel='')
        self.ax.axhline(y=0, alpha=1, linestyle='-', color=vutils.BLACK)
        self.ax.tick_params(width=3)
        plt.setp(self.ax.spines.values(), linewidth=2)
        for (i, trial), con in zip(self.data.groupby('question'), self.ax.containers):
            self.ax.bar_label(con, labels=trial['significance'].to_list(), padding=5, fontsize=12)


def generate_questionnaire_plots(
        mds: list, output_dir: str
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\questionnaire_plots'
    hm = HeatmapQuestionnaire(df=df, output_dir=figures_output_dir)
    hm.create_plot()
    bp = BarPlotTestRetestReliability(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp2 = BarPlotQuestionnaireCorrelation(df=df, output_dir=figures_output_dir)
    bp2.create_plot()


if __name__ == '__main__':
    # Default location for processed raw data
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save output models
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate models and pickle
    generate_questionnaire_plots(mds=raw, output_dir=output)
