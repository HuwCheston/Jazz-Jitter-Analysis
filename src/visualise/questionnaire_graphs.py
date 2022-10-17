import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


# TODO: inherit from a base class?
class ScatterPlotQuestionnaire:
    """
    Creates a scatterplot for each duo/question combination.
    """
    def __init__(self, **kwargs):
        # Get from kwargs (with default arguments)
        self.df: pd.DataFrame = kwargs.get('df', None)
        self.output_dir: str = kwargs.get('output_dir', None)
        self.jitter: bool = kwargs.get('jitter', True)
        self.ax_var: str = kwargs.get('ax_var', 'instrument')
        self.marker_var: str = kwargs.get('marker_var', 'block')
        self.one_reg: bool = kwargs.get('one_reg', False)

        # Initialise an empty attribute where we'll store the facetgrid object
        self.g = None
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
        self._format_plot()
        fname = f'{self.output_dir}\\scatterplot_{self.ax_var}_{self.marker_var}.png'
        return self.g.figure, fname

    def _format_df(self):
        """
        Called from within the class
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

        """
        return [''.join(col) for col in self.df.columns.values]

    def _apply_jitter_for_plotting(self):
        """

        """
        def jitter(x):
            return x + random.uniform(0, .5) - .25

        self.df[self.xvar] = self.df[self.xvar].apply(lambda x: jitter(x))
        self.df[self.yvar] = self.df[self.yvar].apply(lambda x: jitter(x))

    def _create_facetgrid(self):
        """

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

        """
        def scatter(x, y, **kwargs):
            if self.one_reg:
                sns.scatterplot(data=self.df, x=x, y=y, **kwargs)
            else:
                sns.scatterplot(data=self.df, x=x, y=y, style=self.marker_var, **kwargs)

        self.g.map(scatter, self.xvar, self.yvar, s=100, )
        self.g.map(sns.regplot, self.xvar, self.yvar, scatter=False, ci=None)

        for ax in self.g.axes.flatten():
            ax.axline((0, 0), (10, 10), linewidth=2, color=vutils.BLACK, alpha=vutils.ALPHA)

    def _format_plot(self):
        """

        """
        self.g.set_titles('Duo {col_name} - {row_name}')
        self.g.set(xlim=(0, 10), ylim=(0, 10), xlabel='', ylabel='', xticks=[0, 5, 10], yticks=[0, 5, 10])
        self.g.figure.supxlabel(f'{self.xvar.replace("value", "")} rating', y=0.05)
        self.g.figure.supylabel(f'{self.yvar.replace("value", "")} rating', x=0.01)
        self.g.figure.subplots_adjust(bottom=0.12, top=0.93, wspace=0.15, left=0.05, right=0.93)
