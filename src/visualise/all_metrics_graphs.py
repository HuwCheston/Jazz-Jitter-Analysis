import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


class PairPlotAllVariables(vutils.BasePlot):
    """
    Creates an R-style paired plot, showing correlation and distribution of four variables (tempo slope,
    ioi variability, asynchrony rms, and subjective success). Each variable is averaged across instrument and repeat,
    so e.g. we have one value for ioi variability for duo 2, 180ms latency, 0.5x jitter. Tempo slope is given in
    absolute form, to make linear relationships clearer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels: list[str] = kwargs.get(
            'labels', ['Absolute slope\n(BPM/s)', 'IOI variability\n(SD, ms)',
                       'Asynchrony\n(RMS, ms)', 'Self-reported\nsuccess']
        )
        self._jitter_success: bool = kwargs.get('jitter_success', True)
        self._abs_slope: bool = kwargs.get('abs_slope', True)
        self.vars: list[str] = kwargs.get('vars', ['tempo_slope', 'ioi_std', 'pw_asym', 'success'])
        self.df: pd.DataFrame = self._format_df()
        self.counter: int = 0

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting.
        """
        # Define the non-formatted variable names that we want to include
        df = self.df[['trial', 'block', 'latency', 'jitter', 'instrument', *self.vars]]
        # Update variables, if required: need to disable chained assignment warnings here
        pd.options.mode.chained_assignment = None
        if self._jitter_success:
            # Add gaussian noise to success variable
            df['success'] = [val + np.random.normal(0, 0.01) for val in df['success'].to_list()]
        if self._abs_slope:
            # Get absolute tempo slope
            df['tempo_slope'] = df['tempo_slope'].abs()
        pd.options.mode.chained_assignment = 'warn'
        # Group by trial, latency, and jitter, get the mean, then reset the index and return
        return df.groupby(['trial', 'latency', 'jitter']).mean().reset_index()

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate the plot and save in plot decorator.
        """
        # The create plot function here returns our figure to be stored as an attribute defined in the parent class
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\pairplot_all_variables'
        return self.g.figure, fname

    # noinspection PyUnusedLocal
    def _add_label_to_diag(
            self, *args, **kwargs
    ) -> None:
        """
        Adds variable names to diagonal plots.
        """
        # Get the current axis
        ax = plt.gca()
        # Get the required label from our list
        s = self.labels[self.counter]
        # Add the label to the upper centre of the diagonal plot
        ax.annotate(
            s, xy=(0.97, 0.85), transform=ax.transAxes, xycoords='axes fraction', va='center', ha='right', fontsize=30
        )
        # Increment the counter by one for the next plot
        self.counter += 1

    # noinspection PyUnusedLocal
    @staticmethod
    def _reg_coef(
            x: np.ndarray, y: np.ndarray, *args, **kwargs
    ) -> None:
        """
        Adds correlation coefficients and significance asterisks to diagonal plots, with fontsize scaled to coefficient.
        """
        # Get the current axis
        ax = plt.gca()
        # Calculate the correlation between the x and y variables
        r, p = stats.pearsonr(x, y)
        # Convert the p value to asterisks
        ast = vutils.get_significance_asterisks(p)
        # Create the format string, with additional escape characters for nicer kerning between asterisks
        if ast != '':
            s = "${{{}}}".format(round(r, 2)) + ''.join([r"^{{{}}}\,\!".format(a) for a in ast]) + "$"
        # If our correlation is not significant, we won't return any asterisks, so need to catch this
        else:
            s = "${{{}}}$".format(round(r, 2))
        # Add the format string to the middle of the axis, with fontsize scaled to match the absolute r value
        scaling = abs(r)
        if scaling < 0.3:
            scaling = 0.3
        ax.annotate(s, xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=80 * scaling)
        # Disable ticks on the diagonal plots
        ax.tick_params(which='both', axis='both', bottom=False, top=False, left=False, right=False)

    def _create_plot(
            self
    ) -> sns.PairGrid:
        """
        Creates the pair grid object, maps required plot functions, and returns for further modification.
        """
        # Create the pair grid plot
        g = sns.PairGrid(data=self.df, vars=self.vars, height=4.7, aspect=1, corner=False, dropna=True, despine=False,)
        # Map a histogram with KDE estimate to the diagonal plots
        g.map_diag(
            sns.histplot, color=vutils.BLACK, alpha=vutils.ALPHA, stat='count', bins=7, kde=True, line_kws={'lw': 5, }
        )
        # Map a scatter plot to the lower plots
        g.map_lower(sns.scatterplot, s=100, edgecolor=vutils.BLACK, facecolor='#FFFFFF', marker='o', linewidth=2)
        # Map a regression line of best fit to the lower plots
        g.map_lower(sns.regplot, scatter=False, color='#FF0000', ci=None, line_kws={'linewidth': 5})
        # Map the variable titles to the diagonal plots
        g.map_diag(self._add_label_to_diag)
        # Map the regression coefficients to the upper plots
        g.map_upper(self._reg_coef)
        # Return the object for further alteration
        return g

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects, including ticks, labels, spines etc.
        """
        # Iterate through all axis
        for ax in self.g.axes.flatten():
            # Turn off x and y labels
            ax.set(ylabel=None, xlabel=None)
            # Turn off both ticks; we'll add these back in for the required axes later
            ax.tick_params(width=0, length=0, which='both')
            # Set axis thickness
            plt.setp(ax.spines.values(), linewidth=2)
            # Add in the top and right axis to make the axis object a complete box
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
        # If we're using absolute slope, we need to define a different axis limit
        slope_lim = (0, 0.5) if self._abs_slope else (-0.5, 0.5)
        # Iterate through all rows and columns and set required axis limits and step values
        for i, lim, step in zip(range(0, 4), [slope_lim, (0, 100), (0, 400), (0, 10)], [3, 5, 5, 6]):
            ticks = np.linspace(lim[0], lim[1], step)
            self.g.axes[i, i].set(ylim=lim, xlim=lim, yticks=ticks, xticks=ticks)
        # If using absolute tempo slope, adjust axis limit slightly so we don't cut off some markers
        if self._abs_slope:
            self.g.axes[0, 0].set(xlim=(-0.01, 0.5))
        # Iterate through just the first column and last row and add the correct axis ticks back in
        for i in range(0, 4):
            self.g.axes[i, 0].tick_params(width=3, length=7, axis='y')
            self.g.axes[3, i].tick_params(width=3, length=7, axis='x')
        # The axis in the bottom left corner is the only one which requires ticks on both axes
        self.g.axes[3, 0].tick_params(width=3, length=7, axis='both')

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Adjust the spacing between subplots slightly
        self.g.figure.subplots_adjust(wspace=0.15, hspace=0.15, top=0.97, bottom=0.05, left=0.05, right=0.97)


def generate_all_metrics_plots(
    mds: list, output_dir: str,
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\all_metrics_plots'

    pp = PairPlotAllVariables(df=df, output_dir=figures_output_dir)
    pp.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_all_metrics_plots(mds=raw, output_dir=output)
