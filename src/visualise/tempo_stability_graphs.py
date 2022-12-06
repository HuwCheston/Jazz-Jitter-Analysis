import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


@vutils.plot_decorator
def regplot_ioi_std_vs_tempo_slope(
        df: pd.DataFrame, output_dir: str, xvar='ioi_std'
) -> tuple[plt.Figure, str]:
    """
    Creates a regression plot of tempo stability (default metric is median of windowed IOI standard deviations)
    vs tempo slope. Hue of scatterplot corresponds to duo number, marker style to instrument type.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 5))
    plt.rcParams.update({'font.size': vutils.FONTSIZE})
    ax = sns.regplot(data=df, x=xvar, y='tempo_slope', scatter=False, color=vutils.BLACK, ax=ax)
    ax = sns.scatterplot(data=df, x=xvar, y='tempo_slope', hue='trial', palette='tab10', style='instrument', s=100,
                         ax=ax)
    ax.tick_params(width=3, )
    ax.set(ylabel='', xlabel='')
    plt.setp(ax.spines.values(), linewidth=2)
    # Plot a horizontal line at x=0
    ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK)
    # Set axis labels
    fig.supylabel('Tempo slope (BPM/s)', x=0.01)
    fig.supxlabel('Tempo stability (ms)' if xvar == 'ioi_std' else xvar, y=0.12)
    # Format axis positioning and move legend
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:6] + handles[7:], labels=labels[1:6] + labels[7:], ncol=8, frameon=False,
              markerscale=1.6, columnspacing=0.8, bbox_to_anchor=(1, -0.18), )
    ax.figure.subplots_adjust(bottom=0.25, top=0.95, left=0.12, right=0.95)
    fname = f'{output_dir}\\regplot_{xvar}_vs_tempo_slope.png'
    return fig, fname


class HeatmapLaggedLatency(vutils.BasePlot):
    """
    Creates a facetgrid of heatmaps showing mean coefficients for each level of lag, stratified by instrument, jitter
    and trial.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = kwargs.get('var', 'ioi_std_vs_jitter_coefficients')
        if self.df is not None:
            self.df = self._format_df(self.df)
        self.fig, self.ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9.4, 9.4))
        self.cbar_ax = self.fig.add_axes([0.82, 0.2, 0.02, 0.6])

    def _format_df(self, df):
        """
        Formats the dataframe for the plot
        """
        # Explode all of our coefficients into a dataframe
        all_coefs = pd.DataFrame(df[self.var].to_list())
        # Get a list of column names and set them to our new dataframe
        cols = [f'lag_{s}' for s in range(1, all_coefs.shape[1] + 1)]
        all_coefs.columns = cols
        # Concatenate the coefficient dataframe to our primary dataframe
        big_df = pd.concat([df, all_coefs], axis=1)
        # Get the list of columns to melt on
        melters = ['trial', 'block', 'latency', 'jitter', 'instrument']
        # Melt the dataframe, group by instrument, jitter level, and trial, and get mean coefficient at each lag
        big_df = (
            big_df.melt(id_vars=melters, value_vars=cols, value_name='coef', var_name='lag')
                  .sort_values(by=melters)
                  .reset_index(drop=True)
                  .groupby(['jitter', 'instrument', 'trial', 'lag'])
                  .mean()
                  .reset_index(drop=False)
                  .drop(columns=['block', 'latency'])
        )
        # Subset to remove values where no jitter was applied; these coefficients will all be 0 anyway
        return big_df[big_df['jitter'] != 0]

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to create the plot and save with plot decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\heatmap_lagged_latency.png'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the heatmap
        """
        # We use this double for-loop to get separate x and y values for subsetting our axes object
        for x, (jit, grp1) in enumerate(self.df.groupby('jitter')):
            grp1 = grp1.sort_values(by='instrument', ascending=False)
            for y, (ins, grp2) in enumerate(grp1.groupby('instrument', sort=False)):
                # Pivot to get data in correct format for the heatmap
                avg = (
                    grp2.drop(columns=['jitter', 'instrument'])
                        .set_index('trial')
                        .pivot(columns='lag')
                )
                # Create the heatmap
                g = sns.heatmap(
                    data=avg, ax=self.ax[x, y], annot=True, center=0, cbar=x == 0 and y == 0, cbar_ax=self.cbar_ax,
                    fmt='.2f', cmap='vlag', vmin=-0.3, vmax=0.3
                )
                # Easiest to set title now as we have our instrument and jitter values already
                g.set(title=f'{ins}\n{jit}x jitter' if x == 0 else f'{jit}x jitter')

    def _format_ax(self):
        """
        Formats axes-level objects
        """
        # Iterate through all our axes
        for a in self.ax.flatten():
            # Set axes, tick labels
            a.set(
                xlabel='', ylabel='', yticklabels=[duo for duo in self.df['trial'].unique()],
                xticklabels=[num + 1 for num in range(0, self.df['lag'].nunique())]
            )
            # By default seaborn heatmap adds rotation to tick labels, so remove this
            a.tick_params(axis='both', rotation=0)
            # Iterate through minimum and maximum x and y ticks
            for x, y in zip([0, self.df['lag'].nunique()], [0, self.df['trial'].nunique()]):
                a.axhline(y=y, color=vutils.BLACK, lw=4)
                a.axvline(x=x, color=vutils.BLACK, lw=4)

    def _format_fig(self):
        """
        Formats figure-level objects
        """
        # Set figure labels
        self.fig.supxlabel('Lag (s)', y=0.02)
        self.fig.supylabel('Duo number')
        # Set cbar label
        self.cbar_ax.set_ylabel(
            'Average increase in IOI variation (ms) \nper 1ms increase in latency variation', fontsize=20
        )
        # Adjust plot spacing a bit
        self.fig.subplots_adjust(bottom=0.10, top=0.9, right=0.8, left=0.1)


class PointPlotLaggedLatency(vutils.BasePlot):
    """
    Make a pointplot showing lagged timestamps on x-axis and regression coefficients on y. Columns grouped by trial,
    rows grouped by jitter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = kwargs.get('var', 'ioi_std_vs_jitter_coefficients')
        if self.df is not None:
            self.df = self._format_df(self.df)

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_facetgrid()
        self._format_ax()
        self._format_fig()
        # Create filename and return to save
        fname = f'{self.output_dir}\\pointplot_{self.var}.png'
        return self.g.figure, fname

    def _format_df(self, df):
        """
        Formats the dataframe for the plot
        """
        all_coefs = pd.DataFrame(df[self.var].to_list())
        cols = [f'lag_{s}' for s in range(0, all_coefs.shape[1])]
        all_coefs.columns = cols
        big_df = (
            pd.concat([df, all_coefs], axis=1)
              .melt(id_vars=['trial', 'block', 'latency', 'jitter', 'instrument', ],
                    value_vars=cols, value_name='coef', var_name='lag')
              .sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument'])
        )
        big_df = big_df[big_df['jitter'] != 0]
        return big_df.reset_index(drop=True)

    def _create_facetgrid(self):
        """
        Creates the facetgrid and maps plots onto it
        """
        return sns.catplot(
            data=self.df, col='trial', row='jitter', x="lag", y="coef", hue='instrument',
            kind="point", height=5.5, aspect=0.62, dodge=0.2, palette=vutils.INSTR_CMAP, hue_order=['Keys', 'Drums'],
            scale=1.25, estimator=np.median, ci=None
        )

    def _format_ax(self):
        """
        Formats plot subplots and axes
        """
        # Add the reference line now or else it messes up the titles
        self.g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK)
        # Set axes tick parameters and line width
        self.g.set_axis_labels(x_var='', y_var='')
        for ax in self.g.axes.flatten():
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        """
        Formats the overall figure
        """
        # Format titles, labels
        self.g.figure.supxlabel('Lag (s)', y=0.04)
        self.g.figure.supylabel('Coefficient, jitter vs. coupling', x=0.007)
        self.g.set_titles('{col_name}, Jitter: {row_name}x')
        # Set figure properties
        self.g.figure.subplots_adjust(bottom=0.1, top=0.95, left=0.07, right=0.97)
        sns.move_legend(self.g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.01))


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
