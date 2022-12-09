import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


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
        fname = f'{self.output_dir}\\heatmap_lagged_latency'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the heatmap
        """
        # We use this double for-loop to get separate x and y values to subset our axes object
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
        # Set color bar label
        self.cbar_ax.set_ylabel(
            'Average increase in IOI variation (ms) \nper 1ms increase in latency variation', fontsize=20
        )
        # Adjust plot spacing a bit
        self.fig.subplots_adjust(bottom=0.10, top=0.9, right=0.8, left=0.1)


class LinePlotLaggedLatency(vutils.BasePlot):
    """
    Plots a grid of line plots showing the (partial) correlation between latency and
    IOI variability across each jitter level, duo, and musician.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = kwargs.get('var', 'ioi_std_vs_jitter_partial_correlation')
        self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside of the class to generate the figure and save
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        # Create filename and return to save
        fname = f'{self.output_dir}\\lineplot_lagged_latency'
        return self.g.figure, fname

    def _format_df(
            self,
    ) -> pd.DataFrame:
        """
        Coerces the dataframe into the required format for plotting
        """
        # Explode all of our coefficients into a dataframe
        all_coefs = pd.DataFrame(self.df[self.var].to_list())
        # Get a list of column names and set them to our new dataframe
        cols = [f'lag_{s}' for s in range(0, all_coefs.shape[1])]
        all_coefs.columns = cols
        # Concatenate the coefficient dataframe to our primary dataframe
        big_df = pd.concat([self.df, all_coefs], axis=1)
        # Get the list of columns to melt on
        melters = ['trial', 'block', 'latency', 'jitter', 'instrument']
        # Melt the dataframe, group by instrument, jitter level, and trial, and get mean coefficient at each lag
        big_df = (
            big_df.melt(id_vars=melters, value_vars=cols, value_name='coef', var_name='lag')
                  .sort_values(by=melters)
                  .reset_index(drop=True)
        )
        big_df = big_df[big_df['jitter'] != 0]
        all_duos = big_df.copy(deep=True)
        all_duos['trial'] = 6
        return pd.concat([big_df, all_duos], axis=0, ignore_index=True)

    def _create_plot(
            self
    ) -> sns.FacetGrid:
        """
        Creates the facetgrid object and returns
        """
        return sns.relplot(
            data=self.df, col="trial", row='jitter', hue="instrument", hue_order=['Keys', 'Drums'],
            kind='line', aspect=1.625, x='lag', y='coef', errorbar='se', palette=vutils.INSTR_CMAP, height=2.1, lw=3,
        )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects
        """
        # Add the horizontal line in at y=0 to use as our x axis
        self.g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, lw=2)
        # Set a load of axes properties
        self.g.set(
            xticks=[num for num in range(0, 9)], xticklabels=[num for num in range(0, 9)],
            ylabel='', xlabel='', title='', ylim=(-0.35, 0.35)
        )
        # Iterate through every ax
        for ax in self.g.axes.flatten():
            # Set the position of our x axis to the horizontal line we created above
            ax.spines['bottom'].set_position(('data', 0))
            # Adjust the width of our ticks and axes
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
        # Iterate through just the first column of plots and add the y axis label
        for i in range(1, self.g.axes.shape[0] + 1):
            self.g.axes[i - 1, 0].set_ylabel(f'{i / 2}x jitter', fontsize=vutils.FONTSIZE)
        # Iterate through just the first row of plots and add the duo title
        for i in range(self.g.axes.shape[1]):
            if i == self.g.axes.shape[1] - 1:
                self.g.axes[0, i].set_title(f'Average')
            else:
                self.g.axes[0, i].set_title(f'Duo {i + 1}')

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Add the large y and x axes labels to the whole figure
        self.g.figure.supylabel('Correlation ($r$)', x=0.01)
        self.g.figure.supxlabel('Lag (s)', y=0.03)
        # Remove the bottom spine of all the plots
        self.g.despine(left=False, bottom=True)
        # Move the legend positioning
        sns.move_legend(self.g, 'center right', ncol=1, title='Instrument')
        # Adjust the subplot layout
        self.g.figure.subplots_adjust(right=0.89, left=0.09, bottom=0.12, top=0.9, hspace=0.1, wspace=0.1)


def generate_tempo_stability_plots(
        mds: list, output_dir: str
):
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\ioi_variability_plots'
    lp = LinePlotLaggedLatency(df=df, output_dir=figures_output_dir)
    lp.create_plot()


# import scipy.stats as stats
# import numpy as np
# df = []
# for pcm in raw:
#     df.append(pcm.keys_dic)
#     df.append(pcm.drms_dic)
# df = pd.DataFrame(df)
# lp = LinePlotLaggedLatency(df=df,)
# res = []
# for i, g in lp.df.groupby(['jitter', 'instrument', 'lag']):
#     boot = stats.bootstrap((g.coef.values,), np.mean, confidence_level=0.95, n_resamples=1000)
#     res.append((*i, *boot.confidence_interval, np.mean(g.coef.values)))
# res = pd.DataFrame(res, columns=['jitter', 'instrument', 'lag', 'low', 'high', 'actual'])
# g = sns.relplot(data=res, col='jitter', hue='instrument', hue_order=['Keys', 'Drums'], kind='line', x='lag', y='actual', marker='o', errorbar=None, palette=vutils.INSTR_CMAP, lw=3)
# INSTR_CMAP = ['#00ff00', '#9933ff', ]
# g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, lw=2)
#
# for num, (i_, grp_) in enumerate(res.groupby('jitter')):
#     for num_, (i, grp) in enumerate(grp_.groupby('instrument', sort=False)):
#         g.axes[0, num].tick_params(width=3, )
#         plt.setp(g.axes[0, num].spines.values(), linewidth=2)
#         g.axes[0, num].spines['bottom'].set_position(('data', 0))
#         g.axes[0, num].fill_between(grp['lag'], grp['high'], grp['low'], alpha=0.1, color=INSTR_CMAP[num_])
#         g.axes[0, num].set(xticks=[num for num in range(0, 9)], xticklabels=[num for num in range(0, 9)], ylabel='', xlabel='', title=f'Jitter: {i_}x',)
# g.figure.supxlabel('Lag (s)')
# g.figure.supylabel('Correlation ($r$)')
# sns.move_legend(g, 'center right', ncol=1, title='Instrument')
# g.figure.subplots_adjust(right=0.85, left=0.12, bottom=0.12, top=0.9, hspace=0.2)
# plt.show()

if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    generate_tempo_stability_plots(mds=raw, output_dir=output)
