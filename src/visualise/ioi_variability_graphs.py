import numpy as np
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
        self.df = self._bootstrap_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

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
        return self.fig, fname

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
        return big_df

    def _bootstrap_df(
            self, quantiles: tuple[int] = (0.05, 0.95), func=np.mean
    ) -> pd.DataFrame:
        """
        Create a dataframe showing actual mean and boostrapped confidence intervals of lag effects at each jitter level.
        Unit of resampling is the individual performer.
        """
        # Create the pivot table: each row is the data for an individual musician, with columns the coefficient at each
        # amount of lag across both 0.5 and 1.0x jitter scales. As such, coefficients are averaged across conditions
        # with the same amount of latency, e.g. 0.5x/lag_4 contains coefficients obtained for each musician for both
        # 45ms and 180ms latency at 4 seconds of lag
        reshape = self.df.pivot_table(index=['trial', 'instrument'], columns=['jitter', 'lag'], values='coef',
                                      aggfunc=func)
        # Create a list of new dataframes by resampling with replacement (frac=1 means returned df is 100% of length of
        # original). Mean function is used to get the row-wise mean of coefficients, i.e. across musicians, not lag
        samples = [reshape.sample(frac=1, replace=True, random_state=n).mean(axis=0) for n in range(0, 1000)]
        # Combine all of our resampled dataframes together
        boot = pd.concat(samples, axis=1)
        # Extract the 5% and 95% quantile from our samples, as well as the actual mean
        boot_low = boot.quantile(quantiles[0], axis=1).rename('low')
        boot_mu = reshape.mean(axis=0).rename('mean')
        boot_high = boot.quantile(quantiles[1], axis=1).rename('high')
        # Concatenate all the dataframes together
        boot = pd.concat([boot_low, boot_mu, boot_high], axis=1).reset_index(drop=False)
        # Change the formatting of the jitter column to make plotting easier
        boot['jitter'] = boot['jitter'].astype(str) + 'x'
        return boot

    def _create_plot(
            self
    ) -> plt.Axes:
        """
        Create the basic line plot in seaborn and return
        """
        return sns.lineplot(
            data=self.df, hue='jitter', x='lag', y='mean', errorbar=None, palette=vutils.JITTER_CMAP,
            lw=3, ax=self.ax, style='jitter', markers=['o', 's', '^'], markersize=10
        )

    def _format_ax(
            self
    ) -> None:
        """
        Adjust axes-level parameters
        """
        # Fill between the confidence intervals
        for num, (_, jit) in enumerate(self.df.groupby('jitter')):
            self.ax.fill_between(jit['lag'], jit['low'], jit['high'], alpha=vutils.ALPHA, color=vutils.JITTER_CMAP[num])
        # Set axes parameters - labels, axes limits etc
        ticks = [n for n in range(0, 9)]
        self.g.set(
            ylim=(-0.1, 0.3), xlabel=None, ylabel=None, xticks=ticks, xticklabels=ticks
        )
        # Adjust axes parameters
        self.ax.tick_params(width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Set bottom spine to y=0
        self.ax.spines['bottom'].set_position(('data', 0))
        # Hide right and top spines that Seaborn adds by default
        for spine in ['right', 'top']:
            self.ax.spines[spine].set_visible(False)

    def _format_fig(
            self
    ) -> None:
        """
        Adjust figure-level parameters
        """
        # Add axes labels
        self.fig.supxlabel('Lag (s)')
        self.fig.supylabel('Correlation ($r$)')
        # Move legend
        sns.move_legend(
            self.g, 'center right', ncol=1, title='Jitter', frameon=False, bbox_to_anchor=(1.3, 0.5)
        )
        # Adjust padding around plot edges
        self.fig.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.9)


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


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    generate_tempo_stability_plots(mds=raw, output_dir=output)
