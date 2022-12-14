import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from statistics import mean
from matplotlib import animation, pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


class PairGrid(vutils.BasePlot):
    """
    Creates a pairgrid plot for a given x and colour variable.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xvar: str = kwargs.get('xvar', 'correction_partner_onset')
        self.xlim: tuple = kwargs.get('xlim', (-1.5, 1.5))
        self.xlabel: str = kwargs.get('xlabel', None)
        self.cvar: str = kwargs.get('cvar', 'tempo_slope')
        self.clabel: str = kwargs.get('clabel', 'Tempo\nslope\n(BPM/s)\n\n')
        # If we've passed our dataframe
        if self.df is not None:
            self.df = self._format_df()
            self.norm = vutils.create_normalised_cmap(self.df[self.cvar])

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_facetgrid()
        self._format_pairgrid_ax()
        self._format_pairgrid_fig()
        fname = f'{self.output_dir}\\pairgrid_condition_vs_{self.xvar}_vs_{self.cvar}'
        return self.g.fig, fname

    def _format_df(self) -> pd.DataFrame:
        """
        Formats dataframe by adding abbreviation column
        """
        self.df['abbrev'] = self.df['latency'].astype('str') + 'ms/' + round(self.df['jitter'], 1).astype('str') + 'x'
        return self.df.sort_values(by=['latency', 'jitter'])

    def _create_facetgrid(self):
        """
        Creates facetgrid object and plots stripplot
        """
        return sns.catplot(
            data=self.df, x=self.xvar, y='abbrev', row='block', col='trial', hue='instrument', sharex=True, sharey=True,
            hue_order=['Keys', 'Drums'], palette=vutils.INSTR_CMAP, kind='strip', height=5.5, marker='o', aspect=0.62,
            s=100, jitter=False, dodge=False
        )

    def _format_pairgrid_ax(self):
        """
        Formats each axes within a pairgrid
        """
        # Add the reference line first or it messes up the plot titles
        self.g.refline(x=0, alpha=1, linestyle='-', color=vutils.BLACK)
        for num in range(0, 5):
            ax1 = self.g.axes[0, num]
            ax2 = self.g.axes[1, num]
            # When we want different formatting for each row
            ax1.set_title(f'Repeat 1\nDuo {num + 1}' if num == 2 else f'\nDuo {num + 1}', fontsize=vutils.FONTSIZE + 3)
            ax1.set(ylabel='', xlim=self.xlim, )
            ax1.tick_params(bottom=False)
            ax2.set_title(f'Repeat 2' if num == 2 else '', fontsize=vutils.FONTSIZE + 3)
            ax2.set(ylabel='', xlabel='', xlim=self.xlim)
            # When we want the same formatting for both rows
            for x, ax in enumerate([ax1, ax2]):
                # Disable the grid
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)
                # Add the span, shaded according to tempo slope
                d = self.df[
                    (self.df['trial'] == num + 1) & (self.df['block'] == x + 1) & (self.df['instrument'] == 'Keys')
                ].sort_values(['latency', 'jitter'])[self.cvar]
                for n in range(0, 13):
                    coef = vutils.SLOPES_CMAP(self.norm(d.iloc[n]))
                    ax.axhspan(n - 0.5, n + 0.5, facecolor=coef)

    def _format_pairgrid_fig(self) -> None:
        """
        Formats figure-level attributes for a horizontal pairgrid of all conditions
        """
        # Format the figure
        self.g.despine(left=True, bottom=True)
        self.g.fig.supxlabel(self.xlabel, x=0.505, y=0.05)
        self.g.fig.supylabel('Condition', x=0.01)
        # Add the color bar
        position = self.g.fig.add_axes([0.94, 0.2, 0.01, 0.6])
        self.g.fig.colorbar(vutils.create_scalar_cbar(norm=self.norm), cax=position, ticks=vutils.CBAR_BINS)
        position.text(0.0, 0.3, self.clabel, fontsize=vutils.FONTSIZE + 3)  # Super hacky way to add a title...
        # Add the legend
        sns.move_legend(
            self.g, loc='lower center', ncol=2, title=None, frameon=False, markerscale=1.5, fontsize=vutils.FONTSIZE + 3
        )
        # Adjust the plot spacing
        self.g.fig.subplots_adjust(bottom=0.12, top=0.93, wspace=0.15, left=0.11, right=0.93)


class BoxPlot(vutils.BasePlot):
    """
    Creates a figure showing correction to partner coefficients obtained for each performer in a duo, stratified by a
    given variable (defaults to jitter scale). By default, the control condition is included in this plot,
    but this can be excluded by setting optional argument subset to True.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subset: bool = kwargs.get('subset', False)
        self.height: float = kwargs.get('height', 5)
        # Get plot variables
        self.xvar: str = kwargs.get('xvar', 'jitter')
        self.yvar: str = kwargs.get('yvar', 'correction_partner_onset')
        self.ylim: tuple = kwargs.get('ylim', (-1, 1))
        self.ylabel: str = kwargs.get('ylabel', 'Coupling constant')
        # If we're removing 0 latency conditions
        if self.subset:
            self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        # Create the plot
        self.g = self._create_facetgrid()
        # Format the axes and figure
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\boxplot_{self.yvar}_vs_{self.xvar}'
        return self.g.figure, fname

    def _format_df(self):
        """
        Remove 0 latency (control) conditions
        """
        return self.df[self.df['latency'] != 0]

    def _create_facetgrid(self):
        """
        Create the plot and return
        """
        return sns.catplot(
            data=self.df, x=self.xvar, y=self.yvar, col='trial', hue='instrument', kind='box', sharex=True,
            sharey=True, palette=vutils.INSTR_CMAP, boxprops=dict(linewidth=3, ), aspect=0.68,
            whiskerprops=dict(linestyle='-', linewidth=3), flierprops={'markersize': 10}, height=self.height,
        )

    def _format_ax(self):
        """
        Adjust axes-level parameters
        """
        # Add the reference line first or it messes up the plot titles
        self.g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, linewidth=3)
        # Set axes labels, limits
        self.g.set(ylim=self.ylim, xlabel='', ylabel='')
        # Iterate through to set tick and axes line widths
        for num, ax in enumerate(self.g.axes.flatten()):
            ax.set_title(f'Duo {num + 1}')
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        """
        Adjust figure-level parameters
        """
        # Format the x and y axes labels
        self.g.figure.supxlabel(self.xvar.title(), y=0.1, )
        self.g.figure.supylabel(self.ylabel if self.ylabel is not None else self.yvar.replace('_', ' ').title(), x=0.01)
        # Move the legend
        sns.move_legend(self.g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.01), )
        # Adjust the plot size a bit
        self.g.figure.subplots_adjust(bottom=0.25, top=0.9, left=0.06, right=0.98)


class SingleConditionPlot:
    """
    Generate a nice plot of a single performance, showing actual and predicted tempo slope, relative phase adjustments,
    phase correction coefficients to partner and self for both performers
    """
    def __init__(self, **kwargs):
        plt.rcParams.update({'font.size': vutils.FONTSIZE})
        # Get pianist data
        self.keys_df: pd.DataFrame = kwargs.get('keys_df', None)
        self.keys_md: sm.regression.linear_model.RegressionResults = kwargs.get('keys_md', None)
        self.keys_o: pd.DataFrame = kwargs.get('keys_o', None)
        # Get drummer data
        self.drms_df: pd.DataFrame = kwargs.get('drms_df', None)
        self.drms_md: sm.regression.linear_model.RegressionResults = kwargs.get('drms_md', None)
        self.drms_o: pd.DataFrame = kwargs.get('drms_o', None)
        # Get additional data
        self.output_dir: str = kwargs.get('output_dir', None)
        self.metadata: tuple = kwargs.get('metadata', None)
        # Create the matplotlib objects - figure with grid spec, 5 subplots
        self.fig = plt.figure(figsize=(9.4, 15))
        self.ax = vutils.get_gridspec_array(fig=self.fig)
        self.rsquared = mean([self.keys_md.rsquared, self.drms_md.rsquared])
        # Create attributes for plots
        self.polar_num_bins = 15
        self.polar_xt = np.pi / 180 * np.linspace(-90, 90, 5, endpoint=True)

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        # Plot onto the different parts of the grid spec
        self._plot_coefficients()
        self._plot_polar()
        self._plot_slopes()
        # Format the figure and save
        self.fig.suptitle(f'Duo {self.metadata[0]} (repeat {self.metadata[1]}): '
                          f'latency {self.metadata[2]}ms, jitter {self.metadata[3]}x')
        fname = f'{self.output_dir}\\duo{self.metadata[0]}_repeat{self.metadata[1]}' \
                f'_latency{self.metadata[2]}_jitter{self.metadata[3]}'
        return self.fig, fname

    def _plot_coefficients(self):
        """
        For a single condition plot, generate the coefficient stripplots
        """
        # Format the data
        corr = (
            pd.concat([self.keys_md.params, self.drms_md.params], axis=1)
              .rename(columns={0: 'Keys', 1: 'Drums'})
              .transpose()
              .rename(columns={'my_prev_ioi': 'Self', 'asynchrony': 'Partner'})
              .reset_index(drop=False)
        )
        self.ax[1, 1].sharex(self.ax[1, 0])
        # Iterate through data
        for num, st in zip(range(0, 2), ['Keys', 'Drums']):
            drms = corr[corr['index'] == st].melt().loc[2:]
            g = sns.stripplot(
                ax=self.ax[1, num], x='value', y='variable', data=drms, jitter=False, dodge=False, marker='o',
                s=12, color=vutils.INSTR_CMAP[0] if st == 'Keys' else vutils.INSTR_CMAP[1]
            )
            g.set_xlim((-1.5, 1.5))
            g.yaxis.grid(True)
            g.xaxis.grid(False)
            g.axvline(x=0, alpha=1, linestyle='-', color=vutils.BLACK)
            sns.despine(ax=g, left=True, bottom=True)
            g.text(0.05, 0.9, s=st, transform=g.transAxes, ha='center', )
            g.set_ylabel('')
            if num == 0:
                g.axes.set_yticks(g.axes.get_yticks(), g.axes.get_yticklabels())
                g.axes.set_yticklabels(labels=g.axes.get_yticklabels(), va='center')
                g.set_title('Phase Correction', x=1)
                g.set_xlabel('Coefficient', x=1)
            else:
                g.axes.set_yticklabels('')
                g.set_xlabel('')

    def _plot_polar(self):
        """
        For a plot of a single performance, create two polar plots showing phase difference between performers
        """
        # Create the polar plots
        self.ax[0, 1].sharex(self.ax[0, 0])
        for num, (ins, st) in enumerate(zip([self.keys_df, self.drms_df], ['Keys', 'Drums'])):
            ax = self.ax[0, num]
            dat = self._format_polar_data(ins)
            ax.bar(x=dat.idx, height=dat.val, width=0.05, label=st,
                   color=vutils.INSTR_CMAP[0] if st == 'Keys' else vutils.INSTR_CMAP[1])
            self._format_polar_ax(ax=ax, sl=vutils.BLACK, yt=[0, max(ax.get_yticks())])
            ax.set(ylabel='', xlabel='')
            x0, y0, x1, y1 = ax.get_position().get_points().flatten().tolist()
            ax.set_position(
                [x0 - 0.05 if num == 0 else x0 - 0.15, y0 - 0.03, x1 - 0.1 if num == 0 else x1 - 0.2, y1 - 0.5]
            )
            ax.set_facecolor('#FFFFFF')
            ax.text(0, 0.8, s=st, transform=ax.transAxes, ha='center', )
            if num == 0:
                ax.text(1.1, 0.05, s="√Density", transform=ax.transAxes, ha='center', )
                ax.text(-0.2, 0.5, s="Relative Phase (πms)", transform=ax.transAxes, va='center', rotation=90)
                ax.set_title('Relative Phase to Partner', x=1.1)

    def _format_polar_data(self, df: pd.DataFrame,) -> pd.DataFrame:
        """
        Formats the data required for a polar plot by calculating phase of live musician relative to delayed, subsetting
        result into given number of bins, getting the square-root density of these bins.
        """
        # Calculate the amount of phase correction for each beat
        corr = df.asynchrony * -1
        # Cut into bins
        cut = pd.cut(corr, self.polar_num_bins, include_lowest=False).value_counts().sort_index()
        # Format the dataframe
        cut.index = pd.IntervalIndex(cut.index.get_level_values(0)).mid
        cut = pd.DataFrame(cut, columns=['asynchrony']).reset_index(drop=False).rename(
            columns={'asynchrony': 'val', 'index': 'idx'})
        # Multiply the bin median by pi to get number of degrees
        cut['idx'] = cut['idx'] * np.pi
        # Take the square root of the density for plotting
        cut['val'] = np.sqrt(cut['val'])
        return cut

    def _format_polar_ax(self, ax: plt.Axes, sl, yt: list = None) -> None:
        """
        Formats a polar subplot by setting axis ticks and gridlines, rotation properties, and facecolor
        """
        # Used to test if we've provided x ticks already
        test_none = lambda t: [] if t is None else t
        # Set the y axis ticks and gridlines
        ax.set_yticks(test_none(yt))
        ax.set_xticks(test_none(self.polar_xt))
        ax.grid(False)
        # Set the polar plot rotation properties, direction, etc
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # Set the color according to the tempo slope
        ax.patch.set_facecolor(sl)

    def _plot_slopes(self):
        """
        For a single condition plot, generate the tempo slopes graphs.
        """
        ax = self.ax[2, 0]
        # Get actual and predicted rolling slopes
        actual = autils.average_bpms(self.keys_o, self.drms_o)
        predicted = autils.average_bpms(self.keys_df, self.drms_df, elap='elapsed', bpm='predicted_bpm')
        # Plot actual and predicted slopes
        for df, lab in zip((actual, predicted), ('Actual', 'Fitted')):
            ax.plot(df['elapsed'], df['bpm_rolling'], label=lab, linewidth=2)
        # Add metronome line to 120BPM
        ax.axhline(y=120, color=vutils.BLACK, linestyle='--', alpha=vutils.ALPHA, label='Metronome Tempo', linewidth=2)
        ax.text(8, 40, s=f'$r^2$ = {round(self.rsquared, 2)}',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.48, -0.28), title=None, frameon=False, )
        ax.set(xlabel='Performance duration (s)', ylabel='Average tempo (BPM, 8-seconds rolling)', ylim=(30, 160),
               title='Tempo Slope')


class SingleConditionAnimation:
    """
    Creates an animation of actual and predicted tempo slope that should(!) be synchronised to the AV_Manip videos.
    Default FPS is 30 seconds, with data interpolated so plotting look nices and smooth. This can be changed in vutils.
    WARNING: this will take a really long time to complete!!!
    """
    def __init__(self, **kwargs):
        plt.set_loglevel('critical')
        plt.rcParams.update({'font.size': vutils.FONTSIZE})
        # Get attributes from kwargs
        self.keys_df: pd.DataFrame = kwargs.get('keys_df', None)
        self.drms_df: pd.DataFrame = kwargs.get('drms_df', None)
        self.keys_o: pd.DataFrame = kwargs.get('keys_o', None)
        self.drms_o: pd.DataFrame = kwargs.get('drms_o', None)
        self.output_dir: str = kwargs.get('output_dir', None)
        self.metadata: tuple = kwargs.get('metadata', None)
        # Create the matplotlib objects we need
        self.fig = plt.figure(figsize=(7.5, 7.5))
        self.ax = plt.axes(
            xlabel='Performance duration (s)', ylabel='Average tempo (BPM, four-beat rolling window)', xlim=(0, 100),
            ylim=(30, 160), title=f'Duo {self.metadata[0]} (measure {self.metadata[1]}): '
                                  f'latency {self.metadata[2]}ms, jitter {self.metadata[3]}x'
        )
        self.act_line, = self.ax.plot([], [], lw=2, label='Actual tempo')
        self.pred_line, = self.ax.plot([], [], lw=2, label='Fitted tempo')
        # Create additional objects we need
        self.output = vutils.create_output_folder(self.output_dir)
        self.chain = lambda k, d, e, b: vutils.interpolate_df_rows(
            vutils.append_count_in_rows_to_df(autils.average_bpms(df1=k, df2=d, elap=e, bpm=b))
        )
        self.act = self.chain(self.keys_o, self.drms_o, 'elapsed', 'bpm')
        self.pred = self.chain(self.keys_df, self.drms_df, 'elapsed', 'predicted_bpm')

    def init(self,):
        """
        Sets initial state of animations
        """
        self.act_line.set_data([], [])
        self.pred_line.set_data([], [])
        return self.act_line, self.pred_line,

    def animate(self, i: int, ):
        """
        Animates the plot
        """
        f = lambda d: d[d.index <= i]
        a, p = f(self.act), f(self.pred)
        self.act_line.set_data(a['elapsed'].to_numpy(), a['bpm_rolling'].to_numpy())
        self.pred_line.set_data(p['elapsed'].to_numpy(), p['bpm_rolling'].to_numpy())
        return self.act_line, self.pred_line,

    def _format_ax(self):
        """
        Set the axes parameters
        """
        self.ax.tick_params(axis='both', which='both', bottom=False, left=False, )
        self.ax.axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome tempo')
        self.ax.legend()
        plt.tight_layout()

    def create_animation(self):
        """
        Create the animation and save to our directory
        """
        self._format_ax()
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, frames=int(self.act.index.max()),
                                       interval=1000 / vutils.VIDEO_FPS, blit=True)
        anim.save(
            f'{self.output}\\duo{self.metadata[0]}_measure{self.metadata[1]}'
            f'_latency{self.metadata[2]}_jitter{self.metadata[3]}.mp4',
            writer='ffmpeg', fps=vutils.VIDEO_FPS
        )


class RegPlot(vutils.BasePlot):
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.xvars: list[str] = kwargs.get('xvars', ['tempo_slope', 'ioi_std'])
        self.xlabs: list[str] = kwargs.get('xlabs', ['Tempo slope (BPM/s)', 'Tempo stability (ms)'])
        self.fig, self.ax = plt.subplots(1, 2, figsize=(18.8, 9.4), sharex=True, sharey=False)
        self.df = self._format_df(df)

    def _format_df(self, df):
        groupers = ['trial', 'block', 'latency', 'jitter']
        cols = groupers + ['coupling_balance', *self.xvars]
        abs_correction = lambda grp: abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        return pd.DataFrame(
            [[*i, abs_correction(g), *(g[v].mean() for v in self.xvars)] for i, g in df.groupby(groupers)], columns=cols
        )

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\regplot_coupling_balance'
        return self.fig, fname

    def _create_plot(self):
        for num, var in enumerate(self.xvars):
            _ = sns.scatterplot(
                data=self.df, x='coupling_balance', y=var, hue='trial', s=100, style='trial',
                palette=vutils.DUO_CMAP, ax=self.ax[num], legend=None if num == 1 else True
            )
            _ = sns.regplot(
                data=self.df, x='coupling_balance', y=var, x_ci=95, n_boot=100, scatter=None, lowess=True,
                truncate=True, color=vutils.BLACK, ax=self.ax[num], line_kws={'linewidth': 3},
            )

    def _format_ax(self):
        for a, lab in zip(self.ax.flatten(), self.xlabs):
            a.tick_params(width=3, )
            plt.setp(a.spines.values(), linewidth=2)
            a.set(xlabel='Coupling balance', ylabel=lab)
            a.axhline(y=0, linestyle='-', alpha=0.3, color=vutils.BLACK, lw=3)

    def _format_fig(self):
        sns.move_legend(
            self.ax[0], 'center right', ncol=1, title='Duo', frameon=False, bbox_to_anchor=(2.3, 0.5),
            markerscale=1.6, columnspacing=0.2, handletextpad=0.1,
        )
        self.fig.subplots_adjust(left=0.07, right=0.93, bottom=0.1, top=0.95, wspace=0.15)


class BarPlot(vutils.BasePlot):
    """
    Creates a plot showing the coupling coefficients per instrument and duo, designed to look similar to fig 2.(c)
    in Jacoby et al. (2021). However, by default this plot will use the median as an estimator of central tendency,
    rather than mean, due to outlying values. This can be changed by setting the estimator argument to a different
    function that can be called by seaborn's barplot function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.estimator: callable = kwargs.get('estimator', np.median)
        self.yvar: str = kwargs.get('yvar', 'correction_partner')
        self.ylim: tuple[float] = kwargs.get('ylim', (0., 1.5))
        self.ylabel: str = kwargs.get('ylabel', 'Partner coupling')
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_{self.yvar}_vs_instrument'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the two plots
        """
        ax = sns.stripplot(
            data=self.df, x='trial', y=self.yvar, hue='instrument', dodge=True,
            palette='dark:' + vutils.BLACK, s=4, marker='.', jitter=0.1, ax=self.ax
        )
        ax = sns.barplot(
            data=self.df, x='trial', y=self.yvar, hue='instrument', ax=ax, errorbar=('ci', 25),
            palette=vutils.INSTR_CMAP, hue_order=['Keys', 'Drums'], errcolor='#3953a3', errwidth=10,
            estimator=self.estimator, edgecolor=vutils.BLACK, lw=2
        )
        return ax

    def _format_ax(self):
        """
        Set axes-level formatting
        """
        # Set ax formatting
        self.g.tick_params(width=3, )
        self.g.set(ylabel='', xlabel='', ylim=self.ylim)
        plt.setp(self.g.spines.values(), linewidth=2)
        # Plot a horizontal line at x=0
        self.g.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK, linewidth=2)

    def _format_fig(self):
        """
        Set figure-level formatting
        """
        # Set figure labels
        self.fig.supylabel(self.ylabel, x=0.02, y=0.55)
        self.fig.supxlabel('Duo', y=0.02)
        # Format the legend to remove the handles/labels added automatically by the strip plot
        handles, labels = self.g.get_legend_handles_labels()
        self.g.get_legend().remove()
        plt.legend(handles[2:], labels[2:], ncol=1, title='Instrument', frameon=False, bbox_to_anchor=(1, 0.65),
                   markerscale=1.6)
        # Adjust the figure a bit and return for saving in decorator
        self.g.figure.subplots_adjust(bottom=0.15, top=0.95, left=0.14, right=0.8)


class HistPlotR2(vutils.BasePlot):
    """
    Creates histograms of model parameters stratified by trial and instrument, x-axis variable defaults to R-squared
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xvar: str = kwargs.get('xvar', 'r2')
        self.kind: str = kwargs.get('kind', 'hist')

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._format_plot()
        fname = f'{self.output_dir}\\histplot_{self.xvar}'
        return self.g.figure, fname

    def _create_plot(self):
        """
        Creates the plot in seaborn and returns
        """
        return sns.displot(
            self.df, col='trial', kind=self.kind, x=self.xvar, hue="instrument", multiple="stack",
            palette=vutils.INSTR_CMAP, height=vutils.HEIGHT, aspect=vutils.ASPECT,
        )

    def _format_plot(self):
        """
        Format figure and axes level properties (titles, labels, legend)
        """
        self.g.set(xlabel='', ylabel='')
        self.g.set_titles("Duo {col_name}", size=vutils.FONTSIZE)
        self.g.figure.supxlabel(self.xvar.title(), y=0.06)
        self.g.figure.supylabel('Count', x=0.007)
        # Move legend and adjust subplots
        sns.move_legend(self.g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.03),
                        fontsize=vutils.FONTSIZE)
        self.g.figure.subplots_adjust(bottom=0.17, top=0.92, left=0.055, right=0.97)


class BoxPlotR2WindowSize(vutils.BasePlot):
    """
    Creates a boxplot of average R2 values per rolling window size
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.df is not None:
            self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._format_plot()
        fname = f'{self.output_dir}\\boxplot_r2_vs_windowsize'
        return self.g.figure, fname

    def _format_df(self):
        """
        Formats dataframe by dropping nan values, grouping by trial and window size, extracting mean...
        """
        return (
            self.df.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .groupby(['trial', 'win_size'])
                .mean()[['aic', 'r2']]
                .reset_index(drop=False)
        )

    def _create_plot(self):
        """
        Creates the plot in seaborn and returns
        """
        return sns.boxplot(data=self.df, hue='trial', x='win_size', y='r2', color=vutils.INSTR_CMAP[0])

    def _format_plot(self):
        """
        Format figure and axes level properties
        """
        self.g.set(xlabel='Window Size (s)', ylabel='Adjusted R2')
        plt.tight_layout()


class RegPlotGrid(vutils.BasePlot):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xvars: list[str] = kwargs.get('xvars', ['tempo_slope', 'ioi_std', 'pw_asym', 'success'])
        self.xlabs: list[str] = kwargs.get(
            'xlab', ['Tempo slope (BPM/s)', 'IOI variability (SD, ms)', 'Asynchrony (RMS, ms)', 'Self-reported success']
        )
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(18.8, 18.8))
        self.lw = kwargs.get('lw', 5)
        self.ci_frac: float = kwargs.get('ci_frac', 0.66666667)
        self.ci_it: int = kwargs.get('ci_int', 3)
        self.n_boot: int = kwargs.get('n_boot', 1000)
        self.hand, self.lab = None, None

    def _format_df(self):
        test = self.df[self.df['latency'] != 0]
        gps = ['trial', 'latency', 'jitter']
        cols = gps + ['coupling_balance', *self.xvars]
        abs_correction = lambda grp: abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        test = pd.DataFrame(
            [[*i, abs_correction(g, ), *(g[v].mean() for v in self.xvars)] for i, g in test.groupby(gps)], columns=cols
        )
        return test.melt(id_vars=[*gps, 'coupling_balance'], value_vars=self.xvars)

    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        plt.show()

    def _create_plot(self):
        for a, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('variable', sort=False)):
            _ = sns.scatterplot(
                data=grp, x='coupling_balance', y='value', hue='latency', ax=a, palette=vutils.DUO_CMAP,
                style='latency', s=250
            )
            if idx == 'tempo_slope':
                self._add_lowess(ax=a, grp=grp)
            else:
                sns.regplot(
                    data=grp, x='coupling_balance', y='value', ax=a, scatter=False, scatter_kws={'alpha': vutils.ALPHA},
                    color=vutils.BLACK, n_boot=self.n_boot, line_kws={'lw': self.lw},
                )

    def _add_lowess(self, ax: plt.Axes, grp: pd.DataFrame, add_hline: bool = True):
        def lowess(x, y, it, frac):
            return sm_lowess(x, y, it=it, frac=frac, return_sorted=True).T

        # Add in actual line from raw data
        act_x, act_y = lowess(grp['value'], grp['coupling_balance'], self.ci_it, self.ci_frac)
        ax.plot(act_x, act_y, color=vutils.BLACK, lw=self.lw)
        # Bootstrap confidence intervals
        res_x, res_y = [], []
        for n in range(0, self.n_boot):
            # Resample the dataframe, with replacement (resampling unit = performance)
            boot = grp.sample(frac=1, replace=True, random_state=n)
            # Get the lowess model and append to our lists
            sm_x, sm_y = lowess(boot['value'], boot['coupling_balance'], self.ci_it, self.ci_frac)
            res_x.append(sm_x)
            res_y.append(sm_y)
        # Format the resampled dataframes
        fmt = lambda d: pd.DataFrame(d).transpose().reset_index(drop=True)
        fmt_x = fmt(res_x)
        fmt_y = fmt(res_y)
        # Extract the quantiles: median x value, 0.05 and 0.95 quantiles from y
        x_5 = fmt_x.quantile(0.5, axis=1)
        y_05 = fmt_y.quantile(0.05, axis=1)
        y_95 = fmt_y.quantile(0.95, axis=1)
        # Fill between the quantiles
        ax.fill_between(x_5, y_05, y_95, color=vutils.BLACK, alpha=0.2)
        # Add horizontal line at y=0 if required
        if add_hline:
            ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK, linewidth=self.lw)

    def _format_ax(self):
        for ax, lab in zip(self.ax.flatten(), self.xlabs):
            self.hand, self.lab = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_ylabel(lab, fontsize=20)
            ax.set(xlabel='', xticks=np.linspace(0, 1.2, 5))
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.supxlabel('Coupling balance', y=0.02)
        leg = self.fig.legend(self.hand, self.lab, 'center right', ncol=1, title='Latency (ms)', frameon=False,
                              markerscale=2, prop={'size': 20})
        plt.setp(leg.get_title(), fontsize=20)
        self.fig.subplots_adjust(left=0.09, right=0.88, bottom=0.07, top=0.93, wspace=0.2, hspace=0.2)


def generate_phase_correction_plots(
    mds: list, output_dir: str,
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\phase_correction_plots'
    # Create regression table
    vutils.output_regression_table(
        mds=autils.create_model_list(df=df, md=f'correction_partner~C(latency)+C(jitter)+C(instrument)',
                                     avg_groupers=['latency', 'jitter', 'instrument']),
        output_dir=figures_output_dir, verbose_footer=False
    )
    # Create box plot
    bp = BoxPlot(df=df, output_dir=figures_output_dir, yvar='correction_partner', ylim=(-0.2, 1.5))
    bp.create_plot()
    # Create pair grid
    pg = PairGrid(
        df=df, xvar='correction_partner', output_dir=figures_output_dir, xlim=(-0.5, 1.5), xlabel='Coupling constant'
    )
    pg.create_plot()
    # Create regression plots
    rp = RegPlot(df=df, output_dir=figures_output_dir)
    rp.create_plot()
    bar = BarPlot(df=df, output_dir=figures_output_dir)
    bar.create_plot()
    bar = BarPlot(df=df, output_dir=figures_output_dir, yvar='correction_self', ylabel='Self coupling', ylim=(-1, 1))
    bar.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_phase_correction_plots(mds=raw, output_dir=output)
