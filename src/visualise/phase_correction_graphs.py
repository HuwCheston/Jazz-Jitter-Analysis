"""Code for generating plots from the phase correction models"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from statsmodels.stats.multitest import multipletests
from matplotlib import animation, pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
import statsmodels.formula.api as smf
import warnings
from itertools import combinations

from src.analyse.phase_correction_models import PhaseCorrectionModel
import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils

# Define the modules we can import from this file in others
__all__ = [
    'generate_phase_correction_plots'
]


class PairGrid(vutils.BasePlot):
    """
    Creates a pairgrid plot for a given x and colour variable.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xvar: str = kwargs.get('xvar', 'correction_partner_onset')
        self.xlim: tuple = kwargs.get('xlim', (-1.5, 1.5))
        self.xlabel: str = kwargs.get('xlabel', None)
        self.average: bool = kwargs.get('average', True)
        self.cvar: str = kwargs.get('cvar', 'tempo_slope')
        self.clabel: str = kwargs.get('clabel', 'Tempo slope\n(BPM/s)\n\n')
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
        if self.average:
            fname += '_average'
        return self.g.fig, fname

    def _format_df(self) -> pd.DataFrame:
        """
        Formats dataframe by adding abbreviation column
        """
        if self.average:
            self.df = self.df.groupby(['trial', 'latency', 'jitter', 'instrument']).mean().reset_index(drop=False)
        self.df['abbrev'] = self.df['latency'].astype('str') + 'ms/' + round(self.df['jitter'], 1).astype('str') + 'x'
        return self.df.sort_values(by=['latency', 'jitter'])

    def _create_facetgrid(self):
        """
        Creates facetgrid object and plots stripplot
        """
        return sns.catplot(
            data=self.df, x=self.xvar, y='abbrev', row='block', col='trial', hue='instrument', sharex=True, sharey=True,
            hue_order=['Keys', 'Drums'], palette=vutils.INSTR_CMAP, kind='strip', height=5.5, marker='o', aspect=0.62,
            s=125, jitter=False, dodge=False
        )

    def _format_pairgrid_ax(self):
        """
        Formats each axes within a pairgrid
        """
        # Add the reference line first or it messes up the plot titles
        self.g.refline(x=0, alpha=1, linestyle='-', lw=3, color=vutils.BLACK)
        if self.average:
            self.g.refline(y=12.5, alpha=1, linestyle='-', lw=3, color=vutils.BLACK)
            for num in range(0, 5):
                ax = self.g.axes[0, num]
                ax.set_title(f'Duo {num + 1}', fontsize=vutils.FONTSIZE + 3)
                ax.set(ylabel='', xlim=self.xlim, ylim=(12.5, -0.5), xlabel='')
                ax.tick_params(width=3, )
                ax.yaxis.grid(lw=2, alpha=vutils.ALPHA)
                # Add the span, shaded according to tempo slope
                d = self.df[
                    (self.df['trial'] == num + 1) & (self.df['instrument'] == 'Keys')
                ].sort_values(['latency', 'jitter'])[self.cvar]
                for n in range(0, 13):
                    coef = vutils.SLOPES_CMAP(self.norm(d.iloc[n]))
                    ax.axhspan(n - 0.5, n + 0.5, facecolor=coef)
        else:
            for num in range(0, 5):
                ax1 = self.g.axes[0, num]
                ax2 = self.g.axes[1, num]
                # When we want different formatting for each row
                ax1.set_title(
                    f'Repeat 1\nDuo {num + 1}' if num == 2 else f'\nDuo {num + 1}', fontsize=vutils.FONTSIZE + 3
                )
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
        self.g.fig.supxlabel(self.xlabel, x=0.505, y=0.01 if self.average else 0.05)
        self.g.fig.supylabel('Condition (latency/jitter)', x=0.01)
        # Add the color bar
        if self.average:
            position = self.g.fig.add_axes([0.9, 0.1, 0.01, 0.4])
            sns.move_legend(
                self.g, loc='right', ncol=1, title='Instrument', frameon=False, markerscale=2,
                fontsize=vutils.FONTSIZE + 3, bbox_to_anchor=(1.0, 0.8),
            )
            self.g.fig.subplots_adjust(bottom=0.14, top=0.9, wspace=0.17, left=0.11, right=0.885)
            position.text(0, 0.16, self.clabel, fontsize=vutils.FONTSIZE)
        else:
            position = self.g.fig.add_axes([0.94, 0.2, 0.01, 0.6])
            sns.move_legend(
                self.g, loc='lower center', ncol=2, title=None, frameon=False, markerscale=1.5,
                fontsize=vutils.FONTSIZE + 3
            )
            self.g.fig.subplots_adjust(bottom=0.12, top=0.93, wspace=0.15, left=0.11, right=0.93)
            position.text(0.0, 0.3, self.clabel, fontsize=vutils.FONTSIZE + 3)
        cb = self.g.fig.colorbar(vutils.create_scalar_cbar(norm=self.norm), cax=position, ticks=vutils.CBAR_BINS)
        cb.outline.set_linewidth(1.5)
        cb.ax.tick_params(width=3)


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
        self.rsquared = np.mean([self.keys_md.rsquared, self.drms_md.rsquared])
        # Create attributes for plots
        self.polar_num_bins = 15
        self.polar_xt = np.pi / 180 * np.linspace(-90, 90, 5, endpoint=True)

    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        # Plot onto the different parts of the grid spec
        self._plot_coefficients()
        self._plot_polar()
        self._plot_slopes()
        # Format the figure and save
        self.fig.suptitle(f'Duo {self.metadata[0]} (session {self.metadata[1]}): '
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
              .rename(columns={'my_prev_ioi_diff': 'Self', 'asynchrony': 'Partner'})
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
                g.axes.set_yticks(g.axes.get_yticks())
                g.axes.set_yticklabels(labels=g.axes.get_yticklabels(), va='center')
                g.set_title('Phase Correction', x=1)
                g.set_xlabel('Coupling', x=1)
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
        corr = df['asynchrony'] * -1
        # Cut into bins
        cut = pd.cut(corr, self.polar_num_bins, include_lowest=False).value_counts().sort_index()
        # Format the dataframe
        cut.index = pd.IntervalIndex(cut.index.get_level_values(0)).mid
        cut = (
            pd.DataFrame(cut, columns=['asynchrony'])
              .reset_index(drop=False)
              .rename(columns={'asynchrony': 'val', 'index': 'idx'})
        )
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
        bpm = autils.average_bpms(self.keys_o, self.drms_o)
        # Plot actual and predicted slopes
        ax.plot(
            bpm['elapsed'], bpm['bpm_rolling'], linewidth=2, label='Performance Tempo'
        )
        # Add metronome line to 120BPM
        ax.axhline(y=120, color=vutils.BLACK, linestyle='--', alpha=vutils.ALPHA, label='Metronome Tempo', linewidth=2)
        ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.48, -0.28), title=None, frameon=False, )
        ax.set(xlabel='Duration (s)', ylabel='Average tempo (BPM, 8-seconds rolling)', ylim=(30, 160),
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
    """
    Deprecated(?) class for creating regression plots between multiple variables
    """
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
                markers=vutils.DUO_MARKERS, palette=vutils.DUO_CMAP, ax=self.ax[num], legend=None if num == 1 else True
            )
            _ = sns.regplot(
                data=self.df, x='coupling_balance', y=var, x_ci=95, n_boot=vutils.N_BOOT, scatter=None, lowess=True,
                truncate=True, color=vutils.BLACK, ax=self.ax[num], line_kws={'linewidth': 3},
            )

    def _format_ax(self):
        for a, lab in zip(self.ax.flatten(), self.xlabs):
            a.tick_params(width=3, )
            plt.setp(a.spines.values(), linewidth=2)
            a.set(xlabel='Coupling asymmetry', ylabel=lab)
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
        self.estimator: callable = kwargs.get('estimator', np.mean)
        self.errorbar: bool = kwargs.get('errorbar', True)
        self.stripplot: bool = kwargs.get('stripplot', False)
        self.yvar: str = kwargs.get('yvar', 'correction_partner')
        self.ylim: tuple[float] = kwargs.get('ylim', (0., 1))
        self.ylabel: str = kwargs.get('ylabel', 'Coupling coefficient')
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
        if self.stripplot:
            sns.stripplot(
                data=self.df, x='trial', y=self.yvar, hue='instrument', dodge=True,
                palette='dark:' + vutils.BLACK, s=6, marker='.', jitter=0.1, ax=self.ax
            )
        if self.errorbar:
            kwargs = dict(
                errorbar=('ci', 95), errcolor=vutils.BLACK, errwidth=2, n_boot=vutils.N_BOOT, seed=1, capsize=0.1,
            )
        else:
            kwargs = dict(errorbar=None)
        ax = sns.barplot(
            data=self.df, x='trial', y=self.yvar, hue='instrument', ax=self.ax, palette=vutils.INSTR_CMAP,
            hue_order=['Keys', 'Drums'], estimator=self.estimator, edgecolor=vutils.BLACK, lw=2, width=0.8,
            **kwargs
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
        if self.stripplot:
            handles = handles[2:]
            labels = labels[2:]
        plt.legend(
            handles, labels, ncol=1, title='Instrument', frameon=False, bbox_to_anchor=(1, 0.65), markerscale=1.6
        )
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
    Creates a grid of scatter and regression plots, showing relationship between primary variables (tempo slope, ioi
    variability, asynchrony, self-reported success) and coupling balance. Regression is linear for all but tempo slope,
    which is lowess. Distributions are shown in marginal plots.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define plotting attributes
        self.lw = kwargs.get('lw', 5)   # Width of plotted lines
        self.remove_control = kwargs.get('remove_control', False)    # Remove results from control condition
        self.abs_slope: bool = kwargs.get('abs_slope', True)    # Use absolute tempo slope coefficient
        self.error_bar: str = kwargs.get('errorbar', 'sd')  # Type of error bar to plot: either sd or ci
        self.percentiles: tuple[float] = kwargs.get('percentiles', [2.5, 97.5])
        self.ci_frac: float = kwargs.get('ci_frac', 0.66666667)     # Fraction of data to use in lowess curve
        self.ci_it: int = kwargs.get('ci_int', 3)   # Iterations to use in lowess curve
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)   # Number of bootstraps
        self.hand, self.lab = None, None    # Empty variables that will be used to hold handles and labels of legend
        # Define variables to plot, both as they appear in the raw dataframe and the labels that should be on the plot
        self.xvars: list[str] = kwargs.get('xvars', ['tempo_slope', 'ioi_std', 'pw_asym', 'success'])
        self.xlabs: list[str] = kwargs.get(
            'xlab', [
                'Absolute tempo slope (BPM/s)', 'Timing irregularity (SD, ms)',
                'Asynchrony (RMS, ms)', 'Self-reported success'
            ]
        )
        # Format the dataframe
        self.df = self._format_df()
        # Create the figure, main axis, and marginal axis
        self.fig = plt.figure(figsize=(18.8, 18.8))
        self.main_ax, self.marginal_ax = self._init_gridspec_subplots()

    def _init_gridspec_subplots(
            self, widths: tuple[int] = (5, 1, 5, 1), heights: tuple[int] = (1, 5, 5)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialise grid of subplots. Returns two numpy arrays corresponding to main and marginal axes respectively.
        """
        # Initialise subplot grid with desired width and height ratios
        grid = self.fig.add_gridspec(nrows=3, ncols=4, width_ratios=widths, height_ratios=heights)
        # Create a list of index values for each subplot type
        margins = [0, 2, 5, 7, 9, 11]
        mains = [4, 6, 8, 10]
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
            self, success_diff: bool = False
    ) -> pd.DataFrame:
        """
        Coerce the dataframe into the correct format for plotting
        """
        def abs_correction(grp: pd.DataFrame.groupby, var: str = 'correction_partner'):
            ke = grp[grp['instrument'] == 'Keys'][var].mean()
            dr = grp[grp['instrument'] != 'Keys'][var].mean()
            return abs(ke - dr)

        # Create lists of columns for grouping
        gps = ['trial', 'latency', 'jitter']
        cols = gps + ['coupling_balance', *self.xvars]
        if self.abs_slope:
            self.df['tempo_slope'] = abs(self.df['tempo_slope'])
        # If we want the difference between self-reported success values
        if success_diff:
            test = pd.DataFrame(
                [
                    [*i, abs_correction(g), *(g[v].mean() if v != 'tempo_slope' else abs_correction(g, v)
                                              for v in self.xvars)
                     ] for i, g in self.df.groupby(gps)
                ], columns=cols
            )
        # If we want the mean of self-reported success values
        else:
            test = pd.DataFrame(
                [[*i, abs_correction(g), *(g[v].mean() for v in self.xvars)]
                 for i, g in self.df.groupby(gps)], columns=cols
            )
        # Melt the dataframe according to the x variables and return
        return test.melt(id_vars=[*gps, 'coupling_balance'], value_vars=self.xvars)

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate the plot and save in plot decorator
        """
        self._create_plot()
        self._format_main_ax()
        self._format_fig()
        # Format the marginal axes after the figure, otherwise this will affect their position
        self._format_marginal_ax()
        fname = f'{self.output_dir}\\regplot_grid_{self.error_bar}' + '_abs_slope' if self.abs_slope else ''
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Create the plots on both the main and marginal axes
        """
        # Define these variables now so we don't get warnings in PyCharm
        grp, sp = None, None
        # Iterate through our main axes, second row of marginal plots, and variables
        for a, m, (idx, grp) in zip(self.main_ax.flatten(), self.marginal_ax.flatten()[2:],
                                    self.df.groupby('variable', sort=False)):
            # Create the scatter plot on the main axis
            sp = sns.scatterplot(
                data=grp, x='coupling_balance', y='value', hue='trial', ax=a, palette=vutils.DUO_CMAP,
                style='trial', s=250, markers=vutils.DUO_MARKERS, edgecolor=vutils.BLACK
            )
            # Create the kde plot on the marginal axis
            kp = sns.kdeplot(
                data=grp, y='value', hue='trial', ax=m, palette=vutils.DUO_CMAP, legend=False, lw=2,
                multiple='stack', fill=True, common_grid=True, cut=0
            )
            # Set parameters of the kde plot now, as we have access to both it and the scatter plot
            kp.set(yticks=sp.get_yticks(), yticklabels=[], xticks=[], xlabel='', ylabel='', ylim=sp.get_ylim(), )
            # Add a standard linear regression line
            self._add_regression(ax=a, grp=grp)
            # Add a lowess curve to the tempo slope variable plot, with confidence intervals created with bootstrapping
            if idx == 'tempo_slope' and not self.abs_slope:
                self._add_lowess(ax=a, grp=grp)
        # Create the first row of marginal plots, showing distribution of coupling balance
        for m in self.marginal_ax.flatten()[:2]:
            # We can just use the last group and scatter plot here as the data is the same
            kp = sns.kdeplot(
                data=grp, x='coupling_balance', hue='trial', ax=m, palette=vutils.DUO_CMAP, legend=False, lw=2,
                multiple='stack', fill=True, common_grid=True,
            )
            kp.set(xticks=sp.get_xticks(), xticklabels=[], yticks=[], xlim=[-0.03, 1.03], xlabel='', ylabel='')

    def _add_regression(
            self, ax: plt.Axes, grp: pd.DataFrame
    ) -> None:
        """
        Adds in a linear regression fit to a single plot with bootstrapped confidence intervals
        """
        def regress(
                g: pd.DataFrame
        ) -> np.ndarray:
            # Coerce x variable into correct form and add constant
            x = sm.add_constant(g['coupling_balance'].to_list())
            # Coerce y into correct form
            y = g['value'].to_list()
            # Fit the model, predicting y from x
            md = sm.OLS(y, x).fit()
            # Return predictions made using our x vector
            return md.predict(x_vec)

        # Create the vector of x values we'll use when making predictions
        x_vec = sm.add_constant(
            np.linspace(grp['coupling_balance'].min(), grp['coupling_balance'].max(), len(grp['coupling_balance']))
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
        ax.plot(conc['x'], conc['y'], color=vutils.BLACK, lw=self.lw)
        ax.fill_between(conc['x'], conc['high'], conc['low'], color=vutils.BLACK, alpha=0.2)

    def _add_lowess(
            self, ax: plt.Axes, grp: pd.DataFrame, add_hline: bool = True
    ) -> None:
        """
        Adds in a lowess curve to a single plot with bootstrapped confidence intervals
        """
        def lowess(
                x: pd.Series, y: pd.Series, it: int, frac: float
        ) -> np.ndarray:
            """
            Returns a single lowess curve in the form of a numpy array
            """
            return sm_lowess(x, y, it=it, frac=frac, return_sorted=True).T

        # Add in actual lowess curve from raw data
        act_x, act_y = lowess(grp['value'], grp['coupling_balance'], self.ci_it, self.ci_frac)
        ax.plot(act_x, act_y, color=vutils.RED, lw=self.lw)
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
        fmt_y = fmt(res_y)
        # Create empty variable so we don't get errors
        conc = None
        # Extract the quantiles: median x value, 0.025 and 0.975 quantiles from y
        if self.error_bar == 'ci':
            # Get the upper and lower percentiles, row-wise
            y_025 = fmt_y.quantile(self.percentiles[0] / 100, axis=1)
            y_975 = fmt_y.quantile(self.percentiles[1] / 100, axis=1)
            # Concatenate the dataframes together
            conc = pd.concat(
                [pd.Series(act_x).rename('x'), pd.Series(act_y).rename('y'), y_025.rename('low'), y_975.rename('high')],
                axis=1
            )
        elif self.error_bar == 'sd':
            # Get the standard deviation of the bootstrap dataframe and concatenate against the original
            conc = pd.concat(
                [pd.Series(act_x).rename('x'), pd.Series(act_y).rename('y'), fmt_y.std(axis=1).rename('std')], axis=1
            )
            # Add and subtract the standard error to the original values
            conc['high'] = conc['y'] + conc['std']
            conc['low'] = conc['y'] - conc['std']
        # Apply some smoothing to the error bar
        conc.iloc[0:-5] = conc.iloc[0:-5].rolling(10, min_periods=1).mean()
        # Shade error bar in plot
        ax.fill_between(conc['x'], conc['low'], conc['high'], color=vutils.RED, alpha=0.2)
        # Add horizontal line at y=0 if required
        if add_hline and not self.abs_slope:
            ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK, linewidth=self.lw)

    def _format_main_ax(
            self
    ) -> None:
        """
        Applies axes-level formatting to main plots
        """
        # Set IOI variability y limit, as this has a tendency to change depending on which error bar type is used
        self.main_ax[1].set(ylim=(10, 70))
        # Iterate through main axes and labels
        for num, (ax, lab) in enumerate(zip(self.main_ax.flatten(), self.xlabs)):
            # Store our handles and labels here so we can refer to them later
            self.hand, self.lab = ax.get_legend_handles_labels()
            # Remove the legend
            ax.get_legend().remove()
            # Set parameters
            ax.set_ylabel(lab, fontsize=20)
            ax.set(xlabel='', xticks=np.linspace(0, 1.2, 5), xlim=[-0.03, 1.03],)
            # For the top row of plots, we don't want tick labels, so remove them
            if num < 2:
                ax.set_xticklabels([])
            # Adjust tick and axes thickness
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_marginal_ax(
            self,
    ) -> None:
        """
        Applies axes-level formatting to marginal plots
        """
        # Iterate through marginal axes, with a counter
        for i, ax in enumerate(self.marginal_ax.flatten()):
            # Apply formatting to all marginal plots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
            ax.set(yticklabels=[], xticklabels=[],)
            # Apply formatting only to top row of plots
            if i < 2:
                box = ax.get_position()
                box.y0 -= 0.03
                box.y1 -= 0.03
                ax.set_position(box)
                ax.spines['left'].set_visible(False)
            # Apply formatting only to bottom two rows of plots
            else:
                box = ax.get_position()
                box.x0 -= 0.02
                box.x1 -= 0.02
                ax.set_position(box)
                ax.spines['bottom'].set_visible(False)

    def _format_fig(
            self
    ) -> None:
        """
        Apply figure-level formatting to overall image
        """
        # Add in label to x axis
        self.fig.supxlabel('Coupling asymmetry', y=0.02)
        # Add in single legend, using attributes we saved when removing individual legends
        leg = self.fig.legend(
            self.hand, self.lab, loc='center', ncol=1, title='Duo', frameon=False, prop={'size': 20},
            markerscale=2, edgecolor=vutils.BLACK,
        )
        if self.abs_slope:
            leg.set_bbox_to_anchor((0.5, 0.45))
        for handle in leg.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([200])
        plt.setp(leg.get_title(), fontsize=20)
        # Adjust subplot positioning a bit: this will affect marginal positions, so we'll change these later
        self.fig.subplots_adjust(left=0.07, right=1.01, bottom=0.07, top=1.01, hspace=0.2, wspace=0.2)


class ArrowPlotPhaseCorrection(vutils.BasePlot):
    """
    Creates a plot showing the average strength, direction, and balance of the coupling within each duo.
    Designed to look somewhat similar to fig 2.(b) in Jacoby et al. (2021)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self.df.copy(deep=True)
        self.fig, self.ax = self._generate_subplots()

    @staticmethod
    def _generate_subplots(
            small_ax_ratio: float = 0.01
    ):
        """
        Creates subplots with an 'invisible' additional axis, for creating space between plots
        """
        fig, ax_ = plt.subplots(
            nrows=1, ncols=6, figsize=(18.8, 4), sharex=True, sharey=True,
            gridspec_kw=dict(width_ratios=[1, 1, small_ax_ratio, 1, 1, 1])
        )
        ax_[2].set_visible(False)
        return fig, ax_[[0, 1, 3, 4, 5]]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called outside of the class to create plot and save in plot decorator
        """
        self._create_plot()
        self._add_coordination_strategy_brackets()
        # self._add_significance_brackets()
        self._format_fig()
        fname = f'{self.output_dir}\\arrowplot_phase_correction'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the individual subplots
        """
        self.df['trial'] = pd.Categorical(self.df['trial'], [1, 3, 4, 5, 2])
        # Iterate through each ax and duo
        for a, (idx, duo) in zip(self.ax.flatten(), self.df.groupby('trial')):
            # Sort values for each duo
            duo = duo.sort_values(by=['instrument', 'block', 'latency', 'jitter']).reset_index(drop=False)
            # Get coupling constants for each instrument
            drms = duo[duo['instrument'] == 'Drums']['correction_partner'].to_list()
            keys = duo[duo['instrument'] != 'Drums']['correction_partner'].to_list()
            # Now we can zip together the values for each instrument and get the asymmetry and strength
            asym = np.mean([abs(k - d) for k, d in zip(keys, drms)])
            stre = np.mean([k + d for k, d in zip(keys, drms)])
            # Add in the title showing the duo number
            a.set_title(f'Duo {idx}', y=0.85)
            # Add text showing the absolute coupling balance
            a.text(
                0.5, 0.05, f'$Asymmetry$: +{asym:.2f}', ha='center',
                va='center', fontsize=vutils.FONTSIZE + 3
            )
            a.text(
                0.5, -0.05, f'$Strength$: {stre:.2f}', ha='center',
                va='center', fontsize=vutils.FONTSIZE + 3
            )
            # Turn the axis off
            a.axis('off')
            # Iterate through all colours, x values (x2), y values, and text
            for col, col2, col3, x1, x2, y, text, vals in zip(
                    vutils.INSTR_CMAP, vutils.INSTR_CMAP[::-1], [vutils.WHITE, vutils.BLACK],
                    [0, 1], [1, 0], [0.6, 0.35, ], ['Keys', 'Drums'], [drms, keys],
            ):
                # Create the arrow
                a.annotate(
                    '', xy=(x1, y), xycoords=a.transAxes, xytext=(x2, y), textcoords=a.transAxes,
                    arrowprops=dict(
                        edgecolor=vutils.BLACK, lw=1.5, facecolor=col2, mutation_scale=1,
                        width=np.mean(vals) * 15, shrink=0.1, headwidth=20
                    )
                )
                # Add in the mean and standard error
                a.text(
                    0.5, y + 0.1, f'{np.mean(vals):.2f} $({np.std(vals):.2f})$', ha='center', va='center',
                )
                # Add in the rectangle and text showing the instrument and coupling direction
                a.add_patch(
                    plt.Rectangle(
                        (x1 - 0.1, 0.175), width=0.2, height=0.65, clip_on=False, linewidth=3,
                        edgecolor=vutils.BLACK, transform=a.transAxes, facecolor=col
                    )
                )
                a.text(
                    x1, 0.49, text, rotation=90 if x1 == 0 else 270, ha='center', va='center',
                    color=col3, fontsize=vutils.FONTSIZE + 3
                )

    def _add_significance_brackets(
            self
    ) -> None:
        """
        Adds brackets showing confidence intervals for differences between groups
        """
        for ax_num, ci in zip([0, 2], ['[–0.00, 0.12]', '[–0.01, 0.14]']):
            self.ax[ax_num].annotate(
                r"$\Delta\bar{x}$ " + ci, xy=(1.1, -0.15), xytext=(1.1, -0.125), xycoords='axes fraction',
                fontsize=vutils.FONTSIZE, ha='center', va='bottom',
                arrowprops=dict(arrowstyle=f'-[, widthB=8, lengthB=-0.5', lw=2.0)
            )

    def _add_coordination_strategy_brackets(
            self
    ) -> None:
        """
        Adds brackets showing coordination strategies for different duos
        """
        for ax_num, x_pos, strategy, width in zip(
                [0, 3], [1.2, 0.5], ['Democracy', 'Leadership'], [9, 15]
        ):
            self.ax[ax_num].annotate(
                strategy, xy=(x_pos, 1.1), xytext=(x_pos, 1.1), xycoords='axes fraction',
                fontsize=vutils.FONTSIZE + 5, ha='center', va='bottom', fontweight='black',
                bbox=dict(boxstyle='sawtooth', fc='white', lw=2),
                arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=0.75', lw=3)
            )

    def _format_fig(
            self
    ) -> None:
        """
        Adjusts figure-level parameters
        """
        self.fig.subplots_adjust(right=0.95, left=0.05, top=0.8, bottom=0.1, wspace=0.4)


class ArrowPlotModelExplanation(vutils.BasePlot):
    """
    Generates a graphic that explains how the model works. Good luck understanding the code!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=5, figsize=(9.4, 5), sharex=True, sharey=True)
        self.points1 = [[0.5, 0.65], [0.45, 0.65], [0.5, 0.65], [0.53, 0.65], [0.48, 0.65]]
        self.points2 = [[0.55, 0.35], [0.53, 0.35], [0.49, 0.35], [0.55, 0.35], [0.59, 0.35]]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate the plot and save in decorator
        """
        self._create_plot()
        self._format_fig()
        fname = f'{self.output_dir}\\arrowplot_model_explanation'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Create the plot
        """
        # Iterate through each ax
        for n, a in enumerate(self.ax.flatten()):
            a.axis('off')
            if n == 0:
                a.text(1.1, 0.69, s=f'$I_{{k{n - 1}}}^{{1, 1}}$', ha='center', va='center',
                       fontsize=vutils.FONTSIZE - 3)
            elif n == 1:
                a.text(1.1, 0.69, s=f'$I_{{k}}^{{1, 1}}$', ha='center', va='center', fontsize=vutils.FONTSIZE - 3)
            elif n == 2:
                a.text(1.1, 0.69, s=f'$I_{{k+{n - 1}}}^{{1, 1}}$', ha='center', va='center',
                       fontsize=vutils.FONTSIZE - 3)
            a.axvline(x=0.5, ymin=0.3, ymax=0.7, lw=3, color=vutils.BLACK, alpha=vutils.ALPHA, zorder=5)
            a.add_patch(
                plt.Rectangle(
                    (0.25, 0.3), width=0.5, height=0.4, clip_on=False, linewidth=3, alpha=0.2,
                    edgecolor=vutils.BLACK, transform=a.transAxes, facecolor=vutils.BLACK, zorder=1
                )
            )
            a.annotate("", xytext=[self.points2[n][0] - 0.075, self.points2[n][1] - 0.04],
                       xy=[self.points2[n][0] + 0.325, self.points2[n][1] - 0.04],
                       arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=135,angleB=45", lw=3))
            if n == 0:
                a.text(self.points2[n][0] + 0.1, self.points2[n][1] - 0.1, s=f'$d_{{k{n - 1}}}$', ha='center',
                       va='center', fontsize=vutils.FONTSIZE - 3)
            elif n == 1:
                a.text(self.points2[n][0] + 0.1, self.points2[n][1] - 0.1, s=f'$d_{{k}}$', ha='center', va='center',
                       fontsize=vutils.FONTSIZE - 3)
            elif n == 2:
                a.text(self.points2[n][0] + 0.1, self.points2[n][1] - 0.1, s=f'$d_{{k+{n - 1}}}$', ha='center',
                       va='center', fontsize=vutils.FONTSIZE - 3)
            a.scatter(*self.points1[n], color=vutils.INSTR_CMAP[0], s=750, edgecolor=vutils.BLACK, zorder=10, lw=2)
            a.scatter(*self.points2[n], color=vutils.INSTR_CMAP[1], s=750, edgecolor=vutils.BLACK, zorder=10,
                      alpha=vutils.ALPHA, ls='dashed', lw=2)
            self.points2[n][0] += 0.25
            a.scatter(*self.points2[n], color=vutils.INSTR_CMAP[1], s=750, edgecolor=vutils.BLACK, zorder=10, lw=2)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
        for a1 in range(0, 5):
            self.points1[a1][0] -= 0.125
            self.points2[a1][0] -= 0.125
            con = ConnectionPatch(
                xyA=[self.points1[a1][0] - 0.05, self.points1[a1][1]],
                xyB=[self.points2[a1][0] - 0.05, self.points2[a1][1]],
                coordsA='data', coordsB='data', axesA=self.ax[a1],
                axesB=self.ax[a1], zorder=1000000, lw=3, arrowstyle='->',
            )
            self.ax[a1].add_artist(con)
            if a1 < 4:
                con = ConnectionPatch(
                    xyA=self.points1[a1], xyB=self.points1[a1 + 1],
                    coordsA='data', coordsB='data', axesA=self.ax[a1],
                    axesB=self.ax[a1 + 1], lw=3, arrowstyle='->',
                )
                self.ax[a1].add_artist(con)

    def _format_fig(self):
        self.fig.text(
            0.17, 0.55, 'Live\ninstrument', fontsize=vutils.FONTSIZE, va='center', ha='right',
            color=vutils.INSTR_CMAP[0]
        )
        self.fig.text(0.17, 0.25, 'Delayed\ninstrument', fontsize=vutils.FONTSIZE, va='center', ha='right',
                      color=vutils.INSTR_CMAP[1])
        self.fig.text(0.6, 0.03, 'Time', fontsize=vutils.FONTSIZE, va='center', ha='center', color=vutils.BLACK,
                      rotation=0)
        self.ax[0].arrow(0.45, 0.2, 5.75, 0, clip_on=False, transform=self.ax[0].transAxes, lw=3, head_width=0.03,
                         head_length=0.1, facecolor=vutils.BLACK)
        self.ax[0].annotate(f'$I_{{k-1}}^{{1, 1}}-I_{{k}}^{{1, 1}}$ ', xy=(1.70, 0.75), xytext=(1.70, 0.8),
                            xycoords='axes fraction', fontsize=vutils.FONTSIZE, ha='center', va='bottom',
                            bbox=dict(boxstyle='square', fc='white', ls='-', lw=0),
                            arrowprops=dict(arrowstyle='-[, widthB=3.26, lengthB=0.6', lw=2.0))
        self.ax[1].annotate(f'$I_{{k}}^{{1, 1}}-I_{{k+1}}^{{1, 1}}$ ', xy=(1.70, 0.75), xytext=(1.70, 0.8),
                            xycoords='axes fraction',
                            fontsize=vutils.FONTSIZE, ha='center', va='bottom',
                            bbox=dict(boxstyle='square', fc='white', ls='-', lw=3),
                            arrowprops=dict(arrowstyle='-[, widthB=3.26, lengthB=0.6', lw=2.0, ))
        self.ax[1].annotate(f'$α^{{1, 1}}$', xy=(1.1, 0.95), xytext=(1.1, 0.95), xycoords='axes fraction',
                            fontsize=vutils.FONTSIZE + 3, ha='center', va='bottom',
                            bbox=dict(boxstyle='square', fc='white'),
                            arrowprops=dict(arrowstyle='-[, widthB=3.26, lengthB=1.45', lw=4.0))
        self.ax[1].arrow(0.45, 0.955, 0.1, 0, lw=0, clip_on=False, head_width=0.05, head_length=0.15,
                         facecolor=vutils.BLACK)
        self.ax[1].arrow(1.45, 0.955, 0.1, 0, lw=0, clip_on=False, head_width=0.05, head_length=0.15,
                         facecolor=vutils.BLACK)
        self.ax[1].arrow(1.45, 0.5, 0, 0.25, lw=4, head_width=0.1, head_length=0.025, clip_on=False,
                         facecolor=vutils.BLACK)
        self.ax[1].arrow(0.7, 0.5, 0.2, 0, lw=4, clip_on=False, head_width=0.025, head_length=0.1,
                         facecolor=vutils.BLACK)
        self.ax[1].annotate(f'$α^{{1, 2}}$', xy=(1.1, 0.95), xytext=(1.1, 0.5), fontsize=vutils.FONTSIZE + 3,
                            xycoords='axes fraction', bbox=dict(boxstyle='square', facecolor=vutils.WHITE),
                            zorder=100000)
        self.ax[1].text(0.65, 0.515, s=f'$I_{{k}}^{{1, 2}}$', ha='center', va='center', fontsize=vutils.FONTSIZE - 3)
        self.fig.subplots_adjust(bottom=-0.15, top=0.95, right=0.99, left=0.15, wspace=0.4)


class BarPlotCouplingStrengthAsymmetryComparison(vutils.BasePlot):
    """
    Creates grouped barplots showing comparisons of coupling strength and asymmetry between pairwise duo combinations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df()
        self.vars = kwargs.get('vars', ['coupling_strength', 'coupling_asymmetry'])
        self.fig, self.ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(18.8, 10))

    def _format_df(self):
        gps = ['trial', 'latency', 'jitter']
        cols = gps + ['coupling_asymmetry', 'coupling_strength']
        tot_correction = lambda grp: grp.iloc[1]['correction_partner'] + grp.iloc[0]['correction_partner']
        abs_correction = lambda grp: abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        df_f = self.df.groupby(['trial', 'instrument', 'latency', 'jitter']).mean().reset_index()
        return pd.DataFrame(
            [[*i, abs_correction(g), tot_correction(g)] for i, g in df_f.groupby(gps)], columns=cols
        )

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_coupling_strength_asymmetry_comparison'
        return self.fig, fname

    def _create_plot(self):
        for var, col in zip(self.vars, range(0, len(self.vars))):
            li = [1, 0.3, 0.3, 0.3, 0.3]
            comp = mc.MultiComparison(self.df[var], self.df['trial'])
            tab, _, a2 = comp.allpairtest(stats.ttest_ind, method="bonf")
            a2 = pd.DataFrame(a2)
            for i in range(1, 5):
                df_plot = self.df.copy(deep=True)
                df_plot.loc[df_plot['trial'] < i, var] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = sns.barplot(
                        data=df_plot, x='trial', y=var, palette=vutils.DUO_CMAP, errorbar=('ci', 95),
                        errcolor=vutils.BLACK, errwidth=2, edgecolor=vutils.BLACK, lw=2, estimator=np.mean,
                        n_boot=vutils.N_BOOT, seed=1, capsize=0.1, width=0.4, ax=self.ax[i - 1, col],
                    )
                for bar, alpha in zip(g.containers[0], li):
                    bar.set_alpha(alpha)
                li.insert(i, li.pop(i - 1))
                g.set_title(var.split('_')[-1].title() if i == 1 else '')
                y = 1.1 + (i * 0.1)
                for idx, row in a2[a2['group1'] == i].iterrows():
                    pval = vutils.get_significance_asterisks(row['pval_corr'])
                    g.text(x=(((row['group2'] - 1) + (i - 1)) / 2) - 0.1, y=y - 0.05, s=pval, fontsize=vutils.FONTSIZE)
                    g.plot([i - 1, row['group2'] - 1], [y, y], color=vutils.BLACK, lw=2)
                    g.plot([i - 1, i - 1], [y, y - 0.04], color=vutils.BLACK)
                    g.plot([row['group2'] - 1, row['group2'] - 1], [y, y - 0.04], color=vutils.BLACK)
                    y += 0.2

    def _format_ax(self):
        for ax in self.ax.flatten():
            ax.set(xlabel='', ylabel='', ylim=(0, 2))
            ax.tick_params(width=3, which='major')
            ax.tick_params(width=0, which='minor')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.supxlabel('Duo')
        self.fig.supylabel('Coupling', x=0.01)
        self.fig.subplots_adjust(top=0.925, bottom=0.075, right=0.95, left=0.05, wspace=0.05, hspace=0.15)


class PointPlotCouplingStrengthAsymmetry(vutils.BasePlot):
    """
    Creates a plot showing the mean differences between coupling and strength
    for each pair of duos, with confidence intervals obtained via bootstrapping
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18.8, 4))
        self.df = self._format_df()
        self.boot_df = self._generate_bootstraps()

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting
        """
        # Define grouper and column variables
        gps = ['trial', 'latency', 'jitter']
        cols = gps + ['coupling_asymmetry', 'coupling_strength']
        # Define functions for extracting total and absolute correction difference
        tot_correction = lambda grp: grp.iloc[1]['correction_partner'] + grp.iloc[0]['correction_partner']
        abs_correction = lambda grp: abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        # Group to get the average correction per duo, instrument and condition
        df_f = self.df.groupby(['trial', 'instrument', 'latency', 'jitter']).mean().reset_index()
        # Return the absolute and total correction for each condition
        return pd.DataFrame(
            [[*i, abs_correction(g), tot_correction(g)] for i, g in df_f.groupby(gps)], columns=cols
        )

    def _generate_bootstraps(
            self,
    ) -> pd.DataFrame:
        """
        Generate bootstrapped confidence intervals for the mean difference in coupling
        across all duos.
        """
        res = []
        # Iterate through both variables
        for v, nu in zip(['coupling_strength', 'coupling_asymmetry'], range(0, 2)):
            # Iterate through every combination of duos
            for g1, g2 in combinations(self.df['trial'].drop_duplicates().to_list(), 2):
                # Subset dataframe to get just the variable and duos we want
                a1 = self.df[self.df['trial'] == g1][v]
                a2 = self.df[self.df['trial'] == g2][v]
                # Get the actual mean difference
                mea = a2.mean() - a1.mean()
                low, high = vutils.bootstrap_mean_difference(a1, a2)
                # Compute our t-test and extract our p-value
                _, p = stats.ttest_rel(a1, a2)
                # Append the results to our list
                res.append((f'{g1} – {g2}', low, high, mea, v, p))
        # Convert the list to a dataframe
        df = pd.DataFrame(res, columns=['group', 'low', 'high', 'mean', 'variable', 'pval'])
        # Generate our column of asterisks after correcting our p-values
        df['ast'] = self._format_pvals(df=df)
        return df

    @staticmethod
    def _format_pvals(
            df: pd.DataFrame
    ) -> list:
        """
        Format our t-test p-values by correcting them and converting into asterisks
        """
        # Correct our p-values using bonferroni correction
        rejected, p_adjusted, _, alpha_corrected = multipletests(
            df['pval'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False
        )
        # Convert p-values to asterisks
        return [vutils.get_significance_asterisks(p) for p in p_adjusted]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\pointplot_coupling_asymmetry_strength'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the scatter and line plots
        """
        # Iterate through both variables and axis
        for v, ax in zip(['coupling_strength', 'coupling_asymmetry'], self.ax.flatten()):
            # Subset the bootstrapped data for just the variable we want
            sub = self.boot_df[self.boot_df['variable'] == v].reset_index(drop=True)
            # Create the scatter plot of actual means
            sns.scatterplot(
                data=sub, y='group', x='mean', ax=ax, s=125, zorder=100,
                edgecolor=vutils.BLACK, lw=2, marker='o'
            )
            # Set the title (do this now, while we have the variable names)
            ax.set_title(v.replace('_', ' ').capitalize(), y=0.995)
            # Iterate through each row
            for idx, row in sub.iterrows():
                # Add our t-test asterisks above the mean
                # ax.text(row['mean'], idx - 0.1, row['ast'], fontsize=vutils.FONTSIZE)
                # Draw a horizontal line to act as a grid
                ax.hlines(y=idx, xmin=-1, xmax=1, color=vutils.BLACK, alpha=0.2, lw=2, zorder=-1)
                # Draw our 95% confidence intervals around our actual mean
                ax.hlines(y=idx, xmin=row['low'], xmax=row['high'], lw=3, color=vutils.BLACK, zorder=10)
                # Add vertical brackets to our confidence intervals
                for var in ['low', 'high']:
                    ax.vlines(x=row[var], ymin=idx - 0.2, lw=3, ymax=idx + 0.2, color=vutils.BLACK, zorder=-1)

    def _format_ax(
            self
    ) -> None:
        """
        Formats axis level objects
        """
        # Iterate through both axis
        for ax in self.ax.flatten():
            # Remove labels and set axis limits
            ax.set(xlabel='', ylabel='', xlim=(-0.9, 0.9))
            # Set tick and axis width slightly
            plt.setp(ax.spines.values(), linewidth=2)
            ax.tick_params(axis='both', width=3)
            # Add in a vertical line at 0 (no significant difference between duos)
            ax.axvline(x=0, color=vutils.BLACK, ls='--', lw=2, alpha=0.8)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Add in the axis labels
        self.fig.supxlabel('Difference in means')
        self.fig.supylabel('Duos', x=0.01)
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.175, top=0.9, wspace=0.1)


class BarPlotCouplingStrengthAsymmetry(vutils.BasePlot):
    """
    Creates a plot showing the mean differences between coupling and strength
    for each pair of duos, with confidence intervals obtained via bootstrapping
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18.8, 5))
        self.df = self._format_df()

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting
        """
        # Define grouper and column variables
        gps = ['trial', 'latency', 'jitter']
        cols = gps + ['coupling_asymmetry', 'coupling_strength']
        # Define functions for extracting total and absolute correction difference
        tot_correction = lambda grp: grp.iloc[1]['correction_partner'] + grp.iloc[0]['correction_partner']
        abs_correction = lambda grp: abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        # Group to get the average correction per duo, instrument and condition
        df_f = self.df.groupby(['trial', 'instrument', 'latency', 'jitter']).mean().reset_index()
        # Return the absolute and total correction for each condition
        return pd.DataFrame(
            [[*i, abs_correction(g), tot_correction(g)] for i, g in df_f.groupby(gps)], columns=cols
        )

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\barplot_coupling_asymmetry_strength'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the scatter and line plots
        """
        for i, var in zip(range(0, 2), ['coupling_strength', 'coupling_asymmetry']):
            g = sns.barplot(
                data=self.df, x='trial', y=var, ax=self.ax[i], estimator=np.mean,
                errorbar=('ci', 95), n_boot=vutils.N_BOOT, seed=1, capsize=0.1, width=0.4,
                errcolor=vutils.BLACK, errwidth=2, edgecolor=vutils.BLACK, lw=2,
                palette=vutils.DUO_CMAP,
            )
            g.set_title(var.replace('_', ' ').capitalize(), y=1.025)

    def _format_ax(self):
        for ax in self.ax.flatten():
            ax.set(xlabel='', ylabel='', ylim=(0, 1.25))
            ax.tick_params(width=3, which='major')
            ax.tick_params(width=0, which='minor')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(self):
        self.fig.supxlabel('Duo')
        self.fig.supylabel('Value', x=0.01)
        self.fig.subplots_adjust(top=0.85, bottom=0.15, right=0.975, left=0.065, wspace=0.05, hspace=0.15)


class BarPlotPhaseCorrectionModelComparison(vutils.BasePlot):
    """
    Creates barplots showing the difference in R2 between models
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Number of bootstrap samples to use when bootstrapping confidence intervals
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)
        self.group_vars: list[str] = ['trial', 'block', 'latency', 'jitter']
        self.mds: list[str] = kwargs.get('mds', [
            '~C(latency)+C(jitter)+coupling_strength+coupling_balance',
            '~C(latency)+C(jitter)',
            '~coupling_strength+coupling_balance',
            '~C(latency)+C(jitter)+coupling_balance',
            '~C(latency)+C(jitter)+coupling_strength',
        ])
        self.ref_md: str = 'latency+jitter+coupling_strength+coupling_balance'
        self.vars: list[str] = ['tempo_slope', 'ioi_std', 'pw_asym', 'success']
        self.md_names: list[str] = kwargs.get('md_names', [
            'Full model\n(~$L$ + $J$ + $C_a$ + $C_s$)',
            'Testbed only\n(~$L$ + $J$)',
            'Coupling only\n(~$C_a$ + $C_s$)',
            'Testbed + asymmetry\n(~$L$ + $J$ + $C_a$)',
            'Testbed + strength\n(~$L$ + $J$ + $C_s$)',
        ])
        self.titles: list[str] = kwargs.get('titles', [
            'Tempo slope (BPM/s)',
            'Timing irregularity (SD, ms)',
            'Asynchrony (RMS, ms)',
            'Self-reported success'
        ])

        self.df: pd.DataFrame = self._format_df()
        self.mds: pd.DataFrame = self._generate_mds()
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True, figsize=(18.8, 9.4)
        )

    def _format_df(self):
        res = []
        for idx, grp in self.df.groupby(self.group_vars):
            dr = grp[grp['instrument'] != 'Keys']['correction_partner'].iloc[0]
            ke = grp[grp['instrument'] == 'Keys']['correction_partner'].iloc[0]
            di = {'coupling_strength': dr + ke, 'coupling_balance': abs(dr - ke), }
            di.update({k: v for k, v in zip(self.group_vars, idx)})
            di.update({var_: grp[var_].mean() for var_ in self.vars})
            res.append(di)
        res = (
            pd.DataFrame(res)
            .groupby(['trial', 'latency', 'jitter'])
            .mean()
            .reset_index(drop=False)
        )
        res['tempo_slope'] = res['tempo_slope'].abs()
        return res

    @staticmethod
    def _get_conditional_r2(md):
        # Variance explained by the fixed effects: we need to use md.predict() with the underlying data to get this
        var_fixed = md.predict().var()
        # Variance explained by the random effects
        var_random = float(md.cov_re.iloc[0])
        # Variance of the residuals
        var_resid = md.scale
        # Total variance of the model
        total_var = var_fixed + var_random + var_resid
        return (var_fixed + var_random) / total_var

    def _create_mixed_md(self, form: str, df: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                md = smf.mixedlm(form, data=df, groups=df['trial']).fit(reml=False)
                return self._get_conditional_r2(md)
            except:
                return None

    def _generate_mds(self):
        md_res = []
        for var in self.vars:
            for md in self.mds:
                act_r2 = self._create_mixed_md(var + md, self.df)
                boot_r2 = []
                for n in range(self.n_boot):
                    boot_df = self.df.sample(frac=1, replace=True, random_state=n)
                    r2 = self._create_mixed_md(var + md, boot_df)
                    if r2 is not None:
                        boot_r2.append(r2)
                md_res.append({
                    'var': var,
                    'md': md.replace('~', '').replace('C(', '').replace(')', ''),
                    'mean': act_r2,
                    'high': np.quantile(boot_r2, 0.975),
                    'low': np.quantile(boot_r2, 0.025)
                })
        return self._format_md_df(pd.DataFrame(md_res))

    def _format_md_df(self, md_df: pd.DataFrame):
        md_df['var'] = pd.Categorical(md_df['var'], categories=self.vars)
        md_df['low'] = np.abs(md_df['mean'] - md_df['low'])
        md_df['high'] = np.abs(md_df['mean'] - md_df['high'])
        return md_df

    @vutils.plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_phase_correction_model_comparison'
        return self.fig, fname

    def _create_plot(self):
        palette = sns.color_palette('Set2', n_colors=4)
        palette[1], palette[2] = palette[2], palette[1]
        for ax, (i, var), col in zip(self.ax.flatten(), self.mds.groupby('var'), palette):
            g = sns.barplot(
                data=var, x='md', y='mean', ax=ax, edgecolor=vutils.BLACK,
                lw=2, saturation=0.8, alpha=0.8, color=col, width=0.8,
            )
            g.errorbar(
                x=var['md'], y=var['mean'], yerr=(var['low'], var['high']),
                lw=2, ls='none', capsize=5, elinewidth=2, markeredgewidth=2,
                color=vutils.BLACK
            )
            self._add_bar_labels(var, g)

    def _add_bar_labels(self, var: pd.DataFrame, g: plt.Axes):
        var['label'] = var['mean'] - var[var['md'] == self.ref_md].iloc[0]['mean']
        for artist, (i_, lab) in zip(g.containers[0], var.iterrows()):
            if lab['label'] != 0:
                y_pos = artist.get_y() + artist.get_height() + (lab['low'] + lab['high']) + 0.01
                g.text(
                    artist.get_x() + 0.1, y_pos, f'$\Delta{round(lab["label"], 2)}$',
                    ha='left', va='baseline', fontsize=vutils.FONTSIZE - 1
                )

    def _format_ax(self):
        for ax, tit in zip(self.ax.flatten(), self.titles):
            ax.set(
                xticklabels=self.md_names, title=tit, ylabel='', xlabel='',
                ylim=[-0.2, 1.15], yticks=np.linspace(0, 1, 3)
            )
            plt.setp(ax.spines.values(), linewidth=2)
            ax.set_xticklabels(self.md_names, rotation=45, ha='right', rotation_mode='anchor')
            ax.tick_params(axis='both', width=3, )
            ax.axhline(y=0, color=vutils.BLACK, lw=2)

    def _format_fig(self):
        self.fig.supxlabel('Model predictor variables')
        self.fig.supylabel(r'Conditional $R^{2}$', y=0.6)
        self.fig.subplots_adjust(bottom=0.3, top=0.95, left=0.08, right=0.95, hspace=0.2, wspace=0.1)


class BarPlotMixedEffectsRegressionCoefficients(vutils.BasePlot):
    """
    Creates a barplot showing regression coefficients and confidence intervals for mixed effects models
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get parameters for table from kwargs
        self.group_vars: list[str] = ['trial', 'block', 'latency', 'jitter']
        self.categories: list[str] = kwargs.get('categories', ['tempo_slope', 'pw_asym', 'ioi_std', 'success', ])
        self.labels: list[str] = kwargs.get('labels', [
            'Tempo slope (BPM/s)', 'Asynchrony (RMS, ms)', 'Timing irregularity (SD, ms)', 'Self-reported success'
        ])
        self.averaged_vars: list[str] = kwargs.get('averaged_vars', ['tempo_slope', 'pw_asym'])
        self.predictor_ticks: list[str] = kwargs.get('predictor_ticks', [
            'Reference\n(0ms, 0.0x)', 'Latency (ms)', 'Jitter', 'Coupling'
        ])
        self.levels: list[str] = kwargs.get('levels', ['Intercept', '23', '45', '90', '180', '0.5', '1.0', 'Strength',
                                                       'Asymmetry'])
        self.alpha: float = kwargs.get('alpha', 0.05)
        # Format dataframe to get regression results
        self.df = self._format_df()
        self.mds = self._create_mixed_models()
        self.md_dfs = self._extract_betas_ci()
        # Get plotting parameters from kwargs
        self.palette: str = kwargs.get('palette', 'Set2')
        self.errorbar_margin: float = kwargs.get('errorbar_margin', 0.03)
        self.vlines_margin: float = kwargs.get('vlines_margin', 0.9)
        self.add_pvals: bool = kwargs.get('add_pvals', False)
        # Create plotting objects in matplotlib
        self.fig, self.ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(18.8, 18.8))
        self.legend_handles_labels = None  # Used to hold our legend object for later

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting by extracting regression results
        """
        res = []
        for idx, grp in self.df.groupby(self.group_vars):
            dr = grp[grp['instrument'] != 'Keys']['correction_partner'].iloc[0]
            ke = grp[grp['instrument'] == 'Keys']['correction_partner'].iloc[0]
            di = {'coupling_strength': dr + ke, 'coupling_asymmetry': abs(dr - ke)}
            di.update({k: v for k, v in zip(self.group_vars, idx)})
            di.update({var_: grp[var_].mean() for var_ in self.categories})
            res.append(di)
        return (
            pd.DataFrame(res)
              .groupby(['trial', 'latency', 'jitter'])
              .mean()
              .reset_index(drop=False)
        )

    def extract_re_stds(self):
        """
        Extracts standard deviation of random effects specified in mixed model
        """
        return [np.sqrt(float(md.summary().tables[1]['Coef.'].iloc[-1])) for md in self.mds]

    def extract_marginal_conditional_r2(self) -> pd.DataFrame:
        """
        Extracts marginal and conditional r2 from linear mixed models according to procedure outlined
        by Nakagawa and Schielzeth (2013)
        """
        r2 = []
        for var, md in zip(self.categories, self.mds):
            # Variance explained by the fixed effects: we need to use md.predict() with the underlying data to get this
            var_fixed = md.predict().var()
            # Variance explained by the random effects
            var_random = float(md.cov_re.iloc[0])
            # Variance of the residuals
            var_resid = md.scale
            # Total variance of the model
            total_var = var_fixed + var_random + var_resid
            # Calculate the r2 values and append to the model
            r2.append({
                'var': var,
                'conditional_r2': (var_fixed + var_random) / total_var,
                'marginal_r2': var_fixed / total_var
            })
        return pd.DataFrame(r2)

    def _extract_betas_ci(self):
        """
        Extracts beta coefficients and confidence intervals from mixed models
        """
        def formatter(md):
            betas = md.params.rename('beta')
            cis = md.conf_int().rename(columns={0: 'low', 1: 'high'})
            conc = (
                pd.concat([betas, cis], axis=1)
                  .reset_index(drop=False)
                  .rename(columns={'index': 'variable'})
                  .iloc[:-1]
            )
            for s in ['C(latency)[T.', 'C(jitter)[T.', 'coupling_', ']']:
                conc['variable'] = conc['variable'].str.replace(s, '', regex=False)
            conc['variable'] = conc['variable'].str.title()
            return conc

        return [formatter(md) for md in self.mds]

    def _create_mixed_models(self):
        mds = []
        for var in self.categories:
            formula = var + '~' + 'C(latency)+C(jitter)+coupling_strength+coupling_asymmetry'
            md = smf.mixedlm(formula, data=self.df, groups=self.df['trial']).fit(reml=False)
            mds.append(md)
        return mds

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\barplot_mixed_effects_regression_coefs'
        return self.fig, fname

    def _add_errorbars(
            self, grp: pd.DataFrame, ax: plt.Axes
    ) -> None:
        """
        Adds errorbars into a grouped barplot. This is difficult to do in standard matplotlib, so we add errorbars
        manually using the ax.hlines method, providing our upper and lower confidence intervals into this
        """
        # Zip and iterate through each row in the dataframe and patch (bar) in the graph
        for (i, r), p in zip(grp.iterrows(), ax.patches):
            # Get the centre of our x position
            pos = p.get_x() + p.get_width() / 2
            # Add in the centre line of our error bar
            ax.vlines(pos, r['low'], r['high'], color=vutils.BLACK, lw=2)
            # Add in brackets/braces to our error bar, at the top and bottom
            for v in ['low', 'high']:
                ax.hlines(
                    r[v], pos - self.errorbar_margin, pos + self.errorbar_margin, color=vutils.BLACK, lw=2
                )
            # Add in our significance asterisks, if required (off by default)
            if self.add_pvals:
                ypos = r['low_ci'] if abs(r['low_ci']) > abs(r['high_ci']) else r['high_ci']
                ax.text(pos, ypos + self.errorbar_margin, r['pval'])

    def _create_plot(
            self
    ) -> None:
        """
        Creates bar chart + error bars for each subplot, corresponding to one variable
        """
        palette = sns.color_palette(self.palette, n_colors=len(self.md_dfs))
        for ax, md, col in zip(self.ax.flatten(), self.md_dfs, palette):
            md['variable'] = pd.Categorical(md['variable'], self.levels)
            sns.barplot(
                data=md, x='variable', y='beta', ax=ax, errorbar=None, edgecolor=vutils.BLACK,
                lw=2, saturation=0.8, alpha=0.8, color=col
            )
            self._add_errorbars(md, ax)

    def _add_predictor_ticks(
            self, ax: plt.Axes, ticks: list[str] = None
    ) -> None:
        """
        Adds a secondary x axis showing the names of each of our predictor variables
        """
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks([0, 2.5, 5.5, 7.5], self.predictor_ticks if ticks is None else ticks)
        ax2.tick_params(width=3, which='major')
        plt.setp(ax2.spines.values(), linewidth=2)

    def _add_seperating_vlines(
            self, ax: plt.Axes
    ) -> None:
        """
        Adds in vertical lines seperating levels for each predictor on the x axis, e.g. latency, jitter...
        """
        # Get the ticks corresponding to each predictor
        xs = np.array(sorted([p.get_x() for p in ax.patches]))
        idxs = np.argwhere(np.diff(xs) > self.vlines_margin)
        vals = xs[idxs][[0, 4, 6]]
        # Get our axis y limit
        ymi, yma = ax.get_ylim()
        # Iterate through each of the required lines and add it in
        for v in vals:
            ax.vlines(v + self.vlines_margin, ymin=ymi, ymax=yma, color=vutils.BLACK, lw=2, alpha=vutils.ALPHA, ls='--')

    def _format_ax(
            self
    ) -> None:
        """
        Applies required axis formatting
        """
        # Iterate through each axis and label, with a counter
        for count, ax, lab in zip(range(len(self.md_dfs)), self.ax.flatten(), self.labels):
            # Add in a horizontal line at y = 0
            ax.axhline(y=0, color=vutils.BLACK, lw=2)
            # Set our axis labels
            ax.set(xlabel='', ylabel=lab,
                   xticklabels=['Intercept', '23', '45', '90', '180', '0.5x', '1.0x', 'Strength', 'Asymmetry'])
            # Add in our predictor ticks to the top of the first subplot
            ticks = self.predictor_ticks if count == 0 else ['' for _ in range(len(self.predictor_ticks))]
            self._add_predictor_ticks(ax=ax, ticks=ticks)
            # Add vertical lines separating each predictor variable
            self._add_seperating_vlines(ax=ax)
            # Adjust tick formatting
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Applies figure-level formatting, including legend, axis labels etc.
        """
        # Add our axis text and labels in
        self.fig.suptitle('Model predictor variables')
        self.fig.supylabel(r'Coefficient (B)')
        self.fig.supxlabel('Categorical levels')
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(top=0.92, bottom=0.05, left=0.1, right=0.95)


class PointPlotSelfPartnerCouplingByInstrument(vutils.BasePlot):
    """
    Creates a pointplot showing bootstrapped differences in coupling coefficients between members of each duo
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_boot = kwargs.get('n_boot', vutils.N_BOOT)
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, sharex=False, sharey=True, figsize=(18.8, 4)
        )
        self.titles = [r'Self coupling ($\alpha_{i, i}$)', r'Partner coupling ($\alpha_{i, j}$)']
        self.lims = [0.25, 0.9]
        self.handles, self.labels = None, None
        self.df = self._format_df()

    def _format_df(self):
        res = []
        df_ = self.df.groupby(['trial', 'latency', 'jitter', 'instrument']).mean().reset_index(drop=False)
        for v in ['correction_self', 'correction_partner']:
            for idx, grp in df_.groupby(['trial']):
                a1 = grp[grp['instrument'] == 'Keys'][v]
                a2 = grp[grp['instrument'] != 'Keys'][v]
                # Get the actual mean difference
                mea = a2.mean() - a1.mean()
                low, high = vutils.bootstrap_mean_difference(a1, a2, n_boot=self.n_boot)
                # Append the results to our list
                res.append((idx, v, low, high, mea,))
        res = pd.DataFrame(res, columns=['duo', 'variable', 'low', 'high', 'mean'])
        res['duo'] = pd.Categorical(res['duo'], [1, 2, 3, 4, 5])
        res['variable'] = pd.Categorical(res['variable'], ['correction_self', 'correction_partner'])
        return res

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\pointplot_coupling_by_instrument'
        return self.fig, fname

    def _create_plot(self):
        for (idx, grp), ax_ in zip(self.df.groupby('variable'), self.ax.flatten()):
            sns.scatterplot(
                data=grp, y='duo', x='mean', ax=ax_, s=200, zorder=100,
                edgecolor=vutils.BLACK, lw=2, marker='o', hue='duo', style='duo',
                palette=vutils.DUO_CMAP, markers=vutils.DUO_MARKERS
            )
            # Iterate through each row
            for idx_, row in grp.groupby('duo'):
                # Draw a horizontal line to act as a grid
                ax_.hlines(y=idx_, xmin=-1, xmax=1, color=vutils.BLACK, alpha=0.2, lw=2, zorder=-1)
                # Draw our 95% confidence intervals around our actual mean
                ax_.hlines(y=idx_, xmin=row['low'], xmax=row['high'], lw=3, color=vutils.BLACK, zorder=10)
                # Add vertical brackets to our confidence intervals
                for var in ['low', 'high']:
                    ax_.vlines(x=row[var], ymin=idx_ - 0.1, lw=3, ymax=idx_ + 0.1, color=vutils.BLACK, zorder=-1)

    def _format_ax(self):
        # Iterate through both axis
        for ax_, tit, lim in zip(self.ax.flatten(), self.titles, self.lims):
            # Remove labels and set axis limits
            ax_.set(xlabel='', ylabel='', xlim=(-lim, lim), title=tit)
            # Set tick and axis width slightly
            plt.setp(ax_.spines.values(), linewidth=2)
            ax_.tick_params(axis='both', width=3)
            # Add in a vertical line at 0 (no significant difference between duos)
            ax_.axvline(x=0, color=vutils.BLACK, ls='--', lw=2, alpha=0.8)
            self.handles, self.labels = ax_.get_legend_handles_labels()
            ax_.get_legend().remove()

    def _format_fig(self):
        leg = self.fig.legend(
            self.handles, self.labels, ncol=1, title='Duo', frameon=False, bbox_to_anchor=(1, 0.75),
            markerscale=1.6, fontsize=vutils.FONTSIZE
        )
        for handle in leg.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([200])
        plt.setp(leg.get_title(), fontsize=20)
        # Add in the axis labels
        self.fig.supxlabel('Difference in means')
        self.fig.supylabel('Duos', x=0.01)
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(left=0.05, right=0.93, bottom=0.175, top=0.9, wspace=0.1)


def generate_phase_correction_plots(
    mds: list[PhaseCorrectionModel], output_dir: str,
) -> None:
    """
    Generates all plots in this file, with required arguments and inputs
    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\phase_correction_plots'
    # Create regression table
    # vutils.output_regression_table(
    #     mds=autils.create_model_list(df=df, md=f'correction_partner~C(latency)+C(jitter)+C(instrument)',
    #                                  avg_groupers=['latency', 'jitter', 'instrument']),
    #     output_dir=figures_output_dir, verbose_footer=False
    # )
    pp = PointPlotCouplingStrengthAsymmetry(df=df, output_dir=figures_output_dir)
    pp.create_plot()
    pp = PointPlotSelfPartnerCouplingByInstrument(df=df, output_dir=figures_output_dir)
    pp.create_plot()
    bp = BarPlotPhaseCorrectionModelComparison(df=df, output_dir=figures_output_dir, n_boot=500)
    bp.create_plot()
    bp = BarPlotMixedEffectsRegressionCoefficients(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    ap = ArrowPlotPhaseCorrection(df=df, output_dir=figures_output_dir)
    ap.create_plot()

    # Create box plot
    bp = BoxPlot(df=df, output_dir=figures_output_dir, yvar='correction_partner', ylim=(-0.2, 1.5))
    bp.create_plot()
    # Create pair grids
    pg_a = PairGrid(
        df=df, xvar='correction_partner', output_dir=figures_output_dir, xlim=(0, 1.25),
        xlabel='Average coupling to partner', average=True
    )
    pg_a.create_plot()
    pg = PairGrid(
        df=df, xvar='correction_partner', output_dir=figures_output_dir, xlim=(0, 1.5),
        xlabel='Coupling to partner', average=False
    )
    pg.create_plot()
    # Create regression plots
    rp = RegPlotGrid(df=df, output_dir=figures_output_dir, errorbar='ci', abs_slope=False)
    rp.create_plot()
    rp = RegPlotGrid(df=df, output_dir=figures_output_dir, errorbar='ci', abs_slope=True)
    rp.create_plot()
    bar = BarPlot(df=df, output_dir=figures_output_dir)
    bar.create_plot()
    bar = BarPlot(df=df, output_dir=figures_output_dir, yvar='correction_self', ylabel='Self coupling', ylim=(-1, 1))
    bar.create_plot()
    bp = BarPlotCouplingStrengthAsymmetry(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp = BarPlotCouplingStrengthAsymmetryComparison(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    ap = ArrowPlotModelExplanation(output_dir=figures_output_dir)
    ap.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    # TODO: this shouldn't be hardcoded
    raw: list[PhaseCorrectionModel] = autils.load_from_disc(
        r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p'
    )
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_phase_correction_plots(mds=raw, output_dir=output)
