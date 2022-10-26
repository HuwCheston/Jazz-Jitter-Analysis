import pandas as pd
import numpy as np
import statsmodels.api as sm
from statistics import mean
from random import uniform
from matplotlib import animation, patches, pyplot as plt
from matplotlib.transforms import ScaledTranslation
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


class BasePlot:
    """
    Base plotting class from which others inherit
    """
    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': vutils.FONTSIZE})
        # Get from kwargs (with default arguments)
        self.df: pd.DataFrame = kwargs.get('df', None)
        self.output_dir: str = kwargs.get('output_dir', None)
        # Create an empty attribute to store our plot in later
        self.g = None


class PairGrid(BasePlot):
    """
    Creates a scatterplot for each duo/question combination.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xvar: str = kwargs.get('xvar', 'correction_partner_onset')
        self.xlim: tuple = kwargs.get('xlim', (-1.5, 1.5))
        self.xlabel: str = kwargs.get('xlabel', None)
        self.cvar: str = kwargs.get('cvar', 'tempo_slope')
        self.clabel: str = kwargs.get('clabel', 'Tempo\nSlope\n(BPM/s)\n\n')

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
        fname = f'{self.output_dir}\\pairgrid_condition_vs_{self.xvar}_vs_{self.cvar}.png'
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
            s=10, jitter=False, dodge=False
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
            ax1.set_title(f'Measure 1\nDuo {num + 1}' if num == 2 else f'\nDuo {num + 1}', fontsize=vutils.FONTSIZE)
            ax1.set(ylabel='', xlim=self.xlim, )
            ax1.tick_params(bottom=False)
            ax2.set_title(f'Measure 2' if num == 2 else '', fontsize=vutils.FONTSIZE)
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
        self.g.fig.supxlabel(self.xlabel, x=0.53, y=0.05)
        self.g.fig.supylabel('Condition', x=0.01)
        # Add the color bar
        position = self.g.fig.add_axes([0.94, 0.2, 0.01, 0.6])
        self.g.fig.colorbar(vutils.create_scalar_cbar(norm=self.norm), cax=position, ticks=vutils.CBAR_BINS)
        position.text(0., 0.3, self.clabel, fontsize=vutils.FONTSIZE)  # Super hacky way to add a title...
        # Add the legend
        self.g.legend.remove()
        i = self.g.fig.get_axes()[0].legend(loc='lower center', ncol=2, bbox_to_anchor=(2.85, -1.45), title=None,
                                            frameon=False)
        for handle in i.legendHandles:
            handle.set_sizes([100.0])
        # Adjust the plot spacing
        self.g.fig.subplots_adjust(bottom=0.12, top=0.93, wspace=0.15, left=0.11, right=0.93)


class BoxPlot(BasePlot):
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
        self.ylabel: str = kwargs.get('ylabel', None)
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
        fname = f'{self.output_dir}\\boxplot_{self.yvar}_vs_{self.xvar}.png'
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
        self.fig.suptitle(f'Duo {self.metadata[0]} (measure {self.metadata[1]}): '
                          f'latency {self.metadata[2]}ms, jitter {self.metadata[3]}x')
        fname = f'{self.output_dir}\\duo{self.metadata[0]}_measure{self.metadata[1]}' \
                f'_latency{self.metadata[2]}_jitter{self.metadata[3]}.png'
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
    Default FPS is 30 seconds, with data interpolated so plotting look nice and smooth. This can be changed in vutils.
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


class RegPlot(BasePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yvar: str = kwargs.get('yvar', 'tempo_slope')
        self.ylabel: str = kwargs.get('ylabel', 'Tempo slope (BPM/s)')
        self.ylim: tuple = kwargs.get('ylim', None)
        self.xvar: str = kwargs.get('xvar', 'abs_correction')
        self.xlabel: str = kwargs.get('xlabel', 'Coupling balance')
        self.xlim: tuple = kwargs.get('xlim', None)
        self.hline: bool = kwargs.get('hline', True)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 9.4))

    def _format_plot(self):
        """
        Formats a regression plot, including both the figure and axes
        """
        # Set axes level formatting

        self.ax.tick_params(width=3, )
        self.ax.set(
            ylabel='', xlabel='', xlim=self.xlim if self.xlim is not None else self.ax.get_xlim(),
            ylim=self.ylim if self.ylim is not None else self.ax.get_ylim(),
        )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Plot a horizontal line at x=0
        if self.hline:
            self.ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK)
        # Set axis labels
        self.fig.supylabel(self.ylabel, x=0.03)
        self.fig.supxlabel(self.xlabel, y=0.09)


class RegPlotDuo(RegPlot):
    """
    Creates a regression & scatter plot for absolute correction difference between duo members versus another variable
    (defaults to tempo slope, but can be changed by setting yvar argument)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.df is not None:
            self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.ax = self._create_plot()
        self._format_plot()
        # Format axis positioning and move legend
        plt.tight_layout()
        sns.move_legend(
            self.ax, 'lower center', ncol=6, title=None, frameon=False, bbox_to_anchor=(0.45, -0.2), markerscale=1.6,
            columnspacing=0.2, handletextpad=0.1,
        )
        self.ax.figure.subplots_adjust(bottom=0.17, top=0.93, left=0.14, right=0.90)
        # Return with filename to be saved and closed in outer function
        fname = f'{self.output_dir}\\regplot_{self.xvar}_vs_{self.yvar}.png'
        return self.ax.figure, fname

    def _format_df(self):
        self.df = self.df.groupby(by=['trial', 'block', 'latency', 'jitter']).apply(self._get_abs_correction)
        self.df['trial_abbrev'] = self.df['trial'].replace({n: f'Duo {n}' for n in range(1, 6)})
        return self.df

    def _get_abs_correction(self, grp: pd.DataFrame.groupby) -> pd.DataFrame.groupby:
        """
        Function used to calculate absolute coupling difference across duo
        """
        grp['abs_correction'] = abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        return grp.drop_duplicates(subset=[self.xvar, self.yvar])

    def _create_plot(self):
        g = sns.scatterplot(
            data=self.df, x=self.xvar, y=self.yvar, hue='trial_abbrev', style='trial_abbrev',
            s=150, ax=self.ax, palette='tab10'
        )
        g = sns.regplot(
            data=self.df, x=self.xvar, y=self.yvar, scatter=None, truncate=True, ax=g, color=vutils.BLACK
        )
        return g


class RegPlotSingle(RegPlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.df is not None:
            self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.ax = self._create_plot()
        self._format_plot()
        # Create the legend
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles=handles[1:6] + handles[7:], labels=labels[1:6] + labels[7:], ncol=8, frameon=False,
                       markerscale=1.6, columnspacing=0.1, handletextpad=0.1, bbox_to_anchor=(1.15, -0.1))
        self.ax.figure.subplots_adjust(bottom=0.17, top=0.93, left=0.14, right=0.90)
        # Return with filename to be saved and closed in outer function
        fname = f'{self.output_dir}\\regplot_{self.xvar}_vs_{self.yvar}.png'
        return self.ax.figure, fname

    def _format_df(self):
        self.df['trial_abbrev'] = self.df['trial'].replace({n: f'Duo {n}' for n in range(1, 6)})
        # Convert r-squared to percentage
        if self.yvar == 'rsquared':
            self.df['rsquared'] = self.df['rsquared'].apply(lambda x: x * 100)
        return self.df

    def _create_plot(self):
        g = sns.regplot(
            data=self.df, x=self.xvar, y=self.yvar, scatter=False, color=vutils.BLACK, ax=self.ax
        )
        g = sns.scatterplot(
            data=self.df, x=self.xvar, y=self.yvar, hue='trial_abbrev', palette='tab10', style='instrument', s=100, ax=g
        )
        return g


class PointPlotLaggedLatency(BasePlot):
    """
    Make a pointplot showing lagged timestamps on x-axis and regression coefficients on y. Columns grouped by trial,
    rows grouped by jitter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lag_var_name = kwargs.get('lag_var_name', 'pc_l')
        if self.df is not None:
            self.df = self._format_df()

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_facetgrid()
        self._format_ax()
        self._format_fig()
        # Create filename and return to save
        fname = f'{self.output_dir}\\pointplot_{self.lag_var_name}.png'
        return self.g.figure, fname

    def _format_df(self):
        """
        Formats the dataframe for the plot
        """
        # Extract the value vars
        val_vars = [col for col in self.df if self.lag_var_name in col and not col.endswith('_p')]
        self.df = (
            self.df.melt(id_vars=['trial', 'block', 'latency', 'jitter', 'instrument', 'abbrev'],
                         value_vars=val_vars, value_name='coef', var_name='lag')
                   .sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument'])
        )
        self.df.lag = self.df.lag.str.extract(r'(\d+)')
        return self.df

    def _create_facetgrid(self):
        """
        Creates the facetgrid and maps plots onto it
        """
        return sns.catplot(
            data=self.df[self.df['jitter'] != 0], col='trial', row='jitter', x="lag", y="coef", hue='instrument',
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


class NumberLine(BasePlot):
    """
    Creates a numberline showing difference in pairwise asynchrony between duos this experiment during the control
    condition and a corpus of pairwise asynchrony values from other studies and genres
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(corpus_filepath=kwargs.get('corpus_filepath', None))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4 * 2, 4))

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._add_annotations()
        self._format_plot()
        fname = f'{self.output_dir}\\numberline_pairwise_asynchrony.png'
        # TODO: Is this necessary?
        plt.savefig(fname)
        return self.g.figure, fname

    def _format_df(self, corpus_filepath) -> pd.DataFrame:
        """
        Formats the dataframe by concatenating with data from the corpus
        """
        # Read in the corpus data
        corpus = pd.read_excel(io=corpus_filepath, sheet_name=0)
        # Read in the data from the experimental dataframe
        trial = (
            self.df[self.df['latency'] == 0].drop_duplicates(subset='pw_asym')
                                            .groupby(['trial']).mean()[['pw_asym']]
                                            .reset_index(drop=False)
                                            .rename(columns={'trial': 'style'})
        )
        trial['source'] = ''
        trial['style'] = trial['style'].replace({n: f'Duo {n}, this study' for n in range(1, 6)})
        # Concatenate trial and corpus data together
        self.df = pd.concat([corpus.reset_index(drop=True), trial.reset_index(drop=True)]).round(0)
        self.df['this_study'] = (self.df['source'] == '')
        self.df['placeholder'] = ''
        return self.df

    def _create_plot(self):
        """
        Creates the facetgrid object
        """
        return sns.stripplot(
            data=self.df, x='pw_asym', y='placeholder', hue='this_study',
            jitter=False, dodge=False, s=12, ax=self.ax, orient='h'
        )

    def _add_annotations(self):
        """
        Add the annotations onto the plot
        """
        for k, v in self.df.iterrows():
            self.g.annotate(
                text=v['style'] + '\n' + v['source'], xy=(v['pw_asym'], 0), xytext=(v['pw_asym'], -0.3), rotation=45
            )

    def _format_plot(self):
        """
        Formats the plot
        """
        # Add the horizontal line
        self.g.axhline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, linewidth=3)
        # Format the plot
        self.g.set(xlim=(15, 41), xticks=np.arange(15, 41, 5, ), xlabel='', ylabel='')
        self.g.figure.supxlabel('Pairwise asynchrony (ms)', y=0.05)
        sns.despine(left=True, bottom=True)
        plt.subplots_adjust(top=0.34, bottom=0.25, left=0.05, right=0.93)
        plt.yticks([], [])
        plt.legend([], [], frameon=False)


class BarPlot(BasePlot):
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
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_correction_vs_instrument.png'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the two plots
        """
        ax = sns.stripplot(
            data=self.df, x='trial', y=self.yvar, hue='instrument', dodge=True,
            color=vutils.BLACK, s=4, marker='.', jitter=1, ax=self.ax
        )
        ax = sns.barplot(
            data=self.df, x='trial', y=self.yvar, hue='instrument', ax=ax, ci=25, palette=vutils.INSTR_CMAP,
            hue_order=['Keys', 'Drums'], errcolor='#3953a3', errwidth=10, estimator=self.estimator,
            edgecolor=vutils.BLACK, lw=2
        )
        return ax

    def _format_ax(self):
        """
        Set axes-level formatting
        """
        # Set ax formatting
        self.g.tick_params(width=3, )
        self.g.set(ylabel='', xlabel='')
        plt.setp(self.g.spines.values(), linewidth=2)
        # Plot a horizontal line at x=0
        self.g.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK, linewidth=2)

    def _format_fig(self):
        """
        Set figure-level formatting
        """
        # Set figure labels
        self.fig.supylabel('Coupling constant', x=0.02, y=0.55)
        self.fig.supxlabel('Duo', y=0.09)
        # Format the legend to remove the handles/labels added automatically by the strip plot
        handles, labels = self.g.get_legend_handles_labels()
        self.g.get_legend().remove()
        plt.legend(handles[2:], labels[2:], ncol=6, title=None, frameon=False, bbox_to_anchor=(0.7, -0.15),
                   markerscale=1.6)
        # Adjust the figure a bit and return for saving in decorator
        self.g.figure.subplots_adjust(bottom=0.22, top=0.95, left=0.14, right=0.95)


class HistPlotR2(BasePlot):
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
        fname = f'{self.output_dir}\\histplot_{self.xvar}.png'
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


class BoxPlotR2WindowSize(BasePlot):
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
        fname = f'{self.output_dir}\\boxplot_r2_vs_windowsize.png'
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


class ScatterPlotQuestionnaire(BasePlot):
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
        fname = f'{self.output_dir}\\scatterplot_{self.ax_var}_{self.marker_var}.png'
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


class HeatmapQuestionnaire(BasePlot):
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
        fname = f'{self.output_dir}\\heatmap_duo_correlations.png'
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


class BarPlotInterpolatedIOIs(BasePlot):
    """
    Creates a stacked barplot showing the total number of IOIs per duo and the number of these which were interpolated
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fmt = self._format_df()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4, 5))

    def _format_df(self):
        # Group the dataframe and get the sum
        fmt = self.df.groupby(by=['trial', 'instrument']).sum()[['total_beats', 'interpolated_beats']]
        # Create the percentage columns
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
        fname = f'{self.output_dir}\\barplot_total_vs_interpolated_beats.png'
        return self.fig, fname

    def _format_plot(self):
        self.ax.tick_params(width=3, )
        self.ax.set(ylabel='', xlabel='',)
        self.fig.supxlabel('Duo number', y=0.09)
        self.fig.supylabel('Total IOIs', x=0.01)
        plt.setp(self.ax.spines.values(), linewidth=2)
        self.fig.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95)

    def _format_ax(self):
        # Add invisible data to add another legend for instrument
        n1 = [self.ax.bar(0, 0, color=cmap) for i, cmap in zip(range(2), vutils.INSTR_CMAP)]
        l1 = plt.legend(n1, ['Keys', 'Drums'], loc=[0, -0.32], ncol=2, frameon=False, columnspacing=1,
                        handletextpad=0.1)
        self.ax.add_artist(l1)
        # Add invisible data to add another legend for interpolation
        n2 = [self.ax.bar(0, 0, color='gray', hatch=h, alpha=vutils.ALPHA) for i, h in zip(range(2), ['', '//'])]
        l2 = plt.legend(n2, ['No interpolation', 'Interpolation'], loc=[0.37, -0.32], ncol=2, frameon=False,
                        columnspacing=1, handletextpad=0.1)
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


class BarPlotTestRetestReliability(BasePlot):
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
        fname = f'{self.output_dir}\\barplot_test_retest_reliability.png'
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


class BarPlotQuestionnairePercentAgreement(BasePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

    def _format_df(self):
        res = []
        for idx, grp in self.df.groupby(['trial']):
            g = grp.pivot(index=['trial', 'block', 'latency', 'jitter', ], columns='instrument',
                          values=['interaction', 'coordination', 'success'])
            g.columns = [''.join(col) for col in g.columns]
            for s in ['success', 'coordination', 'interaction']:
                lista = g[f'{s}Keys'].to_numpy()
                listb = g[f'{s}Drums'].to_numpy()
                arr = lista == listb
                perc = (len(np.where(arr)[0]) / len(lista)) * 100
                res.append({'trial': idx, 'question': s.title(), 'agreement': perc})
        return pd.DataFrame(res)

    @vutils.plot_decorator
    def create_plot(self):
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_questionnaire_percent_aggrement.png'
        return self.fig, fname

    def _create_plot(self):
        return sns.barplot(
            data=self.data, x='trial', y='agreement', hue='question', ax=self.ax, edgecolor=vutils.BLACK, lw=2
        )

    def _format_fig(self):
        self.fig.supxlabel('Duo', y=0.09)
        self.fig.supylabel('Agreement (%)', x=0.03, y=0.6)
        sns.move_legend(self.ax, 'lower center', ncol=3, title=None, frameon=False, bbox_to_anchor=(0.45, -0.33),
                        markerscale=1.6, handletextpad=0.1, )
        self.fig.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95)

    def _format_ax(self):
        self.g.set(ylim=(0, 100), ylabel='', xlabel='')
        self.ax.tick_params(width=3)
        plt.setp(self.ax.spines.values(), linewidth=2)
