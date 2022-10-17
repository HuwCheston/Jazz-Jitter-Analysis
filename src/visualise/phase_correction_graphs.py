import pandas as pd
import numpy as np
import statsmodels.api as sm
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

import src.visualise.visualise_utils as vutils
import src.analyse.analysis_utils as autils


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
        # Create an empty attribute to store our plot in later when we correct it
        self.g = None
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
        # Initialise empty attribute to store the plot in
        self.g = None
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
        # Set axes labels, limits, titles
        self.g.set(ylim=self.ylim, xlabel='', ylabel='')
        self.g.set_titles("Duo {col_name}", )
        # Iterate through to set tick and axes line widths
        for ax in self.g.axes.flatten():
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
        # Adjust the plotsize a bit
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
        # Get attributes from kwargs
        self.keys_df: pd.DataFrame = kwargs.get('keys_df', None)
        self.drms_df: pd.DataFrame = kwargs.get('drms_df', None)
        self.keys_o: pd.DataFrame = kwargs.get('keys_o', None)
        self.drms_o: pd.DataFrame = kwargs.get('drms_o', None)
        self.output_dir: str = kwargs.get('output_dir', None)
        self.metadata: tuple = kwargs.get('metadata', None)
        # Create the matplotlib objects we need
        self.fig = plt.figure(figsize=(10, 10))
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
        self.xvar: str = kwargs.get('xvar', 'abs_correction')
        self.xlabel: str = kwargs.get('xlabel', 'Coupling strength')
        self.hline: bool = kwargs.get('hline', True)
        self.g = None
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(9.4, 5))

    def _format_plot(self):
        """
        Formats a regression plot, including both the figure and axes
        """
        # Set axes level formatting
        self.ax.tick_params(width=3, )
        self.ax.set(ylabel='', xlabel='')
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Plot a horizontal line at x=0
        if self.hline:
            self.ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK)
        # Set axis labels
        self.fig.supylabel(self.ylabel, x=0.01)
        self.fig.supxlabel(self.xlabel, y=0.09)


class RegPlotAbsCorrection(RegPlot):
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
            self.ax, 'lower center', ncol=6, title=None, frameon=False, bbox_to_anchor=(0.45, -0.33), markerscale=1.6
        )
        self.ax.figure.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95)
        # Return with filename to be saved and closed in outer function
        fname = f'{self.output_dir}\\regplot_{self.xvar}_vs_{self.yvar}.png'
        return self.ax.figure, fname

    def _format_df(self):
        self.df = self.df.groupby(by=['trial', 'block', 'latency', 'jitter']).apply(self._get_abs_correction)
        self.df['trial'] = self.df['trial'].replace({n: f'Duo {n}' for n in range(1, 6)})  # Makes legend fmt easier
        return self.df

    def _get_abs_correction(self, grp: pd.DataFrame.groupby) -> pd.DataFrame.groupby:
        """
        Function used to calculate absolute coupling difference across duo
        """
        grp['abs_correction'] = abs(grp.iloc[1]['correction_partner'] - grp.iloc[0]['correction_partner'])
        return grp.drop_duplicates(subset=[self.xvar, self.yvar])

    def _create_plot(self):
        g = sns.scatterplot(
            data=self.df, x=self.xvar, y=self.yvar, hue='trial', style='trial', s=150, ax=self.ax, palette='tab10'
        )
        g = sns.regplot(
            data=self.df, x=self.xvar, y=self.yvar, scatter=None, truncate=True, ax=g, color=vutils.BLACK
        )
        return g


class RegPlotRSquared(RegPlot):
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
                       markerscale=1.6, columnspacing=0.1, handletextpad=0.1, bbox_to_anchor=(1.05, -0.15), )
        self.ax.figure.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95)
        # Return with filename to be saved and closed in outer function
        fname = f'{self.output_dir}\\regplot_{self.xvar}_vs_{self.yvar}.png'
        return self.ax.figure, fname

    def _format_df(self):
        self.df['trial'] = self.df['trial'].replace({n: f'Duo {n}' for n in range(1, 6)})
        # Convert rsquared to percentage
        if self.yvar == 'rsquared':
            self.df['rsquared'] = self.df['rsquared'].apply(lambda x: x * 100)
        return self.df

    def _create_plot(self):
        g = sns.regplot(
            data=self.df, x=self.xvar, y=self.yvar, scatter=False, color=vutils.BLACK, ax=self.ax
        )
        g = sns.scatterplot(
            data=self.df, x=self.xvar, y=self.yvar, hue='trial', palette='tab10', style='instrument', s=100, ax=g
        )
        return g


class PointPlotLaggedLatency(BasePlot):
    """
    Make a pointplot showing lagged timestamps on x-axis and regression coefficients on y. Columns grouped by trial,
    rows grouped by jitter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g = None
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
        fname = f'{self.output_dir}\\pointplot_lagged_latency_vs_correction.png'
        return self.g.figure, fname

    def _format_df(self):
        """
        Formats the dataframe for the plot
        """
        self.df = (
            self.df.melt(id_vars=['trial', 'block', 'latency', 'jitter', 'instrument', 'abbrev'],
                         value_vars=['lag_1', 'lag_2', 'lag_3', 'lag_4'], value_name='coef', var_name='lag')
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
            scale=1.25
        )

    def _format_ax(self):
        """
        Formats plot subplots and axes
        """
        # Add the reference line now or else it messes up the titles
        self.g.refline(y=0, alpha=1, linestyle='-', color=vutils.BLACK)
        # Set axes tick parameters and linewidth
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
        self.g = None

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
        self.g = None

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
            hue_order=['Keys', 'Drums'], errcolor='#3953a3', errwidth=10, estimator=self.estimator
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


@vutils.plot_decorator
def histplot_r2(
        r: pd.DataFrame, output_dir: str, xvar: str = 'r2', kind: str = 'hist'
) -> tuple[plt.Figure, str]:
    """
    Creates histograms of model parameters stratified by trial and instrument, x-axis variable defaults to R-squared
    """

    # Create the dis plot
    g = sns.displot(r, col='trial', kind=kind, x=xvar, hue="instrument", multiple="stack", palette=vutils.INSTR_CMAP,
                    height=vutils.HEIGHT, aspect=vutils.ASPECT, )
    # Format figure-level properties
    g.set(xlabel='', ylabel='')
    g.set_titles("Duo {col_name}", size=vutils.FONTSIZE)
    g.figure.supxlabel(xvar.title(), y=0.06)
    g.figure.supylabel('Count', x=0.007)
    # Move legend and adjust subplots
    sns.move_legend(g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.03),
                    fontsize=vutils.FONTSIZE)
    g.figure.subplots_adjust(bottom=0.17, top=0.92, left=0.055, right=0.97)
    # Return, with plot_decorator used for saving
    fname = f'{output_dir}\\histplot_{xvar}.png'
    return g.figure, fname


@vutils.plot_decorator
def boxplot_r2_vs_windowsize(
        df: pd.DataFrame, output_dir: str
) -> tuple[plt.Figure, str]:
    """
    Creates a boxplot of average R2 values per rolling window size
    """

    test = df.replace([np.inf, -np.inf], np.nan).dropna().groupby(['trial', 'win_size']).mean()[
        ['aic', 'r2']].reset_index(drop=False)
    g = (
        sns.boxplot(data=test, hue='trial', x='win_size', y='r2', color=vutils.INSTR_CMAP[0])
           .set(xlabel='Window Size (s)', ylabel='Adjusted R2')
    )
    plt.tight_layout()
    fname = f'{output_dir}\\boxplot_r2_vs_windowsize.png'
    return g.figure, fname


# @vutils.plot_decorator
# def pairgrid_correction_vs_condition_iqr(
#         df: pd.DataFrame, value_vars: list, value_name: str, output_dir: str, xvar: str = 'correction_partner',
# ) -> tuple[plt.Figure, str]:
#     """
#     Creates a figure showing pairs of coefficients obtained for each performer in a condition,
#     stratified by block and trial number, with shading according to tempo slope
#     """
#     # Melt the dataframe
#     df = (
#             pd.melt(df, id_vars=['trial', 'block', 'latency', 'jitter', 'instrument', 'tempo_slope'],
#                     value_vars=value_vars, value_name=value_name)
#               .sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument'])
#               .reset_index(drop=False)
#     )
#
#     # Create the abbreviation column, showing latency and jitter
#     df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
#     df = df.sort_values(by=['latency', 'jitter'])
#     # Create the plot
#     plt.rcParams.update({'font.size': vutils.FONTSIZE})
#     pg = sns.catplot(
#         data=df, x=xvar, y='abbrev', row='block', col='trial', hue='instrument',
#         hue_order=['Keys', 'Drums'], palette=vutils.INSTR_CMAP, kind='point', linestyles='', marker='.', s=10,
#         errorbar=lambda v: (min(v), max(v)), estimator=np.median, height=5.5, sharex=True, sharey=True,
#         aspect=0.62, dodge=0.2, plot_kws={'alpha': 1}
#     )
#     ts_df = df.drop_duplicates(subset='tempo_slope', keep='last')
#     norm = vutils.create_normalised_cmap(ts_df['tempo_slope'])
#     # Add the reference line in here or it messes up the plot titles
#     pg.refline(x=0, alpha=1, linestyle='-', color=vutils.BLACK)
#     # Format the axis by iterating through
#     _format_pairgrid_ax(norm, pg, ts_df, xlim=(-1.5, 1.5))
#     _format_pairgrid_fig(pg, norm, xvar=xvar.replace('_', ' ').title() + ' (Q1:Q3)')
#     fname = f'{output_dir}\\pairgrid_condition_vs_{xvar}_iqr.png'
#     return pg.fig, fname
