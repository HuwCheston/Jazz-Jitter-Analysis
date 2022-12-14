import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


class BarPlotInterpolatedIOIs(vutils.BasePlot):
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
        fname = f'{self.output_dir}\\barplot_total_vs_interpolated_beats'
        return self.fig, fname

    def _format_plot(self):
        self.ax.tick_params(width=3, )
        self.ax.set(ylabel='', xlabel='',)
        self.fig.supxlabel('Duo', y=0.02)
        self.fig.supylabel('Total IOIs', x=0.01)
        plt.setp(self.ax.spines.values(), linewidth=2)
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.14, right=0.8)

    def _format_ax(self):
        # Add invisible data to add another legend for instrument
        n1 = [self.ax.bar(0, 0, color=cmap) for i, cmap in zip(range(2), vutils.INSTR_CMAP)]
        l1 = plt.legend(n1, ['Keys', 'Drums'], title='Instrument', bbox_to_anchor=(1, 0.8), ncol=1, frameon=False,)
        self.ax.add_artist(l1)
        # Add invisible data to add another legend for interpolation
        n2 = [self.ax.bar(0, 0, color='gray', hatch=h, alpha=vutils.ALPHA) for i, h in zip(range(2), ['', '//'])]
        l2 = plt.legend(n2, ['No', 'Yes'], bbox_to_anchor=(0.915, 0.5), ncol=1, title='       Interpolation', frameon=False,)
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


class LinePlotZoomCall(vutils.BasePlot):
    """
    Creates a lineplot and histogram showing latency over time for initial zoom call
    """
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerces data into correct format for plotting
        """
        # Get initial zoom array
        sub = (df[(df['latency'] == 180) & (df['jitter'] == 1) & (df['instrument'] == 'Keys') & (df['trial'] == 1) & (
                    df['block'] == 1)])
        zoom_arr = sub['zoom_arr'].values[0]
        # Append timestamps to zoom array
        z = np.c_[zoom_arr, np.linspace(0, 0 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
        sub = pd.DataFrame(z, columns=['latency', 'timestamp'])
        sub['timestamp'] = pd.to_timedelta([timedelta(seconds=val) for val in sub['timestamp']]).seconds
        # Subset zoom array and return
        return sub[sub['timestamp'] <= 90]

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot. Generates and saves in plot decorator
        """
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\lineplot_zoom_call'
        return self.g.figure, fname

    def _create_plot(
            self
    ) -> sns.JointGrid:
        """
        Creates the joint grid
        """
        # Create object in seaborn
        g = sns.JointGrid(data=self.df, x='timestamp', y='latency', )
        # Map line plot to center and histogram to marginal
        g.plot_joint(sns.lineplot, errorbar=None, lw=3, color=vutils.BLACK)
        g.plot_marginals(sns.histplot, kde=True, bins=10, color=vutils.BLACK, line_kws={'lw': 3, })
        # Turn off top marginal plot
        g.ax_marg_x.remove()
        return g

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects
        """
        # Set parameters for both joint and marginal plot
        for ax in [self.g.ax_joint, self.g.ax_marg_y]:
            ax.set(ylabel='', xlabel='')
            ax.tick_params(width=3, )
            plt.setp(ax.spines.values(), linewidth=2)
        # Set parameters for margin plot only
        self.g.ax_marg_y.tick_params(labelbottom=True, bottom=True)
        self.g.ax_marg_y.spines['bottom'].set_visible(True)
        self.g.ax_marg_y.set_ylabel('Density')
        # Set parameters for joint plot only
        self.g.ax_joint.set_xticks([0, 30, 60, 90])
        self.g.ax_joint.spines['top'].set_visible(True)
        self.g.ax_joint.spines['right'].set_visible(True)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure level attributes
        """
        # Set overall size of figure
        self.g.figure.set_size_inches((9.4, 5))
        # Add text to substitute for marginal plot y axis label
        self.g.figure.text(x=0.86, y=0.03, s='Density', fontsize=vutils.FONTSIZE + 3)
        # Add center plot labels
        self.g.figure.supylabel('Latency (ms)')
        self.g.figure.supxlabel('Call duration (s)')
        # Adjust plot spacing slightly
        self.g.figure.subplots_adjust(bottom=0.15, top=1.1, left=0.12, right=0.98, wspace=0.2)


class LinePlotAllConditions(vutils.BasePlot):
    """
    Creates a line plot for all conditions, with one column showing raw latency and another showing windowed standard
    deviation (aka jitter), defaults to 6-second window (8 * 0.75)
    """
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(5, 2, sharex=True, sharey='col', figsize=(18.8, 12))

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Coerce data into correct format for plotting
        """
        # Get the data just for one trial and sort by latency and jitter
        sub = df[(df['trial'] == 1) & (df['instrument'] == 'Keys') & (df['block'] == 1)].reset_index().sort_values(
            by=['latency', 'jitter'])
        res = []
        # Iterate through each row
        for i, r in sub.iterrows():
            # Get the zoom array for this condition
            zoom_arr = r['zoom_arr']
            # Append timestamps to zoom array
            z = np.c_[zoom_arr, np.linspace(0, 0 + (len(zoom_arr) * 0.75), num=len(zoom_arr), endpoint=False)]
            s = pd.DataFrame(z, columns=['latency', 'timestamp'])
            s['timestamp'] = pd.to_timedelta([timedelta(seconds=val) for val in s['timestamp']]).seconds
            # Subset zoom array and return
            res.append((r['latency'], r['jitter'], s))
        # Create a dataframe of all condiitons
        return pd.DataFrame(res, columns=['latency', 'jitter', 'data'])

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Create plot and save in plot decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\lineplot_all_conditions'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Map plots onto axes
        """
        # Iterate through latency values
        for num, (i, g) in enumerate(self.df.groupby(['latency'])):
            # Iterate through jitter scales
            for num_, (i_, g_) in enumerate(g.groupby('jitter')):
                # Plotting latency
                # We don't want to add labels for all our plots
                if num == 1:
                    self.ax[num, 0].plot(
                        g_['data'].values[0]['timestamp'], g_['data'].values[0]['latency'], lw=3, label=f'{i_}x',
                        color=vutils.JITTER_CMAP[num_],
                    )
                else:
                    self.ax[num, 0].plot(
                        g_['data'].values[0]['timestamp'], g_['data'].values[0]['latency'], lw=3,
                        color=vutils.JITTER_CMAP[num_],
                    )
                self.ax[num, 0].set(
                    title=f'{i}ms baseline', ylim=(-25, 300), xticks=[0, 30, 60, 90], yticks=[0, 100, 200, 300]
                )
                # Plotting jitter
                roll = g_['data'].values[0].rolling(window=8, min_periods=1)['latency']
                self.ax[num, 1].plot(
                    g_['data'].values[0]['timestamp'], roll.std(), lw=3, color=vutils.JITTER_CMAP[num_],
                )
                self.ax[num, 1].set(
                    title=f'{i}ms baseline', ylim=(-5, 60), xticks=[0, 30, 60, 90], yticks=[0, 20, 40, 60]
                )
                # Add y labels onto only the middle row of plots
                if num == 2:
                    self.ax[num, 0].set_ylabel('Latency (ms)', labelpad=10, fontsize=vutils.FONTSIZE + 3)
                    self.ax[num, 1].set_ylabel('Latency variability (SD, ms)', labelpad=10, fontsize=vutils.FONTSIZE + 3)

    def _format_ax(
            self
    ) -> None:
        """
        Format axes-level attributes
        """
        for a in self.ax.flatten():
            a.tick_params(width=3, )
            plt.setp(a.spines.values(), linewidth=2)

    def _format_fig(
            self
    ) -> None:
        """
        Format figure-level attributes
        """
        self.fig.supxlabel('Performance duration (s)', )
        self.fig.legend(loc='center right', ncol=1, frameon=False, title='Jitter')
        self.fig.subplots_adjust(top=0.95, bottom=0.09, left=0.07, right=0.9, wspace=0.15, hspace=0.4)


def generate_misc_plots(
    mds: list, output_dir: str,
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\misc_plots'
    stacked_bp = BarPlotInterpolatedIOIs(df=df, output_dir=figures_output_dir)
    stacked_bp.create_plot()
    lp_orig = LinePlotZoomCall(df=df, output_dir=figures_output_dir)
    lp_orig.create_plot()
    lp_all = LinePlotAllConditions(df=df, output_dir=figures_output_dir)
    lp_all.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_misc_plots(mds=raw, output_dir=output)
