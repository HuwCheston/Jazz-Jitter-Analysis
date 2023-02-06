import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils

# Define the objects we can import from this file into others
__all__ = [
    'generate_tempo_slope_plots'
]


class LinePlotTempoSlopes(vutils.BasePlot):
    """
    Creates a lineplot for each individual performance showing average tempo progression over the condition
    """
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self._window_size = kwargs.get('window_size', 8)
        self._ioi_threshold = kwargs.get('ioi_threshold', 0.2)
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(nrows=5, ncols=13, sharex='all', sharey='all', figsize=(18.8, 10))

    def _format_df(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Formats dataframe into correct format for plotting
        """
        res = []
        # Iterate through each condition, both performers
        for idx, grp in df.groupby(['trial', 'block', 'latency', 'jitter']):
            idx = tuple(idx)
            grp = grp.sort_values(by='instrument')
            # Iterate through each performer individually, create dataframe, concatenate
            avg = pd.concat(
                [self._format_performance_df(g['raw_beats'].values[0][0]) for _, g in grp.groupby('instrument')], axis=1
            )
            # Get the rolling average BPM according to the window size
            avg = (60 / avg.mean(axis=1)).rolling(window=timedelta(seconds=self._window_size), min_periods=4).mean()
            # Append the raw data to our list, alongside index variables
            res.append({'trial': idx[0], 'block': idx[1], 'latency': idx[2], 'jitter': idx[3], 'tempo_slope': avg})
        return pd.DataFrame(res)

    def _format_performance_df(
            self, g
    ) -> pd.DataFrame:
        """
        Coerces an individual performance dataframe into the correct format.
        """
        # Create the performance dataframe, sort values by onset, and drop unnecessary columns
        perf_df = pd.DataFrame(g, columns=['my_onset', 'p', 'v']).sort_values('my_onset').drop(columns=['p', 'v'])
        # Get IOIs
        perf_df['my_prev_ioi'] = perf_df['my_onset'].diff().astype(float)
        # Remove ioi values below threshold
        temp = perf_df.dropna()
        temp = temp[temp['my_prev_ioi'] < self._ioi_threshold]
        perf_df = perf_df[~perf_df.index.isin(temp.index)]
        # Recalculate IOI column after removing those below threshold
        perf_df['my_prev_ioi'] = perf_df['my_onset'].diff().astype(float)
        # Resample the dataframe to get mean IOI every second and return
        return autils.resample(perf_df)

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to carry out all plotting functions and save
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\lineplot_tempo_slopes'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates individual lineplots for each condition
        """
        # X counter variable, incremented manually
        x = 0
        # Iterate through each individual condition (both blocks)
        for idx, grp in self.df.groupby(['trial', 'latency', 'jitter']):
            idx = tuple(idx)
            # Y counter variable, equivalent to the current duo number subtract 1
            y = idx[0] - 1
            # Iterate through each individual repeat of a condition
            for i, g in grp.groupby('block'):
                # Plot the individual condition
                pdf = g['tempo_slope'].values[0]
                self.ax[y, x].plot(pdf.index.seconds, pdf, color=vutils.LINE_CMAP[i - 1], lw=2, label=f'Repeat {i}')
            # Set axes titles, labels now as we have access to our index variables already
            if y == 0:
                if x == 0:
                    self.ax[y, x].set_title(f'Latency:   {idx[1]}ms\n   Jitter:     {idx[2]}x', x=-0.05)
                else:
                    self.ax[y, x].set_title(f'{idx[1]}ms\n{idx[2]}x')
            if x == 0:
                self.ax[y, x].set_ylabel(f'Duo {idx[0]}', rotation=90)
            # Increment or reset our x counter variable
            x += 1
            if x > 12:
                x = 0

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects
        """
        # Iterate through all axes
        for ax in self.ax.flatten():
            # Set x and y ticks, axes limits
            ax.set(xlim=(0, 101), ylim=(30, 160), xticks=[0, 50], xticklabels=[0, 50], )
            ax.tick_params(axis='both', which='both', bottom=False, left=False, )
            plt.setp(ax.spines.values(), linewidth=2)
            # Add horizontal line at metronome tempo
            ax.axhline(y=120, color=vutils.BLACK, linestyle='--', alpha=vutils.ALPHA, label='Metronome tempo', lw=2)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Set x and y labels, title
        self.fig.supxlabel('Performance duration (s)', y=0.06)
        self.fig.supylabel(f'Average tempo (BPM, {self._window_size}-seconds rolling)', x=0.01)
        # Add the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        self.fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
        # Reduce the space between plots a bit
        self.fig.subplots_adjust(bottom=0.13, top=0.9, wspace=0.05, hspace=0.05, right=0.98, left=0.08)


class NumberLineTempoSlope(vutils.BasePlot):
    """
    Creates a numberline showing difference in tempo slope between duos this experiment during the control
    condition and a corpus of pairwise asynchrony values from other studies and genres
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(corpus_filepath=kwargs.get('corpus_filepath', None))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9.4 * 2, 5.5))

    @vutils.plot_decorator
    def create_plot(self) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate and save the image.
        """
        self.g = self._create_plot()
        self._add_annotations()
        self._format_plot()
        fname = f'{self.output_dir}\\numberline_tempo_slope'
        return self.g.figure, fname

    def _format_df(self, corpus_filepath) -> pd.DataFrame:
        """
        Formats the dataframe by concatenating with data from the corpus
        """
        # Read in the corpus data
        corpus = pd.read_excel(io=corpus_filepath, sheet_name=1)
        # Read in the data from the experimental dataframe
        trial = self.df[self.df['latency'] == 0]
        trial = (
            trial.drop_duplicates(subset='tempo_slope')
                 .groupby('trial')
                 .mean()
                 .reset_index(drop=False)
                 .rename(columns={'trial': 'style'})
        )
        trial = trial[['style', 'tempo_slope']]
        trial['source'] = ''
        trial['style'] = trial['style'].replace({n: f'Duo {n}, this study' for n in range(1, 6)})
        # Concatenate trial and corpus data together
        self.df = pd.concat([corpus.reset_index(drop=True), trial.reset_index(drop=True)])
        self.df['this_study'] = (self.df['source'] == '')
        self.df['placeholder'] = ''
        return self.df

    def _create_plot(self):
        """
        Creates the facetgrid object
        """
        _ = sns.stripplot(
            data=self.df[self.df['this_study'] == True], x='tempo_slope', y='placeholder', jitter=False, dodge=False,
            s=15, ax=self.ax, orient='h', marker='o', edgecolor=vutils.BLACK, linewidth=2
        )
        return sns.stripplot(
            data=self.df[self.df['this_study'] == False], x='tempo_slope', y='placeholder', jitter=False, dodge=False,
            s=12, ax=self.ax, orient='h', marker='s', edgecolor=vutils.BLACK, linewidth=2
        )

    def _add_annotations(self):
        """
        Add the annotations onto the plot
        """
        for k, v in self.df.iterrows():
            x = v['tempo_slope']
            if v['this_study']:
                self.g.annotate(text=v['style'], xy=(v['tempo_slope'], 0), xytext=(x, -1.45), rotation=315)
            else:
                self.g.annotate(text=v['style'] + '\n' + v['source'], xy=(v['tempo_slope'], 0), xytext=(x, 0.15),
                                rotation=45)

    def _format_plot(self):
        """
        Formats the plot
        """
        # Format the plot
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.tick_params(axis="x", direction="in", pad=-25, width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        self.g.set(xlim=(-0.15, 0.15), ylim=(-1, 1), xticks=np.linspace(-0.15, 0.15, 3), xlabel='', ylabel='')
        self.g.figure.supxlabel('Tempo slope (BPM/s)', y=0.01)
        for text_x, arr_x, lab in zip([0.55, 0.35], [0.9, 0.1], ['Accelerating', 'Decelerating']):
            self.g.annotate(
                f"${lab}$", (arr_x, 0.93), xytext=(text_x, 0.92), annotation_clip=False,
                textcoords='figure fraction', xycoords='figure fraction', fontsize=vutils.FONTSIZE + 3,
                arrowprops=dict(arrowstyle='->', color=vutils.BLACK, lw=4)
            )
        self.g.annotate(
            f"Absolute stability\n(no tempo change)", (0.5, 0.5), xytext=(0.43, 0.7), annotation_clip=False,
            textcoords='figure fraction', xycoords='figure fraction', fontsize=vutils.FONTSIZE + 3,
            arrowprops=dict(arrowstyle='->', color=vutils.BLACK, lw=4)
        )
        sns.despine(left=True, bottom=False)
        # Adjust plot position slightly
        plt.subplots_adjust(top=0.63, bottom=0.18, left=0.03, right=0.97)
        plt.yticks([], [])
        plt.legend([], [], frameon=False)


class BarPlotTempoSlope(vutils.BasePlot):
    """
    Creates two bar plots showing tempo slope against latency and jitter, stratified by duo number
    """
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df[df['instrument'] == 'Keys']
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False, figsize=(18.8, 5))

    @vutils.plot_decorator
    def create_plot(self):
        """
        Called from outside the class to carry out all plotting functions and save the figure
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\barplot_tempo_slope'
        return self.fig, fname

    def _create_plot(self):
        """
        Creates the strip and bar plot for each variable
        """
        for num, var in enumerate(['latency', 'jitter']):
            _ = sns.stripplot(
                data=self.df, x=var, y='tempo_slope', hue='trial', dodge=True, palette='dark:' + vutils.BLACK,
                s=6, marker='.', jitter=0.1, ax=self.ax[num],
            )
            _ = sns.barplot(
                data=self.df, x=var, y='tempo_slope', hue='trial', ax=self.ax[num], errorbar='se', errcolor='#3953a3',
                palette=vutils.DUO_CMAP, errwidth=5, estimator=np.mean, edgecolor=vutils.BLACK, lw=2
            )

    def _format_ax(self):
        """
        Formats axes objects
        """
        # Set ax formatting
        for num, (ax, var) in enumerate(zip(self.ax.flatten(), ['Latency (ms)', 'Jitter'])):
            ax.tick_params(width=3, )
            ax.set(ylabel='Tempo slope (BPM/s)' if num == 0 else '', xlabel=var, ylim=(-0.6, 0.6))
            plt.setp(ax.spines.values(), linewidth=2)
            # Plot a horizontal line at x=0
            ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK, linewidth=2)

    def _format_fig(self):
        """
        Formats figure objects
        """
        handles, labels = None, None
        # Remove legend from all plots but keep handles and labels
        for ax in self.ax.flatten():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
        # Readd the legend, keeping only the handles and labels that we want
        plt.legend(handles[5:], labels[5:], ncol=1, title='Duo', frameon=False, bbox_to_anchor=(1, 0.8))
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07, right=0.93, wspace=0.05)


def generate_tempo_slope_plots(
        mds: list, output_dir: str
):
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\tempo_slopes_plots'
    bp = BarPlotTempoSlope(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    lp = LinePlotTempoSlopes(df=df, output_dir=figures_output_dir)
    lp.create_plot()
    corpus_dir = r"C:\Python Projects\jazz-jitter-analysis\references\corpus.xlsx"
    nl = NumberLineTempoSlope(df=df, output_dir=figures_output_dir, corpus_filepath=corpus_dir)
    nl.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    generate_tempo_slope_plots(raw, output)
