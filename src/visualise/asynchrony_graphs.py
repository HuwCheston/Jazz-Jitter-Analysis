import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils

# Define the objects we can import from this file into others
__all__ = [
    'generate_asynchrony_plots'
]


class NumberLinePairwiseAsynchrony(vutils.BasePlot):
    """
    Creates a numberline showing difference in pairwise asynchrony between duos this experiment during the control
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
        fname = f'{self.output_dir}\\numberline_pairwise_asynchrony'
        return self.g.figure, fname

    def _format_df(self, corpus_filepath) -> pd.DataFrame:
        """
        Formats the dataframe by concatenating with data from the corpus
        """
        # Read in the corpus data
        corpus = pd.read_excel(io=corpus_filepath, sheet_name=0)
        # Read in the data from the experimental dataframe
        trial = self.df[self.df['latency'] == 0]
        trial = (
            trial.drop_duplicates(subset='pw_asym')
                 .groupby('trial')
                 .mean()
                 .reset_index(drop=False)
                 .rename(columns={'trial': 'style'})

        )
        trial = trial[['style', 'pw_asym']]
        trial['source'] = ''
        # Add a bit of noise to trial 3 and 4 so they don't overlap exactly on the graph
        trial.loc[trial['style'] == 3, 'pw_asym'] += 0.5
        trial.loc[trial['style'] == 4, 'pw_asym'] -= 0.5
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
            data=self.df[self.df['this_study'] == True], x='pw_asym', y='placeholder', jitter=False, dodge=False, s=15,
            ax=self.ax, orient='h', marker='o', edgecolor=vutils.BLACK, linewidth=2
        )
        return sns.stripplot(
            data=self.df[self.df['this_study'] == False], x='pw_asym', y='placeholder', jitter=False, dodge=False, s=12,
            ax=self.ax, orient='h', marker='s', edgecolor=vutils.BLACK, linewidth=2
        )

    def _add_annotations(self):
        """
        Add the annotations onto the plot
        """
        for k, v in self.df.iterrows():
            x = v['pw_asym']
            if v['this_study']:
                x -= 0.25
                self.g.annotate(text=v['style'], xy=(v['pw_asym'], 0), xytext=(x, -1.45), rotation=315)
            else:
                x -= 0.5
                self.g.annotate(text=v['style'] + '\n' + v['source'], xy=(v['pw_asym'], 0), xytext=(x, 0.15),
                                rotation=45)

    def _format_plot(self):
        """
        Formats the plot
        """
        # Set axis position
        self.ax.spines['bottom'].set_position(('data', 0))
        # Adjust tick parameters and width
        self.ax.tick_params(axis="x", direction="in", pad=-25, width=3, )
        plt.setp(self.ax.spines.values(), linewidth=2)
        # Set ticks and axis label
        self.g.set(xlim=(15, 45), ylim=(-1, 1), xticks=np.linspace(15, 45, 7), xlabel='', ylabel='')
        plt.yticks([], [])
        self.g.figure.supxlabel('Asynchrony (RMS, ms)', y=0.01)
        # Add arrows and labels showing the direction of the x variable
        for text_x, arr_x, lab in zip([0.6, 0.4], [0.9, 0.1], ['Looser', 'Tighter']):
            self.g.annotate(
                f"${lab}$", (arr_x, 0.93), xytext=(text_x, 0.92), annotation_clip=False,
                textcoords='figure fraction', xycoords='figure fraction', fontsize=vutils.FONTSIZE + 3,
                arrowprops=dict(arrowstyle='->', color=vutils.BLACK, lw=4)
            )
        # Remove the left and bottom axis
        sns.despine(left=True, bottom=False)
        # Adjust plot position slightly
        plt.subplots_adjust(top=0.63, bottom=0.18, left=0.03, right=0.97)
        # Remove the legend
        plt.legend([], [], frameon=False)


def generate_asynchrony_plots(
        mds: list, output_dir: str,
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\asynchrony_plots'

    corpus_dir = r"C:\Python Projects\jazz-jitter-analysis\references\corpus.xlsx"
    nl = NumberLinePairwiseAsynchrony(
        df=df, output_dir=figures_output_dir, corpus_filepath=corpus_dir
    )
    nl.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_asynchrony_plots(mds=raw, output_dir=output)
