import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


class NumberLinePairwiseAsynchrony(vutils.BasePlot):
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
        fname = f'{self.output_dir}\\numberline_pairwise_asynchrony'
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
            x = v['pw_asym']
            if v['this_study']:
                x += 0.1
            self.g.annotate(
                text=v['style'] + '\n' + v['source'], xy=(v['pw_asym'], 0), xytext=(x, -0.3),
                rotation=45
            )

    def _format_plot(self):
        """
        Formats the plot
        """
        # Add the horizontal line
        self.g.axhline(y=0, alpha=1, linestyle='-', color=vutils.BLACK, linewidth=3)
        # Format the plot
        self.g.set(xlim=(17, 41), xticks=np.arange(15, 41, 5, ), xlabel='', ylabel='')
        self.g.figure.supxlabel('Pairwise asynchrony (ms)', y=0.05)
        sns.despine(left=True, bottom=True)
        plt.subplots_adjust(top=0.34, bottom=0.25, left=0.05, right=0.93)
        plt.yticks([], [])
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

    # TODO: corpus should be saved in the root//references directory!
    nl = NumberLinePairwiseAsynchrony(
        df=df, output_dir=figures_output_dir, corpus_filepath=f'{output_dir}\\pw_asymmetry_corpus.xlsx'
    )
    nl.create_plot()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_asynchrony_plots(mds=raw, output_dir=output)
