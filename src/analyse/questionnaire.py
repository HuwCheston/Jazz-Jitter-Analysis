import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
from sklearn.metrics import cohen_kappa_score


class InterRaterReliability:
    """

    """
    def __init__(self, **kwargs):
        self.df: pd.DataFrame = kwargs.get('df', None)
        self.recode: bool = kwargs.get('recode', False)
        self.vrs = ['interaction', 'coordination', 'success']
        self.rater_grouper = 'instrument'
        # If we're recoding variables into three groups
        if self.recode:
            for vr in self.vrs:
                self.df[vr] = self._recode_categorical_variables(s=self.df[vr])

    def format_df(self) -> pd.DataFrame:
        """
        Called from outside the class, returns a dataframe grouped by class containing all the required summary stats
        """
        return pd.DataFrame(
            [(i, s, self._return_interclass_correlation(g, var=s), self._return_cohen_kappa(g, var=s), self._return_kendall_w(g, var=s))
             for s in self.vrs for i, g in self.df.groupby(by=['trial'])],
            columns=['trial', 'variable', 'interclass_correlation', 'cohens_kappa', 'kendals_w']
        )

    def _return_interclass_correlation(self, grp: pd.DataFrame.groupby, var: str) -> float:
        """
        Returns the interclass correlation, in the form of Pearson's R statistic
        """
        g1, g2 = grp.groupby([self.rater_grouper])
        f, _ = stats.pearsonr(g1[1][var].to_numpy(), g2[1][var].to_numpy())
        return f

    def _return_cohen_kappa(self, grp: pd.DataFrame.groupby, var: str) -> float:
        """
        Returns Cohen's kappa value
        """
        g1, g2 = grp.groupby([self.rater_grouper])
        return cohen_kappa_score(g1[1][var].to_numpy(), g2[1][var].to_numpy())

    def _return_kendall_w(self, grp: pd.DataFrame.groupby, var: str) -> float:
        """
        Returns Kendall's w value
        """
        g1, g2 = grp.groupby([self.rater_grouper])
        arr = np.array([g1[1][var].to_numpy(), g2[1][var].to_numpy()])
        m = arr.shape[0]
        n = arr.shape[1]
        denom = m ** 2 * (n ** 3 - n)
        rating_sums = np.sum(arr, axis=0)
        s = n * np.var(rating_sums)
        return 12 * s / denom

    def _recode_categorical_variables(self, s: pd.Series) -> pd.Series:
        """
        Replaces a ordinal variable from 1-9 with three categories, in accordance with those in the questionnaire
        """
        return s.replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9): 3})


class TestRetestReliability:
    def __init__(self, **kwargs):
        self.df = kwargs.get('df', None)
        self.output_dir = kwargs.get('output_dir', None)

    def format_trr_df(self,) -> pd.DataFrame:
        """
        Returns a dataframe of test-retest reliability scores for all variables, stratified by individual duos
        """
        res = []
        warnings.simplefilter("ignore")
        cols = ['trial', 'instrument', 'latency', 'jitter']
        for i, g in self.df.groupby('trial'):
            for ins, g1 in g.groupby('instrument'):
                d = {'trial': i, 'instrument': ins}
                b1, b2 = g1.groupby('block')
                for var in self.df.columns:
                    if var not in cols:
                        try:
                            f = stats.pearsonr(b1[1].sort_values(by=['latency', 'jitter'])[var].to_numpy(),
                                               b2[1].sort_values(by=['latency', 'jitter'])[var].to_numpy())[0]
                        except TypeError:
                            pass
                        else:
                            d.update({var: f})
                res.append(d)
        return pd.DataFrame(res).dropna(axis=1)

    def format_df_for_regplot(self, var: str) -> pd.DataFrame:
        """
        Returns a dataframe that can be provided to RegPlotSingle class to create a regression plot of scores of one
        variable across both measures, for all trials
        """
        data = pd.pivot_table(self.df, index=['trial', 'instrument', 'latency', 'jitter'],
                              columns=['block']).reset_index()
        data.columns = [''.join(tuple(map(str, t))) for t in data.columns.values]
        data['trial_abbrev'] = data['trial'].replace({n: f'Duo {n}' for n in range(1, 6)})
        data['Measure 1'] = data[f'{var}1']
        data['Measure 2'] = data[f'{var}2']
        return data


def questionnaire_analysis(
        raw_data: list[list], output_dir: str
) -> None:

    idx = ['trial', 'block', 'latency', 'jitter', 'instrument', 'thoughts']
    vrs = ['interaction', 'coordination', 'success']
    df = (
        pd.DataFrame([c for i in raw_data for c in i], columns=idx.extend(vrs))
          .sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument'])
          .reset_index(drop=True)
    )
