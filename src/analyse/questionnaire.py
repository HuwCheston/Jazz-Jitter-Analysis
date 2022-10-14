import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score

pd.set_option('display.max_columns', 500)


def return_irr(
        grp: pd.DataFrame.groupby, var: str, rater_grouper: str = 'instrument',
) -> tuple:
    g1, g2 = grp.groupby([rater_grouper])
    f, p = stats.pearsonr(g1[1][var].to_numpy(), g2[1][var].to_numpy())
    return f, p


def return_cohen_kappa(
        grp: pd.DataFrame.groupby, var: str, rater_grouper: str = 'instrument'
) -> float:
    g1, g2 = grp.groupby([rater_grouper])
    return cohen_kappa_score(g1[1][var].to_numpy(), g2[1][var].to_numpy())


def return_kendall_w(
        grp: pd.DataFrame.groupby, var: str, rater_grouper: str = 'instrument'
) -> float:
    g1, g2 = grp.groupby([rater_grouper])
    arr = np.array([g1[1][var].to_numpy(), g2[1][var].to_numpy()])
    m = arr.shape[0]
    n = arr.shape[1]
    denom = m**2*(n**3-n)
    rating_sums = np.sum(arr, axis=0)
    s = n*np.var(rating_sums)
    return 12*s/denom


def recode_categorical_variables(
        d: pd.DataFrame
) -> pd.DataFrame:
    d['success'] = d['success'].replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9): 3})
    d['coordination'] = d['coordination'].replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9): 3})
    d['interaction'] = d['interaction'].replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9): 3})
    return d


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
    irr = pd.DataFrame([(i, s, *return_irr(g, var=s), return_cohen_kappa(g, var=s), return_kendall_w(g, var=s))
                        for s in vrs for i, g in df.groupby(by=['trial'])],
                       columns=['trial', 'variable', 'f', 'p', 'kappa', 'w',])




