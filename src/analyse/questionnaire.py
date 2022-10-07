import pandas as pd
import numpy as np
import scipy.stats as stats

pd.set_option('display.max_columns', 500)


def return_irr(
        grp: pd.DataFrame.groupby, var: str, rater_grouper: str = 'instrument',
) -> tuple:
    g1, g2 = grp.groupby([rater_grouper])
    f, p = stats.pearsonr(g1[1][var].to_numpy(), g2[1][var].to_numpy())
    return f, p


def return_test_retest_reliability(
        df: pd.DataFrame
) -> tuple:
    pass


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
    irr = pd.DataFrame([(i, s, *return_irr(g, var=s)) for s in vrs for i, g in df.groupby(by=['trial'])],
                       columns=['trial', 'variable', 'f', 'p'])

