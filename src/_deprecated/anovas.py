import pandas as pd
import warnings
import matplotlib.pyplot as plt
from bioinfokit.analys import stat
import scipy.stats as stats

from src.analyse.analysis_utils import generate_df
from src._deprecated.linear_regressions import combine_dfs_from_all_trials


def return_levene_results(df: pd.DataFrame) -> tuple[float, float]:
    """Return results from Levene test for homogeneity of variances"""
    # Used to subset dataframe for given latency/jitter value
    lat = lambda x: df[df['latency'] == x]['stdev'].to_list()
    jit = lambda x: df[df['jitter'] == x]['stdev'].to_list()
    # Conduct the statistical test for all groupings within dataframe
    s, p = stats.levene(lat(0), lat(23), lat(45), lat(90), lat(180), jit(0), jit(0.5), jit(1),)
    return s, p


def return_anova(df: pd.DataFrame, mod: str, p_var:str = 'stdev', xfac_var:str = 'latency') -> stat:
    """Return results of a factorial ANOVA"""
    # Bioinfokit returns a FutureWarning for some reason, so disable it
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        res = stat()
        # Conduct the ANOVA
        res.anova_stat(df=df, res_var=p_var, anova_model=mod)
        # Conduct the post-hoc Tukey HSD test to test which groups in the model are significant
        res.tukey_hsd(df=df, res_var=p_var, xfac_var=xfac_var,
                      anova_model=mod)
    return res


def gen_tukey_hsd_plot(df: pd.DataFrame, ax: plt.axis, trial_num: int) -> plt.axis:
    """Generate plot for Tukey HSD post-hoc test for differences in paired group means"""
    # Plot a horizontal line for each pair of conditions, showing range of values
    ax.hlines(range(len(df)), df['Lower'].to_list(), df['Upper'].to_list())
    # Plot a vertical line, showing mean = 0
    ax.vlines(0, -1, len(df) - 1, linestyles='dashed')
    # Format y axis
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f'{x}-{y}' for x, y in zip(df['group1'].to_list(), df['group2'].to_list())])
    ax.set_ylabel('Comparison')
    # Format x axis
    mx = max(df['Upper'].to_list()) + 10
    ax.set_xlim((-abs(mx), mx))
    ax.set_xlabel('Mean difference (ms)')
    # Format title
    ax.set_title(f'Trial {trial_num}')
    return ax


def format_anova_df(res: stat) -> pd.DataFrame:
    """Format the anova results obtained from one trial"""
    # Load the anova results from bioinfokit, round to three decimal places, rename columns and rows
    df = res.anova_summary.round(3).rename(columns={
        'PR(>F)': 'Sig.', 'sum_sq': 'Sum of Squares', 'mean_sq': 'Mean Square'
    }, index={
        'C(latency)': 'Latency', 'C(jitter)': 'Jitter', 'C(latency):C(jitter)': 'Latency: Jitter',
        'C(latency):C(instrument)': 'Latency: Instrument', 'C(latency):C(block)': 'Latency: Block',
        'C(jitter):C(instrument)': 'Jitter: Instrument', 'C(jitter):C(block)': 'Jitter: Block'
    })
    # Set the degrees of freedom column to an int to remove trailing zeros
    df['df'] = df['df'].astype(int)
    # Change column order and return
    return df[['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.']]


def analyse_beat_variance(raw_data: list, output_dir: str) -> pd.DataFrame:
    """
    Analyse variance of crotchet beat durations for each performance vs latency/jitter through factorial ANOVA
    Also runs post-hoc Tukey HSD test to compare means per group pairing
    """
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=True, figsize=(15, 5))
    # Define the model
    mod = 'stdev~C(latency) + C(jitter) + C(latency):C(jitter) + C(latency):C(instrument) ' \
          '+ C(latency):C(block) + C(jitter):C(instrument) + C(jitter):C(block)'
    anovas = []
    # Iterate through all trials
    for trial_num, trial in enumerate(raw_data):
        data = []
        # Iterate through data for each condition in a trial
        for con in trial:
            # Generate the data frame from the midi bpm array
            df = generate_df(data=con['midi_bpm'])
            # Calculate standard deviation of IOI values
            std = df['ioi'].std()
            # Append instrument, standard deviation, latency, and jitter to list
            data.append((con['instrument'], con['block'], con['latency'], con['jitter'], std,))
        # Create a single dataframe per trial
        df = pd.DataFrame(data, columns=['instrument', 'block', 'latency', 'jitter', 'stdev']).sort_values(['latency', 'jitter'])
        # Conduct the ANOVA test per trial
        res = return_anova(df=df, mod=mod)
        anovas.append(format_anova_df(res))
        # Plot results of Tukey HSD pairwise mean comparison
        gen_tukey_hsd_plot(df=res.tukey_summary, ax=ax[trial_num], trial_num=trial_num + 1)
    plt.show()
    # Concat all results together
    bigdf = combine_dfs_from_all_trials(df_list=anovas)
    bigdf.to_csv(f'{output_dir}\\anova_beat_variance_-_latency+jitter.csv', sep=';')
    # Might as well return the dataframe we've created
    return bigdf


def anova_ts_lat_jit(raw_data, output_dir):
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=True, figsize=(15, 5))
    df = pd.DataFrame(raw_data, columns=['trial', 'block', 'latency', 'jitter', 'instrument', 'coeff'])
    mod = 'coeff~C(latency)*C(jitter)+C(latency):C(instrument)+C(latency):C(block)+C(jitter):C(instrument)+C(jitter):C(block)'
    for num, (idx, grp) in enumerate(df.groupby('trial')):
        an = return_anova(grp, mod=mod, p_var='coeff',)
        gen_tukey_hsd_plot(df=an.tukey_summary, ax=ax[num], trial_num=num + 1)
    plt.show()
