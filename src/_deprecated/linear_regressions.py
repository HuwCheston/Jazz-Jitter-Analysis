import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from src.analyse.analysis_utils import generate_df

pd.set_option('display.max_columns', None)


def conduct_ols_regression(df: pd.DataFrame, max_latency: int = 180, mod: str = 'coeff~C(latency)+C(jitter)'):
    """Conduct regression of coefficient vs latency, grouped by trial number"""
    # Subset values according to desired maximum latency
    df = df[df['latency'] <= max_latency].sort_values('latency')
    # Conduct the regression, fit the model, and return the results
    return smf.ols(mod, data=df).fit()


def return_ast(p_col: pd.Series, b_col: pd.Series) -> list:
    """Iterates through significance column and appends asterisks to beta coefficient"""
    li = []
    # Zip significance column and coefficient column together and iterate through both
    for p, b in zip(p_col, b_col):
        if float(p) <= 0.001:
            li.append(str(b) + '***')
        elif float(p) <= 0.01:
            li.append(str(b) + '**')
        elif float(p) <= 0.05:
            li.append(str(b) + '*')
        else:
            li.append(str(b))
    return li


def format_ols_regression_output(results):
    """Formats the results summary table generated from the statsmodels OLS regression"""
    # Load the dataframe into pandas
    df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
    # Round all columns to 3 decimal places, padding with zeroes when necessary
    for col in df.columns:
        df[col] = df[col].map('{:.3f}'.format).str.pad(width=3, side='right', fillchar='0')
    # Add R2 and Adjusted R2 row to bottom of dataframe, with all cells other than first blank
    df.loc[len(df.index)] = [str(round(results.rsquared, 3)).ljust(5, '0'), *('' for _ in range(len(df.columns)-1))]
    df.loc[len(df.index)] = [str(round(results.rsquared_adj, 3)).ljust(5, '0'), *('' for _ in range(len(df.columns)-1))]
    # Rename index
    df = df.rename(columns={'std err': 'SE'})
    return df


def format_mixedlm_regression_output(results) -> pd.DataFrame:
    """Formats the results summary table generated from the statsmodels mixedlm regression"""
    # Load the dataframe into pandas
    df = results.summary().tables[1].apply(pd.to_numeric, errors='coerce')
    # Round all columns to 3 decimal places, padding with zeroes when necessary
    for col in df.columns:
        df[col] = df[col].map('{:.3f}'.format).str.pad(width=3, side='right', fillchar='0')
    # Rename index
    df = df.rename(columns={'Std. Err': 'SE'})
    return df


def combine_dfs_from_all_trials(df_list):
    """Combines dataframes from all trials together and adds hierarchical column headers denoting trial number"""
    return pd.concat([pd.concat({f'Trial {num}': df}, axis=1) for num, df in enumerate(df_list, 1)], axis=1)


def lr_tempo_slope(tempo_slopes_data, output_dir):
    """Conducts regression of latency vs slope coefficients for each trial, """
    mod = 'slope ~ C(latency) * C(jitter) + C(block) + condition'
    # Construct dataframe from raw tempo slopes data
    df = pd.DataFrame(tempo_slopes_data, columns=['trial', 'block', 'condition', 'latency', 'jitter', 'slope'])
    df['group'] = 1
    crossed_md = smf.mixedlm("slope ~ C(latency) + C(jitter)", groups='group',
                             vc_formula={"trial": "0 + C(trial)", "block": "0 + C(block)"}, data=df).fit()
    # # Conduct the regressions for data from each trial
    # res = [conduct_regression(df=group, mod=mod, max_latency=180) for idx, group in df.groupby(['trial'])]
    # # for r in res:
    # #     print(r.summary())
    # # Format the output from each trial individually
    # df_list = [format_regression_output(i) for i in res]
    # # Format the complete dataframe
    # big_df = combine_dfs_from_all_trials(df_list)
    # # Save the dataframe as a .csv file in the output folder
    # big_df.to_csv(f'{output_dir}\\regress_slope_-_latency+jitter.csv', sep=';')
    return df


def lr_beat_variance(raw_data, output_dir):
    # Iterate through all trials
    all_dfs = []
    all_mds = []
    for trial_num, trial in enumerate(raw_data):
        data = []
        # Iterate through data for each condition in a trial
        for con in trial:
            # Generate the data frame from the midi bpm array
            df = generate_df(data=con['midi_bpm'])
            # Calculate standard deviation of IOI values
            std = df['ioi'].std()
            # Append instrument, standard deviation, latency, and jitter to list
            data.append(
                (con['trial'], con['instrument'], con['block'], con['condition'], con['latency'], con['jitter'], std,)
            )
        # Create a single dataframe per trial
        df = pd.DataFrame(data, columns=['trial', 'instrument', 'block', 'condition', 'latency', 'jitter', 'std'])
        all_dfs.append(df)
        # Create a mixed linear model with block number as random effect
        # If we don't sort the values and reset the index, we get an error when creating the model.
        df = df.sort_values(by=['trial', 'block', 'latency']).reset_index(drop=True)
        reg = smf.ols('std ~ C(latency)*C(jitter)+C(instrument)', data=df).fit()
        print(reg.summary())
    # Create the full dataframe with data from all trials
    bigdf = pd.concat(all_dfs).sort_values(by=['trial', 'block', 'latency']).reset_index(drop=True)

