import pandas as pd
import warnings
# pd.read_excel is buggy and gives us loads of warnings in the output, even though it works - so disable warnings
warnings.simplefilter("ignore")
# Set options in pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def format_qualtrics_output(df):
    """Drops metadata columns, parses as list of separate dataframes corresponding to each trial"""
    # Drop metadata and warmup columns - first 34
    cols2skip = [num for num in range(0, 34)]
    df = df.drop(df.columns[[cols2skip]], axis=1)
    # Parse dataframe into chunks of two rows (every trial) and return as list of dataframes
    return [df.iloc[num:num+2, :] for num in range(0, len(df), 2)]


def format_dataframe_per_trial(df, trial_num) -> pd.DataFrame:
    """Create dataframe for each individual trial"""
    # Rename the columns and pivot
    df = pd.concat(
        (pd.DataFrame(
            {
                'trial': trial_num,
                'block': 1 if num <= 13 else 2,
                'condition': num if num <= 13 else num - 13,
                'interaction': df.iloc[:, i],
                'coordination': df.iloc[:, i + 1],
                'success': df.iloc[:, i + 2],
                'thoughts': df.iloc[:, i + 3]
            }) for (num, i) in enumerate(range(0, len(df.columns), 4), 1))
    )
    # Set the index
    df.index.names = ['instrument']
    df = df.reset_index().set_index(['condition', 'instrument'])
    return df


def save_questionnaire(xls_path, df_list):
    """Saves the questionnaire to the processed data folder"""
    # TODO: this should be moved into combine_outputs.py
    with pd.ExcelWriter(xls_path) as writer:
        for (n, df) in enumerate(df_list, 1):
            df.to_excel(writer, f'trial{n}')


def gen_questionnaire_output(input_dir) -> list:
    """Clean questionnaire output: return as list of dataframes, one dataframe per trial (both participants)"""
    # Read the excel spreadsheet created by Qualtrics
    df = pd.read_excel(
        f'{input_dir}/questionnaire_anonymised.xlsx',
        engine="openpyxl",
        skiprows=1,
        index_col=22
    )
    # Format the whole dataframe
    df_list = format_qualtrics_output(df)
    # Split into chunks (one dataframe per trial, two participants per dataframe)
    edited_df = [format_dataframe_per_trial(df, num) for num, df in enumerate(df_list, 1)]
    # Split trial df into list of dictionaries for each condition and return list of lists
    return [df.reset_index().to_dict(orient='records') for df in edited_df]
