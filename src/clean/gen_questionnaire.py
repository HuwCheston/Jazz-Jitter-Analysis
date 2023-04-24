"""Generates questionnaire data and formats"""

import pandas as pd
import numpy as np
import warnings

# pd.read_excel is buggy and gives us loads of warnings in the output, even though it works - so disable warnings
warnings.simplefilter("ignore")
# Set options in pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Define the modules we can import from this file in others
__all__ = [
    'gen_questionnaire_output',
    'gen_perceptual_study_output'
]

# Functions for formatting subjective responses of participants


def _format_qualtrics_output(
        df: pd.DataFrame
) -> list:
    """
    Drops metadata columns, parses as list of separate dataframes corresponding to each trial
    """

    # Drop metadata and warmup columns - first 34
    cols2skip = [num for num in range(0, 34)]
    df = df.drop(df.columns[[cols2skip]], axis=1)
    # Parse dataframe into chunks of two rows (every trial) and return as list of dataframes
    return [df.iloc[num:num + 2, :] for num in range(0, len(df), 2)]


def _format_dataframe_per_trial(
        df: pd.DataFrame, trial_num: int
) -> pd.DataFrame:
    """
    Create dataframe for each individual trial
    """

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


def gen_questionnaire_output(
        input_dir: str
) -> list:
    """
    Clean questionnaire output: return as list of dataframes, one dataframe per trial (both participants)
    """

    # Read the excel spreadsheet created by Qualtrics
    df = pd.read_excel(
        f'{input_dir}/questionnaire_anonymised.xlsx',
        engine="openpyxl",
        skiprows=1,
        index_col=22
    )
    # Format the whole dataframe
    df_list = _format_qualtrics_output(df)
    # Split into chunks (one dataframe per trial, two participants per dataframe)
    edited_df = [_format_dataframe_per_trial(df, num) for num, df in enumerate(df_list, 1)]
    # Split trial df into list of dictionaries for each condition and return list of lists
    return [df.reset_index().to_dict(orient='records') for df in edited_df]

# Functions for formatting subjective responses of perceptual study participants


def _clean_perceptual_data(
        merge: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans perceptual study data by removing answers from participants who failed to complete the study successfully.
    """

    return merge[
        (merge['complete_x'] == True) &
        (merge['failed_x'] == False) &
        (merge['answer_x'].notnull()) &
        (merge['time_taken'].notnull()) &
        (merge['time_taken'] > 45) &    # If below this value, participant finished the task too quickly
        (merge['complete_y'] == True) &
        (merge['failed_y'] == False) &
        (merge['progress'] == 1) &
        (merge['status'] == 'approved')
    ]


def _format_perceptual_data(
        clean: pd.DataFrame
) -> dict:
    """
    Formats perceptual study data by dropping unnecessary columns and setting correct datatypes
    """

    big = pd.concat([clean['definition'].map(eval).apply(pd.Series), clean], axis=1)
    int_cols = [
        'latency', 'jitter', 'hours_of_daily_music_listening',
        'years_of_formal_training', 'age', 'answer_x',
    ]
    for col in int_cols:
        big[col] = big[col].astype(np.int64)
    big['jitter'] = big['jitter'] / 10
    to_keep = [
        'duo', 'session', 'latency', 'jitter', 'id', 'participant_id', 'answer_x',
        'time_taken', 'age', 'gender', 'country', 'formal_education', 'years_of_formal_training',
        'hours_of_daily_music_listening', 'money_from_playing_music', 'feedback',
    ]
    return (
        big.drop(columns=[col for col in big.columns if col not in to_keep])
        .rename({'duo': 'trial', 'session': 'block', 'answer_x': 'answer'}, axis='columns')
        .sort_values(by=['trial', 'block', 'latency', 'jitter'])
        .reset_index(drop=True)
    )


def _perceptual_data_to_dict(
        fmt: pd.DataFrame,
) -> dict:
    """
    Returns perceptual study data as a dictionary, with conditions as the keys and answers as the values (as a list)
    """

    res = []
    for idx_, grp_ in fmt.groupby('trial'):
        res.append([])
        for idx, grp in grp_.groupby(['block', 'latency', 'jitter']):
            idx = list(idx)
            answers = []
            for i_, g_ in grp.iterrows():
                answers.append(
                    g_[[col for col in fmt.columns if col not in ['trial', 'block', 'latency', 'jitter']]].to_dict()
                )
            res[idx_ - 1].append(
                {'trial': idx_, 'block': idx[0], 'latency': idx[1], 'jitter': idx[2], 'perceptual_answers': answers}
            )
    return res


def gen_perceptual_study_output(
        input_dir: str, file_pre: str = 'Database View ', file_post: str = ' -  Dashboard'
) -> dict:
    """
    Clean perceptual study output: return as dictionary, with experimental conditions as keys and answers as values
    """

    ratings_df = pd.read_csv(rf"{input_dir}\{file_pre}SuccessTrial{file_post}.csv")
    participants_df = pd.read_csv(rf"{input_dir}\{file_pre}Participant{file_post}.csv")
    merge = ratings_df.merge(participants_df.rename(columns={'id': 'participant_id'}), on='participant_id')
    clean = _clean_perceptual_data(merge)
    fmt = _format_perceptual_data(clean)
    return _perceptual_data_to_dict(fmt)
