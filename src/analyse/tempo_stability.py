import pandas as pd
from datetime import timedelta

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
from src.visualise.phase_correction_graphs import pairgrid_correction_vs_condition


def extract_rolling_ioi_std(
        df: pd.DataFrame, win_size
) -> pd.Series:
    """
    Runs a sliding window along the dataset and extracts the standard deviation of IOI values
    """
    df['td'] = pd.to_timedelta([timedelta(seconds=val) for val in df['onset']])
    return df.rolling(window=win_size, on='td')['ioi'].std()


# TODO: consider making this its own package?
def extract_npvi(
        s: pd.Series,
) -> float:
    """
    Extracts the normalised pairwise variability index (nPVI) from a column of IOIs
    """
    # Drop nan values and convert to list
    li = s.dropna().tolist()
    # Extract constant term (left side of equation)
    m = 100 / (len(li) - 1)
    # Calculate the right side of the equation
    s = sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(li, li[1:])])
    # Return nPVI
    return s * m


def gen_tempo_stability_df(
        raw_data: list, win_size: int = 8, rolling: bool = True, npvi: bool = True
) -> pd.DataFrame:
    """
    Extract the stability of IOIs from each performances. Can use either the standard deviation of the entire dataset
    (rolling = False), or the median standard deviation from a rolling window applied across the data (rolling = True,
    default). Window size specified in seconds using the win_size argument, defaults to 8.
    """
    # Zip conditions together, for tempo slope
    b = autils.zip_same_conditions_together(raw_data)
    res = []
    # Iterate through both performers
    for z in b:
        for c1, c2 in z:
            # Create dataframes
            keys, drms = autils.generate_df(c1['midi_bpm']), autils.generate_df(c2['midi_bpm'])
            # Extract tempo slope
            tempo_slope = (
                autils.reg_func(autils.average_bpms(keys, drms), ycol='bpm_avg', xcol='elapsed')
                      .params.iloc[1:].values[0]
            )
            # Extract IOI standard deviation
            if rolling:     # If we're using the median of rolling standard deviations
                keys_std = extract_rolling_ioi_std(keys, win_size=timedelta(seconds=win_size)).median()
                drms_std = extract_rolling_ioi_std(drms, win_size=timedelta(seconds=win_size)).median()
            else:   # If we're using the standard deviation of the entire dataset
                keys_std = keys['onset'].std()
                drms_std = drms['onset'].std()
            # Extract IOI normalised pairwise variability index
            keys_npvi = extract_npvi(keys['ioi'])
            drms_npvi = extract_npvi(drms['ioi'])
            # Append the results to the list
            res.append((c1['trial'], c1['block'], c1['latency'], c1['jitter'], c1['instrument'],
                        tempo_slope, keys_std, keys_npvi))
            res.append((c2['trial'], c2['block'], c2['latency'], c2['jitter'], c2['instrument'],
                        tempo_slope, drms_std, drms_npvi))
    # Return as a dataframe
    return (
        pd.DataFrame(res, columns=['trial', 'block', 'latency', 'jitter', 'instrument',
                                   'tempo_slope', 'ioi_std', 'ioi_npvi'])
          .sort_values(by=['trial', 'block', 'latency', 'jitter', 'instrument'])
          .reset_index(drop=True)
    )


def gen_tempo_stability_df_outputs(
        tempo_stability_df: pd.DataFrame, output_dir: str, xvar: str = 'ioi_std',
        xlabel='Median IOI standard deviation, 8-second window (ms)'
) -> None:
    """
    Generates outputs from a dataframe including a tempo stability column
    """
    pairgrid_correction_vs_condition(tempo_stability_df, output_dir=output_dir, xvar=xvar,
                                     xlabel=xlabel,
                                     xlim=(0, tempo_stability_df[xvar].max() + (tempo_stability_df[xvar].max() / 10)))


def gen_tempo_stability_mds(
        tempo_stability_df: pd.DataFrame, md: str = 'ioi_std~C(latency)+C(jitter)+C(instrument)'
) -> list:
    """
    Generates a list of regression outputs from a dataframe that includes a tempo stability column
    """
    return autils.create_model_list(df=tempo_stability_df, avg_groupers=['latency', 'jitter', 'instrument'], md=md)


def gen_tempo_stability_mds_outputs(
        tempo_stability_mds: list, output_dir: str
) -> None:
    """
    Generates outputs from a list of regressions
    """
    vutils.output_regression_table(mds=tempo_stability_mds, output_dir=output_dir)
