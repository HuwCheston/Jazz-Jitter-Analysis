import pandas as pd

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
import src.visualise.tempo_slope_graphs


def gen_avg_tempo(
        raw_data: list
) -> list[tuple]:
    """
    For each condition, generates the average BPM of both performers, sampled every second.
    """
    # Zip both performances per condition together
    b = autils.zip_same_conditions_together(raw_data)
    # Define the function used to subset the raw data
    s = lambda c: (c['trial'], c['block'], c['condition'], c['latency'], c['jitter'])
    # Return the averaged bpms per performance, alongside the rest of the required metadata
    return [(*s(c1), autils.average_bpms(autils.generate_df(c1['midi_bpm']), autils.generate_df(c2['midi_bpm'])))
            for z in b for c1, c2 in z]


def gen_avg_tempo_outputs(
        avg_tempo_list: list[tuple], output_dir: str
) -> None:
    """
    Generate outputs from a list including average tempo across performers
    """
    src.visualise.tempo_slope_graphs.gen_tempo_slope_graph(data=avg_tempo_list, output_dir=output_dir)


def gen_tempo_slope_df(
        avg_tempos: list[tuple]
) -> pd.DataFrame:
    """
    From a list including BPMs averaged across performers, generates a dataframe with one observation per performance,
    with tempo slope calculated as the coefficient of a regression of BPM vs elapsed time
    """
    # Define the function to extract tempo slope coefficient
    # TODO: this should probably be it's own function?
    s = lambda c: autils.reg_func(c, ycol='bpm_avg', xcol='elapsed').params.iloc[1:].values[0]
    # Create the data, applying the function only to the dataframe within each tuple
    res = [(t if not isinstance(t, pd.DataFrame) else s(t) for t in tup) for tup in avg_tempos]
    # Return the output as a dataframe
    return (
        pd.DataFrame(res, columns=['trial', 'block', 'condition', 'latency', 'jitter', 'tempo_slope'])
          .sort_values(by=['trial', 'block', 'latency', 'jitter'])
          .reset_index(drop=True)
    )


def gen_tempo_slope_df_outputs(
        tempo_slope_df: pd.DataFrame, output_dir: str
) -> None:
    """
    Generates outputs from a dataframe that includes a tempo slope column
    """
    src.visualise.tempo_slope_graphs.gen_tempo_slope_heatmap(df=tempo_slope_df, output_dir=output_dir)


def gen_tempo_slope_mds(
        tempo_slope_df: pd.DataFrame, md: str = 'tempo_slope~C(latency)+C(jitter)'
) -> list:
    """
    Generates a list of regression outputs from a dataframe that includes a tempo slope column
    """
    return autils.create_model_list(df=tempo_slope_df, avg_groupers=['latency', 'jitter'], md=md)


def gen_tempo_slope_mds_outputs(
        tempo_slope_mds: list, output_dir: str
) -> None:
    """
    Generates outputs from a dataframe including a tempo slope column
    """
    # Output the regression table
    vutils.output_regression_table(mds=tempo_slope_mds, output_dir=output_dir)
