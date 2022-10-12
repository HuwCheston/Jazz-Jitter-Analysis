# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.analyse.tempo_slope import *
from src.analyse.tempo_stability import *
from src.analyse.phase_correction import *
from src.analyse.questionnaire import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(
        input_filepath: str, output_filepath: str
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info(f'running analysis scripts on processed data in {input_filepath}')

    # LOAD DATA AND PREPARE #
    # Load the data in as list of lists of dictionaries (one list per trial, one dictionary per condition)
    logger.info(f'loading data...')
    data = autils.load_data(input_filepath)
    logger.info(f'loaded data from {len(data)} trials!')

    # TEMPO SLOPES #
    logger.info(f'Analysing tempo slope...')
    # Get average tempos
    avg_tempos = gen_avg_tempo(data)
    gen_avg_tempo_outputs(avg_tempos, output_dir=output_filepath + '\\figures\\tempo_slopes_plots')
    # Get tempo slopes
    tempo_slope_df = gen_tempo_slope_df(avg_tempos)
    gen_tempo_slope_df_outputs(tempo_slope_df, output_dir=output_filepath + '\\figures\\tempo_slopes_plots')
    # Regress tempo slopes
    tempo_slope_mds = gen_tempo_slope_mds(tempo_slope_df)
    gen_tempo_slope_mds_outputs(tempo_slope_mds, output_dir=output_filepath + '\\figures\\tempo_slopes_plots')

    # TEMPO STABILITY #
    logger.info(f'Analysing tempo stability...')
    # Generate IOI stability and NPVI dataframe
    tempo_stability_df = gen_tempo_stability_df(raw_data=data)
    # Regress IOI stability and generate outputs
    gen_tempo_stability_df_outputs(tempo_stability_df, output_dir=output_filepath + '\\figures\\tempo_stability_plots')
    tempo_stability_mds = gen_tempo_stability_mds(tempo_stability_df)
    gen_tempo_slope_mds_outputs(tempo_stability_mds, output_filepath + '\\figures\\tempo_stability_plots')
    # Regress NPVI stability and generate outputs
    gen_tempo_stability_df_outputs(tempo_stability_df, output_dir=output_filepath + '\\figures\\tempo_stability_plots',
                                   xvar='ioi_npvi', xlabel='IOI normalised pairwise variability index (nPVI)')
    npvi_mds = gen_tempo_stability_mds(tempo_stability_df, md='ioi_npvi~C(latency)+C(jitter)+C(instrument)')
    gen_tempo_stability_mds_outputs(npvi_mds, output_filepath + '\\figures\\tempo_stability_plots',)

    # STATIC SYNCHRONISATION MODELS #
    logger.info(f'Creating static phase correction models...')
    static_mds = gen_phase_correction_models(raw_data=data, output_dir=output_filepath)
    gen_phase_correction_model_outputs(static_mds, output_dir=output_filepath)

    # QUESTIONNAIRES #
    logger.info(f'Analysing questionnaires...')
    questionnaire_analysis(raw_data=data, output_dir=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
