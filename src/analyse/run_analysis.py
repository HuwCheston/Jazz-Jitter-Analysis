# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# TODO: fix imports here to use __all__ instead, so that we don't import packages like numpy, pandas etc
from src.analyse.tempo_slope import *
from src.analyse.phase_correction import *


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

    # CREATE MODELS #
    logger.info(f'Creating models...')
    df = gen_phase_correction_models(raw_data=data, output_dir=output_filepath, logger=logger)

    # VISUALISE OUTPUTS
    logger.info(f'Creating tempo slope outputs...')
    avg_tempos = gen_avg_tempo(data)
    gen_avg_tempo_outputs(avg_tempos, output_dir=output_filepath + '\\figures\\tempo_slopes_plots')
    gen_tempo_slope_outputs(df=df, output_dir=output_filepath)

    logger.info(f'Creating tempo stability outputs...')
    gen_tempo_stability_outputs(df, output_dir=output_filepath)

    logger.info(f'Creating phase correction outputs...')
    gen_phase_correction_model_outputs(df, output_dir=output_filepath)

    logger.info(f'Creating questionnaire outputs...')
    gen_questionnaire_outputs(df, output_dir=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
