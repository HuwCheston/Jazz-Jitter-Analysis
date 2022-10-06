# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import src.analyse.analysis_utils as autils
from src.analyse.phase_correction import gen_static_phase_correction_models, \
    gen_rolling_phase_correction_models, gen_static_model_outputs, gen_rolling_model_outputs
from src.analyse.questionnaire import questionnaire_analysis


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

    # SYNCHRONISATION MODELS #
    logger.info(f'Creating static phase correction models...')
    static_mds = gen_static_phase_correction_models(raw_data=data, output_dir=output_filepath)
    gen_static_model_outputs(static_mds, output_dir=output_filepath)

    logger.info(f'Creating rolling phase correction models...')
    rolling_mds = gen_rolling_phase_correction_models(raw_data=data)
    gen_rolling_model_outputs(rolling_mds, output_dir=output_filepath)

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
