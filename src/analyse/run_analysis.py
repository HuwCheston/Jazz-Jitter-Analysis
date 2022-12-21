# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# TODO: fix imports here to use __all__ instead, so that we don't import packages like numpy, pandas etc
import src.analyse.analysis_utils as autils
from src.analyse.phase_correction_models import generate_phase_correction_models
from src.analyse.simulations import generate_phase_correction_simulations_individual, \
    generate_phase_correction_simulations_average


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
    logger.info(f'Generating phase correction models...')
    mds = generate_phase_correction_models(data, logger=logger, output_dir=output_filepath, force_rebuild=False)
    logger.info(f'... models generated!')

    # CREATE INDIVIDUAL SIMULATIONS #
    logger.info(f'Generating average phase correction simulations...')
    sims_avg = generate_phase_correction_simulations_average(
        mds, output_dir=output_filepath, logger=logger, force_rebuild=False
    )
    logger.info(f'... simulations generated!')

    # CREATE INDIVIDUAL SIMULATIONS #
    logger.info(f'Generating individual phase correction simulations...')
    sims = generate_phase_correction_simulations_individual(
        mds, output_dir=output_filepath, logger=logger, force_rebuild=False
    )
    logger.info(f'... simulations generated!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
