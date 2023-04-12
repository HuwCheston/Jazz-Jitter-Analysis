# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import analysis_utils as autils
from phase_correction_models import *
from simulations import *


@click.command()
@click.option('-i', 'input_filepath', type=click.Path(exists=True), default='data\processed')
@click.option('-o', 'output_filepath', type=click.Path(exists=True), default='models')
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path(exists=True))
def main(
        input_filepath: str, output_filepath: str
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'running analysis scripts on processed data in {input_filepath}...')

    # LOAD DATA AND PREPARE #
    # Load the data in as list of lists of dictionaries (one list per trial, one dictionary per condition)
    logger.info(f'loading data...')
    data = autils.load_data(input_filepath)
    logger.info(f'... successfully loaded data from {len(data)} duos!')

    # CREATE MODELS #
    logger.info(f'generating phase correction models...')
    mds, md_inf = generate_phase_correction_models(data, logger=logger, output_dir=output_filepath, force_rebuild=False)
    logger.info(md_inf)

    # CREATE SIMULATIONS WITH COUPLING PARAMETERS - ANARCHY, DEMOCRACY ETC. #
    logger.info(f'generating {autils.NUM_SIMULATIONS} simulations for each coupling paradigm and condition...')
    sims, sim_info = generate_phase_correction_simulations_for_coupling_parameters(
        mds, output_dir=output_filepath, logger=logger, force_rebuild=False, num_simulations=autils.NUM_SIMULATIONS
    )
    logger.info(sim_info)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
