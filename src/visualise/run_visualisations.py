# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.analyse.prepare_data import load_data
from tempo_slope_graphs import gen_tempo_slope_graph, gen_tempo_slope_heatmap, tempo_slope_animation


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'running analysis scripts on processed data in {input_filepath}')

    # LOAD DATA AND PREPARE #
    # Load the data in as list of lists of dictionaries (one list per trial, one dictionary per condition)
    logger.info(f'loading data...')
    data = load_data(input_filepath)

    logger.info(f'generating tempo slope graphs...')
    gen_tempo_slope_graph(raw_data=data, output_dir=output_filepath)
    tempo_slope_animation(raw_data=data, output_dir=output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
