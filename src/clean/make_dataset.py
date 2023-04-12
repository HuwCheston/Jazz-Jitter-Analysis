# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from gen_pm_output import gen_pm_output
from gen_raw_midi_output import gen_raw_midi_output
from gen_questionnaire_output import gen_questionnaire_output
from combine_output import combine_output


@click.command()
@click.option('-i', 'input_filepath', type=click.Path(exists=True), default=r'data\raw')
@click.option('-o', 'output_filepath', type=click.Path(exists=True), default='data\processed')
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    # Initialise logger and output dictionary
    logger = logging.getLogger(__name__)
    logger.info(f'making final data set from raw data in {input_filepath}...')
    output = {}

    # Generate and clean questionnaire data
    logger.info('generating questionnaire...')
    output['quest'] = gen_questionnaire_output(input_dir=input_filepath,)
    logger.info('... done!')

    # Generate and clean raw MIDI data
    logger.info('generating raw MIDI data...')
    output['midi_raw'] = gen_raw_midi_output(input_dir=input_filepath)
    logger.info('... done!')

    # Generate and clean MIDI BPM data
    logger.info('generating MIDI quarter note data...')
    output['midi_bpm'] = gen_pm_output(input_dir=input_filepath,)
    logger.info('... done!')

    # Combine outputs, save, and cleanup
    logger.info('combining all outputs and saving final dataset...')
    raw_data = combine_output(input_dir=input_filepath, output_dir=output_filepath, **output)
    logger.info(f'... saved in {output_filepath}')
    return raw_data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
