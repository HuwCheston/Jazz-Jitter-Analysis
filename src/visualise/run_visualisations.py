"""Central file for running all visualisation functions, called by run.cmd"""

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Import our helper objects
import src.visualise.visualise_utils as vutils

# Import all graphs for performance success
from src.visualise.all_metrics_graphs import *
from src.visualise.tempo_slope_graphs import *
from src.visualise.ioi_variability_graphs import *
from src.visualise.asynchrony_graphs import *
from src.visualise.success_graphs import *

# Import all graphs for phase correction
from src.visualise.phase_correction_graphs import *
from src.visualise.simulations_graphs import *


@click.command()
@click.option('-i', 'input_filepath', type=click.Path(exists=True), default='models')
@click.option('-o', 'output_filepath', type=click.Path(exists=True), default=r'reports\figures')
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'generating graphs from models in {input_filepath}')

    # LOAD DATA AND PREPARE #
    # Load the data in as list of lists of dictionaries (one list per trial, one dictionary per condition)
    logger.info(f'loading models...')
    mds = vutils.load_from_disc(input_filepath, filename='phase_correction_mds.p')
    logger.info(f'... loaded {len(mds)} models!')
    logger.info(f'loading simulations...')
    sims_params = vutils.load_from_disc(input_filepath, filename='phase_correction_sims.p')
    logger.info(f'... loaded {len(sims_params)} simulations!')

    # GENERATE PERFORMANCE SUCCESS PLOTS #
    # ALL METRICS TOGETHER #
    logger.info(f'generating plots for all metrics...')
    generate_all_metrics_plots(mds, output_filepath)
    logger.info(f'... done!')
    # TEMPO SLOPE #
    logger.info(f'generating plots for tempo slope metric...')
    generate_tempo_slope_plots(mds, output_filepath)
    logger.info(f'... done!')
    # TIMING IRREGULARITY #
    logger.info(f'generating plots for timing irregularity metric...')
    generate_tempo_stability_plots(mds, output_filepath)
    logger.info(f'... done!')
    # ASYNCHRONY #
    logger.info(f'generating plots for asynchrony metric...')
    generate_asynchrony_plots(mds, output_filepath)
    logger.info(f'... done!')
    # SUCCESS #
    logger.info(f'generating plots for self-reported success metric...')
    generate_questionnaire_plots(mds, output_filepath)
    logger.info(f'... done!')

    # GENERATE PHASE CORRECTION PLOTS #
    logger.info(f'generating plots for phase correction models...')
    generate_phase_correction_plots(mds, output_filepath)
    logger.info(f'... done!')

    # GENERATE SIMULATION PLOTS #
    logger.info(f'generating plots for simulations...')
    # generate_plots_for_individual_performance_simulations(sims_indiv, output_filepath)
    generate_plots_for_simulations_with_coupling_parameters(sims_params, output_filepath)
    logger.info(f'... done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
