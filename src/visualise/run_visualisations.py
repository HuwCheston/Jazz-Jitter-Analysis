# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Import our helper objects
import src.analyse.analysis_utils as autils

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
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'running analysis scripts on processed data in {input_filepath}')

    # LOAD DATA AND PREPARE #
    # Load the data in as list of lists of dictionaries (one list per trial, one dictionary per condition)
    logger.info(f'loading data...')
    mds = autils.load_from_disc(input_filepath, filename='phase_correction_mds.p')
    sims_indiv = autils.load_from_disc(input_filepath, filename='phase_correction_sims.p')
    sims_params = autils.load_from_disc(input_filepath, filename='phase_correction_sims_average.p')

    # GENERATE PERFORMANCE SUCCESS PLOTS #
    # ALL METRICS TOGETHER #
    generate_all_metrics_plots(mds, output_filepath)
    # TEMPO SLOPE #
    generate_tempo_slope_plots(mds, output_filepath)
    # TIMING IRREGULARITY #
    generate_tempo_stability_plots(mds, output_filepath)
    # ASYNCHRONY #
    generate_asynchrony_plots(mds, output_filepath)
    # SUCCESS #
    generate_questionnaire_plots(mds, output_filepath)

    # GENERATE PHASE CORRECTION PLOTS #
    generate_phase_correction_plots(mds, output_filepath)

    # GENERATE SIMULATION PLOTS #
    generate_plots_for_individual_performance_simulations(sims_indiv, output_filepath)
    generate_plots_for_simulations_with_coupling_parameters(sims_params, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
