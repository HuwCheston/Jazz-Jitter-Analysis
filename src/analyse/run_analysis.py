# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from prepare_data import generate_tempo_slopes, load_data
from linear_regressions import lr_tempo_slope, lr_beat_variance
from granger_causality import gc_event_density_vs_latency_var, gc_ioi_var_vs_latency_var, pearson_r_ioi_var_vs_latency_var
from anovas import analyse_beat_variance, anova_ts_lat_jit
from phase_correction_models import pc_live_ioi_delayed_ioi

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
    ts_raw = generate_tempo_slopes(raw_data=data)
    logger.info(f'loaded data from {len(data)} trials!')

    # SYNCHRONISATION MODELS
    logger.info(f'Creating phase correction models...')
    pc_live_ioi_delayed_ioi(raw_data=data, output_dir=output_filepath)

    # LINEAR REGRESSIONS #
    # logger.info(f'Conducting linear regressions...')
    # lr_ts = lr_tempo_slope(tempo_slopes_data=ts_raw, output_dir=output_filepath)
    # lr_bv = lr_beat_variance(raw_data=data, output_dir=output_filepath)

    # GRANGER CAUSALITY
    # logger.info(f'Estimating Granger causality...')
    # gc_ed = gc_event_density_vs_latency_var(raw_data=data, output_dir=output_filepath)
    # gc_bv = gc_ioi_var_vs_latency_var(raw_data=data, output_dir=output_filepath)

    # ANOVAS
    # logger.info(f'Conducting ANOVAS...')

    # QUESTIONNAIRES


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
