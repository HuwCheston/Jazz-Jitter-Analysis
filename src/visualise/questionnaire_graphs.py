import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import src.visualise.visualise_utils as vutils
from src.analyse.prepare_data import average_bpms


@vutils.plot_decorator
def make_irr_heatmap(
        df: pd.DataFrame, output_dir: str
) -> tuple[plt.Figure, str]:
    pass
