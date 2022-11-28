from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from seaborn import color_palette
from statistics import median
from datetime import timedelta
import numpy as np
import pandas as pd
import functools


# Define constants
from stargazer.stargazer import Stargazer

WIDTH = 6.2677165
HEIGHT = 10.446194166666666666666666666667
ASPECT = 0.6
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'

OFFSET = 8
VIDEO_FPS = 30
CBAR_BINS = np.linspace(-0.5, 0.3, 9, endpoint=True)

# Define the colour palettes
# Function used to shade a color map by given alpha value (can be used in color bars etc)
alpha_func = lambda pal: ListedColormap(np.c_[pal.colors, np.full(len(pal.colors), fill_value=ALPHA)])
SLOPES_CMAP = alpha_func(color_palette('vlag_r', as_cmap=True))     # Used for plotting tempo slopes
INSTR_CMAP = ['#9933ff', '#00ff00']     # Palette used for plotting data that contrasts against slopes color map
LINE_CMAP = ['#1f77b4', '#ff7f0e']     # Original matplotlib colour palette used for manual plotting


def plot_decorator(plotter: callable):
    """
    Decorator applied to any plotting function.
    Used to create a folder, save plot into this, then close it cleanly and exit.
    """
    @functools.wraps(plotter)
    def wrapper(*args, **kwargs):
        # Create the output directory to store the plot
        output = kwargs.get('output_dir', None)
        # If we're accessing this decorator from a class, need to get the output by accessing the class attributes
        if output is None:
            output = args[0].output_dir     # Will be None anyway if no output_dir ever passed to class
        # Create the plot and return the figure
        fig, fname = plotter(*args, **kwargs)
        # If we've provided an output directory, create a folder and save the plot within it
        if output is not None:
            create_output_folder(output)
            fig.savefig(fname, facecolor=WHITE)
        # Close the plot to prevent it from remaining in memory
        plt.close(fig)
    return wrapper


def create_output_folder(out):
    """
    Create a folder to store the plots, with optional subdirectory. Out should be a full system path.
    """
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def create_normalised_cmap(slopes: list) -> TwoSlopeNorm:
    """
    Create a normalised cmap between a minimum, median, and maximum value.
    """
    return TwoSlopeNorm(vmin=min(slopes), vcenter=median(slopes), vmax=max(slopes))


def create_scalar_cbar(norm: TwoSlopeNorm) -> ScalarMappable:
    """
    Creates a scalar colourbar object to be placed on a figure
    """
    return ScalarMappable(norm=norm, cmap=SLOPES_CMAP)


def get_gridspec_array(fig: plt.Figure = None, ncols: int = 2) -> np.array:
    """
    Create an array of axes with unequal numbers of plots per row/column
    Returns an array that can be indexed in the same way as the array normally returned by plt.subplots()
    """
    # Create the figure, if we haven't already provided one
    if fig is None:
        fig = plt.figure()
    # Create the grid spec
    gs = fig.add_gridspec(3, ncols, height_ratios=[1, 2, 3], wspace=0.1, hspace=0.4, top=0.92, bottom=0.09)
    # Create an array of axes with uneven numbers of plots per row
    return np.array(
        [[*(fig.add_subplot(gs[1, num], projection='polar') for num in range(0, ncols))],  # For the phase diff polar
         [*(fig.add_subplot(gs[0, num]) for num in range(0, ncols))],  # For the phase correction coefficients
         [fig.add_subplot(gs[2, :]), np.nan]]  # For the tempo slope
    )


def append_count_in_rows_to_df(df_avg_slopes: pd.DataFrame, time_col: str = 'elapsed') -> pd.DataFrame:
    """
    Appends empty rows to a dataframe corresponding with every second of the count-in where no notes were played.
    Input dataframe should be in the format returned by average_bpms (i.e. average BPM across performers per second)
    Used when creating videos of tempo slopes.
    """
    # Create a new dataframe of the extra rows we need to go from 0 to performance start
    add = pd.DataFrame([(num, np.nan, np.nan) for num in range(0, int(df_avg_slopes.loc[0][time_col]))],
                       columns=df_avg_slopes.columns)
    # Concat the new dataframe and our input
    return pd.concat([add, df_avg_slopes]).reset_index(drop=True)


def interpolate_df_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional rows to a tempo slope dataframe, used when upscaling video FPS.
    """
    # Create the new index
    new_indx = np.linspace(df.index.min(), df.index.max(),
                           (df.shape[0] - 1) * (VIDEO_FPS - 1) + df.shape[0])
    # Reindex the input dataframe and interpolate according to it
    return df.reindex(new_indx).interpolate(method='index').reset_index(drop=False)


def get_xrange(data, min_off: int = -OFFSET, max_off: int = -OFFSET) -> tuple:
    return (
        min([d[-1]['elapsed'].min() for d in data]) + min_off,
        max([d[-1]['elapsed'].max() for d in data]) + max_off,
    )


def get_yrange(data, min_off: int = -OFFSET, max_off: int = -OFFSET) -> tuple:
    return (
        min([d[-1]['bpm_rolling'].min() for d in data]) + min_off,
        max([d[-1]['bpm_rolling'].max() for d in data]) + max_off
    )


def output_regression_table(
        mds: list, output_dir: str, verbose_footer: bool = False
) -> None:
    """
    Create a nicely formatted regression table from a list of regression models ordered by trial, and output to html.
    """

    def get_cov_names(name: str) -> list[str]:
        k = lambda x: float(x.partition('T.')[2].partition(']')[0])
        # Try and sort the values by integers within the string
        try:
            return [o for o in sorted([i for i in out.cov_names if name in i], key=k)]
        # If there are no integers in the string, return unsorted
        except ValueError:
            return [i for i in out.cov_names if name in i]

    def format_cov_names(i: str, ext: str = '') -> str:
        # If we've defined a non-default reference category the stats models output looks weird, so catch this
        if ':' in i:
            lm = lambda s: s.split('C(')[1].split(')')[0].title() + ' (' + s.split('[T.')[1].split(']')[0] + ')'
            return lm(i.split(':')[0]) + ': ' + lm(i.split(':')[1])
        if 'Treatment' in i:
            return i.split('C(')[1].split(')')[0].title().split(',')[0] + ' (' + i.split('[T.')[1].replace(']', ')')
        else:
            base = i.split('C(')[1].split(')')[0].title() + ' ('
            return base + i.split('C(')[1].split(')')[1].title().replace('[T.', '').replace(']', '') + ext + ')'

    # Create the stargazer object from our list of models
    out = Stargazer(mds)
    # Get the original co-variate names
    l_o, j_o, i_o, int_o = (get_cov_names(i) for i in ['latency', 'jitter', 'instrument', 'Intercept'])
    orig = [item for sublist in [l_o, j_o, i_o, int_o] for item in sublist]
    # Format the original co-variate names so they look nice
    lat_fm = [format_cov_names(s, 'ms') for s in l_o]
    jit_fm = [format_cov_names(s, 'x') for s in j_o]
    instr_fm = [format_cov_names(s) for s in i_o]
    form = [item for sublist in [lat_fm, jit_fm, instr_fm, int_o] for item in sublist]
    # Format the stargazer object
    out.custom_columns([f'Duo {i}' for i in range(1, len(mds) + 1)], [1 for _ in range(1, len(mds) + 1)])
    out.show_model_numbers(False)
    out.covariate_order(orig)
    out.rename_covariates(dict(zip(orig, form)))
    t = out.dependent_variable
    out.dependent_variable = ' ' + out.dependent_variable.replace('_', ' ').title()
    # If we're removing some statistics from the bottom of our table
    if not verbose_footer:
        out.show_adj_r2 = False
        out.show_residual_std_err = False
        out.show_f_statistic = False
    # Create the output folder
    fold = create_output_folder(output_dir)
    # Render to html and write the result
    with open(f"{fold}\\regress_{t}.html", "w") as f:
        f.write(out.render_html())


class BasePlot:
    """
    Base plotting class from which others inherit
    """
    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': FONTSIZE})
        # Get from kwargs (with default arguments)
        self.df: pd.DataFrame = kwargs.get('df', None)
        self.output_dir: str = kwargs.get('output_dir', None)
        # Create an empty attribute to store our plot in later
        self.g = None


def resample(
        perf: pd.DataFrame, col: str = 'my_onset', resample_window: str = '1s'
) -> pd.DataFrame:
    """
    Resamples an individual performance dataframe to get mean of every second.
    """
    # Create the timedelta index
    idx = pd.to_timedelta([timedelta(seconds=val) for val in perf[col]])
    # Create the offset: 8 - first onset time
    offset = timedelta(seconds=8 - perf.iloc[0][col])
    # Set the index, resample to every second, and take the mean
    return perf.set_index(idx).resample(resample_window, offset=offset).mean()
