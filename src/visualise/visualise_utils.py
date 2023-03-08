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
WIDTH = 6.2677165
HEIGHT = 10.446194166666666666666666666667
ASPECT = 0.6
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'
RED = '#FF0000'

OFFSET = 8
VIDEO_FPS = 30
CBAR_BINS = np.linspace(-0.5, 0.3, 9, endpoint=True)
N_BOOT = 10000

# Define the colour palettes
# Function used to shade a color map by given alpha value (can be used in color bars etc)
alpha_func = lambda pal: ListedColormap(np.c_[pal.colors, np.full(len(pal.colors), fill_value=ALPHA)])
SLOPES_CMAP = alpha_func(color_palette('vlag_r', as_cmap=True))     # Used for plotting tempo slopes
INSTR_CMAP = ['#9933ff', '#00ff00']     # Palette used for plotting data that contrasts against slopes color map
LINE_CMAP = ['#1f77b4', '#ff7f0e']     # Original matplotlib colour palette used for manual plotting
DUO_CMAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']      # Colour palette used for duos
DUO_MARKERS = ['o', 'X', 's', 'P', 'D']     # Marker cycle used for duos
JITTER_CMAP = ['#FF0000', '#00FF00', '#0000FF']     # Colour palette used for different jitter levels
JITTER_MARK = ['o', 's', '^']    # Marker cycle used for jitter levels
JITTER_LS = ['-', '--', 'dotted']


def plot_decorator(plotter: callable):
    """
    Decorator applied to any plotting function.
    Used to create a folder, save plot into this, then close it cleanly and exit.
    """
    @functools.wraps(plotter)
    def wrapper(*args, **kwargs):
        # Define the filetypes we want to save the plot as
        filetypes = ['png', 'svg']
        # Create the output directory to store the plot
        output = kwargs.get('output_dir', None)
        # If we're accessing this decorator from a class, need to get the output by accessing the class attributes
        if output is None:
            output = args[0].output_dir     # Will be None anyway if no output_dir ever passed to class
        # Create the plot and return the figure
        try:
            fig, fname = plotter(*args, **kwargs)
            # If we've provided an output directory, create a folder and save the plot within it
            if output is not None:
                create_output_folder(output)
                # Iterate through all filetypes and save the plot as each type
                for filetype in filetypes:
                    fig.savefig(f'{fname}.{filetype}', format=filetype, facecolor=WHITE)
        # We've forgotten to return anything from our plotter
        except TypeError:
            pass
        else:
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


def get_significance_asterisks(
        p: float
) -> str:
    """
    Converts a raw p-value into asterisks, showing significance boundaries.
    """
    # We have to iterate through in order from smallest to largest, or else we'll match incorrectly
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''


def break_axis(
        ax1: plt.Axes, ax2: plt.Axes, d: float = 0.015
) -> None:
    """
    Add a vertical breaks into two axis to show change in scale
    """
    # Create kwargs dictionary for ax 1
    kwargs = dict(transform=ax1.transAxes, color=BLACK, clip_on=False)
    # Turn off bottom spine
    ax1.spines['bottom'].set_visible(False)
    # Plot diagonal lines to show axis break
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    # Update kwargs dictionary for second axis
    kwargs.update(transform=ax2.transAxes)
    # Turn off top spine
    ax2.spines['top'].set_visible(False)
    # Plot diagonal lines to show axis break
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, d), (1 - d, 1 + d), **kwargs)


def bootstrap_mean_difference(
    a1: pd.Series, a2: pd.Series, quantile: float = 0.025, n_boot: int = N_BOOT
):
    """
    Helper function to bootstrap the mean difference between two arrays (a1, a2). Number of bootstraps is given by the
    N_BOOT constant if not provided. Quantile is set to 2.5, for 95% confidence intervals.
    """
    # Get our bootstrapped means for both arrays
    m1 = np.array([a1.sample(frac=1, replace=True, random_state=n).mean() for n in range(0, n_boot)])
    m2 = np.array([a2.sample(frac=1, replace=True, random_state=n).mean() for n in range(0, n_boot)])
    # Get the mean difference for each bootstrap iteration
    diff = np.subtract(m2, m1)
    # Get our upper and lower quantiles
    low = np.quantile(diff, quantile)
    high = np.quantile(diff, 1 - quantile)
    return low, high
