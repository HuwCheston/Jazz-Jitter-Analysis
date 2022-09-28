from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from seaborn import color_palette
from statistics import median
import numpy as np
import pandas as pd
import functools


# Define constants
ALPHA = 0.4
BLACK = '#ffffff'
OFFSET = 8
VIDEO_FPS = 30
CBAR_BINS = np.linspace(-0.5, 0.3, 9, endpoint=True)

# Define the colour palettes
# Function used to shade a cmap by given alpha value (can be used in colorbars etc)
alpha_func = lambda pal: ListedColormap(np.c_[pal.colors, np.full(len(pal.colors), fill_value=ALPHA)])
SLOPES_CMAP = alpha_func(color_palette('vlag_r', as_cmap=True))     # Used for plotting tempo slopes
INSTR_CMAP = ['#9933ff', '#00ff00']     # Palette used for plotting data that contrasts against slopes_cmap
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
        # Create the plot and return the figure
        fig, fname = plotter(*args, **kwargs)
        # Save the figure in the form output directory + fname
        if output is not None:
            d = create_output_folder(output)
            fig.savefig(d + fname)
        # Close the plot to prevent it remaining in memory
        plt.close(fig)
    return wrapper


def create_output_folder(out):
    """
    Create a folder to store the plots, with optional subdirectory. Out should be a full system path.
    Optional keyword arguments:
    Parent:     first subdirectory, usually what the plot depicts i.e. tempo_slopes
    Child:      second subdirectory, usually the type of plot i.e. point plot, polar plot...
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
    if fig is None:
        fig = plt.figure()
    # Create the gridspec
    gs = fig.add_gridspec(3, ncols, height_ratios=[1, 2, 3],
                          wspace=0.1, hspace=0.3, top=0.92, bottom=0.07)
    # Create an array of axes with uneven numbers of plots per row
    return np.array(
        [[*(fig.add_subplot(gs[1, num], projection='polar') for num in range(0, ncols))],   # For the phase diff polar
         [*(fig.add_subplot(gs[0, num]) for num in range(0, ncols))],   # For the phase correction coefficients
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
