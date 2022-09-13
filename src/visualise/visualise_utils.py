from pathlib import Path
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
from seaborn import color_palette
from statistics import median

# Define constants
ALPHA = 0.4
OFFSET = 8

# Function used to shade a cmap by given alpha value (can be used in colobars etc)
cmap_alpha = lambda pal: ListedColormap(np.c_[pal.colors, np.full(len(pal.colors), fill_value=ALPHA)])

# Define the colour palettes
slopes_cmap = cmap_alpha(color_palette('vlag_r', as_cmap=True))     # Used for plotting tempo slopes
data_cmap_contrast = ['#9933ff', '#00ff00']     # Palette used for plotting data that contrasts against slopes_cmap
data_cmap_orig = ['#1f77b4', '#ff7f0e']     # Original matplotlib colour palette used for manual plotting


def create_output_folder(out: str, parent: str = 'default', child: str = None):
    """
    Create a folder to store the plots, with optional subdirectory. Out should be a full system path.
    Optional keyword arguments:
    Parent:     first subdirectory, usually what the plot depicts i.e. tempo_slopes
    Child:      second subdirectory, usually the type of plot i.e. point plot, polar plot...
    """
    if child is None:
        output_path = out + f'\\figures\\{parent}'
    else:
        output_path = out + f'\\figures\\{parent}\\{child}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path


def create_normalised_cmap(slopes: list) -> TwoSlopeNorm:
    """
    Create a normalised cmap between a minimum, median, and maximum value.
    """
    return TwoSlopeNorm(vmin=min(slopes), vcenter=median(slopes), vmax=max(slopes))


def create_scalar_cbar(norm: TwoSlopeNorm) -> ScalarMappable:
    """
    Creates a scalar colourbar object to be placed on a figure
    """
    return ScalarMappable(norm=norm, cmap=slopes_cmap)
