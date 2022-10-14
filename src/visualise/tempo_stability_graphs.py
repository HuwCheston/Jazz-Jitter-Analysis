import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import src.visualise.visualise_utils as vutils


@vutils.plot_decorator
def regplot_ioi_std_vs_tempo_slope(
        df: pd.DataFrame, output_dir: str, xvar='ioi_std'
) -> tuple[plt.Figure, str]:
    """
    Creates a regression plot of tempo stability (default metric is median of windowed IOI standard deviations)
    vs tempo slope. Hue of scatterplot corresponds to duo number, marker style to instrument type.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 5))
    plt.rcParams.update({'font.size': vutils.FONTSIZE})
    ax = sns.regplot(data=df, x=xvar, y='tempo_slope', scatter=False, color=vutils.BLACK, ax=ax)
    ax = sns.scatterplot(data=df, x=xvar, y='tempo_slope', hue='trial', palette='tab10', style='instrument', s=100,
                         ax=ax)
    ax.tick_params(width=3, )
    ax.set(ylabel='', xlabel='')
    plt.setp(ax.spines.values(), linewidth=2)
    # Plot a horizontal line at x=0
    ax.axhline(y=0, linestyle='-', alpha=1, color=vutils.BLACK)
    # Set axis labels
    fig.supylabel('Tempo slope (BPM/s)', x=0.01)
    fig.supxlabel('Tempo stability (ms)' if xvar == 'ioi_std' else xvar, y=0.12)
    # Format axis positioning and move legend
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:6] + handles[7:], labels=labels[1:6] + labels[7:], ncol=8, frameon=False,
              markerscale=1.6, columnspacing=0.8, bbox_to_anchor=(1, -0.18), )
    ax.figure.subplots_adjust(bottom=0.25, top=0.95, left=0.12, right=0.95)
    fname = f'{output_dir}\\regplot_{xvar}_vs_tempo_slope.png'
    return fig, fname
