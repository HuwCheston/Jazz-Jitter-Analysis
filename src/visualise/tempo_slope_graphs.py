import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from itertools import groupby

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils


@vutils.plot_decorator
def gen_tempo_slope_graph(data, output_dir, block_num: int = None,) -> tuple[plt.Figure, str]:
    """
    Creates a graph with subplots showing the average tempo trajectory of every condition in a trial.
    Repeats of one condition are plotted as different coloured lines on the same subplot.
    """
    # Set colormap properties
    vutils.create_normalised_cmap(
        slopes=[autils.reg_func(d[5], xcol='elapsed', ycol='bpm_avg').params.iloc[1:].values[0] for d in data]
    )
    plt.rcParams.update({'font.size': vutils.FONTSIZE})
    # Calculate number of discrete trials and conditions within dataset
    fn = lambda nu: max(data, key=itemgetter(nu))[nu]
    n_trials = fn(0)
    n_conditions = fn(2)
    # Construct figure and axis object
    fig, ax = plt.subplots(nrows=n_trials, ncols=n_conditions, sharex='all', sharey='all', figsize=(6.2677165*3, 10))
    # Iterate through data from each trial
    for k, g in groupby(data, itemgetter(0)):
        # Sort the data from each trial by block, latency, and jitter and subset for required block number
        li = sorted(list(g), key=lambda e: (e[0], e[1], e[3], e[4]))
        if block_num is not None:
            li = [t for t in li if t[1] == block_num]
        # Iterate through each condition and plot
        for num, i in enumerate(li):
            _format_ax(num_iter=num, con_data=i, ax=ax, n_conditions=n_conditions, )
    # Format the figure
    fig = _format_fig(fig=fig,)
    # Save the result to the output_filepath
    fname = f'{output_dir}\\tempo_slopes.png'
    return fig, fname


def _format_ax(num_iter: int, con_data: tuple, ax: plt.Axes, n_conditions: int = 13, ):
    """
    Plots the data for one condition on one subplot of the overall figure
    """
    # Generate X and Y coordinates and subset axis object
    x = con_data[0] - 1
    y = num_iter if num_iter < n_conditions else num_iter - n_conditions    # Plot data from both blocks on one subplot
    condition_axis = ax[x, y]
    con_data[5]['elapsed'] -= 8
    plt.setp(condition_axis.spines.values(), linewidth=2)
    # Plot the data on required subplot
    condition_axis.plot(con_data[5]['elapsed'], con_data[5]['bpm_rolling'],
                        label=f'Repeated Measure {con_data[1]}', color=vutils.LINE_CMAP[con_data[1] - 1],
                        linewidth=2)
    condition_axis.tick_params(axis='both', which='both', bottom=False, left=False,)
    # If this condition is either in the first row or column, add the required label
    if x == 0:
        condition_axis.set_title(f'{con_data[3]}ms\n{con_data[4]}x')
    if num_iter == 0:
        condition_axis.set_ylabel(f'Duo {con_data[0]}', rotation=90)
    # Add the reference column as a horizontal line
    if con_data[1] == 2:
        condition_axis.axhline(y=120, color=vutils.BLACK, linestyle='--', alpha=vutils.ALPHA, label='Metronome Tempo',
                               linewidth=2)


def _format_fig(fig: plt.Figure, ) -> plt.Figure:
    """
    Formats the overall figure, setting axis limits, adding labels/titles, configuring legend
    """
    # Set x and y axis limit for figure according to max/min values of data
    plt.xlim(0, 93)
    plt.ylim(30, 160)
    # Set x and y labels, title
    fig.supxlabel('Performance duration (s)', y=0.06)
    fig.supylabel('Average tempo (BPM, 8-seconds rolling)', x=0.01)
    # Call tight_layout only once we've finished formatting but before we, add the legend
    plt.tight_layout()
    # Add the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
    # Reduce the space between plots a bit
    plt.subplots_adjust(bottom=0.13, wspace=0.05, hspace=0.05, right=0.98, left=0.08)
    return fig


@vutils.plot_decorator
def gen_tempo_slope_heatmap(df, output_dir,):
    # Set colormap properties
    df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
    n_trials = max(df['trial'])
    fig, ax = plt.subplots(nrows=n_trials, ncols=1, sharex='all', sharey='all', figsize=(15, 8))
    norm = vutils.create_normalised_cmap(slopes=df['tempo_slope'])
    for idx, grp in df.groupby(by=['trial']):
        piv = grp.pivot_table(index=['latency', 'jitter', 'abbrev'], columns='block', values='tempo_slope').reset_index(
            drop=False).set_index('abbrev').drop(columns=['latency', 'jitter']).transpose()
        sns.heatmap(piv, ax=ax[idx - 1], cmap=vutils.SLOPES_CMAP, cbar=False, norm=norm, linewidth=1, linecolor='w')
        ax[idx - 1].xaxis.set_ticks_position('top')
        ax[idx - 1].yaxis.set_ticks_position('right')
        ax[idx - 1].tick_params(axis='y', labelrotation=0)
        ax[idx - 1].tick_params(axis='x', which='both', bottom=False, top=False, labelsize=12)
        ax[idx - 1].set(xlabel='', ylabel=f'Duo {idx}')
        if idx != 1:
            ax[idx - 1].tick_params(axis='x', labeltop=False)
    fig.suptitle('Performance Tempo Slopes')
    fig.supylabel('Duo Number', x=0.01)
    fig.supxlabel('Condition')
    position = fig.add_axes([0.95, 0.2, 0.01, 0.6])
    fig.colorbar(vutils.create_scalar_cbar(norm=norm), cax=position)
    position.text(0, 0.3, 'Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    plt.subplots_adjust(bottom=0.05, wspace=0.05, hspace=0.15, right=0.90, left=0.05, top=0.91)
    plt.text(-3, -0.18, 'Measure Number', rotation=-90, fontsize=12)
    fname = f'{output_dir}\\tempo_slopes_heatmap.png'
    return fig, fname
