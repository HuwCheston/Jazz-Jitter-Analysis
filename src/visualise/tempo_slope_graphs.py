import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
from statistics import median
from operator import itemgetter
from itertools import groupby
from src.analyse.prepare_data import generate_df, average_bpms, zip_same_conditions_together, reg_func

cmap = sns.color_palette('vlag_r', as_cmap=True)
# Count-in offset for each performance
offset = 8


def return_data(raw_data) -> list[tuple]:
    # For each condition, generate the rolling BPM average across the ensemble
    b = zip_same_conditions_together(raw_data)
    s = lambda c: (c['trial'], c['block'], c['condition'], c['latency'], c['jitter'])   # to subset the raw data
    return [(*s(c1), average_bpms(generate_df(c1['midi_bpm']), generate_df(c2['midi_bpm']))) for z in b for c1, c2 in z]


def gen_tempo_slope_graph(raw_data, output_dir, regplot: bool = False, block_num: int = 1):
    """
    Creates a graph with subplots showing the average tempo trajectory of every condition in a trial.
    Repeats of one condition are plotted as different coloured lines on the same subplot.
    """
    # For each condition, generate the rolling BPM average across the ensemble
    data = return_data(raw_data)
    # Set colormap properties
    slopes = [reg_func(d[5], xcol='elapsed', ycol='bpm_avg').params.iloc[1:].values[0] for d in data]
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(slopes), vcenter=median(slopes), vmax=max(slopes))
    xrange = (
        min([d[-1]['elapsed'].min() for d in data]) - offset,
        median([d[-1]['elapsed'].median() for d in data]) - offset,
        max([d[-1]['elapsed'].max() for d in data]) - offset,
    )
    # Calculate number of discrete trials and conditions within dataset
    fn = lambda nu: max(data, key=itemgetter(nu))[nu]
    n_trials = fn(0)
    n_conditions = fn(2)
    # Construct figure and axis object
    fig, ax = plt.subplots(nrows=n_trials, ncols=n_conditions, sharex='all', sharey='all', figsize=(15, 8))
    # Iterate through data from each trial
    for k, g in groupby(data, itemgetter(0)):
        # Sort the data from each trial by block, latency, and jitter and subset for required block number
        li = sorted(list(g), key=lambda e: (e[0], e[1], e[3], e[4]))
        li = [t for t in li if t[1] == block_num]
        # Iterate through each condition and plot
        for num, i in enumerate(li):
            plot_avg_tempo_for_condition(num_iter=num, con_data=i, ax=ax, n_conditions=n_conditions, regplot=regplot, norm=norm, xrange=xrange)
    # Format the figure
    fig = format_figure(fig=fig, data=data, xrange=xrange, norm=norm, block_num=block_num)
    # Save the result to the output_filepath
    fig.savefig(f'{output_dir}\\figures\\tempo_slopes_measure{block_num}.png')


def plot_avg_tempo_for_condition(num_iter: int, con_data: tuple, ax: plt.Axes, n_conditions: int = 13, regplot: bool = False, xrange: tuple = (8, 50.5, 93), norm=None,):
    """
    Plots the data for one condition on one subplot of the overall figure
    """
    # Generate X and Y coordinates and subset axis object
    x = con_data[0] - 1
    y = num_iter if num_iter < n_conditions else num_iter - n_conditions    # Plot data from both blocks on one subplot
    condition_axis = ax[x, y]
    con_data[5]['elapsed'] -= 8
    cs = sns.color_palette(["#1f77b4", '#ff7f0e'])
    # Plot the data on required subplot
    if not regplot:
        condition_axis.plot(con_data[5]['elapsed'], con_data[5]['bpm_rolling'], label=f'Performance Tempo', color='#000000')
    else:
        sns.regplot(data=con_data[5], x='elapsed', y='bpm_rolling', scatter=False,
                    ax=condition_axis, ci=None, color=cs[con_data[1] - 1])
        condition_axis.set(ylabel='', xlabel='')
    condition_axis.tick_params(axis='both', which='both', bottom=False, left=False,)
    # If this condition is either in the first row or column, add the required label
    if x == 0:
        condition_axis.set_title(f'{con_data[3]}ms/{con_data[4]}x')
    if num_iter == 0:
        condition_axis.set_ylabel(f'Duo {con_data[0]}', rotation=90)
    # Add the reference column as a horizontal line
    condition_axis.axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome Tempo')
    # Shade the plot according to the tempo slope coefficient
    coef = reg_func(con_data[5], xcol='elapsed', ycol='bpm_avg').params.iloc[1:].values[0]
    condition_axis.axvspan(xrange[0], xrange[2], alpha=0.4, facecolor=cmap(norm(coef)))


def format_figure(fig: plt.Figure, data: list, xrange: tuple = (8, 50.5, 93), norm=None, block_num:int = 1) -> plt.Figure:
    """
    Formats the overall figure, setting axis limits, adding labels/titles, configuring legend
    """
    # Set x and y axis limit for figure according to max/min values of data
    plt.ylim(min([d[-1]['bpm_rolling'].min() for d in data]), max([d[-1]['bpm_rolling'].max() for d in data]))
    plt.xlim(xrange[0], xrange[2])
    # Set x and y labels, title
    fig.supxlabel('Performance Duration (s)', y=0.05)
    fig.supylabel('Average tempo (BPM, four-beat rolling window)', x=0.01)
    fig.suptitle(f'Performance Tempo: Measure {block_num}')
    # Call tight_layout only once we've finished formatting but before we, add the legend
    plt.tight_layout()
    # Add the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
    # Add the colorbar
    position = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position,)
    position.text(0, 0.3, 'Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    # Reduce the space between plots a bit
    plt.subplots_adjust(bottom=0.11, wspace=0.05, hspace=0.05, right=0.93)
    return fig


def gen_tempo_slope_heatmap(raw_data, output_dir,):
    data = return_data(raw_data)
    # Set colormap properties
    slopes = [(d[0], d[1], d[3], d[4], reg_func(d[5], xcol='elapsed', ycol='bpm_avg').params.iloc[1:].values[0]) for d in data]
    df = pd.DataFrame(slopes, columns=['trial', 'block', 'latency', 'jitter', 'slope']).sort_values(
        by=['trial', 'block', 'latency', 'jitter'])
    df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
    n_trials = max(df['trial'])
    fig, ax = plt.subplots(nrows=n_trials, ncols=1, sharex='all', sharey='all', figsize=(15, 8))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(df['slope']), vcenter=median(df['slope']), vmax=max(df['slope']))
    for idx, grp in df.groupby(by=['trial']):
        piv = grp.pivot_table(index=['latency', 'jitter', 'abbrev'], columns='block', values='slope').reset_index(
            drop=False).set_index('abbrev').drop(columns=['latency', 'jitter']).transpose()
        sns.heatmap(piv, ax=ax[idx - 1], cmap=cmap, cbar=False, norm=norm, linewidth=1, linecolor='w')
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
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position, )
    position.text(0, 0.3, 'Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    plt.subplots_adjust(bottom=0.05, wspace=0.05, hspace=0.15, right=0.95)
    plt.text(-3, -0.18, 'Measure Number', rotation=-90, fontsize=12)
    fig.savefig(f'{output_dir}\\figures\\tempo_slopes_heatmap.png')
