import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from statistics import median
from operator import itemgetter
from itertools import groupby
from src.analyse.prepare_data import generate_df, average_bpms, zip_same_conditions_together, reg_func

cmap = sns.color_palette('vlag_r', as_cmap=True)
# Count-in offset for each performance
offset = 8


def gen_tempo_slope_graph(raw_data, output_dir, regplot: bool = False):
    """
    Creates a graph with subplots showing the average tempo trajectory of every condition in a trial.
    Repeats of one condition are plotted as different coloured lines on the same subplot.
    """
    # For each condition, generate the rolling BPM average across the ensemble
    b = zip_same_conditions_together(raw_data)
    s = lambda c: (c['trial'], c['block'], c['condition'], c['latency'], c['jitter'])   # to subset the raw data
    data = [(*s(c1), average_bpms(generate_df(c1['midi_bpm']), generate_df(c2['midi_bpm']))) for z in b for c1, c2 in z]
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
        # Sort the data from each trial by block, latency, and jitter
        li = sorted(list(g), key=lambda e: (e[0], e[1], e[3], e[4]))
        # Iterate through each condition and plot
        for num, i in enumerate(li):
            plot_avg_tempo_for_condition(num_iter=num, con_data=i, ax=ax, n_conditions=n_conditions, regplot=regplot, norm=norm, xrange=xrange)
    # Format the figure
    fig = format_figure(fig=fig, data=data, xrange=xrange, norm=norm)
    # Save the result to the output_filepath
    fig.savefig(f'{output_dir}\\figures\\tempo_slopes_{"rolling_mean" if not regplot else "regression"}.png')


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
        condition_axis.plot(con_data[5]['elapsed'], con_data[5]['bpm_rolling'], label=f'Measure {con_data[1]}')
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
    # Add the reference column as a horizontal line once for each subplot
    if con_data[1] == 2:
        condition_axis.axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome Tempo')
    # Shade the quadrant of the plot according to the tempo slope coefficient
    coef = reg_func(con_data[5], xcol='elapsed', ycol='bpm_avg').params.iloc[1:].values[0]
    condition_axis.axvspan(xrange[0] if con_data[1] == 1 else xrange[1], xrange[1] if con_data[1] == 1 else xrange[2],
                           alpha=0.5, facecolor=cmap(norm(coef)))
    if num_iter == 0:
        condition_axis.text(x=xrange[0]+2, y=142, s='Measure 1', rotation='horizontal', fontsize=6)
        condition_axis.text(x=xrange[2]-43, y=142, s='Measure 2', rotation='horizontal', fontsize=6)


def format_figure(fig: plt.Figure, data: list, xrange: tuple = (8, 50.5, 93), norm = None) -> plt.Figure:
    """
    Formats the overall figure, setting axis limits, adding labels/titles, configuring legend
    """
    # Set x and y axis limit for figure according to max/min values of data
    plt.ylim(min([d[-1]['bpm_rolling'].min() for d in data]), max([d[-1]['bpm_rolling'].max() for d in data]))
    plt.xlim(xrange[0], xrange[2])
    # Set x and y labels, title
    fig.supxlabel('Performance Duration (s)', y=0.05)
    fig.supylabel('Average tempo (BPM, four-beat rolling window)', x=0.01)
    fig.suptitle('Performance Tempo Slopes')
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
