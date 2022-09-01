import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from src.analyse.prepare_data import generate_df, average_bpms, zip_same_conditions_together


def gen_tempo_slope_graph(raw_data, output_dir):
    """
    Creates a graph with subplots showing the average tempo trajectory of every condition in a trial.
    Repeats of one condition are plotted as different coloured lines on the same subplot.
    """
    # For each condition, generate the rolling BPM average across the ensemble
    b = zip_same_conditions_together(raw_data)
    s = lambda c: (c['trial'], c['block'], c['condition'], c['latency'], c['jitter'])   # to subset the raw data
    data = [(*s(c1), average_bpms(generate_df(c1['midi_bpm']), generate_df(c2['midi_bpm']))) for z in b for c1, c2 in z]
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
            plot_avg_tempo_for_condition(num_iter=num, condition_data=i, ax=ax, n_conditions=n_conditions)
    # Format the figure
    fig = format_figure(fig=fig, data=data)
    # Save the result to the output_filepath
    fig.savefig(f'{output_dir}\\figures\\tempo_slopes.png')


def plot_avg_tempo_for_condition(num_iter: int, condition_data: tuple, ax: plt.Axes, n_conditions: int = 13):
    """
    Plots the data for one condition on one subplot of the overall figure
    """
    # Generate X and Y coordinates and subset axis object
    x = condition_data[0] - 1
    y = num_iter if num_iter < n_conditions else num_iter - n_conditions    # Plot data from both blocks on one subplot
    condition_axis = ax[x, y]
    # Plot the data on required subplot
    condition_axis.plot(condition_data[5]['elapsed'] - 8,
                        condition_data[5]['bpm_rolling'],
                        label=f'Measure {condition_data[1]}')
    condition_axis.tick_params(axis='both', which='both', bottom=False, left=False,)
    # If this condition is either in the first row or column, add the required label
    if x == 0:
        condition_axis.set_title(f'{condition_data[3]}ms/{condition_data[4]}x')
    if num_iter == 0:
        condition_axis.set_ylabel(f'Duo {condition_data[0]}', rotation=90)
    # Add the reference column as a horizontal line once for each subplot
    if condition_data[1] == 2:
        condition_axis.axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome Tempo')


def format_figure(fig: plt.Figure, data: list) -> plt.Figure:
    """
    Formats the overall figure, setting axis limits, adding labels/titles, configuring legend
    """
    # Count-in offset for each performance
    offset = 8
    # Set x and y axis limit for figure according to max/min values of data
    plt.ylim(
        min([d[-1]['bpm_rolling'].min() for d in data]),
        max([d[-1]['bpm_rolling'].max() for d in data])
    )
    plt.xlim(
        min([d[-1]['elapsed'].min() for d in data]) - offset,
        max([d[-1]['elapsed'].max() for d in data]) - offset
    )
    # Set x and y labels, title
    fig.supxlabel('Performance Duration (s)', y=0.05)
    fig.supylabel('Average tempo (BPM, four-beat rolling window)', x=0.01)
    fig.suptitle('Performance Tempo Slopes')
    # Call tight_layout only once we've finished formatting but before we add the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.11, wspace=0.05, hspace=0.05)  # Reduce the space between plots a bit
    # Add the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False)
    return fig
