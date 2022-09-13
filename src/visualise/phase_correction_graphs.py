import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import statsmodels.formula.api as smf
import src.visualise.visualise_utils as vutils


def make_pairgrid(df, output: str, xvar: str = 'correction_partner'):
    """
    Creates a figure showing pairs of coefficients obtained for each performer in a condition,
    stratified by block and trial number, with shading according to tempo slope
    """
    output = vutils.create_output_folder(output, parent='phase_correction_models', child='pairgrid')
    # Create the abbreviation column, showing latency and jitter
    df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
    df = df.sort_values(by=['latency', 'jitter'])
    # Sort the palette for the shading
    norm = vutils.create_normalised_cmap(df['tempo_slope'])
    # Create the plot
    g = sns.catplot(
        data=df, x=xvar, y='abbrev', row='block', col='trial', hue='instrument', hue_order=['Keys', 'Drums'],
        palette=sns.color_palette(vutils.data_cmap_contrast), kind='strip', height=4, sharex=True, sharey=True,
        aspect=0.6, s=7, jitter=False, dodge=False, marker='D',
    )
    # Format the axis by iterating through
    for num in range(0, 5):
        # When we want different formatting for each row
        g.axes[0, num].set(title=f'Measure 1\nDuo {num + 1}' if num == 2 else f'\nDuo {num + 1}', ylabel='',
                           xlim=(-1, 1), )
        g.axes[0, num].tick_params(bottom=False)
        g.axes[1, num].set(title='Measure 2' if num == 2 else '', ylabel='', xlabel='', xlim=(-1, 1))
        # When we want the same formatting for both rows
        for x in range(0, 2):
            # Disable the grid
            g.axes[x, num].xaxis.grid(False)
            g.axes[x, num].yaxis.grid(True)
            # Add on a vertical line at x=0
            g.axes[x, num].axvline(alpha=vutils.ALPHA, linestyle='-', color='#000000')
            # Add the span, shaded according to tempo slope
            d = df[
                (df['trial'] == num + 1) & (df['block'] == x + 1) & (df['instrument'] == 'Keys')
                ].sort_values(['latency', 'jitter'])['tempo_slope']
            for n in range(0, 13):
                coef = vutils.slopes_cmap(norm(d.iloc[n]))
                g.axes[x, num].axhspan(n - 0.5, n + 0.5, facecolor=coef)
    _format_pairgrid_fig(g, norm)
    g.savefig(f'{output}\\condition_vs_{xvar}_pointplot.png')


def _format_pairgrid_fig(g, norm):
    # Format the figure
    g.despine(left=True, bottom=True)
    g.fig.supxlabel('Correction to Partner', x=0.53, y=0.04)
    g.fig.supylabel('Condition', x=0.01)
    # Add the colorbar
    position = g.fig.add_axes([0.95, 0.3, 0.01, 0.4])
    g.fig.colorbar(vutils.create_scalar_cbar(norm=norm), cax=position, )
    position.text(0, 0.3, ' Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    # Add the legend
    g.legend.remove()
    g.fig.get_axes()[0].legend(loc='lower center', ncol=2, bbox_to_anchor=(2.85, -1.36), title=None, frameon=False)
    # Adjust the plot spacing
    g.fig.subplots_adjust(bottom=0.10, top=0.94, wspace=0.15, left=0.1, right=0.93)


def create_prediction_plots(pred_list: list[tuple], output_dir: str):
    test_li = [item for item in pred_list if item[0] == 4 and item[1] == 1 and item[2] == 45 and item[3] == 0]
    km = test_li[0][4]
    md = smf.ols('live_next_ioi~live_prev_ioi+live_delayed_onset+live_delayed_ioi', data=km).fit()
    f = lambda d: (60 / d.dropna()).rolling(8).mean()
    plt.plot(f(km['live_prev_ioi']), label='Data')
    plt.plot(f(pd.Series(
        np.insert(np.cumsum(md.predict()) + km['live_prev_onset'].loc[0], 0, km['live_prev_onset'].loc[0])).diff()),
             label='Predicted')
    plt.legend()
    plt.show()


def make_polar(nn_list: list[tuple], output_dir: str,) -> None:
    """
    Creates circle plots showing relative phase (i.e. playing before or after reference) for both musicians for a trial.
    """
    # Create the folder to store the plots
    output_path = vutils.create_output_folder(output_dir, parent='phase_correction_models', child='polar_plots')
    # Subset the data for only the block we are plotting
    sorter = lambda e: (e[0], e[1], e[2], e[3], e[4])
    to_plot = [t for t in sorted(nn_list, key=sorter)]
    # Create the subplots
    fig, ax = plt.subplots(nrows=10, ncols=13, subplot_kw=dict(projection="polar"), figsize=(22, 14), sharex='row', )
    # Create the indexers
    col_ind, row_ind = 0, 0
    # Sort the palette for the shading
    norm = vutils.create_normalised_cmap(slopes=[t[-1] for t in sorted(nn_list, key=sorter)])
    # Iterate through the data
    for (trial, block, latency, jitter, keys, drms, slope) in to_plot:
        # Subset the subplots
        a = ax[row_ind, col_ind]
        # Plot the keys and drums data
        for ins, st in zip([keys, drms], ['Keys', 'Drums']):
            dat = _format_polar_data(ins, num_bins=15)
            a.bar(x=dat.idx, height=dat.val, width=0.05, label=st,
                  color=vutils.data_cmap_contrast[0] if st == 'Keys' else vutils.data_cmap_contrast[1])
        # Adjust formatting for all plots
        maxy = max(a.get_yticks())
        _format_polar_ax(ax=a, sl=vutils.slopes_cmap(norm(slope)))
        # Adjust formatting for top row of plots
        if trial == 1 and block == 1:
            if col_ind == 0:
                a.set_title(f'{latency}ms/{jitter}x\nDuo {trial}', fontsize='xx-large', y=0.9, )
            elif col_ind == 6:
                a.set_title(f'{latency}ms/{jitter}x\nMeasure {block}', fontsize='xx-large', y=0.9, )
            else:
                a.set_title(f'{latency}ms/{jitter}x\n', fontsize='xx-large', y=0.9, )
            a.text(-0.1, 0.1, s='–90°', transform=a.transAxes, ha='left', fontsize='large')
            a.text(0.5, 0.8, s='0°', transform=a.transAxes, ha='center', fontsize='large')
            a.text(1.1, 0.1, s='90°', transform=a.transAxes, ha='right', fontsize='large')
        # Adjust formatting for all other rows
        else:
            if col_ind == 0 and block == 1:
                a.set_title(f'Duo {trial}', fontsize='xx-large', y=0.75, )
            elif col_ind == 6:
                a.set_title(f'Measure {block}', fontsize='xx-large', y=0.75, )
        # Adjust formatting for first column of plots
        if col_ind == 0:
            a.text(-0.1, 0.6, s=str(int(round(maxy, 0))), transform=a.transAxes, fontsize='large')
        # Increase counter
        row_ind = row_ind + 1 if col_ind == 12 else row_ind
        col_ind = col_ind + 1 if col_ind < 12 else 0
    # Format the overall figure
    hand, lab = ax[-1, -1].get_legend_handles_labels()
    _format_polar_fig(fig, hand, lab, norm)
    # Add horizontal and vertical lines separating subplots
    _add_horizontal_lines_polar(fig)
    # Save figure
    fig.savefig(f'{output_path}\\polarplot.png')


def _add_vertical_lines_polar(fig: plt.Figure, start: float = 0.093, step: float = 0.2134,
                              y: np.array = np.array([0.055, 0.945]), num_lines: int = 4):
    """
    Adds horizontal lines seperating the subplots of a polar plot by latency baseline
    """
    x = start
    for _ in range(1, num_lines):
        fig.add_artist(Line2D([x, x], y, alpha=vutils.ALPHA, linestyle='-', color='#000000'))
        x += step


def _add_horizontal_lines_polar(fig: plt.Figure, start: float = 0.78, step: float = -0.1825,
                                x: np.array = np.array([0.025, 0.94]), num_lines: int = 4):
    """
    Adds horizontal lines seperating the subplots of a polar plot by duo number.
    """
    y = start
    for _ in range(0, num_lines):
        fig.add_artist(Line2D(x, [y, y], alpha=vutils.ALPHA, linestyle='-', color='#000000'))
        y += step


def _format_polar_data(df: pd.DataFrame, num_bins: int = 10) -> pd.DataFrame:
    """
    Formats the data required for a polar plot by calculating phase of live musician relative to delayed, subsetting
    result into given number of bins, getting the square-root density of these bins.
    """
    # Calculate the amount of phase correction for each beat
    corr = df.live_prev_onset - df.delayed_prev_onset
    # Cut into bins
    cut = pd.cut(corr, num_bins, include_lowest=False).value_counts().sort_index()
    # Format the dataframe
    cut.index = pd.IntervalIndex(cut.index.get_level_values(0)).mid
    cut = pd.DataFrame(cut, columns=['val']).reset_index(drop=False).rename(columns={'index': 'idx'})
    # Multiply the bin median by pi to get number of degrees
    cut['idx'] = cut['idx'] * np.pi
    # Take the square root of the density for plotting
    cut['val'] = np.sqrt(cut['val'])
    return cut


def _format_polar_ax(ax: plt.Axes, sl) -> None:
    """
    Formats a polar subplot by setting axis ticks and gridlines, rotation properties, and facecolor
    """
    # Set the y axis ticks and gridlines
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    # Set the polar plot rotation properties, direction, etc
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # Set the color according to the tempo slope
    ax.patch.set_facecolor(sl)


def _format_polar_fig(fig: plt.Figure, hand, lab, norm):
    """
    Format the overall polar plot figure, including setting titles, positioning colorbars, legends etc.
    """
    # Add figure-wise titles and labels
    fig.supylabel('√Density', x=0.01, fontsize='xx-large')
    fig.supxlabel("Relative Phase to Partner's Onsets (πms)", y=0.03, fontsize='xx-large')
    # Create and position colourbar
    position = fig.add_axes([0.95, 0.2, 0.01, 0.6])
    fig.colorbar(vutils.create_scalar_cbar(norm=norm), cax=position,)
    position.tick_params(labelsize=17.7)
    position.text(0, 0.3, 'Slope\n(BPM/s)\n', fontsize='xx-large')  # Super hacky way to add a title...
    # Adjust positioning slightly
    fig.subplots_adjust(bottom=0.03, top=0.96, wspace=0.3, left=0.03, right=0.94, hspace=-0.1)
    # Create and position legend
    plt.legend(hand, lab, ncol=2, loc='lower center', bbox_to_anchor=(-46, -0.34), title=None,
               frameon=False, fontsize='xx-large')
