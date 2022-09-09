import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from statistics import median
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import pathlib

cmap = sns.color_palette('vlag_r', as_cmap=True)
offset = 8


def create_output_folder(out):
    # Create a folder to store the plots
    output_path = out + '\\figures\\phase_correction_graphs'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path


def format_ax(ax: plt.Axes, duo_num: int = 0, set_margins: bool = False) -> None:
    """
    Formats a matplotlib axis by setting tick paramaters, axis label and title
    """
    ax.tick_params(axis='both', which='both', bottom=False, left=False, )
    ax.set(xlabel='', ylabel='', title=f'Duo {duo_num}')
    # Use this if we're adding a rugplot and need to extend the margins of the x axis
    if set_margins:
        xl, yl = ax.get_xlim()
        ax.set_xlim(xl - 1, yl)


def format_fig(fig: plt.Figure, xlab: str = 'x', ylab: str = 'y',) -> None:
    """
    Formats a matplotlib figure by adding xlabel, ylabel, adjusting subplot size, then tightening the layout
    """
    fig.supylabel(ylab, x=0.01,)
    fig.supxlabel(xlab, y=0.085,)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.05, wspace=0.05, hspace=0.05)


def format_label(label: str, inbetween: str = ' to ', end: str = '') -> str:
    """
    Formats an axis label by appending to the input string
    """
    label = label.title().split(sep='_')
    label.insert(1, inbetween)
    label.append(end)
    return ''.join(label)


def plot_barplot(df, output, xvar: str = 'latency', yvar: str = 'correction_partner', hvar: str = 'instrument'):
    """
    Creates joint barplot and scatter plot for phase correction model.
    Includes scatter of all values and group means as bar columns. Grouped by hvar.
    """
    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex='all', sharey='all', figsize=(15, 5))
    # We use these to store legend handles and labels, because matplotlib doesn't like working with multiple sns plots
    hand, lab = (None, None)
    # Iterate through each trial individually
    grouper = df.groupby('trial')
    for idx, grp in grouper:
        x = idx - 1
        # Subset for means for barplot
        means = grp.groupby([xvar, hvar]).mean().reset_index(drop=False)
        # Create swarmplot
        sns.swarmplot(data=grp, x=xvar, y=yvar, hue=hvar, ax=ax[x], s=2, hue_order=['Keys', 'Drums'],
                      color='#000000', dodge=True,)
        # Create rugplot
        sns.rugplot(data=grp, y=yvar, hue=hvar, hue_order=['Keys', 'Drums'], ax=ax[x], legend=False, height=0.05)
        # Create barplot
        sns.barplot(data=means, x=means[xvar], y=yvar, hue=hvar, ax=ax[x], hue_order=['Keys', 'Drums'],
                    palette=sns.color_palette(["#1f77b4", '#ff7f0e']))
        # Store handles and labels then remove legend
        hand, lab = ax[-1].get_legend_handles_labels()
        ax[x].legend([], [], frameon=False)
        # Format axis
        format_ax(ax=ax[x], duo_num=idx)
        xl, yl = ax[x].get_xlim()
        ax[x].set_xlim(xl - 0.5, yl)
    # Format the figure, readd the legend and move, then save
    format_fig(fig, ylab=format_label(yvar), xlab=xvar.title())
    plt.legend(hand[2:], lab[2:], ncol=2, loc='lower center', bbox_to_anchor=(-1.75, -0.25), title=None, frameon=False)
    fig.savefig(f'{output}\\{xvar}_vs_{yvar}_barplot.png')


def plot_regplot(df, output, xvar: str = 'correction_partner', yvar: str = 'tempo_slope', hvar: str = 'instrument'):
    """
    Creates regression plot for phase correction model.
    Includes scatter of all values and seperate regression lines for each category in hvar.
    """
    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex='all', sharey='all', figsize=(15, 5))
    cs = sns.color_palette(["#1f77b4", '#ff7f0e'])
    # Iterate through each trial individually
    grouper = df.groupby('trial')
    for idx, grp in grouper:
        x = idx - 1
        # Create the scatterplot, with hue = instrument
        sns.scatterplot(x=xvar, y=yvar, hue=hvar, hue_order=['Keys', 'Drums'], data=grp, ax=ax[x], palette=cs,
                        legend=True if idx == len(grouper) else False)
        # For each instrument, plot a separate regression line
        for c, (_, g) in zip(reversed(cs), grp.groupby(hvar)):
            sns.regplot(x=xvar, y=yvar, data=g, scatter=False, ax=ax[x], color=c, ci=None,)
        # Format the axes
        format_ax(ax=ax[x], duo_num=idx)
    # Format the figure, move the legend, then save
    format_fig(fig=fig, xlab=format_label(xvar), ylab=format_label(yvar, inbetween=' ', end=' (BPM/s)'))
    sns.move_legend(ax[-1], ncol=2, loc='lower center', bbox_to_anchor=(-1.75, -0.25), title=None, frameon=False)
    fig.savefig(f'{output}\\{xvar}_vs_{yvar}_regplot.png')


def plot_pairgrid(df, output: str, xvar: str = 'correction_partner'):
    """
    Creates a figure showing pairs of coefficients obtained for each performer in a condition,
    stratified by block and trial number, with shading according to tempo slope
    """
    # Create the abbreviation column, showing latency and jitter
    df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
    df = df.sort_values(by=['latency', 'jitter'])
    # Sort the palette for the shading
    slopes = df['tempo_slope']
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(slopes), vcenter=median(slopes), vmax=max(slopes))
    # Create the plot
    g = sns.catplot(
        data=df, x=xvar, y='abbrev', row='block', col='trial', hue='instrument', hue_order=['Keys', 'Drums'],
        palette=sns.color_palette(['#4daf4a', '#f781bf']), kind='strip', height=4, sharex=True, sharey=True, aspect=0.6,
        s=7, jitter=False, dodge=False, marker='D',
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
            g.axes[x, num].axvline(alpha=0.4, linestyle='-', color='#000000')
            # Add the span, shaded according to tempo slope
            d = df[
                (df['trial'] == num + 1) & (df['block'] == x + 1) & (df['instrument'] == 'Keys')
                ].sort_values(['latency', 'jitter'])['tempo_slope']
            for n in range(0, 13):
                coef = cmap(norm(d.iloc[n]))
                g.axes[x, num].axhspan(n-0.5, n+0.5, alpha=0.4, facecolor=coef)
    # Format the figure
    g.despine(left=True, bottom=True)
    g.fig.supxlabel('Correction to Partner', x=0.53, y=0.04)
    g.fig.supylabel('Condition', x=0.01)
    # Add the colorbar
    position = g.fig.add_axes([0.95, 0.3, 0.01, 0.4])
    cb = g.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position,)
    cb.set_alpha(0.5)
    position.text(0, 0.3, ' Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    # Add the legend
    g.legend.remove()
    g.fig.get_axes()[0].legend(loc='lower center', ncol=2, bbox_to_anchor=(2.85, -1.36), title=None, frameon=False)
    # Adjust the plot spacing and save
    g.fig.subplots_adjust(bottom=0.10, top=0.94, wspace=0.15, left=0.1, right=0.95)
    g.savefig(f'{output}\\condition_vs_{xvar}_pointplot.png')


def plot_kde(df, output: str, xvar: str = 'correction_partner', ):
    """
    Creates kernel density estimate plot for phase correction model.
    """
    # Create subplots
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(15, 5), sharex='all', sharey='all')
    # Iterate through each trial individually
    grouper = df.groupby('trial')
    for idx, grp in grouper:
        x = idx - 1
        # Create the kdeplot
        sns.kdeplot(data=grp, x=xvar, hue="instrument", ax=ax[x], hue_order=['Keys', 'Drums'],
                    legend=True if idx == len(grouper) else False)
        # Format axis
        format_ax(ax=ax[x], duo_num=idx)
    # Format figure, move legend to bottom, then save into output directory
    format_fig(fig=fig, xlab=format_label(xvar), ylab='Density')
    sns.move_legend(ax[-1], ncol=2, loc='lower center', bbox_to_anchor=(-1.75, -0.25), title=None, frameon=False)
    fig.savefig(f'{output}\\{xvar}_kdeplot.png')


def create_plots(df: pd.DataFrame, output_dir: str):
    """
    Creates plots for phase correction models.
    """
    output_path = create_output_folder(output_dir)
    # Plot pairgrid
    plot_pairgrid(df=df, output=output_path, xvar='correction_partner_onset')
    plot_pairgrid(df=df, output=output_path, xvar='correction_partner_ioi')
    # # Create barplots (latency vs correction to partner/correction to self)
    # plot_barplot(df, output=output_path, xvar='latency', yvar='correction_partner')
    # plot_barplot(df, output=output_path, xvar='latency', yvar='correction_self')
    # # # Create barplots (jitter vs correction to partner/correction to self)
    # plot_barplot(df[df['latency'] != 0], output=output_path, xvar='jitter', yvar='correction_partner')
    # plot_barplot(df[df['latency'] != 0], output=output_path, xvar='jitter', yvar='correction_self')


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


def format_polar_plot_data(df, num_bins: int = 10) -> pd.DataFrame:
    # Calculate the amount of phase correction for each beat
    corr = df.live_prev_onset - df.delayed_prev_onset
    # Cut into bins
    cut = pd.cut(corr, num_bins, include_lowest=False).value_counts().sort_index()
    # Format the dataframe
    cut.index = pd.IntervalIndex(cut.index.get_level_values(0)).mid
    cut = pd.DataFrame(cut, columns=['val']).reset_index(drop=False).rename(columns={'index': 'idx'})
    cut['idx'] = cut['idx'] * np.pi
    return cut


def create_polar_plots(nn_list, output_dir: str, instr: str = 'Keys'):
    output_path = create_output_folder(output_dir)
    i = 0
    fig, ax = plt.subplots(nrows=5, ncols=13, subplot_kw=dict(projection="polar"), figsize=(25, 8), )
    for (trial, block, latency, jitter, keys, drms) in sorted(nn_list, key=lambda e: (e[0], e[1], e[2], e[3], e[4])):
        dat = format_polar_plot_data(keys if instr == 'Keys' else drms, num_bins=15)
        a = ax[trial - 1, i]
        a.bar(x=dat.idx, height=dat.val, width=0.05, label=f'Measure {block}')
        if trial != 1:
            a.set_xticklabels('')
        a.set_yticks([0, max(a.get_yticks())], labels=['', str(int(max(a.get_yticks())))])
        a.set_thetamin(-45)
        a.set_thetamax(45)
        a.set_theta_zero_location('N')
        a.set_theta_direction(-1)
        a.set_title(f'{latency}ms/{jitter}x' if trial == 1 else '')
        i = i + 1 if i < 12 else 0
    fig.subplots_adjust(bottom=0.05, top=0.95, wspace=0.05, left=0.01, right=0.99, hspace=0.03)
    plt.legend(ncol=4)
    fig.savefig(f'{output_path}\\{instr}_polarplot.png')
