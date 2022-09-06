import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


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

def plot_pairgrid(df, output: str):
    # Create pivot table
    # piv = df.pivot_table('correction_partner', ['latency', 'jitter', 'instrument', 'block'], 'trial').reset_index(
    #     drop=False).sort_values(by=['latency', 'jitter', 'block'])
    # piv['abbrev'] = piv['latency'].astype(str) + '/' + round(piv['jitter'], 1).astype(str) + '/' + piv['block'].astype(
    #     str)
    # piv = piv.drop(columns=['latency', 'jitter', 'block'])
    # Create the pairgrid
    # g = sns.PairGrid(piv, x_vars=piv.columns[1:6], y_vars="abbrev", hue='instrument',
    #                  height=10, aspect=0.25, hue_order=['Keys', 'Drums'],
    #                  palette=sns.color_palette(["#1f77b4", '#ff7f0e']))
    df['abbrev'] = df['latency'].astype('str') + '/' + round(df['jitter'], 1).astype('str')
    g = sns.FacetGrid(df, hue='instrument', row='block', col='trial', hue_order=['Keys', 'Drums'],
                      palette=sns.color_palette(["#1f77b4", '#ff7f0e']))
    g.map(sns.stripplot, 'correction_partner', 'abbrev')


    plt.show()
    # # Use the same x axis limits on all columns and add better labels
    # g.set(xlim=(-1, 1), xlabel="Correction to Partner", ylabel="")
    # # Use semantically meaningful titles for the columns
    # titles = [f"Duo {num}" for num in range(1, 6)]
    # for ax, title in zip(g.axes.flat, titles):
    #     # Set a different title for each axes
    #     ax.set(title=title)
    #     # Make the grid horizontal instead of vertical
    #     ax.xaxis.grid(False)
    #     ax.yaxis.grid(True)
    # sns.despine(left=True, bottom=True)
    # plt.show()


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


def create_plots(df, output_dir):
    """
    Creates plots for phase correction models.
    """
    # Create a folder to store the plots
    output_path = output_dir + '\\figures\\phase_correction_graphs'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # Plot pairgrid
    plot_pairgrid(df=df, output=output_path)
    # # Create barplots (latency vs correction to partner/correction to self)
    # plot_barplot(df, output=output_path, xvar='latency', yvar='correction_partner')
    # plot_barplot(df, output=output_path, xvar='latency', yvar='correction_self')
    # # # Create barplots (jitter vs correction to partner/correction to self)
    # plot_barplot(df[df['latency'] != 0], output=output_path, xvar='jitter', yvar='correction_partner')
    # plot_barplot(df[df['latency'] != 0], output=output_path, xvar='jitter', yvar='correction_self')