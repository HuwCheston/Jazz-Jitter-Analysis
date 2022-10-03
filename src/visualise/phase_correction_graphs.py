import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from stargazer.stargazer import Stargazer

import src.visualise.visualise_utils as vutils
from src.analyse.prepare_data import average_bpms


@vutils.plot_decorator
def make_pairgrid(df: pd.DataFrame, output_dir: str,
                  xvar: str = 'correction_partner_onset', xlim=(-1, 1)) -> tuple[plt.Figure, str]:
    """
    Creates a figure showing pairs of coefficients obtained for each performer in a condition,
    stratified by block and trial number, with shading according to tempo slope
    """
    # Create the abbreviation column, showing latency and jitter
    df['abbrev'] = df['latency'].astype('str') + 'ms/' + round(df['jitter'], 1).astype('str') + 'x'
    df = df.sort_values(by=['latency', 'jitter'])
    # Sort the palette for the shading
    norm = vutils.create_normalised_cmap(df['tempo_slope'])
    # Create the plot
    pg = sns.catplot(
        data=df, x=xvar, y='abbrev', row='block', col='trial', hue='instrument', hue_order=['Keys', 'Drums'],
        palette=vutils.INSTR_CMAP, kind='strip', height=4, sharex=True, sharey=True,
        aspect=0.6, s=7, jitter=False, dodge=False,
    )
    # Add the reference line in here or it messes up the plot titles
    pg.refline(x=0, alpha=vutils.ALPHA, linestyle='-', color=vutils.BLACK)
    # Format the axis by iterating through
    for num in range(0, 5):
        # When we want different formatting for each row
        pg.axes[0, num].set(title=f'Measure 1\nDuo {num+1}' if num == 2 else f'\nDuo {num+1}', ylabel='', xlim=xlim)
        pg.axes[0, num].tick_params(bottom=False)
        pg.axes[1, num].set(title='Measure 2' if num == 2 else '', ylabel='', xlabel='', xlim=xlim)
        # When we want the same formatting for both rows
        for x in range(0, 2):
            # Disable the grid
            pg.axes[x, num].xaxis.grid(False)
            pg.axes[x, num].yaxis.grid(True)
            # Add the span, shaded according to tempo slope
            d = df[
                    (df['trial'] == num + 1) & (df['block'] == x + 1) & (df['instrument'] == 'Keys')
                ].sort_values(['latency', 'jitter'])['tempo_slope']
            for n in range(0, 13):
                coef = vutils.SLOPES_CMAP(norm(d.iloc[n]))
                pg.axes[x, num].axhspan(n - 0.5, n + 0.5, facecolor=coef)
    _format_pairgrid_fig(pg, norm, xvar=xvar.replace('_', ' ').title())
    fname = f'\\pairgrid_condition_vs_{xvar}.png'
    return pg.fig, fname


def _format_pairgrid_fig(g: sns.FacetGrid, norm, xvar: str = 'Correction') -> None:
    # Format the figure
    g.despine(left=True, bottom=True)
    g.fig.supxlabel(xvar, x=0.53, y=0.04)
    g.fig.supylabel('Condition', x=0.01)
    # Add the colorbar
    position = g.fig.add_axes([0.95, 0.3, 0.01, 0.4])
    g.fig.colorbar(vutils.create_scalar_cbar(norm=norm), cax=position, ticks=vutils.CBAR_BINS)
    position.text(0, 0.3, ' Slope\n(BPM/s)\n', fontsize=12)  # Super hacky way to add a title...
    # Add the legend
    g.legend.remove()
    g.fig.get_axes()[0].legend(loc='lower center', ncol=2, bbox_to_anchor=(2.85, -1.36), title=None, frameon=False)
    # Adjust the plot spacing
    g.fig.subplots_adjust(bottom=0.10, top=0.94, wspace=0.15, left=0.1, right=0.93)


@vutils.plot_decorator
def make_correction_boxplot_by_variable(df: pd.DataFrame, output_dir: str, xvar: str = 'jitter', ylim=(-1, 1),
                                        yvar: str = 'correction_partner_onset', subset=False) -> tuple[plt.Figure, str]:
    """
    Creates a figure showing correction to partner coefficients obtained for each performer in a duo, stratified by a
    given variable (defaults to jitter scale). By default, the control condition is included in this plot,
    but this can be excluded by setting optional argument subset to True.
    """
    # Format the data to exclude the control condition
    if subset:
        df = df[df['latency'] != 0]
    # Create the plot
    bp = sns.catplot(
        data=df, x=xvar, y=yvar, col='trial', hue='instrument', kind='box', sharex=True,
        sharey=True, palette=vutils.INSTR_CMAP, height=4, aspect=0.6,
    )
    bp.refline(y=0, alpha=vutils.ALPHA, linestyle='-', color=vutils.BLACK)
    # Adjust axes-level parameters
    bp.set(ylim=ylim, xlabel='', ylabel='', )
    bp.set_titles("Duo {col_name}")
    for ax in bp.axes.flat:
        ax.yaxis.set_ticks(np.linspace(-1, 1, 5, endpoint=True))
    # Adjust figure-level parameters
    bp.figure.supxlabel(xvar.title(), y=0.06)
    bp.figure.supylabel(yvar.replace('_', ' ').title(), x=0.007)
    sns.move_legend(bp, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.01))
    # Adjust plot spacing
    bp.figure.subplots_adjust(bottom=0.17, top=0.92, left=0.055, right=0.97)
    # Save the plot
    fname = f'\\boxplot_{yvar}_vs_{xvar}.png'
    return bp.figure, fname


@vutils.plot_decorator
def make_polar(nn_list: list[tuple], output_dir: str) -> tuple[plt.Figure, str]:
    """
    Creates circle plots showing relative phase (i.e. playing before or after reference) for both musicians for a trial.
    """
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
                  color=vutils.INSTR_CMAP[0] if st == 'Keys' else vutils.INSTR_CMAP[1])
        # Adjust formatting for all plots
        maxy = max(a.get_yticks())
        _format_polar_ax(ax=a, sl=vutils.SLOPES_CMAP(norm(slope)))
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
    # Add horizontal lines separating subplots
    _add_horizontal_lines_polar(fig)
    # Save figure
    fname = '\\polarplot_relative_phase.png'
    return fig, fname


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


def _format_polar_ax(ax: plt.Axes, sl, yt: list = None, xt: list = None) -> None:
    """
    Formats a polar subplot by setting axis ticks and gridlines, rotation properties, and facecolor
    """
    # Used to test if we've provided xticks already
    test_none = lambda t: [] if t is None else t
    # Set the y axis ticks and gridlines
    ax.set_yticks(test_none(yt))
    ax.set_xticks(test_none(xt))
    ax.grid(False)
    # Set the polar plot rotation properties, direction, etc
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # Set the color according to the tempo slope
    ax.patch.set_facecolor(sl)


def _format_polar_fig(fig: plt.Figure, hand, lab, norm) -> None:
    """
    Format the overall polar plot figure, including setting titles, positioning colorbars, legends etc.
    """
    # Add figure-wise titles and labels
    fig.supylabel('√Density', x=0.01, fontsize='xx-large')
    fig.supxlabel("Relative Phase to Partner's Onsets (πms)", y=0.03, fontsize='xx-large')
    # Create and position colourbar
    position = fig.add_axes([0.95, 0.2, 0.01, 0.6])
    fig.colorbar(vutils.create_scalar_cbar(norm=norm), cax=position, ticks=vutils.CBAR_BINS)
    position.tick_params(labelsize=17.7)
    position.text(0, 0.3, 'Slope\n(BPM/s)\n\n', fontsize='xx-large')  # Super hacky way to add a title...
    # Adjust positioning slightly
    fig.subplots_adjust(bottom=0.03, top=0.96, wspace=0.3, left=0.03, right=0.94, hspace=-0.1)
    # Create and position legend
    plt.legend(hand, lab, ncol=2, loc='lower center', bbox_to_anchor=(-46, -0.34), title=None,
               frameon=False, fontsize='xx-large')


@vutils.plot_decorator
def make_single_condition_phase_correction_plot(keys_df: pd.DataFrame, drms_df: pd.DataFrame,
                                                keys_md, drms_md,
                                                keys_o: pd.DataFrame, drms_o: pd.DataFrame,
                                                output_dir: str,
                                                meta: tuple = ('nan', 'nan', 'nan', 'nan'),) -> tuple[plt.Figure, str]:
    """
    Generate a nice plot of a single performance, showing actual and predicted tempo slope, relative phase adjustments,
    phase correction coefficients to partner and self for both performers
    """
    # Create the matplotlib objects - figure with gridspec, 5 subplots
    fig = plt.figure(figsize=(8, 8))
    ax = vutils.get_gridspec_array(fig=fig)
    # Plot onto the different parts of the gridspec
    _single_fig_slopes(ax, keys_df=keys_df, drms_df=drms_df, keys_o=keys_o, drms_o=drms_o)
    _single_fig_polar(ax, drms_df, keys_df)
    _single_fig_coefficients(ax, drms_md, keys_md)
    # Format the figure and save
    fig.suptitle(f'Duo {meta[0]} (measure {meta[1]}): latency {meta[2]}ms, jitter {meta[3]}x')
    fname = f'\\duo{meta[0]}_measure{meta[1]}_latency{meta[2]}_jitter{meta[3]}.png'
    return fig, fname


def _single_fig_polar(ax: plt.Axes, drms_df: pd.DataFrame, keys_df: pd.DataFrame) -> None:
    """
    For a plot of a single performance, create two polar plots showing phase difference between performers
    """
    # Create the polar plots
    ax[0, 1].sharex(ax[0, 0])
    for num, (ins, st) in enumerate(zip([keys_df, drms_df], ['Keys', 'Drums'])):
        dat = _format_polar_data(ins, num_bins=15)
        ax[0, num].bar(x=dat.idx, height=dat.val, width=0.05, label=st,
                       color=vutils.INSTR_CMAP[0] if st == 'Keys' else vutils.INSTR_CMAP[1])
        _format_polar_ax(ax=ax[0, num], sl=vutils.BLACK, xt=np.pi / 180 * np.linspace(-90, 90, 5, endpoint=True),
                         yt=[0, max(ax[0, num].get_yticks())])
        ax[0, num].set(ylabel='', xlabel='')
        x0, y0, x1, y1 = ax[0, num].get_position().get_points().flatten().tolist()
        ax[0, num].set_position([x0-0.1 if num == 0 else x0-0.2, y0-0.03, x1-0.05 if num == 0 else x1-0.2, y1-0.5])
        ax[0, num].text(-0.1, 0.8, s=st, transform=ax[0, num].transAxes, ha='center',)
        if num == 0:
            ax[0, num].text(1.5, 0.05, s="√Density", transform=ax[0, num].transAxes, ha='center',)
            ax[0, num].text(-0.3, 0.5, s="Relative Phase (πms)",
                            transform=ax[0, num].transAxes, va='center', rotation=90)
            ax[0, num].set_title('Relative Phase to Partner', x=1.5)


def _single_fig_coefficients(ax: plt. Axes, drms_md, keys_md) -> None:
    corr = pd.concat([keys_md.params, drms_md.params], axis=1).rename(
        columns={0: 'Keys', 1: 'Drums'}).transpose().rename(
        columns={'live_prev_ioi': 'Self', 'live_delayed_onset': 'Partner'}).reset_index(drop=False)
    ax[1, 1].sharex(ax[1, 0])
    for num, st in zip(range(0, 2), ['Keys', 'Drums']):
        drms = corr[corr['index'] == st].melt().loc[2:]
        g = sns.stripplot(ax=ax[1, num], x='value', y='variable', data=drms, jitter=False, dodge=False, marker='D', s=7,
                          color=vutils.INSTR_CMAP[0] if st == 'Keys' else vutils.INSTR_CMAP[1])
        g.set_xlim((-1.5, 1.5))
        g.yaxis.grid(True)
        g.xaxis.grid(False)
        g.axvline(alpha=vutils.ALPHA, linestyle='-', color=vutils.BLACK)
        sns.despine(ax=g, left=True, bottom=True)
        g.text(0.05, 0.9, s=st, transform=g.transAxes, ha='center', )
        g.set_ylabel('')
        if num == 0:
            g.axes.set_yticks(g.axes.get_yticks(), g.axes.get_yticklabels())
            g.axes.set_yticklabels(labels=g.axes.get_yticklabels(), va='center')
            g.set_title('Phase Correction', x=1)
            g.set_xlabel('Coefficient', x=1)
        else:
            g.axes.set_yticklabels('')
            g.set_xlabel('')


def _single_fig_slopes(ax: plt.Axes, keys_df: pd.DataFrame, drms_df: pd.DataFrame,
                       keys_o: pd.DataFrame, drms_o: pd.DataFrame):
    # Plot the actual and predicted rolling tempo slope
    z = zip((average_bpms(keys_o, drms_o), average_bpms(keys_df, drms_df, elap='elapsed', bpm='predicted_bpm')),
            ('Actual', 'Fitted'))
    for df, lab in z:
        ax[2, 0].plot(df['elapsed'], df['bpm_rolling'], label=lab)
    ax[2, 0].axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome Tempo')
    ax[2, 0].legend()
    ax[2, 0].set(xlabel='Performance Duration (s)', ylabel='Average tempo (BPM)', ylim=(30, 160), title='Tempo Slope')


def make_single_condition_slope_animation(keys_df, drms_df, keys_o: pd.DataFrame, drms_o: pd.DataFrame,
                                          output_dir, meta: tuple = ('nan', 'nan', 'nan', 'nan'),):
    """
    Creates an animation of actual and predicted tempo slope that should(!) be synchronised to the AV_Manip videos.
    Default FPS is 30 seconds, with data interpolated so plotting look nice and smooth. This can be changed in vutils.
    WARNING: this will take a really long time to complete!!!
    """
    def init():
        act_line.set_data([], [])
        pred_line.set_data([], [])
        return act_line, pred_line,

    def animate(i: int,):
        f = lambda d: d[d.index <= i]
        a, p = f(act), f(pred)
        act_line.set_data(a['elapsed'].to_numpy(), a['bpm_rolling'].to_numpy())
        pred_line.set_data(p['elapsed'].to_numpy(), p['bpm_rolling'].to_numpy())
        return act_line, pred_line,

    output = vutils.create_output_folder(output_dir)
    chain = lambda k, d, e, b: vutils.interpolate_df_rows(
        vutils.append_count_in_rows_to_df(average_bpms(df1=k, df2=d, elap=e, bpm=b)))
    act = chain(keys_o, drms_o, 'elapsed', 'bpm')
    pred = chain(keys_df, drms_df, 'elapsed', 'predicted_bpm')

    # Create the matplotlib objects we need
    fig = plt.figure()
    ax = plt.axes(xlabel='Performance duration (s)',
                  ylabel='Average tempo (BPM, four-beat rolling window)',
                  xlim=(0, 100), ylim=(30, 160),
                  title=f'Duo {meta[0]} (measure {meta[1]}): latency {meta[2]}ms, jitter {meta[3]}x')
    act_line, = ax.plot([], [], lw=2, label='Actual tempo')
    pred_line, = ax.plot([], [], lw=2, label='Fitted tempo')
    # Set the axes parameters
    ax.tick_params(axis='both', which='both', bottom=False, left=False, )
    ax.axhline(y=120, color='r', linestyle='--', alpha=0.3, label='Metronome Tempo')
    ax.legend()
    # Create the animation and save to our directory
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=int(act.index.max()),
                                   interval=1000 / vutils.VIDEO_FPS, blit=True)
    anim.save(f'{output}\\duo{meta[0]}_measure{meta[1]}_latency{meta[2]}_jitter{meta[3]}.mp4',
              writer='ffmpeg', fps=vutils.VIDEO_FPS)


def output_regression_table(mds: list, output_dir: str, verbose_footer: bool = False) -> None:
    """
    Create a nicely formatted regression table from a list of regression models ordered by trial, and output to html.
    """
    def get_cov_names(name: str) -> list[str]:
        k = lambda x: float(x.partition('T.')[2].partition(']')[0])
        # Try and sort the values by integers within the string
        try:
            return [o for o in sorted([i for i in out.cov_names if name in i], key=k)]
        # If there are no integers in the string, return unsorted
        except ValueError:
            return [i for i in out.cov_names if name in i]

    def format_cov_names(i: str, ext: str = '') -> str:
        # If we've defined a non-default reference category the statsmodels output looks weird, so catch this
        if ':' in i:
            lm = lambda s: s.split('C(')[1].split(')')[0].title() + ' (' + s.split('[T.')[1].split(']')[0] + ')'
            return lm(i.split(':')[0]) + ': ' + lm(i.split(':')[1])
        if 'Treatment' in i:
            return i.split('C(')[1].split(')')[0].title().split(',')[0] + ' (' + i.split('[T.')[1].replace(']', ')')
        else:
            base = i.split('C(')[1].split(')')[0].title() + ' ('
            return base + i.split('C(')[1].split(')')[1].title().replace('[T.', '').replace(']', '') + ext + ')'

    # Create the stargazer object from our list of models
    out = Stargazer(mds)
    # Get the original covariate names
    l_o, j_o, i_o, int_o = (get_cov_names(i) for i in ['latency', 'jitter', 'instrument', 'Intercept'])
    orig = [item for sublist in [l_o, j_o, i_o, int_o] for item in sublist]
    # Format the original covariate names so they look nice
    lat_fm = [format_cov_names(s, 'ms') for s in l_o]
    jit_fm = [format_cov_names(s, 'x') for s in j_o]
    instr_fm = [format_cov_names(s) for s in i_o]
    form = [item for sublist in [lat_fm, jit_fm, instr_fm, int_o] for item in sublist]
    # Format the stargazer object
    out.custom_columns([f'Duo {i}' for i in range(1, len(mds) + 1)], [1 for _ in range(1, len(mds) + 1)])
    out.show_model_numbers(False)
    out.covariate_order(orig)
    out.rename_covariates(dict(zip(orig, form)))
    t = out.dependent_variable
    out.dependent_variable = ' ' + out.dependent_variable.replace('_', ' ').title()
    # If we're removing some statistics from the bottom of our table
    if not verbose_footer:
        out.show_adj_r2 = False
        out.show_residual_std_err = False
        out.show_f_statistic = False
    # Create the output folder
    fold = vutils.create_output_folder(output_dir + '\\phase_correction')
    # Render to html and write the result
    with open(f"{fold}\\regress_{t}.html", "w") as f:
        f.write(out.render_html())


@vutils.plot_decorator
def make_trial_hist(r: pd.DataFrame, output_dir: str, xvar: str = 'r2', kind: str = 'hist') -> tuple[plt.Figure, str]:
    """
    Creates histograms of model parameters stratified by trial and instrument, x-axis variable defaults to R-squared
    """
    # Create the ditsplot
    g = sns.displot(r, col='trial', kind=kind, x=xvar, hue="instrument", multiple="stack", palette=vutils.INSTR_CMAP,
                    height=4, aspect=0.6, )
    # Format figure-level properties
    g.set(xlabel='', ylabel='')
    g.set_titles("Duo {col_name}", size=12)
    g.figure.supxlabel(xvar.title(), y=0.06)
    g.figure.supylabel('Count', x=0.007)
    # Move legend and adjust subplots
    sns.move_legend(g, 'lower center', ncol=2, title=None, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=12)
    g.figure.subplots_adjust(bottom=0.17, top=0.92, left=0.055, right=0.97)
    # Return, with plot_decorator used for saving
    fname = f'\\{xvar}_hist'
    return g.figure, fname
