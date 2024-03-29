"""Code for generating plots that combine all performance success metrics"""

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils
from src.analyse.phase_correction_models import PhaseCorrectionModel

__all__ = [
    'generate_all_metrics_plots'
]


class PairPlotAllVariables(vutils.BasePlot):
    """
    Creates an R-style paired plot, showing correlation and distribution of four variables (tempo slope,
    ioi variability, asynchrony rms, and subjective success). Each variable is averaged across instrument and repeat,
    so e.g. we have one value for ioi variability for duo 2, 180ms latency, 0.5x jitter. Tempo slope is given in
    absolute form, to make linear relationships clearer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels: list[str] = kwargs.get(
            'labels', ['|Tempo slope|\n(|BPM/s|)', 'Asynchrony\n(RMS, ms)',
                       'Timing\nirregularity\n(SD, ms)', 'Performer-\nreported\nsuccess', 'Listener-\nreported\nsuccess']
        )
        self._jitter_success: bool = kwargs.get('jitter_success', True)
        self._abs_slope: bool = kwargs.get('abs_slope', True)
        self._avg_sessions: bool = kwargs.get('average_sessions', True)
        self.error_bar: str = kwargs.get('error_bar', 'sd')
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)
        self.percentiles: tuple[float] = kwargs.get('percentiles', (2.5, 97.5))
        self.vars: list[str] = kwargs.get('vars', [
            'tempo_slope', 'pw_asym', 'ioi_std', 'success', 'perceptual_answer_mean'
        ])
        self.df: pd.DataFrame = self._format_df()
        self.counter: int = 0

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting.
        """
        # Define the non-formatted variable names that we want to include
        df = self.df[['trial', 'block', 'latency', 'jitter', 'instrument', *self.vars]]
        # Update variables, if required: need to disable chained assignment warnings here
        pd.options.mode.chained_assignment = None
        if self._jitter_success:
            # Add gaussian noise to success variable
            df['success'] = [val + np.random.normal(0, 0.01) for val in df['success'].to_list()]
            df['perceptual_answer_mean'] = [val + np.random.normal(0, 0.01) for val in df['perceptual_answer_mean'].to_list()]
        if self._abs_slope:
            # Get absolute tempo slope
            df['tempo_slope'] = df['tempo_slope'].abs()
        pd.options.mode.chained_assignment = 'warn'
        # Group by trial, latency, and jitter, get the mean, then reset the index and return
        if self._avg_sessions:
            return df.groupby(['trial', 'latency', 'jitter']).mean().reset_index()
        else:
            return df

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class to generate the plot and save in plot decorator.
        """
        # The create plot function here returns our figure to be stored as an attribute defined in the parent class
        self.g = self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = f'{self.output_dir}\\pairplot_all_variables'
        return self.g.figure, fname

    # noinspection PyUnusedLocal
    def _add_label_to_diag(
            self, *args, **kwargs
    ) -> None:
        """
        Adds variable names to diagonal plots.
        """
        # Get the current axis
        ax = plt.gca()
        # Get the required label from our list
        s = self.labels[self.counter]
        # Add the label to the upper centre of the diagonal plot
        y = 0.85 if self.counter < 2 else 0.8
        ax.annotate(
            s, xy=(0.97, y), transform=ax.transAxes, xycoords='axes fraction', va='center', ha='right', fontsize=30
        )
        # Increment the counter by one for the next plot
        self.counter += 1

    def reg_plot(
            self, *args, **_
    ) -> None:
        """
        Adds a regression line with bootstrapped confidence intervals to a plot
        """
        def regress(
                g: pd.DataFrame
        ) -> np.ndarray:
            # Coerce x variable into correct form and add constant
            x = sm.add_constant(g['x'].to_list())
            # Coerce y into correct form
            y = g['y'].to_list()
            # Fit the model, predicting y from x
            md = sm.OLS(y, x).fit()
            # Return predictions made using our x vector
            return md.predict(x_vec)

        x_, y_ = args
        grp = pd.concat([x_, y_], axis=1)
        grp.columns = ['x', 'y']
        # Create the vector of x values we'll use when making predictions
        x_vec = sm.add_constant(
            np.linspace(x_.min(), x_.max(), len(x_))
        )
        # Create model predictions for a regression fitted on our actual data
        act_predict = regress(grp)
        # Create bootstrapped regression predictions
        res = [regress(grp.sample(frac=1, replace=True, random_state=n)) for n in range(0, self.n_boot)]
        # Create empty variable to hold concatenated dataframe, so we don't get warnings
        conc = None
        # Convert bootstrapped predictions to dataframe and get standard deviation for each x value
        if self.error_bar == 'ci':
            # Convert bootstrapped predictions to dataframe and get required percentiles for each x value
            boot_res = (
                pd.DataFrame(res)
                  .transpose()
                  .apply(np.nanpercentile, axis=1, q=self.percentiles)
                  .apply(pd.Series)
                  .rename(columns={0: 'low', 1: 'high'})
            )
            # Concatenate bootstrapped standard errors with original predictions
            # No need to add error terms to our original data, we just plot the percentiles
            conc = pd.concat([pd.Series(x_vec[:, 1]).rename('x'), pd.Series(act_predict).rename('y'), boot_res], axis=1)
        elif self.error_bar == 'sd':
            # Convert bootstrapped predictions to dataframe and get standard deviation for each x value
            boot_res = pd.DataFrame(res).transpose().std(axis=1).rename('sd')
            # Concatenate bootstrapped standard errors with original predictions
            conc = pd.concat([pd.Series(x_vec[:, 1]).rename('x'), pd.Series(act_predict).rename('y'), boot_res], axis=1)
            # Get confidence intervals by adding and subtracting error to original y values
            conc['high'] = conc['y'] + conc['sd']
            conc['low'] = conc['y'] - conc['sd']
        # Plot the resulting data
        plt.plot(conc['x'], conc['y'], color=vutils.RED, lw=5)
        plt.fill_between(conc['x'], conc['high'], conc['low'], color=vutils.RED, alpha=0.2)

    # noinspection PyUnusedLocal
    @staticmethod
    def _reg_coef(
            x: np.ndarray, y: np.ndarray, *args, **kwargs
    ) -> None:
        """
        Adds correlation coefficients and significance asterisks to diagonal plots, with fontsize scaled to coefficient.
        """
        # Get the current axis
        ax = plt.gca()
        # Calculate the correlation between the x and y variables
        r, p = stats.pearsonr(x, y)
        # Convert the p value to asterisks
        ast = vutils.get_significance_asterisks(p)
        # Create the format string, with additional escape characters for nicer kerning between asterisks
        if ast != '':
            s = "${{{}}}".format(round(r, 2)) + ''.join([r"^{{{}}}\,\!".format(a) for a in ast]) + "$"
        # If our correlation is not significant, we won't return any asterisks, so need to catch this
        else:
            s = "${{{}}}$".format(round(r, 2))
        # Add the format string to the middle of the axis, with fontsize scaled to match the absolute r value
        scaling = abs(r)
        if scaling < 0.3:
            scaling = 0.3
        ax.annotate(s, xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=80 * scaling)
        # Disable ticks on the diagonal plots
        ax.tick_params(which='both', axis='both', bottom=False, top=False, left=False, right=False)

    def _create_plot(
            self
    ) -> sns.PairGrid:
        """
        Creates the pair grid object, maps required plot functions, and returns for further modification.
        """
        # Create the pair grid plot
        g = sns.PairGrid(data=self.df, vars=self.vars, height=4.7, aspect=1, corner=False, dropna=True, despine=False,)
        # Map a histogram with KDE estimate to the diagonal plots
        g.map_diag(
            sns.histplot, color=vutils.BLACK, alpha=vutils.ALPHA, stat='count', bins=7, kde=True, line_kws={'lw': 5, }
        )
        # Map a scatter plot to the lower plots
        g.map_lower(sns.scatterplot, s=100, edgecolor=vutils.BLACK, facecolor=vutils.WHITE, marker='o', linewidth=2)
        # Map a regression line of best fit with bootstrapped confidence intervals to the lower plots
        g.map_lower(self.reg_plot)
        # Map the variable titles to the diagonal plots
        g.map_diag(self._add_label_to_diag)
        # Map the regression coefficients to the upper plots
        g.map_upper(self._reg_coef)
        # Return the object for further alteration
        return g

    def _format_ax(
            self
    ) -> None:
        """
        Formats axes-level objects, including ticks, labels, spines etc.
        """
        # Iterate through all axis
        for ax in self.g.axes.flatten():
            # Turn off x and y labels
            ax.set(ylabel=None, xlabel=None)
            # Turn off both ticks; we'll add these back in for the required axes later
            ax.tick_params(width=0, length=0, which='both')
            # Set axis thickness
            plt.setp(ax.spines.values(), linewidth=2)
            # Add in the top and right axis to make the axis object a complete box
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
        # If we're using absolute slope, we need to define a different axis limit
        slope_lim = (0, 0.5) if self._abs_slope else (-0.5, 0.5)
        # Iterate through all rows and columns and set required axis limits and step values
        for i, lim, step in zip(range(0, len(self.vars)), [slope_lim, (0, 400), (0, 100), (1, 9), (1, 9)], [3, 5, 5, 3, 3]):
            ticks = np.linspace(lim[0], lim[1], step)
            self.g.axes[i, i].set(ylim=lim, xlim=lim, yticks=ticks, xticks=ticks)
        # If using absolute tempo slope, adjust axis limit slightly, so we don't cut off some markers
        if self._abs_slope:
            self.g.axes[0, 0].set(xlim=(-0.01, 0.5))
        # Iterate through just the first column and last row and add the correct axis ticks back in
        for i in range(0, 5):
            self.g.axes[i, 0].tick_params(width=3, length=7, axis='y')
            self.g.axes[4, i].tick_params(width=3, length=7, axis='x')
        # The axis in the bottom left corner is the only one which requires ticks on both axes
        self.g.axes[4, 0].tick_params(width=3, length=7, axis='both')

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Adjust the spacing between subplots slightly
        self.g.figure.subplots_adjust(wspace=0.15, hspace=0.15, top=0.97, bottom=0.05, left=0.05, right=0.97)


class RegressionTableAllMetrics:
    """
    Creates an R-style regression table for all performance success metrics.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = self._format_df(df)
        self.model_list = self._generate_models()
        self.output_dir = output_dir

    def create_tables(self):
        for k, v in self.model_list.items():
            tab = self._output_regression_table(v)
            with open(f"{self.output_dir}\\regress_{k}.html", "w") as f:
                f.write(tab.render_html())

    @staticmethod
    def _format_df(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        # Need to catch FutureWarnings related to dropping invalid columns in a group by
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            return (
                df.groupby(['trial', 'latency', 'jitter', 'instrument'])
                  .mean(numeric_only=False)
                  .reset_index(drop=False)
            )

    def _generate_models(
            self,
    ) -> dict:
        res = {
            'ioi_std': [],
            'pw_asym': [],
            'success': [],
            'tempo_slope': []
        }
        for idx, grp in self.df.groupby('trial'):
            for var in ['ioi_std', 'pw_asym', 'success']:
                md = smf.ols(f'{var}~C(latency)+C(jitter)+C(instrument)', data=grp).fit()
                res[var].append(md)
            avg = grp.groupby(['latency', 'jitter']).mean().reset_index(drop=False)
            md = smf.ols(f'tempo_slope~C(latency)+C(jitter)', data=avg).fit()
            res['tempo_slope'].append(md)
        return res

    @staticmethod
    def _get_cov_names(
            out, name: str
    ) -> list[str]:
        k = lambda x: float(x.partition('T.')[2].partition(']')[0])
        # Try and sort the values by integers within the string
        try:
            return [o for o in sorted([i for i in out.cov_names if name in i], key=k)]
        # If there are no integers in the string, return unsorted
        except ValueError:
            return [i for i in out.cov_names if name in i]

    @staticmethod
    def _format_cov_names(
            i: str, ext: str = ''
    ) -> str:
        # If we've defined a non-default reference category the stats models output looks weird, so catch this
        if ':' in i:
            lm = lambda s: s.split('C(')[1].split(')')[0].title() + ' (' + s.split('[T.')[1].split(']')[0] + ')'
            return lm(i.split(':')[0]) + ': ' + lm(i.split(':')[1])
        if 'Treatment' in i:
            return i.split('C(')[1].split(')')[0].title().split(',')[0] + ' (' + i.split('[T.')[1].replace(']', ')')
        else:
            base = i.split('C(')[1].split(')')[0].title() + ' ('
            return base + i.split('C(')[1].split(')')[1].title().replace('[T.', '').replace(']', '') + ext + ')'

    def _output_regression_table(
            self, mds: list,
    ) -> Stargazer:
        """
        Create a nicely formatted regression table from a list of regression models ordered by trial, and output to html
        """

        # Create the stargazer object from our list of models
        out = Stargazer(mds)
        # Get the original co-variate names
        l_o, j_o, i_o, int_o = (self._get_cov_names(out, i) for i in ['latency', 'jitter', 'instrument', 'Intercept'])
        orig = [item for sublist in [l_o, j_o, i_o, int_o] for item in sublist]
        # Format the original co-variate names so they look nice
        lat_fm = [self._format_cov_names(s, 'ms') for s in l_o]
        jit_fm = [self._format_cov_names(s, 'x') for s in j_o]
        instr_fm = [self._format_cov_names(s) for s in i_o]
        form = [item for sublist in [lat_fm, jit_fm, instr_fm, int_o] for item in sublist]
        # Format the stargazer object
        out.custom_columns([f'Duo {i}' for i in range(1, len(mds) + 1)], [1 for _ in range(1, len(mds) + 1)])
        out.show_model_numbers(False)
        out.covariate_order(orig)
        out.rename_covariates(dict(zip(orig, form)))
        _ = out.dependent_variable
        out.dependent_variable = ' ' + out.dependent_variable.replace('_', ' ').title()
        # If we're removing some statistics from the bottom of our table
        out.show_adj_r2 = False
        out.show_residual_std_err = False
        out.show_f_statistic = False
        out.significance_levels([0.05, 0.01, 0.001])
        return out


class PointPlotRepeatComparisons(vutils.BasePlot):
    """
    Creates a pointplot showing bootstrapped mean differences in variables across each experimental session.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = ['tempo_slope', 'ioi_std', 'pw_asym', 'success', 'perceptual_answer_mean']
        self.titles = [
            'Tempo slope (BPM/s)',
            'Asynchrony (RMS, ms)',
            'Timing irregularity (SD, ms)',
            'Performer-reported Success',
            'Listener-reported Success'
        ]
        self.xlims = [0.2, 9, 70, 2, 2]
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=3, sharex=False, sharey=True, figsize=(18.8, 9.4)
        )
        self.ax.flatten()[-1].axis('off')
        self.boot_df = self._generate_bootstraps()
        self.handles, self.labels = None, None

    def _generate_bootstraps(
            self
    ) -> pd.DataFrame:
        """
        Generate our bootstrapped difference in means between sessions for each metric and duo combination
        """
        # Initialise empty list to hold results
        res = []
        # Iterate through each duo
        for idx, grp in self.df.groupby('trial'):
            # Iterate through each of the metrics we want to extract
            for met in self.metrics:
                # Extract both arrays
                a1 = grp[grp['block'] == 1][met]
                a2 = grp[grp['block'] == 2][met]
                # Get the actual mean difference between arrays
                mea = a2.mean() - a1.mean()
                # Get the lower and upper bands of our confidence intervals and append everything
                low, high = vutils.bootstrap_mean_difference(a1, a2, n_boot=vutils.N_BOOT)
                # We append index as a string so that it is automatically treated as a category in seaborn
                res.append((str(idx), met, low, high, mea,))
        # Return everything as a dataframe
        return pd.DataFrame(res, columns=['duo', 'metric', 'low', 'high', 'mean'])

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\pointplot_repeat_comparisons'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the plots for each metric and duo
        """
        # Iterate over all of our metrics and axes at the same time
        for (i, v), ax in zip(
                self.boot_df.groupby('metric', sort=False), self.ax.flatten()
        ):
            # Create our scatter plots
            sns.scatterplot(
                data=v, y='duo', x='mean', ax=ax, s=125, zorder=100, edgecolor=vutils.BLACK, lw=2,
                palette=vutils.DUO_CMAP, markers=vutils.DUO_MARKERS, hue='duo', legend=True, style='duo'
            )
            # Iterate through each row
            for idx, row in v.groupby('duo'):
                # Draw our horizontal lines to act as a grid
                ax.hlines(y=idx, xmin=-100, xmax=100, color=vutils.BLACK, alpha=0.2, lw=2, zorder=-1)
                # Draw our 95% confidence intervals around our actual mean
                ax.hlines(y=idx, xmin=row['low'], xmax=row['high'], lw=3, color=vutils.BLACK, zorder=10)
                # Add vertical brackets to our confidence intervals
                for var in ['low', 'high']:
                    ax.vlines(
                        x=row[var], ymin=int(idx) - 1.1, lw=3, ymax=int(idx) - 0.9, color=vutils.BLACK, zorder=-1
                    )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axis-level objects
        """
        # Iterate over each axis, x axis limit, and title
        for ax, lim, tit in zip(
                self.ax.flatten(), self.xlims, self.titles,
        ):
            self.handles, self.labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            # Set axis properties
            ax.set(xlabel='', ylabel='', xlim=(-lim, lim))
            ax.set_title(tit, y=1.01)
            # Set tick and axis width slightly
            plt.setp(ax.spines.values(), linewidth=2)
            ax.tick_params(axis='both', width=3)
            # Add in a vertical line at 0 (no significant difference between duos)
            ax.axvline(x=0, color=vutils.BLACK, ls='--', lw=2, alpha=0.8)

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        lgnd = plt.legend(
            self.handles, self.labels, title='Duo', frameon=False, ncol=1, loc='right', markerscale=1.5,
            fontsize=vutils.FONTSIZE + 3, edgecolor=vutils.BLACK, bbox_to_anchor=(0.65, 0.55),
        )
        for handle in lgnd.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([100])
        # Add in the axis labels
        self.fig.suptitle('Objective evaluations', fontsize=vutils.FONTSIZE + 7)
        self.fig.text(0.4, 0.475, 'Subjective evaluations', fontsize=vutils.FONTSIZE + 7)
        self.fig.supxlabel('Difference in mean across both sessions')
        self.fig.supylabel('Duos', x=0.01)
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.88, wspace=0.1, hspace=0.5)


class BarPlotRegressionCoefficients(vutils.BasePlot):
    """
    Creates a barplot showing regression coefficients and confidence intervals for numerous multiple regression models
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get parameters for table from kwargs
        self.categories: list[str] = kwargs.get('categories', ['tempo_slope', 'pw_asym', 'ioi_std', 'success', 'perceptual_answer_mean'])
        self.labels: list[str] = kwargs.get('labels', [
            'Tempo slope (BPM/s)', 'Asynchrony (RMS, ms)', 'Timing irregularity (SD, ms)', 'Performer-reported Success', 'Listener-reported Success'
        ])
        self.averaged_vars: list[str] = kwargs.get('averaged_vars', ['tempo_slope', 'pw_asym', 'perceptual_answer_mean'])
        self.predictor_ticks: list[str] = kwargs.get('predictor_ticks', [
            'Reference\n(0ms, 0.0x, Drums)', 'Latency (ms)', 'Jitter', 'Instrument'
        ])
        self.levels: list[str] = kwargs.get('levels', ['Intercept', '23', '45', '90', '180', '0.5', '1.0', 'Keys'])
        self.yticks = kwargs.get('yticks', [(-0.6, 0.6, 7), (-50, 200, 6), (-10, 40, 6), (-5, 10, 4), (-5, 10, 4)])
        self.alpha: float = kwargs.get('alpha', 0.05)
        # Format dataframe to get regression results
        self.df = self._format_df()
        # Get plotting parameters from kwargs
        self.x_fontsize: int = kwargs.get('x_fontsize', 60)
        self.errorbar_margin: float = kwargs.get('errorbar_margin', 0.03)
        self.vlines_margin: float = kwargs.get('vlines_margin', 0.25)
        self.add_pvals: bool = kwargs.get('add_pvals', False)
        # Create plotting objects in matplotlib
        self.fig, self.ax = plt.subplots(
            nrows=len(self.categories),
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=(18.8, 4.7 * len(self.categories))
        )
        self.legend_handles_labels = None   # Used to hold our legend object for later

    def _extract_regression_parameters(
            self, reg
    ) -> pd.DataFrame:
        """
        Extracts regression parameter from statsmodels ols fit and coerces into a dataframe
        """
        # Extract confidence intervals
        ci = reg.conf_int(alpha=self.alpha).rename(columns={0: 'low', 1: 'high'})
        # Extract beta coefficients
        beta = reg.params.rename('beta')
        # Concatenate confidence intervals and coefficients
        full = pd.concat([beta, ci], axis=1)
        # Add in our p-values, now converted into asterisks
        full['pval'] = [vutils.get_significance_asterisks(p) for p in reg.pvalues.to_list()]
        return full

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Coerces dataframe into correct format for plotting by extracting regression results
        """
        # Melt the dataframe to convert our response variables from rows to columns
        test = self.df.melt(
            id_vars=['trial', 'block', 'latency', 'jitter', 'instrument'], value_vars=self.categories
        )
        res = []
        # Iterate through each trial
        for idx, grp in test.groupby('trial'):
            # Iterate through each response variable
            for i, g in grp.groupby('variable'):
                # Define our initial model formula and rename our value column
                md = f'{i}~C(latency)+C(jitter)'
                g = g.rename(columns={'value': i})
                # Variables which are averaged across the duo, i.e. tempo slope, asynchrony
                if i in self.averaged_vars:
                    avg = g.groupby(['trial', 'latency', 'jitter']).mean().reset_index(drop=False)
                # Variables which are unique for each performer, i.e. timing irregularity and self-reported success
                else:
                    avg = g.groupby(['trial', 'latency', 'jitter', 'instrument']).mean().reset_index(drop=False)
                    # For these variables, add in instrument as a predictor
                    md += '+C(instrument)'
                # Create and fit our regression model
                reg = smf.ols(data=avg, formula=md).fit()
                # Extract the parameters we want from our dataframe
                full = self._extract_regression_parameters(reg)
                # Iterate through each row
                for pred, row in full.iterrows():
                    # Append the required results
                    res.append({
                        'trial': idx,
                        'var': i,
                        'predictor': str(pred).split(')[T.')[-1].split(']')[0],   # We modify the string slightly here
                        'beta': row['beta'],
                        'pval': row['pval'],
                        'low_ci': row['low'],
                        'high_ci': row['high']
                    })
        # Create the dataframe and set the correct column types
        df = pd.DataFrame(res)
        df['var'] = pd.Categorical(df['var'], self.categories)
        df['predictor'] = pd.Categorical(df['predictor'], self.levels)
        return df.sort_values(by=['trial', 'predictor']).reset_index(drop=True)

    def _add_dummy_instrument_var(
            self, df: pd.DataFrame, var_name: str = 'tempo_slope'
    ) -> pd.DataFrame:
        """
        Add in dummy instrument variable for metrics which don't use this as a predictor, to enable us to keep the x
        axis consistent for all subplots
        """
        # Iterate through each trial number
        for i in range(self.df['trial'].nunique()):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                # Append a new row to our dataframe with the dummy variable in
                df = df.append(
                    {'trial': i + 1, 'var': var_name, 'predictor': 'Keys', 'beta': 0, 'low_ci': 0, 'high_ci': 0},
                    ignore_index=True)
        return df

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\barplot_regression_coefs'
        return self.fig, fname

    def _add_errorbars(
            self, grp: pd.DataFrame, ax: plt.Axes
    ) -> None:
        """
        Adds errorbars into a grouped barplot. This is difficult to do in standard matplotlib, so we add errorbars
        manually using the ax.hlines method, providing our upper and lower confidence intervals into this
        """
        # Zip and iterate through each row in the dataframe and patch (bar) in the graph
        for (i, r), p in zip(grp.iterrows(), ax.patches):
            # Get the centre of our x position
            pos = p.get_x() + p.get_width() / 2
            # Add in the centre line of our error bar
            ax.vlines(pos, r['low_ci'], r['high_ci'], color=vutils.BLACK, lw=2)
            # Add in brackets/braces to our error bar, at the top and bottom
            for v in ['low_ci', 'high_ci']:
                ax.hlines(
                    r[v], pos - self.errorbar_margin, pos + self.errorbar_margin, color=vutils.BLACK, lw=2
                )
            # Add in our significance asterisks, if required (off by default)
            if self.add_pvals:
                ypos = r['low_ci'] if abs(r['low_ci']) > abs(r['high_ci']) else r['high_ci']
                ax.text(pos, ypos + self.errorbar_margin, r['pval'])

    def _add_x_to_averaged_vars(
            self, ax: plt.Axes
    ) -> None:
        """
        Adds in an X to an axes to show that certain variables were not included as predictors
        """
        # Get all of our X positions, sort them, and extract the final 5 (equivalent to our keys variable, all duos)
        xs = sorted([p.get_x() for p in ax.patches])[-5:]
        # Add in the text
        ax.text((xs[2] + xs[3]) / 2, y=0, s='X', fontsize=self.x_fontsize, ha='center', va='center_baseline')

    def _create_plot(
            self
    ) -> None:
        """
        Creates bar chart + error bars for each subplot, corresponding to one variable
        """
        # Iterate through our subplots and each variable
        for ax, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('var')):
            # Add in our dummy instrument variable rows to the groups that need it
            if idx in self.averaged_vars:
                grp = self._add_dummy_instrument_var(grp, var_name=idx)
            # Sort our table here so that our error bars match our bar heights
            grp['predictor'] = pd.Categorical(grp['predictor'], self.levels)
            grp = grp.sort_values(by=['trial', 'predictor']).reset_index(drop=True)
            # Create the bar plot
            sns.barplot(
                data=grp, x='predictor', y='beta', hue='trial', ax=ax, errorbar=None, edgecolor=vutils.BLACK,
                lw=2, palette=vutils.DUO_CMAP, saturation=0.8, alpha=0.8
            )
            # Add in our error bars to the plot
            self._add_errorbars(grp, ax)
            # Add in Xs if required to axis
            if idx in self.averaged_vars:
                self._add_x_to_averaged_vars(ax)

    def _add_predictor_ticks(
            self, ax: plt.Axes, ticks: list[str] = None
    ) -> None:
        """
        Adds a secondary x axis showing the names of each of our predictor variables
        """
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks([0, 2.5, 5.5, 7], self.predictor_ticks if ticks is None else ticks)
        ax2.tick_params(width=3, which='major')
        plt.setp(ax2.spines.values(), linewidth=2)

    def _add_seperating_vlines(
            self, ax: plt.Axes
    ) -> None:
        """
        Adds in vertical lines seperating levels for each predictor on the x axis, e.g. latency, jitter...
        """
        # Get the ticks corresponding to each predictor
        xs = np.array(sorted([p.get_x() for p in ax.patches]))
        idxs = np.argwhere(np.diff(xs) > self.vlines_margin)
        vals = xs[idxs][[0, 4, 6]]
        # Get our axis y limit
        ymi, yma = ax.get_ylim()
        # Iterate through each of the required lines and add it in
        for v in vals:
            ax.vlines(v + self.vlines_margin, ymin=ymi, ymax=yma, color=vutils.BLACK, lw=2, alpha=vutils.ALPHA, ls='--')

    def _format_ax(
            self
    ) -> None:
        """
        Applies required axis formatting
        """
        ylims = [(-0.7, 0.7), (-50, 250), (-20, 50), (-9, 12), (-9, 12)]
        # Iterate through each axis and label, with a counter
        for count, ax, lab, yt, yl in zip(
                range(self.df['var'].nunique()), self.ax.flatten(), self.labels, self.yticks, ylims
        ):
            # Add in a horizontal line at y = 0
            ax.axhline(y=0, color=vutils.BLACK, lw=2)
            # Set our axis labels
            ax.set(
                xlabel='',
                ylabel=lab,
                xticklabels=['Intercept', '23', '45', '90', '180', '0.5x', '1.0x', 'Keys'],
                yticks=np.linspace(*yt),
                ylim=yl
            )
            # Add in our predictor ticks to the top of the first subplot
            ticks = self.predictor_ticks if count == 0 else ['' for _ in range(len(self.predictor_ticks))]
            self._add_predictor_ticks(ax=ax, ticks=ticks)
            # Add vertical lines separating each predictor variable
            self._add_seperating_vlines(ax=ax)
            # Adjust tick formatting
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)
            # Save our legend handles and labels, then remove the legend -- we'll add this back in later
            self.legend_handles_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    def _format_fig(
            self
    ) -> None:
        """
        Applies figure-level formatting, including legend, axis labels etc.
        """
        # Add our legend back in
        hand, lab = self.legend_handles_labels
        self.fig.legend(
            hand, lab, loc='right', ncol=1, title='Duo', frameon=False,
            markerscale=2, fontsize=vutils.FONTSIZE + 3, bbox_to_anchor=(1.0, 0.5),
        )
        # Add our axis text and labels in
        self.fig.suptitle('Model predictor variables')
        self.fig.supylabel(r'Coefficient (B)')
        self.fig.supxlabel('Categorical levels')
        # Adjust the subplot positioning slightly
        self.fig.subplots_adjust(top=0.935, bottom=0.05, left=0.1, right=0.9)


class BarPlotModelComparison(vutils.BasePlot):
    """
    Creates a barplot comparing criteria across models with a different number of predictors
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Dataframe formatting attributes
        self.categories: list[str] = kwargs.get('categories', ['tempo_slope', 'ioi_std', 'pw_asym', 'success', 'perceptual_answer_mean', ])
        self.labels: list[str] = kwargs.get('labels', [
            'Tempo slope (BPM/s)', 'Timing irregularity (SD, ms)',  'Asynchrony (RMS, ms)', 'Performer-reported success', 'Listener-reported success',
        ])
        self.averaged_vars: list[str] = kwargs.get('averaged_vars', ['tempo_slope', 'pw_asym', 'perceptual_answer_mean'])
        self.full_mds = [
            'C(latency)+C(jitter)+C(instrument)',
            'C(latency)+C(jitter)',
            'C(instrument)',
            'C(latency)+C(instrument)',
            'C(jitter)+C(instrument)',
            'C(latency)',
            'C(jitter)',
        ]
        self.partial_mds = [
            'C(latency)+C(jitter)',
            'C(latency)',
            'C(jitter)',
        ]
        # Format the dataframe
        self.df = self._format_df()
        # Create subplots
        self.fig, self.ax = plt.subplots(
            nrows=3, ncols=2, sharex=False, sharey='all', figsize=(18.8, 14.1),
            gridspec_kw=dict(width_ratios=[3, 7])
        )
        self.ax.flatten()[-1].axis('off')
        # Visualisation attributes
        self.ci: str = kwargs.get('ci', 'se')  # Type of error bar to plot

    def _format_df(
            self
    ) -> pd.DataFrame:
        """
        Creates models for each performance success metric and extracts required critiera (e.g. R2, AIC)
        """
        # Melt the dataframe to convert our response variables from rows to columns
        test = self.df.melt(
            id_vars=['trial', 'block', 'latency', 'jitter', 'instrument'],
            value_vars=self.categories
        )
        res = []
        # Iterate through each trial
        for idx, grp in test.groupby('trial'):
            # Iterate through each response variable
            for i, g in grp.groupby('variable'):
                g = g.rename(columns={'value': i})
                # Variables which are averaged across the duo, i.e. tempo slope, asynchrony
                if i in self.averaged_vars:
                    avg = g.groupby(['trial', 'latency', 'jitter']).mean().reset_index(drop=False)
                    mds = self.partial_mds
                # Variables which are unique for each performer, i.e. timing irregularity and self-reported success
                else:
                    avg = g.groupby(['trial', 'latency', 'jitter', 'instrument']).mean().reset_index(drop=False)
                    # For these variables, add in instrument as a predictor
                    mds = self.full_mds
                # Iterate through all the required models
                for md in mds:
                    # Create and fit our regression model
                    reg = smf.ols(data=avg, formula=f'{i}~{md}').fit()
                    # Extract the comparison criteria and append to our list
                    res.append({
                        'trial': idx,
                        'var': i,
                        'md': md,
                        'r2_adj': reg.rsquared_adj,
                    })
        # Create the dataframe and set the variable column as categorical -- needed for correct ordering!
        res = pd.DataFrame(res)
        res['var'] = pd.Categorical(res['var'], self.categories)
        return res

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class, creates the plot and saves in plot decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\barplot_model_comparison'
        return self.fig, fname

    def _create_plot(
            self
    ) -> None:
        """
        Creates the barplots for each performance success metric
        """
        palette = sns.color_palette('Set2', n_colors=5)
        palette[1], palette[2] = palette[2], palette[1]
        # Iterate through each axis and performance success variable
        for ax, (idx, grp), col in zip(self.ax.flatten(), self.df.groupby('var'), palette):
            # Melt dataframe into correct format for plotting
            grp = grp.melt(id_vars=['trial', 'md'], value_vars=['r2_adj', ])
            # Create the barplot
            g = sns.barplot(
                data=grp, x='md', y='value', ax=ax, estimator=np.mean,
                errorbar=self.ci, edgecolor=vutils.BLACK, lw=2, saturation=0.8, alpha=0.8,
                color=col, errwidth=2, capsize=0.05, width=0.8, errcolor=vutils.BLACK
            )
            if grp['md'].str.contains('instrument').sum() > 0:
                self._add_bar_labels(grp, g, 'C(latency)+C(jitter)+C(instrument)', self.full_mds)
            else:
                self._add_bar_labels(grp, g, 'C(latency)+C(jitter)', self.partial_mds)

    @staticmethod
    def _add_bar_labels(grp: pd.DataFrame, g: plt.Axes, ref: str, md_list: list,):
        ref_mean = grp[grp['md'] == ref]['value'].mean()
        grp['md_'] = pd.Categorical(grp['md'], md_list)
        for artist, (i_, lab) in zip(g.containers[0], grp.groupby('md_')):
            txt = lab["value"].mean() - ref_mean
            if txt != 0:
                y_pos = artist.get_y() + artist.get_height() + (lab['value'].sem() * 1.6)
                if y_pos < 0.1:
                    y_pos = 0.15
                g.text(
                    artist.get_x() + 0.05, y_pos, f'$\Delta{round(txt, 2)}$',
                    ha='left', va='baseline', fontsize=vutils.FONTSIZE - 1
                )

    def _format_ax(
            self
    ) -> None:
        """
        Formats axis-level attributes
        """
        # Iterate through each axis and label together
        for ax, lab in zip(self.ax.flatten(), self.labels):
            # Set title and axis labels
            ax.set(title=lab, ylabel='', xlabel='', ylim=[-0.2, 1.15], )
            # Set x axis ticks and tick parameters
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)
            # Add horizontal line at y = 0
            ax.axhline(y=0, color=vutils.BLACK, lw=2)
            # Hack alert
            if lab == 'Performer-reported success':
                t = [
                    'Full model\n(~$L$ + $J$ + $I$)',
                    'Testbed only\n(~$L$ + $J$)',
                    'Instrument only\n(~$I$)',
                    'Latency + instrument\n(~$L$ + $I$)',
                    'Jitter + instrument\n(~$J$ + $I$)',
                    'Latency only\n(~$L$)',
                    'Jitter only\n(~$J$)',
                ]
                ax.set_xticks(ax.get_xticks(), labels=t, rotation=45, ha='right', rotation_mode='anchor')
                ax.tick_params(axis='x', which='major', pad=20)
            elif lab == 'Listener-reported success':
                t = [
                    'Full model\n(~$L$ + $J$)',
                    'Latency only\n(~$L$)',
                    'Jitter only\n(~$J$)',
                ]
                ax.set_xticks(ax.get_xticks(), labels=t, rotation=45, ha='right', rotation_mode='anchor')
            else:
                ax.set(xticklabels=[])

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level attributes
        """
        # Add in x and y axis labels
        self.fig.supxlabel('Model predictor variables')
        self.fig.supylabel(r'Adjusted $R^{2}$', y=0.6)
        # Adjust plot positioning slightly
        self.fig.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.95, hspace=0.3, wspace=0.1)


class RegPlotTestRetestReliability(vutils.BasePlot):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df)
        self.fig = plt.figure(figsize=(18.8, 12))
        self.vars = ['tempo_slope', 'ioi_std', 'pw_asym', 'success', 'perceptual_answer_mean']
        self.main_ax, self.top_marginal_ax, self.right_marginal_ax = self._init_gridspec_subplots()
        self.handles, self.labels = None, None
        self.lims = [(-0.6, 0.35), (0, 80), (0, 250), (0, 10), (0, 10)]
        self.ticks = [(-0.5, 0.0), (0, 50), (0, 100, 200), (1, 5, 9), (1, 5, 9)]
        self.titles = [
            'Tempo slope',
            'Timing irregularity',
            'Asynchrony',
            'Performer success',
            'Listener success',
        ]

    def _init_gridspec_subplots(
            self, widths: tuple[int] = (5, 0.5, 5, 0.5, 5, 0.5), heights: tuple[int] = (0.5, 5, 0.5, 5)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialise grid of subplots. Returns two numpy arrays corresponding to main and marginal axes respectively.
        """
        # Initialise subplot grid with desired width and height ratios
        grid = self.fig.add_gridspec(
            nrows=4, ncols=6,
            width_ratios=widths, height_ratios=heights,
            wspace=0.3, hspace=0.5
        )
        # Create a list of index values for each subplot type
        top_margins = [0, 2, 4, 12, 14]
        right_margins = [7, 9, 11, 19, 21]
        mains = [6, 8, 10, 18, 20]
        # Create empty lists to hold subplots
        top_marginal_ax = []
        right_marginal_ax = []
        main_ax = []
        # Iterate through all the subplots we want to create
        for i in range(len(heights) * len(widths)):
            # Create the subplot object
            ax = self.fig.add_subplot(grid[i // len(widths), i % len(widths)])
            # Append the subplot to the desired list
            if i in top_margins:
                top_marginal_ax.append(ax)
            elif i in right_margins:
                right_marginal_ax.append(ax)
            elif i in mains:
                main_ax.append(ax)
            # If we don't want to keep the axis, turn it off so it is hidden from the plot
            else:
                ax.axis('off')
        # Return the figure, main subplots (as numpy array), and marginal subplots (as numpy array)
        return np.array(main_ax), np.array(top_marginal_ax), np.array(right_marginal_ax)

    @staticmethod
    def _format_df(df):
        piv = df.pivot(['trial', 'latency', 'jitter', 'instrument'], 'block',
                       ['tempo_slope', 'ioi_std', 'pw_asym', 'success', 'perceptual_answer_mean']).reset_index(
            drop=False)
        piv.columns = pd.Index([str(e[0]) + str(e[1]) for e in piv.columns.tolist()])
        # piv['tempo_slope1'] = piv['tempo_slope1'].abs()
        # piv['tempo_slope2'] = piv['tempo_slope2'].abs()
        np.random.seed(2)
        piv['success1'] += np.random.normal(0, 0.2, len(piv))
        piv['success2'] += np.random.normal(0, 0.2, len(piv))
        return piv

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the class; creates plot and returns to vutils.plot_decorator for saving.
        """
        self._create_plot()
        self._format_main_ax()
        self._format_marginal_ax()
        self._format_fig()
        self._move_marginal_ax()
        self._move_marginal_ax_y()
        fname = f'{self.output_dir}\\regplot_test_retest_reliability'
        return self.fig, fname

    def _create_plot(
            self
    ):
        """
        Creates scatter plots for both simulated and actual tempo slope values
        """
        # Create the grid, but don't map any plots onto it just yet
        for main_ax, var, top_margin, right_margin in zip(
                self.main_ax.flatten(), self.vars, self.top_marginal_ax, self.right_marginal_ax
        ):
            sns.scatterplot(
                data=self.df, x=var + '1', y=var + '2', hue='trial', palette=vutils.DUO_CMAP,
                style='trial', markers=vutils.DUO_MARKERS, s=100, edgecolor=vutils.BLACK, ax=main_ax
            )
            sns.regplot(
                data=self.df, x=var + '1', y=var + '2', ax=main_ax, scatter=False,
                n_boot=vutils.N_BOOT, line_kws=dict(color=vutils.BLACK, lw=3)
            )
            self._add_correlation_results(ax=main_ax, var=var, position=var == 'success')
            for margin, kwargs, in zip([top_margin, right_margin],
                                       [dict(x=var + '1'), dict(y=var + '2')]):
                sns.kdeplot(
                    data=self.df, hue='trial', palette=vutils.DUO_CMAP, legend=False, lw=2,
                    multiple='stack', fill=True, common_grid=True, cut=0, ax=margin, **kwargs
                )

    def _add_correlation_results(
            self, ax, var, position=False
    ) -> None:
        """
        Adds the results of a linear correlation onto the plot
        """
        # Calculate the correlation, get r and p values
        r, p = stats.pearsonr(self.df[var + '1'], self.df[var + '2'])
        # Format correlation results into a string
        s = f'$r$ = {round(r, 2)}{vutils.get_significance_asterisks(p)}'
        # Add the annotation onto the plot
        pos = (0.21, 0.9) if not position else (0.79, 0.1)
        ax.annotate(
            s, pos, xycoords='axes fraction', fontsize=vutils.FONTSIZE, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=vutils.ALPHA, edgecolor=vutils.BLACK),
        )

    def _move_marginal_ax(self):
        for ax in self.right_marginal_ax.flatten():
            pos1 = ax.get_position()  # get the original position
            pos2 = [pos1.x0 - 0.02, pos1.y0 - 0.001, pos1.width, pos1.height]
            ax.set_position(pos2)

    def _move_marginal_ax_y(self):
        for ax in self.top_marginal_ax.flatten():
            pos1 = ax.get_position()  # get the original position
            pos2 = [pos1.x0, pos1.y0 - 0.06, pos1.width, pos1.height]
            ax.set_position(pos2)

    def _format_marginal_ax(self):
        for top_margin, right_margin, lim, tit, tick in zip(
                self.top_marginal_ax.flatten(), self.right_marginal_ax.flatten(), self.lims, self.titles, self.ticks
        ):
            top_margin.set(xlim=lim, ylabel='', xlabel='', xticklabels=[], yticks=[], xticks=tick)
            top_margin.set_title(tit, fontsize=vutils.FONTSIZE + 5, y=1.1)
            top_margin.spines['left'].set_visible(False)
            right_margin.set(ylim=lim, xlabel='', ylabel='', yticklabels=[], xticks=[], yticks=tick)
            right_margin.spines['bottom'].set_visible(False)
            for ax in [top_margin, right_margin]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(width=3, which='major')
                plt.setp(ax.spines.values(), linewidth=2)

    def _format_main_ax(self):
        for ax, lim, tick in zip(self.main_ax.flatten(), self.lims, self.ticks):
            # Get the axes limit from minimum and maximum values across both simulated and original data
            ax.set(xlim=lim, ylim=lim, xticks=tick, yticks=tick, xlabel='', ylabel='')
            # Set the top and right spines of the joint plot to visible
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            # Add the diagonal line to the joint ax
            ax.axline(
                xy1=(0, 0), xy2=(1, 1), linewidth=3, transform=ax.transAxes,
                color=vutils.BLACK, alpha=vutils.ALPHA, ls='--'
            )
            ax.tick_params(width=3, which='major')
            plt.setp(ax.spines.values(), linewidth=2)
            # Store our handles and labels
            self.handles, self.labels = ax.get_legend_handles_labels()
            # Remove the legend
            ax.get_legend().remove()

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure object, setting legend and padding etc.
        """
        # Add the legend back in
        self.labels = [int(float(lab)) for lab in self.labels]
        lgnd = plt.legend(
            self.handles, self.labels, title='Duo', frameon=False, ncol=1, loc='right', markerscale=1.5,
            fontsize=vutils.FONTSIZE + 3, edgecolor=vutils.BLACK, bbox_to_anchor=(-5, 0.5),
        )
        self.fig.suptitle('Objective evaluations', fontsize=vutils.FONTSIZE + 7)
        self.fig.text(0.4, 0.475, 'Subjective evaluations', fontsize=vutils.FONTSIZE + 7)
        self.fig.supxlabel('Experiment session 1')
        self.fig.supylabel('Experiment session 2')
        # Set the legend font size
        plt.setp(lgnd.get_title(), fontsize=vutils.FONTSIZE + 3)
        # Set the legend marker size and edge color
        for handle in lgnd.legendHandles:
            handle.set_edgecolor(vutils.BLACK)
            handle.set_sizes([100])
        # Adjust subplots positioning a bit to fit in the legend we've just created
        self.fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.95)


class PointPlotDuoStats(vutils.BasePlot):
    """
    Creates a pointplot showing bootstrapped mean for each level of latency and jitter predictor variables
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.boot_df = pd.DataFrame(self._format_df())
        self.fig, self.ax = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=True, figsize=(18.8, 5))

    def _format_df(self):
        df_ = self.df.groupby(['trial', 'latency', 'jitter', 'instrument'], as_index=False).mean(numeric_only=True)
        # df_ = df_[df_['latency'] != 0]
        # Iterate over each independent variable
        for iv in ['latency', 'jitter']:
            # Iterate over each level of the iv
            for lvl, grp in df_.groupby(iv, sort=False):
                # Iterate over each dependent variable
                for dv in ['tempo_slope', 'ioi_std', 'pw_asym', 'success', 'perceptual_answer_mean']:
                    print(iv, dv, lvl)
                    # Get the actual mean
                    true_mean = grp[dv].mean()
                    # Get the bootstrap data
                    boots = []
                    for i in range(vutils.N_BOOT):
                        random.seed(i)
                        trials = random.choices(grp['trial'].unique(), k=5)
                        values = pd.concat([grp[grp['trial'] == t] for t in trials], axis=0)[dv]
                        boots.append(values.mean())
                    # Return the results
                    yield {
                        'level': str(lvl),
                        'iv': iv,
                        'dv': dv,
                        'mean': true_mean,
                        'low': np.quantile(boots, 0.025),
                        'high': np.quantile(boots, 0.975)
                    }

    @vutils.plot_decorator
    def create_plot(
            self
    ) -> tuple[plt.Figure, str]:
        """
        Called from outside the plot to generate the figure, format, and save in decorator
        """
        self._create_plot()
        self._format_ax()
        self._format_fig()
        # Save the plot
        fname = f'{self.output_dir}\\pointplot_duo_stats'
        return self.fig, fname

    def _create_plot(self):
        for a, (i, v) in zip(self.ax.flatten(), self.boot_df.groupby('dv', sort=False)):
            for (i_, v_) in v.groupby('iv', sort=False):
                a.scatter(
                    v_['mean'], v_['level'], zorder=5, s=100, lw=1, edgecolor=vutils.BLACK,
                    color=vutils.DUO_CMAP[0] if i_ == 'latency' else vutils.DUO_CMAP[1],
                    marker=vutils.DUO_MARKERS[0] if i_ == 'latency' else vutils.DUO_MARKERS[1],
                )

            l = v['mean'] - v['low']
            h = v['high'] - v['mean']
            a.errorbar(
                x=v['mean'], y=v['level'], xerr=[l, h], ls='none', lw=2,
                color=vutils.BLACK, zorder=1, capsize=5, markeredgewidth=2
            )
            a.axhline(
                y=4.5, xmin=0, xmax=1, ls='dashed', lw=2,
                color=vutils.BLACK, alpha=vutils.ALPHA
            )

    def _format_ax(self):
        titles = [
            'Tempo slope (BPM/s)',
            'Timing irregularity (SD, ms)',
            'Asynchrony (RMS, ms)',
            'Performer-reported Success',
            'Listener-reported Success'
        ]
        self.ax[0].set(xlim=(-0.25, 0.25))
        self.ax[0].text(x=-0.35, y=6, s='Jitter', rotation=90, ha='center', va='center')
        self.ax[0].text(x=-0.35, y=2, s='Latency (ms)', rotation=90, ha='center', va='center')
        for tit, a in zip(titles, self.ax.flatten()):
            a.set(xlabel=tit)
            plt.setp(a.spines.values(), linewidth=2)
            a.tick_params(axis='both', width=3)

    def _format_fig(self):
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15, wspace=0.12)


def generate_all_metrics_plots(
    mds: list[PhaseCorrectionModel], output_dir: str,
) -> None:
    """
    Generates all plots in this file, with required arguments and inputs
    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\all_metrics_plots'
    ppds = PointPlotDuoStats(df=df, output_dir=figures_output_dir)
    ppds.create_plot()
    pp_ = PointPlotRepeatComparisons(df=df, output_dir=figures_output_dir)
    pp_.create_plot()
    rp = RegPlotTestRetestReliability(df=df, output_dir=figures_output_dir,)
    rp.create_plot()
    pp = PairPlotAllVariables(df=df, output_dir=figures_output_dir, error_bar='ci')
    pp.create_plot()
    bp = BarPlotRegressionCoefficients(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    bp = BarPlotModelComparison(df=df, output_dir=figures_output_dir)
    bp.create_plot()
    rt = RegressionTableAllMetrics(df, output_dir=figures_output_dir)
    rt.create_tables()


if __name__ == '__main__':
    import logging
    import os

    # Configure logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # Default location for phase correction models
    logger.info(f"Making graphs from data in {os.path.abspath(r'../../models')}")
    raw = autils.load_from_disc(
        r'..\..\models', filename='phase_correction_mds.p'
    )
    # Default location to save plots
    output = r"..\..\reports"
    # Generate phase correction plots from models
    generate_all_metrics_plots(mds=raw, output_dir=output)
