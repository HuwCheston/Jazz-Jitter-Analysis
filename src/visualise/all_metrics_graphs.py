import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from stargazer.stargazer import Stargazer
import statsmodels.formula.api as smf

from src.analyse.phase_correction_models import PhaseCorrectionModel
import src.analyse.analysis_utils as autils
import src.visualise.visualise_utils as vutils

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
            'labels', ['Tempo slope\n(BPM/s)', 'Timing\nirregularity\n(SD, ms)',
                       'Asynchrony\n(RMS, ms)', 'Self-reported\nsuccess']
        )
        self._jitter_success: bool = kwargs.get('jitter_success', True)
        self._abs_slope: bool = kwargs.get('abs_slope', True)
        self.error_bar: str = kwargs.get('error_bar', 'sd')
        self.n_boot: int = kwargs.get('n_boot', vutils.N_BOOT)
        self.percentiles: tuple[float] = kwargs.get('percentiles', (2.5, 97.5))
        self.vars: list[str] = kwargs.get('vars', ['tempo_slope', 'ioi_std', 'pw_asym', 'success'])
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
        if self._abs_slope:
            # Get absolute tempo slope
            df['tempo_slope'] = df['tempo_slope'].abs()
        pd.options.mode.chained_assignment = 'warn'
        # Group by trial, latency, and jitter, get the mean, then reset the index and return
        return df.groupby(['trial', 'latency', 'jitter']).mean().reset_index()

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
        y = 0.85 if self.counter != 1 else 0.8
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
        # Create empty variable to hold concatenated dataframe so we don't get warnings
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
        for i, lim, step in zip(range(0, 4), [slope_lim, (0, 100), (0, 400), (1, 9)], [3, 5, 5, 3]):
            ticks = np.linspace(lim[0], lim[1], step)
            self.g.axes[i, i].set(ylim=lim, xlim=lim, yticks=ticks, xticks=ticks)
        # If using absolute tempo slope, adjust axis limit slightly so we don't cut off some markers
        if self._abs_slope:
            self.g.axes[0, 0].set(xlim=(-0.01, 0.5))
        # Iterate through just the first column and last row and add the correct axis ticks back in
        for i in range(0, 4):
            self.g.axes[i, 0].tick_params(width=3, length=7, axis='y')
            self.g.axes[3, i].tick_params(width=3, length=7, axis='x')
        # The axis in the bottom left corner is the only one which requires ticks on both axes
        self.g.axes[3, 0].tick_params(width=3, length=7, axis='both')

    def _format_fig(
            self
    ) -> None:
        """
        Formats figure-level objects
        """
        # Adjust the spacing between subplots slightly
        self.g.figure.subplots_adjust(wspace=0.15, hspace=0.15, top=0.97, bottom=0.05, left=0.05, right=0.97)


class RegressionTableAllMetrics:

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


def generate_all_metrics_plots(
    mds: list[PhaseCorrectionModel], output_dir: str,
) -> None:
    """

    """
    df = []
    for pcm in mds:
        df.append(pcm.keys_dic)
        df.append(pcm.drms_dic)
    df = pd.DataFrame(df)
    figures_output_dir = output_dir + '\\figures\\all_metrics_plots'

    pp = PairPlotAllVariables(df=df, output_dir=figures_output_dir, error_bar='ci')
    pp.create_plot()
    rt = RegressionTableAllMetrics(df, output_dir=figures_output_dir)
    rt.create_tables()


if __name__ == '__main__':
    # Default location for phase correction models
    raw = autils.load_from_disc(r"C:\Python Projects\jazz-jitter-analysis\models", filename='phase_correction_mds.p')
    # Default location to save plots
    output = r"C:\Python Projects\jazz-jitter-analysis\reports"
    # Generate phase correction plots from models
    generate_all_metrics_plots(mds=raw, output_dir=output)
