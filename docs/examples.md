# Examples
## Loading serialised objects
If you want to work with the serialised dataset, models, or simulations in a Python session, you'll first need to temporarily add the directory to where these classes are defined to your PATH:

```
import sys
import os
sys.path.append("PATH_TO_\src\analyse")
```
   
Now, to unserialise the models and simulations:

```
import dill
mds = dill.load(open("PATH_TO\models\phase_correction_mds.p", "rb"))
sims = dill.load(open("PATH_TO\models\phase_correction_sims.p", "rb"))
```

## Working with `PhaseCorrectionModel` instances

The `mds` variable we defined above is an interable containing an instance of the `PhaseCorrectionModel` class for each performance in the corpus. We can access the model `md` for each individual performance, therefore, with code like:
    
```
for md in mds:
    do something
```

### `PhaseCorrectionModel` attributes 

There are a number of helpful attributes within this class that we can access to gain more information about each performance. For example, to print the results of the nearest neighbour matching process, you can access the `md.keys_nn` or `md.drms_nn` attribute. If you want to access the underlying [regression object in StatsModels](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html), you can use the `md.keys_md` and `md.drms_md` attributes, for example `md.keys_md.summary()` or `md.drms_md.params`. 

### Printing a summary dataframe

We can print a dataframe containing summary statistics for all the performances in the corpus (e.g. the individual coupling coefficients) by accessing the `md.keys_dic` and `md.drms_dic` attributes in a loop, for example:
    
```
import pandas as pd
res = []
for md in mds:
    res.append(md.keys_dic) # Results dictionary for pianist
    res.append(md.drms_dic) # Results dictionary for drummer
df = pd.DataFrame(res)
df
```

### Accessing ensemble-level characteristics

The `df` contained above will contain one row for each performer, meaning that the data for a single performance in the corpus occupies two rows (given two performers in each ensemble).

Certain characteristics referred to in the paper are instead measured across an ensemble; for instance, coupling asymmetry is calculated as the absolute difference between coupling coefficients obtained from both musicians. To obtain these characteristics, we will need to group `df`, e.g.:

```
res = []
grp_vars = ['trial', 'block', 'latency', 'jitter']
for idx, grp in df.groupby(grp_vars):
    dr = grp[grp['instrument'] != 'Keys']['correction_partner'].iloc[0]
    ke = grp[grp['instrument'] == 'Keys']['correction_partner'].iloc[0]
    di = {'coupling_strength': dr + ke, 'coupling_asymmetry': abs(dr - ke)}
    di.update({k: v for k, v in zip(grp_vars, idx)})
    res.append(di)
df_ensemble = pd.DataFrame(res)
```

The resulting `df_ensemble` will have 6 columns, containing the duo and session number, the latency and jitter values of the condition, and the coupling asymmetry and strength values obtained across both members of the duo. Note that this grouping is carried out automatically when creating figures and visualisations.

For more information on the `PhaseCorrectionModel` class, consult the documentation and code.

## Working with `Simulation` instances

The `sims` list, likewise, is an iterable of `Simulation` class instances, where each instance corresponds to one condition and simulation paradigm. Taking the 45 ms latency, 0.5x jitter experimental condition as an example, we'd have one Simulation instance for all the individual simulations made under using the `democracy` paradigm and another instance for the simulations made using the `anarchy` paradigm. 

Just as before, we can access each individual Simulation `sim` instance by iterating like:

```
for sim in sims:
    do something
```

### `Simulation` attributes

Inside each `sim` instance, we can access every individual keyboard simulation with `sim.keys_simulations` and every drum simulation with `sim.drms_simulations`. These attributes are lists of dataframes, with each dataframe corresponding to one simulation. So, to match the data for one ensemble simulation together:

```
matched = [(drms_sim, keys_sim) for drms_sim, keys_sim in zip(sim.keys_simulations, sim.drms_simulations)]
```

### `Simulation` helper functions

There are three helper functions provided to obtain averages across all the simulations contained within a single `sim` instance, which can be used when comparing across paradigms. These are:
- `sim.get_average_tempo_slope()`
- `sim.get_average_ioi_variability()`
- `sim.get_average_pairwise_asynchrony()`

Each function takes in a keyword argument `func`, which can be used to define the method used to obtain the average. Any function should be compatible here, as long as it returns a single value from a Pandas series of dtype `float`. Additional keyword arguments passed into these helper functions will be passed as `**kwargs` to `func`.

### Printing a summary dataframe

As with the `PhaseCorrectionModel` class, you can print a dataframe containing summary statistics for all the simulations by accessing `sim.results_dic` in a loop, for example:

```
res = pd.DataFrame([sim.results_dic for sim in sims])
```

## Working with the graphs

The code for generating the graphs, contained within the `\src\visualise` directory, is generally provided *as-is*, and not intended to be accessed or imported directly outside this project or used for different datasets. However, some small changes can still be made. 

Aesthetic changes to the plots can be made by altering the constant variables defined at the start of `visualise_utils.py` contained in `src\visualise\`. You can also make aesthetic & statistic changes to the plots by adjusting various keyword arguments used in the individual plotting functions called in `src\visualise\run_visualisations.py`.