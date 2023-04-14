# Troubleshooting

## Reduce processing time

On slower and less powerful systems, it may take a while for `run.cmd` to complete when reproducing the original dataset, models, simulations, and figures. This page includes some possible solutions for speeding this process up.

### Adjusting the default number of simulations

The number of simulations used per paradigm and condition in the original paper was 500. This results in 52,000 simulations being generated whenever `run.cmd` is executed, which can take quite a long time to complete.

On less powerful systems, it may be desirable to reduce the number of simulations that are created, e.g. to 250, 100. This can be accomplished by changing the constant `NUM_SIMULATIONS` defined at the top of the file `analysis_utils.py` contained in `src\analyse`.

Note that each simulation is, inherently, a stochastic process. This means that running a dramatically reduced number of simulations may lead to very different results to those contained in our original paper. Likewise, it should also be expected that there will be some slight variation in simulation results, regardless of how many simulations are created.

### Reducing the number of samples when bootstrapping confidence intervals

The default number of bootstrap samples used when creating error bars across all figures contained in the paper is 10,000. To increase performance on slower systems, you can lower this value by changing the constant `N_BOOT` defined at the top of the file `visualise_utils.py` contained in `src\visualise`.
