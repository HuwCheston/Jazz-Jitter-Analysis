# Code for: Trade-offs in Coordination Strategies for Duet Jazz Performances Subject to Network Delay and Jitter

![A video of performers in the experiment for this project](https://user-images.githubusercontent.com/97224401/231712093-133cafa0-dffe-4a23-945d-5249c4385bab.gif "Performer GIF")

**Paper**: [![DOI (Preprint)](http://img.shields.io/badge/DOI-10.1525/mp.2024.42.1.48-blue)](https://doi.org/10.1525/mp.2024.42.1.48)

**Dataset**: [![DOI (Dataset)](https://zenodo.org/badge/DOI/10.5281/zenodo.7773824.svg)](https://doi.org/10.5281/zenodo.7773824)

This repository is associated with the paper `Trade-offs in Coordination Strategies for Duet Jazz Performances Subject to Network Delay and Jitter`, published in *Music Perception* and includes scripts for reproducing the analysis, models, computer simulations, audio-visual stimuli, and figures contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual jazz duo performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is [freely accessible on Zenodo](https://doi.org/10.5281/zenodo.7773824).

## Quick start

To build the dataset, models, simulations, figures, and audio-visual stimuli used in the original paper, follow the instructions in [Getting started](https://huwcheston.github.io/Jazz-Jitter-Analysis/getting-started.html).

After building, for examples on how to use the dataset in your own Python sessions, take a look at [Examples](https://huwcheston.github.io/Jazz-Jitter-Analysis/examples.html).

For help, including advice on building the dataset on slower systems, see [Troubleshooting](https://huwcheston.github.io/Jazz-Jitter-Analysis/troubleshooting.html).

For the source code documentation (not comprehensive or pretty), go to [Documentation](https://huwcheston.github.io/Jazz-Jitter-Analysis/_autosummary/src.html#).

Finally, to explore the models and data created here without having to install or run anything locally, check out [Notebooks](https://huwcheston.github.io/Jazz-Jitter-Analysis/notebooks.html).

## Requirements

- Windows 10 (other operating systems may work but are not tested)
- Python 3.10+
- Git
- FFmpeg (**only required for reproducing stimuli used in perceptual study**)

## Future

As this repository relates to a research paper, I don't envisage making many future changes. However, if you encounter any bugs or have suggestions, please open an issue and I'll take a look.

## Repository structure

    ```
    ├── data
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump. Extract the `data.zip` file from Zenodo here!
    ├── docs                <- The Sphinx .html docs, accessible at https://huwcheston.github.io/Jazz-Jitter-Analysis/
    ├── models              <- Serialized models and simulations.
    ├── notebooks           <- Jupyter notebooks used in data exploration, also accesible on CoLab.
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            
    │   └── figures         <- Generated graphics and figures included in the original paper
    ├── src                 <- Source code for the project.
    │   ├── clean           <- Scripts to generate final dataset from raw data
    │   ├── analysis        <- Scripts to turn final dataset into models and simulations
    │   ├── visualise       <- Scripts to create exploratory and results oriented visualisations
    │   └── muxer.py        <- Executable python script for reproducing audio-visual stimuli
    ├── LICENSE
    ├── README.md           <- The file you're reading currently!
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment
    ├── run.cmd             <- Batch file to run all required scripts, generate models & graphs
    ├── setup.py            <- Makes src installable            
    └── test_environment.py <- Used in process of checking venv created by run.cmd
    ```

## Citation

If you reference any aspect of this work, please cite the paper in the format below:

```
@article{10.1525/mp.2024.42.1.48,
    author = {Cheston, Huw and Cross, Ian and Harrison, Peter M. C.},
    title = "{Trade-offs in Coordination Strategies for Duet Jazz Performances Subject to Network Delay and Jitter}",
    journal = {Music Perception},
    volume = {42},
    number = {1},
    pages = {48-72},
    year = {2024},
    month = {09},
    issn = {0730-7829},
    doi = {10.1525/mp.2024.42.1.48},
    url = {https://doi.org/10.1525/mp.2024.42.1.48},
    eprint = {https://online.ucpress.edu/mp/article-pdf/42/1/48/833292/mp.2024.42.1.48.pdf},
}
```
