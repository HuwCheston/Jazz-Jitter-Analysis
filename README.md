# Code from: Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances
![output](https://user-images.githubusercontent.com/97224401/231712093-133cafa0-dffe-4a23-945d-5249c4385bab.gif)

This repository is associated with the paper “Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances”, published in JOURNAL, and includes scripts for reproducing the analysis, models, computer simulations, audio-visual stimuli, and figures contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual jazz duo performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is [freely accessible on Zenodo](https://doi.org/10.5281/zenodo.7773824).

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
- FFmpeg (**only for reproducing combined audio-visual stimuli**)

## Future

As this repository relates to a research paper, I don't envisage making many future changes. However, if you encounter any bugs or have suggestions, please open an issue and I'll take a look.

## Repository structure

    ```
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── docs               <- The Sphinx .html docs, accessible at https://huwcheston.github.io/Jazz-Jitter-Analysis/
    ├── models             <- Serialized models and model predictions (simulations)
    ├── notebooks          <- Jupyter notebooks used in data exploration.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            
    │   └── figures        <- Generated graphics and figures included in the original paper
    ├── src                <- Source code for the project.
    │   ├── clean          <- Scripts to generate final dataset from raw data
    │   ├── analysis       <- Scripts to turn final dataset into models and simulations
    │   ├── visualise      <- Scripts to create exploratory and results oriented visualisations
    │   └── muxer.py       <- Executable python script for reproducing audio-visual stimuli
    ├── LICENSE
    ├── README.md
    ├── run.cmd            <- Batch file to run all required scripts, generate models & graphs
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── setup.py           <- Makes src installable            
    └── test_environment   <- Used in process of chec
    ```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
