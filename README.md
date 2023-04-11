# Code from: Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances

This repository is associated with the paper “Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances”, published in JOURNAL, and includes scripts for reproducing the analysis, computer simulations, and graphs contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is hosted on Zenodo at [10.5281/zenodo.7773824](https://doi.org/10.5281/zenodo.7773824)

## Requirements
- Python 3.10+
- Git
- [Reaper](https://www.reaper.fm/) (*only if you wish to open the DAW sessions used to clean the MIDI performance files*)
- FFmpeg, accessible via PATH (*only if you wish to generate muxed audio-video recordings from the raw data*)

## Install

The following instructions are based on the commands required to recreate our original models and graphs on a local machine running Windows 10, with Python and Git already installed and configured. While they should work similarly on other operating systems, full compatibility is not guaranteed.

1. First, clone this repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Jazz-Jitter-Analysis
```

2. Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.7773824). You should have the following files:
    - five RAR archives in the form duo_X.rar;
    - an Excel spreadsheet, questionnaire_anonymised.xlsx;
    - two .csv files containing mappings for different MIDI instruments;
    - a single .csv file containing the latency array used in the experiments
    - *You do not need to download the file perceptual_study_videos.rar unless you wish to replicate the perceptual component of the paper, the code for which is hosted in a separate repository acccessible [via this link](https://github.com/HuwCheston/2023-duo-success-analysis).*
    
3. Extract the files you just downloaded into the following place inside \data\raw
    - The .csv and .xlsx files should be extracted directly into \data\raw
    - the contents of the 'raw' folder inside each .RAR should be extracted directly into the folder for the corresponding duo inside \data\raw\avmanip_output: for example, duo_1.rar\raw should be extracted into avmanip_output\trial_1. 
    - likewise, the contents of the 'cleaned_midi_files' folder in each .RAR should be extracted into the corresponding directory inside \data\raw\midi_bpm_cleaning: so, duo_1.rar\cleaned_midi_files should be extracted into \data\raw\midi_bpm_cleaning\trial_1, creating two folders inside this directory (block_1 and block_2)
    - the video files inside the muxed_performances folder in each .RAR can either be ignored or extracted to the \data\raw\muxed_performances folder: these are not used in any analysis and are provided for reference only.
        
    Your \data\raw folder should then contain:

    ```
    ├── avmanip_output
    │   ├── trial_1  
    │   │   ├── Block 1
    │   │   │   └── Folders inside correspond to each performance for this duo and block
    │   │   ├── Block 2
    │   │   └── Warm-Up
    │   ├── trial_2 
    │   │   ├── Block 1
    │   │   ├── Block 2
    │   │   ├── Warm-Up
    │   │   └── Polar Data
    │   │   │   └── Note that this folder will not be present for every duo.
    │   ├── trial_3
    │   │   └── ...
    │   ├── trial_4
    │   │   └── ...
    │   └── trial_5
    │   │   └── ...
    ├── midi_bpm_cleaning
    |   ├── trial_1  
    │   │   ├── block_1
    │   │   │   └── Individual MIDI files inside correspond to each performance
    │   │   └── block_2
    |   ├── trial_2
    │   │   └── ...
    |   ├── trial_3
    │   │   └── ...
    |   ├── trial_4
    │   │   └── ...
    |   └── trial_5
    │   │   └── ...
    ├── muxed_performances
    |   └── Optional directory to extract .mp4 files to, not used in analysis
    ├── drums_midi_mapping.csv
    ├── keys_midi_mapping.csv
    ├── latency_array.csv
    └── questionnaire_anonymised.xlsx    
    ```

4. Open a new command prompt in the root directory of the repository and execute:
    ```
    run.cmd
    ```
    This script will do the following:
    - Install pip, setuptools, wheel, and virtualenv into your default Python environment
    - Create a new virtual environment in the repository folder, activate it, and test that it meets all requirements
    - Install the required packages listed in requirements.txt
    - Create the final dataset from the raw data (MIDI files, video data, and questionnaire responses)
    - Create the phase correction models and simulations (takes a while) for each performance and save into \models
    - Create the graphs (takes even longer)

Once the command has finished, you can access the final dataset in \data\processed, the models and simulations in \models, and the graphs in \figures\reports. The dataset, models, and simulations are saved as [Python pickle files](https://docs.python.org/3/library/pickle.html) and can be unserialized using any appropriate module. The graphs are saved as .png and .svg files which can be opened in many applications.

## Project Organization

    ```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── models             <- Serialized models and model predictions (simulations)
    ├── notebooks          <- Jupyter notebooks used in data exploration. Naming convention is a number (for ordering), the creator's surname, and a short `-` delimited description, e.g `1.0-cleese-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── clean          <- Scripts to generate final dataset from raw data
    │   │   └── make_dataset.py
    │   ├── analysis       <- Scripts to turn final dataset into models and simulations
    │   │   └── run_analysis
    │   └── visualise      <- Scripts to create exploratory and results oriented visualisations
    │       └── visualize.py
    └── run.cmd            <- Batch file to run all required scripts
    ```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
