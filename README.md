# Code from: Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances
![output](https://user-images.githubusercontent.com/97224401/231712093-133cafa0-dffe-4a23-945d-5249c4385bab.gif)

This repository is associated with the paper “Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances”, published in JOURNAL, and includes scripts for reproducing the analysis, models, computer simulations, audio-visual stimuli, and figures contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual jazz duo performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is [freely accessible on Zenodo](https://doi.org/10.5281/zenodo.7773824).

## Requirements
- Windows 10 (other operating systems may work but are not tested)
- Python 3
- Git
- FFmpeg (**only for reproducing combined audio-visual stimuli**)

## Quick start
### Reproduce models, simulations, and figures from original paper
1. First, clone this repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Jazz-Jitter-Analysis
```

2. Download our data from [Zenodo](https://doi.org/10.5281/zenodo.7773824). You'll need the **data.zip** file and the 6 corresponding volumes, **data.z01 – data.z06**
    
    *You do not need to download the file perceptual_study_videos.rar unless you wish to replicate the perceptual component of the paper, the code for which is hosted in a separate repository acccessible [via this link](https://github.com/HuwCheston/2023-duo-success-analysis).*
    
3. Open the data.zip file (you may need to install a tool for opening multi-part zip files, such as [WinRAR](https://www.win-rar.com/)) and extract the contents into \data\raw. This folder should then look like:
    ```
    ├── avmanip_output     <- The raw MIDI, audio, and video output from each performance
    │   ├── trial_1  
    │   │   ├── Block 1    <- Folders inside correspond to each performance for this session.
    │   │   ├── Block 2
    │   │   └── Warm-Up
    │   ├── trial_2 
    │   │   ├── Block 1
    │   │   ├── Block 2
    │   │   ├── Warm-Up
    │   │   └── Polar Data <- Note that this folder will not be present for every duo.
    │   ├── trial_3
    │   ├── trial_4
    │   └── trial_5
    ├── midi_bpm_cleaning  <- The cleaned MIDI files (quarter note onset positions)
    │   ├── trial_1  
    │   │   ├── block_1    <- Individual MIDI files inside correspond to each performance.    
    │   │   └── block_2
    │   ├── trial_2
    │   ├── trial_3
    │   ├── trial_4
    │   └── trial_5
    ├── muxed_performances <- Audio-video .mp4 files, not used in analysis
    ├── drums_midi_mapping.csv
    ├── keys_midi_mapping.csv
    ├── latency_array.csv
    └── questionnaire_anonymised.xlsx    
    ```

4. Open a new command prompt in the root directory of the repository and execute:
    ```
    run.cmd
    ```

    This script will:
    - Create a new virtual environment in the repository root folder and install the required dependencies into it. 
    - Generate the final dataset from the raw data dump downloaded above
    - Generate the models and simulations from the dataset
    - Generate the figures used in the original paper and supplementary material. 
    
    Reproducing the models is fairly quick and optimised (on a system with a 2.6GHz CPU and 16GB of RAM, it takes about three minutes), but reproducing the simulations will typically take a lot longer; probably *at least twenty minutes* with the default number of simulations per paradigm/condition (500). The script will log each model or simulation it creates with a timestamp so you can be sure that the process hasn't stalled.
    
5. Once the command has finished, you can access the final dataset in \data\processed, the models and simulations in \models, and the figures in \figures\reports. The dataset, models, and simulations are saved as [Python pickle files](https://docs.python.org/3/library/pickle.html) and should be unserialized using the [Dill](https://dill.readthedocs.io/en/latest/) module. Note that trying to unserialize using the Pickle module itself (i.e. with the pickle.load() function) is not supported and will more than likely produce errors. The figures are saved as .png and .svg files which can be opened in many different applications.

6. The script will not attempt to rebuild the processed dataset, models, or simulations if it detects that these have already been created. This can help e.g. if you wish to adjust simulation parameters without rebuilding the entire set of models. To force a rebuild you'll need to delete or rename the processed files in the above folders and run the script again.

### Reproduce combined audio-visual stimuli from perceptual component of paper
The raw dataset contains separate audio and video files (both with and without latency/jitter) for each performer inside the \data\raw\avmanip_output folder. These can be combined to create muxed audio-video recordings, i.e. a single video file containing both audio and video tracks, sync'ed together, with any combination of live or delayed audio/video. The perceptual component of this study used the delayed audio and video from both the pianist and drummer.

1. Follow steps 1–3 in the section [**Reproduce models, simulations, and figures from original paper**](https://github.com/HuwCheston/Jazz-Jitter-Analysis/edit/main/README.md#reproduce-models-simulations-and-figures-from-original-paper). You don't need to build the models or simulations, just get the data into the correct place within the overall filestructure.

2. Download and install [FFmpeg](https://ffmpeg.org/) and ensure that it can be accessed on your PATH. To check this, open a command prompt and type in
    ```
    ffmpeg
    ```
    If you don't see any errors, you're good to go!

3. Open a new command prompt in the root directory of the repository and execute:
    ```
    python mixer.py
    ```
    
    This script will take all the delayed audio and video files from both musicians (located in \data\raw\avmanip_output) and mux them together into a new folder (default \data\raw\muxed_performances\kdelay_ddelay). 
    
4. By default, the script will recreate the stimuli used in the perceptual component of our paper, i.e. with delay applied to the audio and video footage from both performers. You can customize the output by setting the `--keys` and `--drums` flags to either "Live" or "Delay". So, to recreate the experience that the keyboard player had in the experiment (i.e. live keyboard audio and video, delayed drummer audio and video), you'd use:
   
    ```
    python mixer.py --drums "Delay" --keys "Live"
    ```
    
5. You can also change a few other options with how the videos look by passing in further FFmpeg commands and flags. Not all of these are currently implemented: you can see those that are with:
    
    ```
    python mixer.py --help
    ```    

## Examples
To unserialise the models and print a dataframe containing summary stats for the whole corpus (e.g. coupling coefficients):
    ```

    To read in the simulations and print summary results:

    ```
    # These lines are necessary when unpickling custom classes
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'src\\analyse'))
    
    # Load in the simulations
    import pickle
    sims = pickle.load(open("\models\phase_correction_sims.p", "rb"))
    
    # Get all the results dictionaries and convert to a dataframe
    import pandas as pd
    df_avg = pd.DataFrame([sim.results_dic for sim in sims])
    ```


## Future

As this repository relates to a research paper, I don't envisage making many future changes. However, if you encounter any bugs or have suggestions, please open an issue and I'll take a look.

## Repository structure

    ```
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Serialized models and model predictions (simulations)
    ├── notebooks          <- Jupyter notebooks used in data exploration.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            
    │   └── figures        <- Generated graphics and figures included in the original paper
    ├── src                <- Source code for the project.
    │   ├── clean          <- Scripts to generate final dataset from raw data
    │   ├── analysis       <- Scripts to turn final dataset into models and simulations
    │   └── visualise      <- Scripts to create exploratory and results oriented visualisations
    ├── LICENSE
    ├── README.md
    ├── muxer.py           <- Executable python script for reproducing audio-visual stimuli
    ├── run.cmd            <- Batch file to run all required scripts, generate models & graphs
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── setup.py           <- Makes src installable            
    └── test_environment   <- Used in process of chec
    ```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
