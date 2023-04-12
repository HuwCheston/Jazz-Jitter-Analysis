# Code from: Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances

This repository is associated with the paper “Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances”, published in JOURNAL, and includes scripts for reproducing the analysis, computer simulations, and graphs contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is hosted on Zenodo at [10.5281/zenodo.7773824](https://doi.org/10.5281/zenodo.7773824)

## Requirements
- Windows (other operating systems may work but are not tested)
- Python 3.10+
- Git

## Quick start
### Reproduce models, simulations, and figures from original paper
1. First, clone this repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Jazz-Jitter-Analysis
```

2. Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.7773824). This is the **data.zip** file and the 6 corresponding volumes, **data.z01 – data.z06**
    
    *You do not need to download the file perceptual_study_videos.rar unless you wish to replicate the perceptual component of the paper, the code for which is hosted in a separate repository acccessible [via this link](https://github.com/HuwCheston/2023-duo-success-analysis).* The instructions in the section 
    
3. Open the data.zip file (you may need to install a tool for opening multi-part zip files, such as [WinRAR](https://www.win-rar.com/)) and extract the contents into \data\raw. This folder should then look like:
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
    |   └── Audio-video .mp4 files, not used in analysis
    ├── drums_midi_mapping.csv
    ├── keys_midi_mapping.csv
    ├── latency_array.csv
    └── questionnaire_anonymised.xlsx    
    ```

4. Open a new command prompt in the root directory of the repository and execute:
    ```
    run.cmd
    ```

    This script will create a new virtual environment, install the required packages into it, create the final dataset from the raw data dump, create the models and simulations from the dataset, and finally create the graphs used in the original paper. Once the command has finished, you can access the final dataset in \data\processed, the models and simulations in \models, and the graphs in \figures\reports. The dataset, models, and simulations are saved as [Python pickle files](https://docs.python.org/3/library/pickle.html) and can be unserialized using any appropriate module e.g. [Dill](https://dill.readthedocs.io/en/latest/). The graphs are saved as .png and .svg files which can be opened in many applications. 

    Generating the models is fairly quick (takes approximately 3 minutes with 2.6GHz CPU, 16GB RAM), however generating the simulations will typically take a lot longer (20 minutes+). To speed things up, the script will not attempt to rebuild the processed dataset, models, or simulations if it detects these have already been created. This can help e.g. if you wish to adjust simulation parameters without rebuilding the entire set of models. To force a rebuild you'll need to delete the processed files in the above folders.

### Reproduce audio-visual stimuli from perceptual component
The raw dataset contains separate audio and video files (both with and without latency/jitter) for each performer inside the \data\raw\avmanip_output folder. These can be combined to create muxed audio-video recordings, i.e. a single video file containing both audio and video tracks, sync'ed together, with any combination of live or delayed audio/video. The perceptual component of this study used the delayed audio and video from both the pianist and drummer.

1. Follow steps 1–3 in the section **Reproduce models, simulations, and figures from original paper**. You don't need to build the models or simulations, just get the data into the correct place within the overall filestructure.

2. Download and install [FFmpeg](https://ffmpeg.org/) and ensure that it can be accessed on your PATH. To check this, open a command prompt and type in
    ```
    ffmpeg
    ```
    If you don't see any errors, you're good to go!

3. Open a new command prompt in the root directory of the repository and execute:
    ```
    python mixer.py
    ```
    
    This script will take all the delayed audio and video files from both musicians (located in \data\raw\avmanip_output) and mux them together into a new folder (default \data\raw\muxed_performances). 
    
4. You can customize the output by setting the `--keys` and `--drums` flags to either "Live" or "Delay". So, to recreate the experience that the keyboard player had in the experiment (i.e. live keyboard audio and video, delayed drummer audio and video), you'd use:
   
    ```
    python mixer.py --drums "Delay" --keys "Live"
    ```
    
    These flags default to "Delay" for both musicians, which will recreate the stimuli used in the perceptual component of this paper, i.e. the video files contained in perceptual_study_videos.rar on the [Zenodo](https://doi.org/10.5281/zenodo.7773824) dataset.
    
5. You can also change a few other options with how the videos look by passing in further commands, just as you would with FFmpeg. Not all of these are implemented: you can see those that are with:
    
    ```
    python mixer.py --help
    ```    
    
## Future

As this repository relates to a research paper, I don't envisage making many future changes. However, if you encounter any bugs or have suggestions, please open an issue and I'll take a look.

## Repository structure

    ```
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Serialized models and model predictions (simulations)
    ├── notebooks          <- Jupyter notebooks used in data exploration.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            
    │   └── figures        <- Generated graphics and figures included in the original paper
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── setup.py           
    ├── src                <- Source code for the project.
    │   ├── __init__.py    
    │   ├── clean          <- Scripts to generate final dataset from raw data
    │   │   └── make_dataset.py
    │   ├── analysis       <- Scripts to turn final dataset into models and simulations
    │   │   └── run_analysis.py
    │   └── visualise      <- Scripts to create exploratory and results oriented visualisations
    │       └── visualize.py
    └── run.cmd            <- Batch file to run all required scripts, generate models & graphs
    ```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
