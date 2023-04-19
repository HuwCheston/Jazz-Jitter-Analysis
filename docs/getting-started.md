# Getting started
This page contains instructions for reproducing the data structures used in the original paper.

**Tip:** *If you just want to see how our models and simulations work without having to install anything locally, check out this online notebook:* <a target="_blank" href="https://colab.research.google.com/github/HuwCheston/Jazz-Jitter-Analysis/blob/main/notebooks/0.1-cheston-modelling-one-performance.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Reproduce models, simulations, and figures
0. Make sure you have Python 3.10 installed and accessible at the top of your `PATH`, before any other versions of Python. Running `python` on a command terminal should open a new Python 3.10 session, not any other version of Python.


1. Clone our repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Jazz-Jitter-Analysis
```


2. Download our raw data from [Zenodo](https://doi.org/10.5281/zenodo.7773824). You'll need the `data.zip` file and all the corresponding volumes (in the form `data.z**`)
    
    *You do not need to download the file perceptual_study_videos.rar unless you wish to replicate the perceptual component of the paper, the code for which is hosted in a separate repository acccessible [via this link](https://github.com/HuwCheston/2023-duo-success-analysis).*
  
  
3. Open the `data.zip` file (you may need to install a tool for opening multi-part zip files, such as [WinRAR](https://www.win-rar.com/)) and extract all the contents (three folders + one file) into `\data\raw`. This folder should then look like:
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
    ├── muxed_performances <- Audio-video .mp4 files, not used currently but might be in the future
    └── questionnaire_anonymised.xlsx    
    ```


4. Open a new command prompt in the root directory of the repository and execute:
    ```
    run.cmd
    ```

    This script will:
    - Create a new virtual environment in the repository root folder, set some environment variables, then install the required dependencies. 
    - Generate the final dataset from the raw data dump downloaded above
    - Generate the models and simulations from the dataset
    - Generate the figures used in the original paper and supplementary material. 
    
    Reproducing the models is fairly quick and optimised (on a system with a 2.6GHz CPU and 16GB of RAM, it takes about three minutes), but reproducing the simulations will typically take a lot longer; probably *at least twenty minutes* with the default number of simulations per paradigm/condition (`500`). The script will log each model or simulation it creates with a timestamp so you can be sure that the process hasn't stalled.
   
 
5. Once the command has finished, you can access the final dataset in `\data\processed`, the models and simulations in `\models`, and the figures in `\figures\reports`. The dataset, models, and simulations are saved as [Python pickle files](https://docs.python.org/3/library/pickle.html) and should be unserialised using the `dill.load()` function in the [Dill](https://dill.readthedocs.io/en/latest/) module. Note that trying to unserialise using the Pickle module itself (i.e. with the `pickle.load()` function) is not supported and will more than likely produce errors due to how the custom classes are serialised. The figures are saved as `.png` and `.svg` files which can be opened in many different applications.


6.   If you execute `run.cmd` again, you'll notice that the script will not attempt to rebuild the processed dataset, models, or simulations if it detects that these have already been created successfully during a previous build. This can help e.g. if you wish to adjust simulation parameters without rebuilding the entire set of models. To force a rebuild you'll need to delete or rename the processed files in their respective output folders, before executing `run.cmd` again.

Now that you've built the dataset, models, and simulations, see [Examples](./examples.html) for guidance on how to work with these files in your own Python sessions.

## Reproduce combined audio-visual stimuli
The raw dataset contains separate audio and video files (both with and without latency/jitter) for each performer inside the `\data\raw\avmanip_output` folder. These can be combined to create muxed audio-video recordings, i.e. a single video file containing both audio and video tracks, sync'ed together, with any combination of live or delayed audio/video. The perceptual component of this study used the delayed audio and video from both the pianist and drummer.

1. Follow steps 1–3 in the section **Reproduce models, simulations, and figures from original paper** above. You don't need to build the models or simulations, just get the data into the correct place within the overall filestructure.


2. Download and install [FFmpeg](https://ffmpeg.org/). Any recent version should work: versions `6.0` and `5.1.2` have both been tested. You then need to ensure that FFmpeg can be accessed on your `PATH`. To check this, open a command prompt and type in
    
   ```
   ffmpeg -version
   ```

   If you don't see any errors, you're good to go! Otherwise, you may need to manually add the containing `ffmpeg.exe` (usually `ffmpeg\bin`) to your `PATH`.


3. Open a new command prompt in the root directory of the repository and execute:
    
   ```
   python src\muxer.py
   ```
    
   This script will take all the delayed audio and video files from both musicians (located in `\data\raw\avmanip_output`) and mux them together into a new folder (default `\data\raw\muxed_performances\`). You can choose a custom input and output directory by passing the `-i` and `-o` flags when calling `muxer.py`.

    
4. By default, the script will recreate all of the stimuli used in the perceptual component of our paper, i.e. a 47-second excerpt with delay applied to the audio and video footage from both performers. You can customize the output by setting the `--keys` and `--drums` flags to either `"Live"` or `"Delay"` when calling `muxer.py`. So, to recreate the experience that the keyboard player had in the experiment (i.e. live keyboard audio and video, delayed drummer audio and video), you'd use:
   
   ```
   python src\muxer.py --drums "Delay" --keys "Live"
   ```
    
   If you only want to mux a few performances, you can use the `--d`, `--s`, `--l`-, and `--j` flags to choose the duos, experimental sessions, latency values, and jitter scalings to mux. So, to render only the performance of duo 2 in the second session with 90 ms of latency and 1.0x jitter:

   ```
   python src\muxer.py --d 2 --s 2 --l 90 --j 1.0
   >>> __main__ - INFO - Muxing performances by duos: 2
   >>> __main__ - INFO - Muxing sessions: 2
   >>> __main__ - INFO - Muxing latency values: 90 ms
   >>> __main__ - INFO - Muxing jitter values: 1.0 x
   >>> __main__ - INFO - Muxing keys Delay, drums Delay
   >>> __main__ - INFO - Found 1 performances to mux!
   ```
   
   These flags accept multiple values: if you want all performance of the democracy duos 1 and 3 with 45 ms of latency and both jitter conditions (0.5x and 1.0x), you'd use:

   ```
   python src\muxer.py --d 1 --d 3 --l 45 --j 0.5 --j 1.0
   >>> __main__ - INFO - Muxing performances by duos: 1, 3
   >>> __main__ - INFO - Muxing sessions: 1, 2
   >>> __main__ - INFO - Muxing latency values: 45 ms
   >>> __main__ - INFO - Muxing jitter values: 0.5, 1.0 x
   >>> __main__ - INFO - Muxing keys Live, drums Delay
   >>> __main__ - INFO - Found 8 performances to mux!
   ```

   The `-ss` and `-to` flags work exactly as their FFmpeg counterparts do, allowing you to specify and input and output timestamp to seek by when rendering. So `-ss 00:00:06` and `-to 00:00:53` (the default options) will cut from 6 seconds to 53 seconds in the video. These were the commands used when creating the perceptual study stimuli.


6. You can also change a few other options related to how the videos look by passing in further FFmpeg commands and flags. Not all of the commands supported by FFmpeg are currently implemented here. You can see those that are by typing:
    
   ```
   python src\muxer.py --help
   ```
   
   Finally, if you're experiencing issues, you can turn on logging in FFmpeg by passing the `-verbose` flag to `muxer.py`. This will directly pipe the output of FFmpeg through to the console, which can be helpful when debugging.

## Reproduce latency measurement array
By default, the latency time series (the sequence of delay values applied iteratively by our software testbed, created by measuring network latency on Zoom Meetings) is not reproduced when the raw dataset is built. Instead, this is stored in the repository under `references\latency_array.csv`. We made the decision to 'hard-code' the latency time series in this way so that any future changes to the onset detection algorithms used to parse the network delay times from our original recording do not change our overall analysis.

We have created a notebook file that allows you to reproduce this latency time series from our original Zoom Meetings recording: hit the button below to run it online. Note that, while the output latency time series will be very similar, it may not be identical to the one currently in the repository as the [onset detection algorithms used in Librosa](https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect) are liable to change and vary depending on source audio. We do not support the use of any re-generated time series when reproducing our analysis for this reason.

Run the notebook online here: <a target="_blank" href="https://colab.research.google.com/github/HuwCheston/Jazz-Jitter-Analysis/blob/main/notebooks/0.1-cheston-jitter-measurement.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


