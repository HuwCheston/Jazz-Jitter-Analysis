"""Separate executable .py file for creating combined audio-video stimuli from all raw datasets"""

import subprocess
import os
import re
import logging
import datetime
import time
import math
import click
import json


class AVMuxer:
    """
    Uses FFmpeg to mux raw audio and video footage from each performance in the corpus together.
    """

    def __init__(
            self, input_dir, output_dir, keys_ext: str = 'Delay', drms_ext: str = 'Delay', **kwargs
    ):
        # The extension we should use to identify our performances (live or delayed)
        self.keys_ext = keys_ext
        self.drms_ext = drms_ext
        self.ext = {
            'keys_audio': self.keys_ext,
            'drums_audio': self.drms_ext,
            'keys_video': 'View' if self.keys_ext == 'Delay' else 'Rec',
            'drums_video': 'View' if self.drms_ext == 'Delay' else 'Rec',
        }

        # Input directory where our performance audio + video is found
        self.input_dir = input_dir
        # Output directory where we'll store our combined footage
        self.output_dir = output_dir
        self.output_dir = self._create_output_folder()
        # A logger we can use to provide our own messages
        self.logger = kwargs.get('logger', None)
        self.log_individual_progress = kwargs.get('log_individual_progress', False)
        self.verbose: bool = kwargs.get('verbose', False)
        # Audio attributes
        self.audio_in_ftype: tuple[str] = kwargs.get('audio_in_ftype', '.wav')
        self.audio_codec: str = kwargs.get('audio_codec', 'aac')
        # Timestamps for seeking
        self.input_ts = time.strftime('%H:%M:%S', time.strptime(kwargs.get('input_ts', "00:00:06"), '%H:%M:%S'))
        self.output_ts = time.strftime('%H:%M:%S', time.strptime(kwargs.get('output_ts', "00:00:53"), '%H:%M:%S'))
        # Video attributes
        self.video_in_ftype: tuple[str] = kwargs.get('video_in_ftype', ('.mkv', '.avi'))
        self.video_out_ftype: str = kwargs.get('video_out_ftype', 'mp4')
        self.video_codec: str = kwargs.get('video_codec', 'libx264')
        self.video_bitrate: str = kwargs.get('video_bitrate', '1500k')
        self.video_px_fmt: str = kwargs.get('video_px_fmt', 'yuv420p')
        self.video_resolution: str = kwargs.get('video_resolution', '1280:720')
        self.video_crf: str = str(kwargs.get('video_crf', 28))
        self.video_fps: str = str(kwargs.get('video_fps', 30))
        self.duo_1_crf_mod: int = kwargs.get('duo_1_crf_mod', 4)
        self.video_crop: str = str(kwargs.get('video_crop', 13))
        self.duo_1_video_crop: str = str(kwargs.get('duo_1_video_crop', 31))
        self.video_border_width: int = str(kwargs.get('video_border_width', 5))
        # FFmpeg attributes
        self.ffmpeg_filter = kwargs.get(
            'ffmpeg_filter',
            f"[0]crop=iw-{self.video_crop}:ih-{self.video_crop}:0:0,"
            f"scale={self.video_resolution},pad=iw+{self.video_border_width}:color=black[vidL];"
            f"[1]crop=iw-{self.video_crop}:ih-{self.video_crop}:0:0,"
            f"scale={self.video_resolution},pad=iw+{self.video_border_width}:color=black[vidR];"
            f"[vidL][vidR]hstack=inputs=2,format={self.video_px_fmt}[vid];"
            "[2]dynaudnorm=f=100:maxgain=5:gausssize=31[audL];"
            "[3]dynaudnorm=f=100:maxgain=5:gausssize=21[audR];"
            "[audL][audR]amix=inputs=2:duration=longest[aud]",
        )
        self.ffmpeg_preset: str = kwargs.get('ffmpeg_preset', 'ultrafast')
        self.input_ts: int = kwargs.get('input_ts', '00:00:06')
        self.output_ts: float = kwargs.get('output_ts', '00:01:38')
        # The dictionary containing all of our matched filenames that will be used to create combined videos
        self.fname_dic: dict = self._get_matched_fnames()

    @staticmethod
    def _get_dict_key(
            dirpath: str, chars: tuple[str] = ('d', 's', 'l', 'j')
    ) -> str:
        """
        Gets the key for our dictionary of filenames.
        Format is in the form: duoX_sessionX_latencyX_jitterX
        """
        return '_'.join([s + n for s, n in zip(list(chars), [re.findall(r'\d+', dirpath)[n] for n in [0, 1, 3, 4]])])

    def _get_matched_fnames(
            self
    ) -> dict:
        """
        For every performance, matches audio and video filenames for both performers
        """
        dic = {}
        # Walk through our input directory
        for (dirpath, dirnames, filenames) in os.walk(self.input_dir):

            # Iterate through each filename in our folder
            for filename in filenames:
                # Get our complete filepath
                join = os.sep.join([dirpath, filename])
                # If this isn't from an experimental condition (i.e warm-up), skip and continue to the next performance
                if re.search(r'- (.*?)\\', join) is None or 'BPM' in dirpath:
                    continue
                # Get our dictionary key and add it into our results if we haven't got it already
                key = self._get_dict_key(dirpath)
                if key not in dic:
                    dic[key] = {}
                if filename.endswith(self.audio_in_ftype):
                    ins = 'keys' if 'keys' in filename.lower() else 'drums'
                    if self.ext[f'{ins}_audio'] not in filename:
                        continue
                    dic[key][self._get_audio_fpath(filename)] = join
                elif filename.endswith(self.video_in_ftype):
                    if key[1] == '1':
                        ins = 'drums' if 'cam1' in filename else 'keys'
                    else:
                        ins = 'keys' if 'cam1' in filename else 'drums'
                    if self.ext[f'{ins}_video'] not in filename:
                        continue
                    dic[key][self._get_video_fpath(filename, key)] = join
        # Return our filepath dictionary
        return dic

    @staticmethod
    def _get_video_fpath(
            filename: str, key: list
    ) -> str:
        """
        Returns the type of video that is described by this filename
        """
        # cam1 = Drums, cam2 = Keys for duo 1 only
        if key[1] == '1':
            if 'cam1' in filename:
                return 'drms_vid'
            elif 'cam2' in filename:
                return 'keys_vid'
        # Otherwise, cam1 = keys, cam2 = drums
        else:
            if 'cam1' in filename:
                return 'keys_vid'
            elif 'cam2' in filename:
                return 'drms_vid'

    @staticmethod
    def _get_audio_fpath(
            filename: str
    ) -> str:
        """
        Returns the type of audio that is described by this filename
        """
        if 'Keys' in filename:
            return 'keys_wav'
        elif 'Drums' in filename:
            return 'drms_wav'

    def mux_all_performances(
            self, duos: tuple = (1, 2, 3, 4, 5), sessions: tuple = (1, 2), latency: tuple = (0, 23, 45, 90, 180),
            jitter: tuple = (0, 0.5, 1.0)
    ) -> None:
        """
        Muxes all performances together. Duos should be an iterable containing the duo numbers we want to mux.
        """
        fmt = lambda it: ", ".join(str(i) for i in it)
        if self.logger is not None:
            self.logger.info(f'Muxing performances by duos: {fmt(duos)}')
            self.logger.info(f'Muxing sessions: {fmt(sessions)}')
            self.logger.info(f'Muxing latency values: {fmt(latency)} ms')
            self.logger.info(f'Muxing jitter values: {fmt(jitter)} x')
            self.logger.info(f'Muxing keys {self.keys_ext}, drums {self.drms_ext}')
        # Get the performances we want to mux
        performances_to_mux = self._get_performances_to_mux(duos, jitter, latency, sessions)
        if self.logger is not None:
            self.logger.info(f"Found {len(performances_to_mux)} performances to mux!")
        for num, (k, v) in enumerate(performances_to_mux.items()):
            self._log_progress_bar(current=num, end=len(performances_to_mux), perf=f'Muxing ({k})')
            # Get the filename
            fname = fr'{self.output_dir}\{k}_k{self.keys_ext.lower()}_d{self.drms_ext.lower()}.{self.video_out_ftype}'
            # Mux the individual audio and video files together
            self._mux_performance(k, v, fname, )
        if self.logger is not None:
            self.logger.info(f'Muxing finished!')

    def _get_performances_to_mux(
            self, duos: tuple, jitter: tuple, latency: tuple, sessions: tuple
    ) -> dict:
        """
        Gets the filenames of performances to mux from given input iterables
        """

        performances_to_mux = {}
        # We need to convert our jitter values to integers, as we can't have full stops (.) in our file names!
        jitter = tuple(int(float(i) * 10) for i in jitter)
        latency = tuple(int(i) for i in latency)
        # Iterate through all of our filenames
        for k, v in self.fname_dic.items():
            # Remove unnecessary characters from filenames to get list of integers
            p = [int(v.translate({ord(c): None for c in 'dslj'})) for v in k.split('_')]
            # Add filename and values to our dictionary if we want it
            if p[0] in duos and p[1] in sessions and p[2] in latency and p[3] in jitter:
                performances_to_mux[k] = v
        return performances_to_mux

    def _create_output_folder(
            self
    ) -> str:
        """
        Creates an output folder according to the given parameters
        """
        # Gets the string of our new directory
        new_dir = os.sep.join([self.output_dir, f'k{self.keys_ext.lower()}_d{self.drms_ext.lower()}'])
        # Create the directory if it doesn't exist and return our directory filepath
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir

    def _log_progress_bar(
            self, current: float, end: float, perf: str
    ) -> None:
        """
        Logs current progress of the ffmpeg conversion, including current time, overall duration, and a progress bar
        """

        def bar() -> str:
            # Get our current progress percentage
            perc = (current / end) * 100
            # Get the nearest ten of our current progress, for conversion to progress bar
            nearest_ten = int(10 * math.ceil(float(perc) / 10))
            # Return our progress bar
            return f'[{("=" * int(nearest_ten / 10)) + (" " * (10 - int(nearest_ten / 10)))} {round(perc, 2)}%]'

        if self.logger is not None:
            self.logger.info(f'{perf}: {bar()} (completed: {current}, total: {end})')

    @staticmethod
    def _read_timedelta(
            line: str
    ) -> float:
        """
        Converts the reported time in ffmpeg to a Python timedelta object and gets number of seconds
        """
        reg = re.search('\d\d:\d\d:\d\d', line)
        try:
            x = time.strptime(reg.group(0), '%H:%M:%S')
            return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        except AttributeError:
            return None

    def _read_ffmpeg_output(
            self, process: subprocess.Popen, k: str, timeout_mins: int = 3
    ):
        """
        Reads the output from ffmpeg, line by line, and logs a progress bar
        """
        duration = []
        start = time.time() + (timeout_mins * 60)
        while True:
            if time.time() > start:
                self.logger.warning(f'{k}: encoding timed out!')
                break
            # Read a line from the subprocess
            line = process.stdout.readline()
            if self.verbose:
                self.logger.info(line)
            # Break out of the loop once we no longer receive lines from ffmpeg
            if not line:
                break
            # Try and convert the line into a timedelta object
            td = self._read_timedelta(line)
            # The first few lines of the ffmpeg output, i.e. the duration of each input object
            # The current process of ffmpeg
            if 'Duration' in line:
                duration.append(td)
            if 'time' in line and self.log_individual_progress and td is not None:
                self._log_progress_bar(current=td, end=max(duration), perf=k)

    def _get_video_crop_params(
            self, perf: str,
    ) -> tuple:
        """
        Gets the crop parameters from the given arguments
        """
        if int(perf[1]) == 1:
            return self.ffmpeg_filter.replace(f":ih-{self.video_crop}", f":ih-{self.duo_1_video_crop}")
        else:
            return self.ffmpeg_filter

    @staticmethod
    def _get_video_duration(
            filename: str
    ) -> str:
        """
        Gets the duration of an input video using ffprobe, in the format HH:MM:SS
        """
        command = f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"'
        result = subprocess.check_output(command, shell=True).decode()
        fields = json.loads(result)['streams'][0]
        try:
            return time.strftime('%H:%M:%S', time.gmtime(float(fields['duration'])))
        except KeyError:
            return fields['tags']['DURATION'][:8]

    def _get_output_timestamp(
            self, v: dict
    ) -> None:
        """
        If the given output timestamp video is longer than the duration of the input videos, set this to the duration
        """
        keys_dur = self._get_video_duration(v['keys_vid'])
        drms_dur = self._get_video_duration(v['drms_vid'])
        if self.output_ts > max([keys_dur, drms_dur]):
            self.output_ts = max([keys_dur, drms_dur])

    def _mux_performance(
            self, k: str, v: dict, fname: str,
    ) -> None:
        """
        Combine both video files together
        """
        filt = self._get_video_crop_params(perf=k)
        self._get_output_timestamp(v)
        # Initialise our ffmpeg command
        command = [
            'ffmpeg',  # Opens FFMPEG
            '-i', v["keys_vid"],  # Keys video filepath
            '-i', v["drms_vid"],  # Drums video filepath
            '-i', v['keys_wav'],  # Keys audio filepath
            '-i', v['drms_wav'],  # Drums audio filepath,
            '-ss', self.input_ts,
            '-to', self.output_ts,
            '-filter_complex', filt,  # Ffmpeg filter
            '-map', '[vid]',  # Maps video files
            '-map', '[aud]',  # Maps audio files
            '-b:v', self.video_bitrate,  # Sets video bitrate, default 1200kbps
            '-c:a', self.audio_codec,  # Select audio codec, default aac
            '-c:v', self.video_codec,  # Select video codec, default libx264
            '-crf', self.video_crf if int(k[1]) != 1 else str(int(self.video_crf) + self.duo_1_crf_mod),
            '-preset', self.ffmpeg_preset,  # Select video output preset, default ultrafast
            '-r', self.video_fps,
            fname, '-y'
        ]
        # Run the process, with the required kwargs for realtime logging
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        # Start to read our ffmpeg output and log in the console
        self._read_ffmpeg_output(process, k)
        self._check_filesize(fname, k)

    def _check_filesize(
            self, fname: str, performance: str, max_fsize: int = 75
    ):
        """
        Checks size of a muxed video and raises a warning if it's suspicously large (encoding issue)
        """
        fsize = os.path.getsize(fname) >> 20
        if fsize > max_fsize and self.logger is not None:
            self.logger.warning(f'{performance}: size is suspicously large ({fsize} MB), check output!')


@click.command()
@click.option(
    '-i', 'input_dir', default=os.path.abspath(r'.\\data\\raw\\avmanip_output'), type=str,
    help=r'The path to your \avmanip_output directory, containing the raw input files',
)
@click.option(
    '-o', 'output_dir', default=os.path.abspath(r'.\\data\\raw\\muxed_performances'), type=str,
    help=r'The path to save the output files. A new folder will be created in this location',
)
@click.option(
    '-preset', 'ffmpeg_preset', default='ultrafast', help='FFmpeg preset, defaults to "ultrafast"', type=str
)
@click.option(
    '-b:v', 'video_bitrate', default='1500k', help='Video bitrate, defaults to "1500k"', type=str
)
@click.option(
    '-c:v', 'video_codec', default='libx264', help='Video codec, defaults to "libx264"', type=str
)
@click.option(
    '-c:a', 'audio_codec', default='aac', help='Audio codec, defaults to "aac"', type=str
)
@click.option(
    '-format', 'video_format', default='yuv420p', help='Video pixel format, defaults to "yuv420p"', type=str
)
@click.option(
    '-scale', 'video_resolution', default='1280:720', help='Video resolution, defaults to "1280:720"', type=str
)
@click.option(
    '-crf', 'video_crf', default=28, help='Video crf, defaults to 28', type=int
)
@click.option(
    '-r', 'video_fps', default=30, help='Video fps, defaults to 30', type=int
)
@click.option(
    '-bw', 'video_border_width', default=5, help='Width of border separating videos, defaults to 5', type=int,
)
@click.option(
    '-ss', 'input_ts', default="00:00:06", help='Input timestamp in the form HH:MM:SS, defaults to 00:00:06', type=str,
)
@click.option(
    '-to', 'output_ts', default="00:00:53", type=str,
    help='Output timestamp in the form HH:MM:SS, defaults to 00:00:53',
)
@click.option(
    '-verbose', 'verbose', is_flag=True, default=False, show_default=True, help="Pipe ffmpeg output to console",
)
@click.option(
    '--keys', 'keys_ext', default='Delay', type=click.Choice(['Delay', 'Live']),
    help='Keyboard audio and video type, defaults to "Delay"'
)
@click.option(
    '--drums', 'drms_ext', default='Delay', type=click.Choice(['Delay', 'Live'], case_sensitive=False),
    help='Drums audio and video type, defaults to "Delay"'
)
@click.option(
    '--d', 'duos', default=(1, 2, 3, 4, 5), type=click.IntRange(1, 5, clamp=False), multiple=True,
    help="A duo to mux. Pass multiple arguments (e.g. --d 1, --d 3) to mux multiple duos. "
         "Defaults to all available duos."
)
@click.option(
    '--s', 'sessions', default=(1, 2), type=click.IntRange(1, 2, clamp=False), multiple=True,
    help="A session to mux. Pass multiple arguments (e.g. --s 1, --s 2) to mux multiple sessions. "
         "Defaults to all available sessions."
)
@click.option(
    '--l', 'latency', default=("0", "23", "45", "90", "180"), type=click.Choice(["0", "23", "45", "90", "180"]), multiple=True,
    help="Latency values to mux. Pass multiple arguments (e.g. --l 0, --l 45) to mux multiple values. "
         "Defaults to all latency values."
)
@click.option(
    '--j', 'jitter', default=("0.0", "0.5", "1.0"), type=click.Choice(["0.0", "0.5", "1.0"]), multiple=True,
    help="Jitter values to mux. Pass multiple arguments (e.g. --j 0.5, --j 1.0) to mux multiple values. "
         "Defaults to all jitter values."
)
def generate_muxed_performances(
        input_dir: str, output_dir: str, ffmpeg_preset: str, video_bitrate: str, video_codec: str,
        audio_codec: str, video_format: str, video_resolution: str, video_crf: int, video_fps: int,
        video_border_width: int, input_ts: str, output_ts: str, verbose: bool, keys_ext: str, drms_ext: str,
        duos: tuple, sessions: tuple, latency: tuple, jitter: tuple
) -> None:
    """
    Generates all muxed performances from an input and output directory
    """
    # Initialise the logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logging.info(f'Input directory: {input_dir}')

    # Create the AVMuxer with required settings
    muxer = AVMuxer(
        input_dir, output_dir, logger=logger, ffmpeg_preset=ffmpeg_preset, video_bitrate=video_bitrate,
        video_codec=video_codec, audio_codec=audio_codec, video_format=video_format, video_resolution=video_resolution,
        video_crf=video_crf, video_fps=video_fps, video_border_width=video_border_width, input_ts=input_ts,
        keys_ext=keys_ext, drms_ext=drms_ext, verbose=verbose, output_ts=output_ts
    )
    logging.info(f'Output directory: {muxer.output_dir}')
    # Mux all the performances
    muxer.mux_all_performances(duos=duos, sessions=sessions, latency=latency, jitter=jitter)


if __name__ == '__main__':
    generate_muxed_performances()
