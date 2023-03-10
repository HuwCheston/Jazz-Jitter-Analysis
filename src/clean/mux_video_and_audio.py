import subprocess
import os
import re
import logging


class AVMuxer:
    """
    This class is used to mux audio and video footage from each performance together
    """
    def __init__(self, input_dir, output_dir, ext: str = 'Delay', **kwargs):
        # TODO: implement this more!
        # The extension we should use to identify our performances (live or delayed)
        self.ext = ext
        # Input directory where our performance audio + video is found
        self.input_dir = input_dir
        # Output directory where we'll store our combined footage
        self.output_dir = output_dir
        # Whether or not to show logging messages from FFMPEG
        self.report_ffmpeg: bool = kwargs.get('report_ffmpeg', False)
        self.subprocess_kwargs = self._get_ffmpeg_kwargs()
        # A logger we can use to provide our own messages
        self.logger = kwargs.get('logger', None)
        # Audio attributes
        self.audio_ftype: tuple[str] = kwargs.get('audio_ftype', '.wav')
        self.audio_filter: str = kwargs.get('audio_filter', 'amix=inputs=2:duration=longest')
        self.audio_output_ftype: str = kwargs.get('audio_output_ftype', 'aac')
        # Video attributes
        self.video_filter: str = kwargs.get('video_filter', 'hstack,format=yuv420p')
        self.video_ftype: tuple[str] = kwargs.get('video_ftype', ('.mkv', '.avi'))
        self.video_name: str = kwargs.get('video_name', 'View')
        self.video_output_ftype: str = kwargs.get('video_output_ftype', 'avi')
        self.video_codec: str = kwargs.get('video_codec', 'libx264')
        self.video_crf: str = kwargs.get('video_crf', '18')
        self.video_preset: str = kwargs.get('video_preset', 'ultrafast')
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
                # If this is a video file, add it as such
                if filename.endswith(self.video_ftype) and self.video_name in filename:
                    dic[key][self._get_video_fpath(filename, key)] = join
                # Else if this is an audio file, add it as such
                elif filename.endswith(self.audio_ftype):
                    dic[key][self._get_audio_fpath(filename)] = join
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
        # TODO: I should probably change this at some point
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

    def _get_ffmpeg_kwargs(
            self
    ) -> dict:
        """
        Returns a dictionary of keyword arguments for use in subprocess.command processes piped to FFMPEG
        """
        # These arguments tell Python not to log from FFMPEG
        if not self.report_ffmpeg:
            return dict(stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            return dict()

    def mux_all_performances(
            self, duos: tuple = (1, 2, 3, 4, 5)
    ) -> None:
        """
        Muxes all performances together
        """
        if self.logger is not None:
            self.logger.info(f'Muxing all performances!')
        # Iterate through all of our performances
        for k, v in self.fname_dic.items():
            # If we're skipping over this duo, continue on
            if int(k[1]) not in duos:
                continue
            if self.logger is not None:
                self.logger.info(f'Muxing: {k}')
            # Get all of our filenames
            audio_fname = fr'{self.output_dir}\{k}_audio.{self.audio_output_ftype}'
            video_fname = fr'{self.output_dir}\{k}_video.{self.video_output_ftype}'
            all_fname = fr'{self.output_dir}\{k}_all.avi'
            # Mux the audio and video separately
            self._mux_audio(v, audio_fname)
            self._mux_video(v, video_fname)
            # Mux the audio and video together
            self._mux_audio_video(all_fname, video_fname, audio_fname)
            # Cleanup the interim files
            self._cleanup_interim_files([audio_fname, video_fname])
        if self.logger is not None:
            self.logger.info(f'Muxing finished!')

    def _mux_audio(
            self, v: dict, fname: str,
    ) -> None:
        """
        Combines both audio files together
        """
        command = [
            'ffmpeg',  # Opens FFMPEG
            '-i', v["keys_wav"],  # Keys wav as input
            '-i', v["drms_wav"],  # Drums wav as input
            '-filter_complex', self.audio_filter,  # Output as mix
            fname, '-y'  # Output directory, overwrite if needed
        ]
        subprocess.call(command, **self.subprocess_kwargs)

    def _mux_video(
            self, v: dict, fname: str,
    ) -> None:
        """
        Combine both video files together
        """
        command = [
            'ffmpeg',   # Opens FFMPEG
            '-i', v["keys_vid"],    # Keys video filepath
            '-i', v["drms_vid"],    # Drums video filepath
            '-filter_complex', self.video_filter,   # Output both videos side-by-side
            '-c:v', self.video_codec,   # Select video codec, default lib x 264
            '-crf', self.video_crf,     # Select video crf, default 18
            '-preset', self.video_preset,   # Select video output preset, default ultrafast
            fname, '-y'     # Output directory, overwrite if needed
        ]
        subprocess.call(command, **self.subprocess_kwargs)

    def _mux_audio_video(
            self, fname: str, video_fname: str, audio_fname: str,
    ) -> None:
        """
        Muxes together the muxed video and audio files
        """
        # Combine video and audio
        command = [
            'ffmpeg',   # Opens FFMPEG
            '-i', video_fname,    # Video filepath
            '-i', audio_fname,    # Audio file[ath
            '-c:v', 'copy',     # Copies the video output filetype
            '-c:a', self.audio_output_ftype,    # Sets audio output filetype
            fname, '-y'     # Output directory, overwrite if needed
        ]
        subprocess.call(command, **self.subprocess_kwargs)

    @staticmethod
    def _cleanup_interim_files(
            files: list
    ) -> None:
        """
        Removes interim audio and video files
        """
        # Iterate through files passed in as a list and try to remove them
        for file in files:
            try:
                os.remove(file)
            # We use the try-except block in case the interim file is missing, for whatever reason
            except OSError:
                pass


def generate_muxed_performances(input_dir: str, output_dir: str, logger_=None):
    mux = AVMuxer(input_dir, output_dir, logger=logger_)
    mux.mux_all_performances()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    input_d = r"C:\Python Projects\jazz-jitter-analysis\data\raw\avmanip_output"
    output_d = r"C:\Python Projects\jazz-jitter-analysis\data\raw\muxed_performances"
    generate_muxed_performances(input_d, output_d, logger)
