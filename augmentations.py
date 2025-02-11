import scipy.interpolate
import subprocess
import numpy as np
import soundfile as sf
import io


class ScalePitchOrTempo:
    def __init__(self, p=0.3, fs=16000, steps_range=[-3, 3], tempo_range=[0.8, 1.2],
                 warp_range=[0.9, 1.1]):
        self.p = p
        self.fs = fs
        self.steps_range = steps_range
        self.tempo_range = tempo_range
        self.warp_range = warp_range

    def augment(self, audio):
        """
        Shifts the pitch, adjusts the tempo, or applies time-warping to an audio signal with high quality.

        Args:
            audio (numpy.ndarray): The input audio signal (1D array).

        Returns:
            numpy.ndarray: Augmented audio signal.
        """
        if np.random.rand() < self.p:
            # Randomly choose which augmentation to apply
            # choice = np.random.choice(['pitch', 'tempo', 'warp'])
            choice = 'warp'
            if choice == 'pitch':
                # Pitch shift augmentation using librosa
                n_steps = np.random.uniform(self.steps_range[0], self.steps_range[1])
                audio = self.pitch_shift_sox_pipe(audio, self.fs, n_steps)

            elif choice == 'tempo':
                # Tempo augmentation using librosa
                tempo_factor = np.random.uniform(self.tempo_range[0], self.tempo_range[1])
                audio = self.tempo_adjust_sox_pipe(audio, self.fs, tempo_factor)

            elif choice == 'warp':
                # Time warp augmentation using high-quality linear interpolation
                warp_factor = np.random.uniform(self.warp_range[0], self.warp_range[1])
                audio = self.time_warp(audio, warp_factor)

        return audio

    def time_warp(self, audio, warp_factor=1.0):
        """
        Time-warp the audio by a given warp factor using linear interpolation for high quality.

        Args:
            audio (numpy.ndarray): The input audio signal (1D array).
            warp_factor (float): The factor by which to warp the time. >1 for speed-up, <1 for slow-down.

        Returns:
            numpy.ndarray: Time-warped audio signal.
        """
        # Create an index array for the original audio
        original_indices = np.arange(len(audio))
        # Create new indices based on the warp factor
        new_indices = np.linspace(0, len(audio) - 1, int(len(audio) * warp_factor))
        # Use linear interpolation to create the time-warped signal
        interpolator = scipy.interpolate.interp1d(original_indices, audio, kind='linear', fill_value='extrapolate')
        warped_audio = interpolator(new_indices)

        return warped_audio


    def pitch_shift_sox_pipe(self, audio_np, sample_rate, n_steps):
        # Convert NumPy array to WAV data in memory
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_np, sample_rate, format='WAV')
            wav_buffer.seek(0)

            # Run Sox through subprocess with pipe
            process = subprocess.Popen(
                ['sox', '-', '-t', 'wav', '-', 'pitch', str(n_steps * 100)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            output_audio, _ = process.communicate(input=wav_buffer.read())

        # Convert processed WAV bytes back to NumPy
        with io.BytesIO(output_audio) as output_buffer:
            processed_audio, _ = sf.read(output_buffer)

        return processed_audio

    def pitch_shift_sox_pipe(self, audio_np, sample_rate, n_steps):
        # Convert NumPy array to WAV data in memory
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_np, sample_rate, format='WAV')
            wav_buffer.seek(0)

            # Run Sox through subprocess with pipe
            process = subprocess.Popen(
                ['sox', '-', '-t', 'wav', '-', 'pitch', str(n_steps * 100)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            output_audio, _ = process.communicate(input=wav_buffer.read())

        # Convert processed WAV bytes back to NumPy
        with io.BytesIO(output_audio) as output_buffer:
            processed_audio, _ = sf.read(output_buffer)

        return processed_audio

    def tempo_adjust_sox_pipe(self, audio_np, sample_rate, tempo_factor):
        """
        Adjusts the tempo of an audio signal using Sox, without changing the pitch.

        Args:
            audio_np (numpy.ndarray): The input audio signal (1D array).
            sample_rate (int): The sample rate of the audio signal.
            tempo_factor (float): The factor by which to adjust the tempo. >1 for speed-up, <1 for slow-down.

        Returns:
            numpy.ndarray: Tempo-adjusted audio signal.
        """
        # Convert NumPy array to WAV data in memory
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_np, sample_rate, format='WAV')
            wav_buffer.seek(0)

            # Run Sox through subprocess with pipe to adjust the tempo
            process = subprocess.Popen(
                ['sox', '-', '-t', 'wav', '-', 'tempo', str(tempo_factor)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            output_audio, _ = process.communicate(input=wav_buffer.read())

        # Convert processed WAV bytes back to NumPy
        with io.BytesIO(output_audio) as output_buffer:
            processed_audio, _ = sf.read(output_buffer)

        return processed_audio




