import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelScale

class ConstMaskLayer(nn.Module):
    def __init__(self, bands=120, fs=16000, fft_channels=640, dropout_p=0.0, dropout_rate=0.2, bandmask_p=0.0,
                 max_bankmask_width=0.2, timemask_p=0.0):
        """
        Applies constant masking, random dropout, bandmask, and time masking augmentations.

        Args:
            bands (int): Number of bands for bandmask.
            fs (int): Sample rate of the audio.
            fft_channels (int): Number of FFT channels.
            dropout_p (float): Probability of applying dropout.
            dropout_rate (float): Proportion of spectrogram elements to drop.
            bandmask_p (float): Probability of applying bandmask augmentation.
            maxwidth (float): Maximum width of frequency bands to mask.
            timemask_p (float): Probability of applying time mask.
        """
        super().__init__()
        self.fs = fs
        self.bands = bands
        self.nfft = fft_channels
        self.dropout_p = dropout_p
        self.dropout_rate = dropout_rate
        self.bandmask_p = bandmask_p
        self.max_bankmask_width = max_bankmask_width
        self.timemask_p = timemask_p
        # self.const_value = 0.3
        mel = MelScale(n_mels=bands, sample_rate=fs, n_stft=fft_channels // 2 + 1, f_min=40, f_max=fs // 2)

        # Register the filterbank as a buffer
        # self.register_buffer('mel_fb', mel.fb.clone())

        # self.mel_fb = self.mel_fb.to(spectrum.device)  # Get the filterbank matrix
        dominant_filters = torch.argmax(mel.fb, dim=1)  # Find the dominant mel filter for each frequency bin

        # Create a mask for mel_fb that keeps only the dominant filters
        mel_fb_masked = torch.zeros_like(mel.fb)
        mel_fb_masked[torch.arange(mel.fb.size(0)), dominant_filters] = mel.fb[
        torch.arange(mel.fb.size(0)), dominant_filters]
        self.register_buffer('mel_fb_masked', mel_fb_masked)
    def forward(self, spectrum):
        """
        Apply constant masking augmentation to an audio spectrum.

        Args:
            spectrum (torch.Tensor): The input audio spectrogram (complex tensor) with shape (batch_size, 1, freq_channels, time_frames, 2).

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        if not self.training:
            # Only apply augmentation during training
            return spectrum


        batch_size, _, freq_channels, time_frames, _ = spectrum.size()

        # Generate random decisions for each example in the batch
        apply_dropout = torch.rand(batch_size, device=spectrum.device) < self.dropout_p
        apply_bandmask = torch.rand(batch_size, device=spectrum.device) < self.bandmask_p
        apply_timemask = torch.rand(batch_size, device=spectrum.device) < self.timemask_p

        # Apply dropout mask
        if self.dropout_p > 0:
            const_rand_vec = torch.rand(batch_size, device=spectrum.device) * 0.4 + 0.1
            const_rand_mat = const_rand_vec[:, None, None, None] * torch.full_like(spectrum[..., 0], 1)
            rand_mask = torch.rand(batch_size, 1, freq_channels, time_frames,
                                   device=spectrum.device) < self.dropout_rate
            dropout_mask = (apply_dropout[:, None, None, None] * rand_mask)
            spectrum[..., 0] = torch.where(dropout_mask, const_rand_mat, spectrum[..., 0])
            spectrum[..., 1] = torch.where(dropout_mask, torch.full_like(spectrum[..., 1], 0), spectrum[..., 1])

        # Apply bandmask
        if self.bandmask_p > 0:
            # const_rand_vec = torch.rand(1, device=spectrum.device)[0] * 0.4 + 0.1
            const_rand_vec = torch.rand(batch_size, device=spectrum.device) * 0.4 + 0.1
            const_rand_mat = const_rand_vec[:, None, None, None] * torch.full_like(spectrum[..., 0], 1)

            # Step 1: Create a [batch_size, bands] matrix with True in one entry per example
            max_near_bands = int(self.max_bankmask_width * self.bands)
            n_near_bands = torch.randint(1, max_near_bands + 1, (1,), device=spectrum.device).item()
            selected_bands = torch.randint(0, self.bands - n_near_bands, (batch_size,), device=spectrum.device)
            band_selection_mask = torch.zeros((batch_size, self.bands), device=spectrum.device)
            band_selection_mask[torch.arange(batch_size), selected_bands] = 1

            # Step 2: Extend to n near bands using convolution for efficiency
            kernel = torch.ones((1, 1, n_near_bands), device=spectrum.device, dtype=torch.float32)
            band_selection_mask = band_selection_mask.unsqueeze(1)  # Shape: [batch_size, 1, bands]
            band_selection_mask = F.conv1d(band_selection_mask, kernel, padding=n_near_bands-1).squeeze(1)
            band_selection_mask = band_selection_mask[:, :self.bands]
            # Step 3: Use Mel filterbank to get mask in the frequency domain
            freq_band_mask = torch.matmul(band_selection_mask, self.mel_fb_masked.transpose(0,1)).bool()

            # Apply the band mask to the spectrum
            freq_band_mask = freq_band_mask.view(batch_size, 1, freq_channels, 1)
            freq_band_mask = (apply_bandmask[:, None, None, None] * freq_band_mask)


            spectrum[..., 0] = torch.where(freq_band_mask, const_rand_mat, spectrum[..., 0])
            spectrum[..., 1] = torch.where(freq_band_mask, torch.full_like(spectrum[..., 1], 0), spectrum[..., 1])
        # Apply time mask
        if self.timemask_p > 0:
            const_rand_vec = torch.rand(batch_size, device=spectrum.device) * 0.4 + 0.1
            const_rand_mat = const_rand_vec[:, None, None, None] * torch.full_like(spectrum[..., 0], 1)
            time_mask_len = torch.randint(1, 3, (batch_size,), device=spectrum.device)
            time_starts = torch.randint(0, time_frames - time_mask_len.max(), (batch_size,), device=spectrum.device)

            # Vectorized application of time mask
            time_mask = torch.zeros_like(spectrum[..., 0], dtype=torch.bool)
            # batch_indices = torch.arange(batch_size, device=spectrum.device)

            for i in range(batch_size):
                if apply_timemask[i]:
                    start = time_starts[i]
                    length = time_mask_len[i]
                    if start + length <= time_frames:  # Ensure valid range
                        time_mask[i, :, :, start:start + length] = True

            spectrum[..., 0] = torch.where(time_mask, const_rand_mat, spectrum[..., 0])
            spectrum[..., 1] = torch.where(time_mask, torch.full_like(spectrum[..., 1], 0), spectrum[..., 1])

        return spectrum
