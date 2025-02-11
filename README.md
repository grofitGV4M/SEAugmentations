SEAugmentations
Speech Enhancement Augmentations inspired by DeepSpeech

This repository provides a set of speech enhancement augmentations designed specifically for networks that estimate a complex mask for applying on the input spectrum.

The augmentations are inspired by DeepSpeech's techniques but have been adapted for Speech Enhancement (SE) methods, particularly focusing on spectral masking.

📂 Repository Structure
1️⃣ augmentations.py
Contains augmentations that can be applied in parallel for audio in the time domain.
Uses SoX for time and frequency scaling, but these transformations are currently not used in the training process.
The current training uses only warping technique, which also modify the frequency content of the audio.
2️⃣ AugmentLayer.py
Implements a PyTorch layer for augmentations.
Specifically designed for Speech Enhancement (SE) networks that estimate and apply a complex spectral mask.
🛠 Dependencies
This project uses the following Python libraries:

import scipy.interpolate
import subprocess
import numpy as np
import soundfile as sf
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelScale

📌 Installation
pip install numpy scipy soundfile torch torchaudio


For SoX-based transformations, install SoX:
# Ubuntu
sudo apt-get install sox

# macOS (using Homebrew)
brew install sox
