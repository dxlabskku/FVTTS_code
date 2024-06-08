import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import read
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mono_path', default='data/mono', type=str, help= 'mono_path')
parser.add_argument('--spec_path', default='data/spec', type=str, help= 'spec_path')
parser.add_argument('--max_wav_value', default=1258791, type=int, help= 'max_wav_value')
parser.add_argument('--filter_length', default=1024, type=int, help= 'filter_length')
parser.add_argument('--sampling_rate', default=44100, type=int, help= 'sampling_rate')
parser.add_argument('--hop_length', default=256, type=int, help= 'hop_length')
parser.add_argument('--win_length', default=1024, type=int, help= 'win_length')
args = parser.parse_args()

max_wav_value = args.max_wav_value
filter_length = args.filter_length
sampling_rate = args.sampling_rate
hop_length = args.hop_length
win_length = args.win_length

hann_window = {}
def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
    
mono_path = args.mono_path
spec_path = args.spec_path
if not os.path.exists(spec_path):
    os.mkdir(spec_path)

wav_list = os.listdir(mono_path)
for w in tqdm(wav_list):
    filename = f"{mono_path}/{w}"
    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = os.path.basename(filename).replace(".wav", ".spec.pt")
    spec_o = spectrogram_torch(audio_norm, filter_length, sampling_rate, hop_length, win_length, center=False)
    spec = torch.squeeze(spec_o, 0)
    torch.save(spec, f"{spec_path}/{spec_filename}")