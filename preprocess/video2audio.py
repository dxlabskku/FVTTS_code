import os
import subprocess
from tqdm import tqdm
import warnings
from pydub import AudioSegment
import glob

warnings.filterwarnings(action='ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', default='data/trainval', type=str, help= 'video_dir')
parser.add_argument('--audio_path', default='data/audio', type=str, help= 'audio_path')
parser.add_argument('--mono_path', default='data/mono', type=str, help= 'mono_path')
args = parser.parse_args()

video_dir = args.video_dir
audio_path = args.audio_path
mono_path = args.mono_path
if not os.path.exists(audio_path):
    os.mkdir(audio_path)
if not os.path.exists(mono_path):
    os.mkdir(mono_path)

# video -> audio
data_list = os.listdir(video_dir)
for idx, name in enumerate(data_list):
    print(f'{idx}/ {len(data_list)}')
    video_file = [f for f in os.listdir(f"{video_dir}/{name}") if f.endswith('mp4')]
    for v in tqdm(video_file):
        video_path = f"{video_dir}/{name}/{v}"
        save_path = f"{audio_path}/{name}_{v[:-4]}.wav"
        command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {save_path}" 
        subprocess.call(command, shell=True)

# stereo -> mono
audio_list = glob.glob(audio_path)
for audio_path in tqdm(audio_list):
    a = os.path.basename(audio_path)
    save_path = f"{mono_path}/{a}"
    sound = AudioSegment.from_wav(audio_path)
    sound = sound.set_channels(1)
    sound.export(save_path, format="wav")