import os
import argparse
from text import cleaners

parser = argparse.ArgumentParser()
parser.add_argument('--text_path', default='data/trainval', type=str, help= 'text_path')
parser.add_argument('--mono_path', default='data/mono', type=str, help= 'mono_path')
parser.add_argument('--file_name', default='LRS3_valid.txt.cleaned', type=str, help= 'file_name')
args = parser.parse_args()

text_path = args.text_path
file_name = args.file_name
mono_path = args.mono_path


cleaner = getattr(cleaners, "english_cleaners2")
data_list = os.listdir(text_path)

with open (file_name, 'w') as lrs:
    for idx in range(len(data_list)):
        name = data_list[idx]
        text_file = [f for f in os.listdir(f"{text_path}/{name}") if f.endswith('txt')]
        for t in text_file:
            with open (f"{text_path}/{name}/{t}", 'r') as f:
                text_data = f.readline()
                text_data = text_data.split(":  ")[1].strip()
            clean_text = cleaner(text_data)
            lrs.write(f"{mono_path}/{name}_{t[:-4]}.wav|{name}_{t[:-4]}|{clean_text}|{text_data}\n")
