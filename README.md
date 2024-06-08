# FVTTS : Face-based Voice Synthesis for Text-to-Speech

This repository is the official implementation of FVTTS.

## Preprocessing
**1. Prepare your data into `'data'`. Build your dataset by setting up the following directory structure:**

```
data
├── trainval                   
|   ├── speaker1        # i.e. tMSU6k5SWXg
|   |   ├── text1.txt           # text file of first utterance  (i.e. 50001.txt)
|   |   └── video1.mp4          # video file of first utterance (i.e. 50001.mp4)
|   |   ├── text2.txt           # text file of second utterance  (i.e. 50002.txt)
|   |   └── video2.mp4          # video file of second utterance (i.e. 50002.mp4)
|   |   |
|   |   |
|   ├── speaker2
|   |   
├── test

```

**2. Prepare `shape_predictor_68_face_landmarks.dat` for facial landmark detection.**

```
wget   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 

bunzip2 /content/shape_predictor_68_face_landmarks.dat.bz2
```


**3. Preprocess the data by run following codes.**
   
```
python video2img.py --video_dir data/trainval --img_path data/image --emb_path data/img_emb --landmark_path shape_predictor_68_face_landmarks.dat # split image from video

python video2audio.py --video_dir data/trainval --audio_path data/audio --mono_path data/mono  # split audio from video

python audio2spec.py --mono_path data/mono --spec_path data/spec # calculate mel-spectrogram of each audio sample
```

**4. Generate the file 'training.txt' for training.**
   
```
python file.py --text_path data/trainval --mono_path data/mono --file_name LRS3_valid.txt.cleaned
```

## Train
After prepare and preprocess the data, train the model on your data

```
python FVTTS.py --training_files 'LRS3_valid.txt.cleaned' --validation_files 'LRS3_test.txt.cleaned
```

## Inference
For inference you need to prepare the face image of speakers.

See `inference.ipynb` for the examples of inference. 
