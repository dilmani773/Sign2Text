# Sign2Text

Translating American Sign Language (ASL) into English text from body, hand, and face keypoints.

## Overview

The project has two parts:

1. **Research baseline.** Sentence-level ASL-to-English translation on How2Sign, using MediaPipe Holistic keypoints, a Transformer pose encoder, and a pretrained T5 decoder.
2. **Live demo (in progress).** A webcam app that recognizes a fixed set of 20 everyday ASL phrases, reusing the encoder trained on How2Sign.

## Pipeline

```
webcam / video ─► MediaPipe Holistic (543 keypoints per frame)
               ─► 85 selected keypoints, shoulder-normalized, hand shape features (384 per frame)
               ─► Transformer pose encoder
               ─► T5 decoder ─► English text
```

## Data

[How2Sign Holistic](https://www.kaggle.com/datasets/psewmuthu/how2sign-holistic): MediaPipe Holistic keypoints for How2Sign (Duarte et al., CVPR 2021), about 31k sentence clips.

Findings from preparing the data:
- Keypoint order in this dataset is pose [0:33], face [33:501], left hand [501:522], right hand [522:543]. Confirmed by plotting and by the hand wrist z = 0 check.
- Hips and legs are outside the video frame, so normalization uses the shoulders.
- About 6% of clips were removed: misaligned (too few frames for the sentence), over 40 words, or over 900 frames.

## Results (How2Sign test set, BLEU-4)

| Run | Changes | Val BLEU | Test BLEU |
|-----|---------|----------|-----------|
| 1 | Baseline: 4-layer encoder + t5-small | 2.03 | 1.67 |
| 2 | 6-layer encoder, bag-of-words helper loss, T5 frozen for 3 epochs, no-repeat decoding | 1.94 | 2.02 |

How2Sign is hard: the English sentences are loose translations, with 16k words, and 40% of them appear only once. Published systems that reach higher scores usually pretrain on much larger datasets first.

## Repository layout

```
training/   data pipeline, model, training script (run on Kaggle GPU)
demo/       webcam phrase recorder (run locally)
results/    training history and test predictions per run
```

## Usage

**Training (Kaggle, GPU T4, Internet on):**
```python
!git clone https://github.com/dilmani773/Sign2Text.git /kaggle/working/code
!pip install -q sacrebleu
!cd /kaggle/working && python code/training/s2t_phase5_train.py > train_log.txt 2>&1
```

**Tests (no GPU needed):**
```
python training/s2t_phase1_dataset.py
python training/s2t_phase34_model.py
python training/s2t_phase5_train.py --smoke
python demo/record_phrases.py --selftest
```

**Recording phrases (laptop with webcam, Python 3.10 to 3.12):**
```
pip install -r demo/requirements.txt
python demo/record_phrases.py
```

## Author

Hiyumi Dilmani Suriyapperuma (Inaaya), Computer Engineering, University of Peradeniya.
