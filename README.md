# Vehicle Damage Classification for Insurance Claim Verification

## Overview

Insurance claim processing is time-consuming when done manually. This project builds a Convolutional Neural Network (CNN) that classifies vehicle damage type from a single photo, supporting automated triage of insurance claims at submission.

The model classifies damage into six categories:

| Category | Description |
|---|---|
| Crack | Surface crack in bodywork |
| Scratch | Paint scratch |
| Dent | Body dent from impact |
| Tire flat | Flat or punctured tyre |
| Glass shatter | Shattered windscreen or window |
| Lamp broken | Broken headlamp or tail light |

## Project Structure

```
vehicle-damage-classifier/
│
├── models/
│   ├── model_1.h5
│   ├── model_2.h5
│   ├── model_3.h5
│   └── model_4.h5          
│
├── sample_images/                
│   └── one_sample_per_cat.png
│
├── notebooks/
│   └── vehicle_damage_cnn.ipynb 
│
├── requirements.txt
└── README.md
```

## Model

### Architecture

A three-block CNN trained from scratch:

```
Input (150×150×3)
  └─ Conv2D(32, 3×3, ReLU) + MaxPool(2×2)
  └─ Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
  └─ Conv2D(128, 3×3, ReLU) + MaxPool(2×2)
  └─ Flatten → Dense(64, ReLU) → Dropout(0.3) → Dense(6, Softmax)
```


Four configurations were trained and compared. 

| Model | Optimiser | Batch Size | Dropout | Early Stopping | Batch Norm |
|---|---|---|---|---|---|
| M1 | SGD (lr=0.01, mom=0.9) | 32 | 0.5 | No | No |
| M2 | Adam (lr=0.001) | 32 | 0.3 | Yes (patience=5) | No |
| M3 | SGD (lr=0.01, mom=0.9) | 64 | 0.3 | Yes (patience=5) | No |
| M4 | Adam (lr=0.001) | 32 | 0.3 | Yes (patience=5) | Yes |


## Dataset

**Vehicle Damage Insurance Verification:** available on [Kaggle](https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest).

- ~7,200 labelled training images across 6 classes
- 80/20 train/validation split 
- Training-time augmentation: rotation, horizontal flip, brightness variation, zoom


## Future Work
Future improvement would likely require transfer learning with a pretrained architecture such as ResNet, EfficientNet or MobileNet, combined with stronger data augmentation and class-balancing techniques.

