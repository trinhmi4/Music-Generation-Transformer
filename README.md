# Music Generation Transformer
## Introduction
This project use transformer architecture to produce music. Input to the model will be a sequence of 40 midi notes and the output will be prediction for the next note. 
## Data Source
The dataset that was being used was downloaded from <a href="https://colinraffel.com/projects/lmd/" target="_blank">The Lakh MIDI Dataset</a>. The model uses Clean MIDI subset.
## Data Split
Due to restriction in RAM, we had to pick a random subset of 100 songs from Clean MIDI to work with. The chosen songs are stored in <a href="https://drive.google.com/drive/folders/1ffu0J6SJt_soSpeH1jP68LV0c-MUVdV2?usp=sharing" target="_blank">Google Drive</a>. Out of 100 songs, 60 are training data, 20 are validation data, and the remaining is test data. We decided to split by songs instead of tokenizing the notes, then splitting train-validation-test because we would like to make sure no song would appear in both training and test set, since several parts of a song might sound similar, and thus increase in test accuracy. This would not be a good indicator for how well our model performs on unseen data.
## Data Summary
## Data Transformation
## Model Figure
## Model Parameters
## Model Examples
## Training Curve
## Hyperparameter Tuning
## Quantitative Measures
## Quantitative and Qualitative Results
## Justification of Results
## Ethical Consideration
Since the model learned from existing data, so if someone use the model to generate music and make
money on it, this can be thought of as using some original artists’ work without rewarding or giving
credit to them. Moreover, AI generated music would also be unfair to the artists since they had to
spent a great amount of hours to create art whereas the model learnt from them and is able to create
art at a much faster rate. Therefore, this seems to invalidate real artists’ effort.
## Author
Ali: Data transformation, provided the starting code for transformer
Ramzi: Worked on fixing the code for positional encoding
Minh: Data summary, write up readme file
