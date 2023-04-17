# Music Generation Transformer
## Introduction
This project use a transformer model to produce music. Transformer is a type of neural network architecture that exploits attention mechanism, and normally take a sequence of inputs and produce a sequence of output. In our project, input to the model will be a sequence of 40 midi notes and the output will be a prediction for the next note. 
## Data Source
The dataset that was being used was downloaded from <a href="https://colinraffel.com/projects/lmd/" target="_blank">The Lakh MIDI Dataset</a>. The model uses Clean MIDI subset.
## Data Split
Due to restriction in RAM, we had to pick a random subset of 51 songs from Clean MIDI to work with. The chosen songs are stored in <a href="https://drive.google.com/drive/folders/1ffu0J6SJt_soSpeH1jP68LV0c-MUVdV2?usp=sharing" target="_blank">Google Drive</a>. 60% are training data, 20% are validation data, and the remaining is test data. We decided to split by songs instead of tokenizing the notes, then splitting train-validation-test because we would like to make sure no song would appear in both training and test set, since several parts of a song might have the same melody, and thus increase in test accuracy. This would not be a good indicator for how well our model performs on unseen data.
## Data Summary
Below is the summary of 99 songs' length initially:

Count| 51 |
--- | --- |
Mean | 7683.37 |
Std | 9038.92 |
Min | 430 |
25% | 3548.5 |
50%| 5360 |
75%| 8489 |
Max| 60878 |


According to the table, the weighted average song length (number of notes in a song) is 7683.37 while the median of the song length is 5360. This indicates that the data is heavily right skewed, which caused by extremely large observation. Moreover, the correct measure of central tendency here is median due to the existence of extreme outliers that affected both mean and standard deviation. Therefore, we decided to remove the song whose duration is exceptionally long so that the effect of this song (if it ends up in training set) will not outweigh the effect of the majority of other songs. Before removing outliers, we would like to check the histogram:

<img src="images/hist.png" width="500">

Two methods for detecting outliers were being used were detecting outliers
by z-score and detecting outliers by IQR. In the first method, any data point having z-score above 3
or below 3 would be categorized an outlier. In the second method, any data point outside the interval
(Q1 − 1.5IQR, Q3 + 1.5IQR) is considered an outlier. However, the first method did not work since the
number of outlier based on that method is 1 only. This is due to extreme outlier that increased the
mean, and enlarged variance. The second method worked better since IQR is robust to outliers. After
looking at the outliers, every song whose length is above Q3 + 1.5IQR of all songs' length will be removed.

Below is the summary of 46 songs' length after removing outliers:

Count| 46 |
--- | --- |
Mean | 5473.34 |
Std | 3038.59|
Min | 430 |
25% | 3446 |
50%| 5036 |
75%| 7173 |
Max| 12890 |

The histogram for distribution of songs' length after removing outliers is illustrated below:

<img src="images/hist-after-outliers.png" width="500">

Moreover, we should also look at the distribution for frequency of notes. This is because in the end, we would like to make sure our model performs at least better than a model always predicting the most frequent note. The mode (most frequent MIDI note) is 42, and the histogram of note frequency distribution is shown below:

<img src="images/note_freq.png" width="500">

## Data Transformation

Since there are 128 possible notes on a MIDI device, and they range from 0 to 127, we created one hot vector with 128 entries.For example, a middle C is a note number 60 will be denoted by number 1 at index 60 and 0 everywhere else. Moreover, a note 0 never appears in any of the song, so we used a note 0 (numbered 1 at the first index and 0 everywhere else) to indicate the end of a song. 

We first fetch every note from a specific song, then we form sequences of 40 notes each as inputs to our model, and the note immediately the $40^{th}$ note is used as ground truth. For example, one valid pair of input-output is (($1^{st}$ note, $2^{nd}$ note,..., $40^{th}$ note), $41^{st}$ note).

Next, we convert each note in our input into one hot vector. Therefore, we have $40$ notes $\times$ $128$ entries per note = 5120 entries for one input data point. We implement the above process for every training points in our training set. 

## Model Figure

Insert model diagram here

## Model Parameters

Count parameters here, where the parameters come from

## Model Examples

The first example is from test set and the model predicted correctly:

The second example is from test set and the model predicted incorrectly:

## Training Curve

Below is the training curve of our final model where blue line represents performance on training set and orange represents model performance on validation set:

INSERT TRAINING CURVE HERE

## Hyperparameter Tuning

Our hyperparameter is .... We tuned the hyperparameters by running the model with NUMBER OF different combinations of hyperparmeters that includes .... 

## Quantitative Measures

Loss Cross Entropy was being used to measure the error of the model since the output is categorical (an integer from 0 to 127), and percentage of correct prediction is used as accuracy rate.
$$L_{CE} = -\sum_{i = 1}^{n}t_i \log{y_i}$$
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Number of data points}}$$

## Quantitative and Qualitative Results

The best training accuracy is.... with training loss of ....

The best validation accuracy is.... with training loss of ....

The test accuracy is ....


## Justification of Results

Suppose there is a model that always predict the most frequent note (42), then the test accuracy of the hypothetical model is ...

Therefore, the result obtain from our transformer model is acceptable. Moreover, predicting music is a difficult task since ...

## Ethical Consideration
Since the model learned from existing data, so if someone use the model to generate music and make
money on it, this can be thought of as using some original artists’ work without rewarding or giving
credit to them. Moreover, AI generated music would also be unfair to the artists since they had to
spent a great amount of hours to create art whereas the model learnt from them and is able to create
art at a much faster rate. Therefore, this seems to invalidate real artists’ effort.

LIMITATION OF THE MODEL AND TRAINING DATA.
## Author
Ali: Data transformation, provided the starting code for transformer\
Ramzi: Worked on fixing the code for positional encoding\
Minh: Data exploration, write up readme file
