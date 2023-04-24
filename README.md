# Music Generation Transformer
## Introduction
This project use a transformer model to produce music. Transformer is a type of neural network architecture that exploits attention mechanism, and normally take a sequence of inputs and produce a sequence of output. In our project, input to the model will be a sequence of 250 midi notes and the output will be a prediction for the next note. 
## Data Source
The dataset that was being used was downloaded from <a href="https://colinraffel.com/projects/lmd/" target="_blank">The Lakh MIDI Dataset</a>. The model uses Clean MIDI subset.
## Data Split
Due to restriction in computational resources, we had to pick a random subset of 100 songs from Clean MIDI to work with. The chosen songs are stored in <a href="https://drive.google.com/drive/folders/1ffu0J6SJt_soSpeH1jP68LV0c-MUVdV2?usp=sharing" target="_blank">Google Drive</a>. 60% are training data, 20% are validation data, and the remaining is test data. We decided to split by songs instead of tokenizing the notes, then splitting train-validation-test because we would like to make sure no song would appear in both training and test set, since several parts of a song might have the same melody, and thus increase in test accuracy. This would not be a good indicator for how well our model performs on unseen data.
## Data Summary
Below is the summary of 99 songs' length initially:

Count| 99 |
--- | --- |
Mean | 8111 |
Std |7468|
Min | 430 |
25% | 4293 |
50%|  6161|
75%| 9450|
Max| 60878 |


According to the table, the weighted average song length (number of notes in a song) is 8111 while the median of the song length is 6161. This indicates that the data is heavily right skewed, which caused by extremely large observation. Moreover, the correct measure of central tendency here is median due to the existence of extreme outliers that affected both mean and standard deviation. Therefore, we decided to remove the song whose duration is exceptionally long so that the effect of this song (if it ends up in training set) will not outweigh the effect of the majority of other songs. Before removing outliers, we would like to check the histogram:

<img src="images/hist100.png" width="500">

After looking at the outliers, every song whose length is at least 20000 will be removed.

Below is the summary of 46 songs' length after removing outliers:

Count| 94 |
--- | --- |
Mean | 6899 |
Std | 4076 |
Min | 430 |
25% | 4111 |
50%| 6081|
75%| 8993 |
Max| 19514 |

The histogram for distribution of songs' length after removing outliers is illustrated below:

<img src="images/hist94.png" width="500">

Moreover, we should also look at the distribution for frequency of notes. This is because in the end, we would like to make sure our model performs at least better than a model always predicting the most frequent note. The mode (most frequent MIDI note) is 42, and the histogram of note frequency distribution is shown below:

<img src="images/note_freq94.png" width="500">

## Data Transformation

We first fetch every note from a specific song, then we form sequences of 250 notes each as inputs to our model, and the note immediately the $250^{th}$ note is used as ground truth. For example, one valid pair of input-output is (($1^{st}$ note, $2^{nd}$ note,..., $250^{th}$ note), $251^{st}$ note).

## Model Figure

Our transformer is a bit different than the standard transformer since we only output one note due to restricted computational resource. The number of times the encoder and decoder got repeated is 5.However, if the project is replicated to produce sequence of notes instead of one note, we can parsed in predicted output to the decoder and use masked multi head attention with a layer "Add and Norm" to combine with output from encode. 

<img src="images/Transformer.drawio.png" width="500">

Below is further details for Multihead Attention layer and Add and Norm layer:

For Multihead attention layer, the input is the output of model input embedding and position encoding layers, which give (Number of samples, Sequence Length, Embedding Size) shaped matrix. There are three distinct linear layers for query, key, and value since they have their own weights.

<img src="images/Attention.drawio.png" width="500">

The add step is a residual connection $F(x) + x$, and followed by normalization layer.

<img src="images/Norm.drawio.png" width="500">

The Linear layer is just applied the following formula
$$z = Wx + b$$

Softmax layer is the implementation of the following formula
$$y_i = \dfrac{e^{z_i}}{\sum e^{z_j}}$$

Then the predicted output would be the category that receives highest probability.
## Model Parameters

Below is the summary of number of parameters in our model:

Layer 1 : input_7 = 0 trainable weights

Layer 2 : embedding_6 = 65536 trainable weights

Layer 3 : layer_normalization_60 = 1024 trainable weights

Layer 4 : positional_encoding_6 = 0 trainable weights

Layer 5 : multi_head_attention_27 = 1046522 trainable weights

Layer 6 : add_54 = 0 trainable weights

Layer 7 : layer_normalization_61 = 1024 trainable weights

Layer 8 : dense_33 = 262656 trainable weights

Layer 9 : add_55 = 0 trainable weights

Layer 10 : layer_normalization_62 = 1024 trainable weights

Layer 11 : multi_head_attention_28 = 1046522 trainable weights

Layer 12 : add_56 = 0 trainable weights

Layer 13 : layer_normalization_63 = 1024 trainable weights

Layer 14 : dense_34 = 262656 trainable weights

Layer 15 : add_57 = 0 trainable weights

Layer 16 : layer_normalization_64 = 1024 trainable weights

Layer 17 : multi_head_attention_29 = 1046522 trainable weights

Layer 18 : add_58 = 0 trainable weights

Layer 19 : layer_normalization_65 = 1024 trainable weights

Layer 20 : dense_35 = 262656 trainable weights

Layer 21 : add_59 = 0 trainable weights

Layer 22 : layer_normalization_66 = 1024 trainable weights

Layer 23 : multi_head_attention_30 = 1046522 trainable weights

Layer 24 : add_60 = 0 trainable weights

Layer 25 : layer_normalization_67 = 1024 trainable weights

Layer 26 : dense_36 = 262656 trainable weights

Layer 27 : add_61 = 0 trainable weights

Layer 28 : layer_normalization_68 = 1024 trainable weights

Layer 29 : multi_head_attention_31 = 1046522 trainable weights

Layer 30 : add_62 = 0 trainable weights

Layer 31 : layer_normalization_69 = 1024 trainable weights

Layer 32 : dense_37 = 262656 trainable weights

Layer 33 : add_63 = 0 trainable weights

Layer 34 : layer_normalization_70 = 1024 trainable weights

Layer 35 : global_average_pooling1d_6 = 0 trainable weights

Layer 36 : dropout_6 = 0 trainable weights

Layer 37 : dense_38 = 65664 trainable weights


## Model Examples

The first example is from test set and the model predicted correctly:

Input sequence: [[76 74 50 59 53 57 59 53 57 50 66 62 74 61 73 66 62 50 57 38 74 66 62 59 35 59 59 35 62 74 38 69 71 61 73 74 62 50 66 62 35 38 59 62 35 49 35 61 66 49 35 62 54 59 74 62 74 62 66 74 59 35 59 35 74 62 62 74 66 35 62 54 47 59 62 59 53 54 59 53 59 66 62 74 61 74 62 62 62 35 59 66 66 59 62 59 59 57 47 74 62 62 74 54 66 35 62 59 38 66 59 53 38 62 59 54 59 53 38 66 38 47 35 66 59 59 35 66 62 59 54 66 71 57 59 53 66 62 59 53 59 54 62 71 59 30 73 66 62 59 35 59 59 35 35 54 54 66 59 62 30 73 35 76 59 53 35 59 53 35 54 66 59 62 54 59 59 66 59 62 62 74 54 35 76 74 59 53 62 59 53 59 62 59 54 59 69 57 35 59 35 61 59 35 69 57 59 71 62 59 74 54 35 66 35 38 73 66 59 53 38 62 59 53 38 59 54 59 71 57 69 62 38 62 35 54 59 35 59 59 35 61 54 59 62 47 35 57 69 57]]
  
prediction vs output: 71 71

Model prediction: 71

The second example is from test set and the model predicted incorrectly:

Input sequence: [[66 28 28 28 28 61 61 42 42 61 61 28 28 28 28 35 64 46 64 35 46 28 28 28 28 42 42 64 64 28 28 28 28 43 38 61 61 43 38 40 47 61 61 52 28 28 28 28 45 38 50 42 40 42 73 76 81 45 38 28 50 28 80 28 28 76 47 71 35 46 52 40 40 35 46 28 28 28 28 59 59 42 42 59 59 28 28 28 28 38 46 61 61 38 46 61 61 28 28 64 28 28 64 42 42 28 28 28 28 35 46 35 46 28 28 28 28 42 42 64 64 28 28 28 28 38 61 61 46 38 46 61 61 28 28 28 28 42 42 71 76 80 28 28 28 28 81 35 46 73 76 35 46 28 28 28 28 42 59 59 42 59 59 28 28 28 28 38 61 46 61 38 46 61 61 28 28 64 64 28 28 42 42 28 28 28 28 35 46 35 46 28 28 28 28 42 42 64 64 28 28 66 66 28 28 43 38 38 43 40 66 66 28 28 28 28 38 42 42 38 73 81 76 28 28 28 28 35 46 40 76 80 71 35 46 28 28 59 64 28 28 42 42 59 64 28 61 28 66 28]]
  
prediction vs output: 38 28

A sample audio that contains music generated by our model is in example/sample999.mid

## Training Curve

Below is the training curve of our final model where blue line represents performance on training set and orange represents model performance on validation set:

<img src="images/training_curve.png" width="500">
<img src="images/training_curve2.png" width="500">

## Hyperparameter Tuning

Our hyperparameter is learning rate, decay steps, decay rate, dropout rate, number of layers, number of heads, and model size. We tuned the hyperparameters by running the model with different combinations of hyperparmeters

Hyperparameters| Set of values |
--- | --- |
Model size | 128, 256, 384, 512 |
Number of Heads | 4,6,8|
Number of layers | 3,4,5|
Dropout rate | 0.1, 0.2, 0.3|
Learning rate | ($e^{-6}, e^{-4}$)|
Decay steps | 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000|
Decay rate | 0.9,0.91,0.92,...,0.99|

Finally, the best combination of hyperparameters is model size 512, number of heads 6, number of layers 5, dropout rate 0.1, learning rate, $5.6189e^{-5}$, decay steps 8000, decay rate 0.9
## Quantitative Measures

Loss function that was being used is sparse categorical cross entropy since output is an integer instead of one hot encoded.
$$L_{CE} = \sum_{i=1}^{N} t_i \log{y_i}$$
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Number of data points}}$$

## Quantitative and Qualitative Results

The final model training accuracy is 65.31% with training loss of 1.2071

The final model validation accuracy is 54.79% with validation loss of 1.7070

The test accuracy is 58.69%.

## Justification of Results

Suppose there is a model that always predict the most frequent note 42, then the test accuracy of the hypothetical model is 6% (calculate the number of times ground truth is 42 divides the number of test data points).

Therefore, the result obtain from our transformer model is acceptable. Moreover, predicting music is a difficult task since music depends on the artist, and different artist might have their own style, let alone if different music genre will have different melody. It is difficult to capture the pattern in art work since it offers lots of freedom and creativity. Moreover, the model was trained on mostly pop music, which means there is no guarantee the model will work well on other genres such as blues. The project can be extended to be a model whose input is sequence of notes, and output sequence of notes (until end of song) as well. This might be a better use of transformer since it makes use of decoder architecture that being parsed in the predicted output to generate even more notes. However, due to restricted computational resource, we could only build a model that predicts the next note given a sequence of 250 notes.

## Ethical Consideration
Since the model learned from existing data, so if someone use the model to generate music and make
money on it, this can be thought of as using some original artists’ work without rewarding or giving
credit to them. Moreover, AI generated music would also be unfair to the artists since they had to
spent a great amount of hours to create art whereas the model learnt from them and is able to create
art at a much faster rate. Therefore, this seems to invalidate real artists’ effort.

## Author
Ali: Data transformation, pair coding with Ramzi to write code for transformer
Ramzi: pair coding with Ali to write code for transformer, provide code for graphing model performance
Minh: Data exploration, write up readme file

## References
- Lab 5 from CSC311 Fall 2022
