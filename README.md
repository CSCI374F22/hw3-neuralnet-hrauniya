# Austin Alcancia and Harsha Rauniyar

1. 

Program input: python neural.py monks1.csv 2 0.1 0.5 2250 0.5 

(training_set = 0.5, random_seed = 2250)

a. accuracy test set of neural network: 0.9537

b. accuracy test set of logistic: 0.7222

c. Confidence Interval

neural network = [0.9125, 0.9949]

logistic = [0.6344, 0.8010]

The neural network had the highest accuracy. The difference was pretty statistically significant, since the neural network had a confidence interval from 91% to 99% while the logistic interval was only between 63% and 80% in terms of accuracy.

4. 

a. Five different thresholds (random seed 1000)

accuracy on test set (0.05 threshold) = 0.4894

accuracy on test set (0.1 threshold) = 0.7524

accuracy on test set (0.5 threshold) = 0.9246

accuracy on test set (0.9 threshold) = 0.9246

accuracy on test set (0.95 threshold) = 0.9246

As the threshold increased the overall accuracy also increased. 

b. Recalls

0.05 threshold:
recall_0 = 0.4582
recall_1 = 0.8718

0.1 threshold
recall_0 = 0.7741
recall_1 = 0.4871

0.5 threshold 
recall_0 = 0.9246
recall_1 = 0

0.9 threshold 
recall_0 = 0.9246
recall_1 = 0

0.95 threshold 
recall_0 = 0.9246
recall_1 = 0

As the threshold increased, recall_0 became more accurate, however recall_1 had a drastic decrease in accuracy. This is most likely due to the predictions being under the higher thresholds, therefore recall_1 decreases as threshold increases.

c. I think the threshold of 0.05 would be the best at predicting seismic events since it had the highest recall_1 at 0.87. This means that 87% of the times it predicted there would be a seismic event it was correct. The downside to this is that the overall accuracy is only 48% so there may be many false positive predictions, however its better to have a false positive than predict incorrectly. 