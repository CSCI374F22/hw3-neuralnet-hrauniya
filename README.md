# Neural Network Experiments
Building a single layer artificial neural network with backpropagation algorithm.

## Research Analysis

 **1) For the monks1.csv data set, pick a training set percentage and a random seed (document both in your README). Use 2 hidden neurons, a learning rate n = 0.1, and a threshold of 0.5.**

Program input: python neuralnet.py monks1.csv 2 0.1 0.5 2250 0.5 

(training_set = 0.5, random_seed = 2250)

**a. What is the test set accuracy you observed for your neural network?**
--> accuracy test set of neural network: 0.9537

**b. Repeat using the same training set percentage, random seed, and learning rate but
with your logistic regression solution from Homework 2. What is the test set
accuracy you observed for your logistic regression model on the monks1.csv data
set?**
--> accuracy test set of logistic: 0.7222

**c. Create 95% confidence intervals for both accuracies. Which learner (neural
network or logistic regression) had the highest accuracy? Was the difference
statistically significant? What are the implications of your results?**

--> Confidence Interval (test_set size = 108)

neural network = [0.9141, 0.9933]

logistic = [0.6377, 0.8070]

The neural network had the highest accuracy. The difference was pretty statistically significant, since the neural network had a confidence interval from 91% to 99% while the logistic interval was only between 63% and 80% in terms of accuracy.

**2.For the mnist_5v8.csv data set, pick a random seed. Use a learning rate of n = 0.001, a
training set percentage of 60%, and a threshold of 0.5. Create a neural network with each
of the following numbers of neurons: 2, 5, 10, 20, and 50**

**a) What is the test set accuracy you observed for each number of neurons? Plot a
line chart (using the tool of your choice: Excel, R, matplotlib in Python, etc.) of
the test set accuracy as the number of neurons increased. Include your line chart
as an image in your GitHub repository**

-->With a random seed of 12345,learning rate of 0.01, training set percentage of 60%, and threshold of 0.5, the following were the accuracies achieved on the test given the number of hidden neurons.

2 hidden neurons: 0.825
5 hidden neurons:0.85
10 hidden neurons:0.90
20 hidden neurons:0.925
50 hidden neurons:0.925

A line chart observing this trend is attached.

**b) How did the accuracy change as the number of hidden neurons change? Why do you think this result occurred?
--> The accuracy for the test set became greater as the number of neurons in the hidden layer was increased. As we can see when there were 2 hidden neurons the accuracy was 0.825 but the accuracy gradually increases as the number of neurons is increased to 20. The accuracy is at 0.925 at 20 hidden neurons, and it does not change when the accuracy is at 50.**

This behavior occurs because increasing the number of neurons lets the network learn more underlying patterns to predict correctly. For example, for the mnist_5v8.csv dataset, increasing the number of neurons may allow the network to learn the smaller shapes that give rise to the number 5 and 8. Learning the build up of the shape of the numbers will allow the network to predict correctly.

**3.For the mnist_5v8.csv data set, use the three learning rates n = 0.001, 0.01, 0.1. Use the
number of neurons that gave the highest accuracy in Q2 (in case of ties, use the smallest
number of neurons that tied for the highest accuracy), a training percentage of 60%, a
threshold of 0.5, and the same random seed used in Q2. Track the accuracy on both the
training set the validation set after each epoch of Backpropagation (i.e., after you feed
the entire training set in).**

**a) Plot the accuracy of the network on the training set for each epoch on a single line
chart (again using your favorite tool) for all three learning rates (each learning
rate should be a separate line on the same chart, where the x-axis is the epoch and
the y-axis is the training set accuracy) . Include your line chart as an image in
your GitHub repository**
--> uploaded to github

**b)Plot the accuracy of the network on the validation set for each epoch on a single
line chart (again using your favorite tool) for all three learning rates (each
learning rate should be a separate line on the same chart, where the x-axis is the
epoch and the y-axis is the validation set accuracy) . Include your line chart as an
image in your GitHub repository.**
--> uploaded to github

**c)Compare the training set accuracy across the three learning rates. What trends do
you observe in your line charts? What do you think this implies about choosing a
learning rate?**
--> The accuracy for the 0.001 starts somewhere above 0.7, but then decreases and increases like a hockey stick. This learning rate has the highest starting accuracy. However, its final accuracy is still lower than the other two learning rates.The learning rate with 0.01 starts somewhere below 0.6 and gradually increases. After it hits 0.9 accuracy it’s slope decreases but it still comes to a stand-still at an accuracy of 1. The learning rate of 0.1 reaches a high accuracy very fast leading to a very high slope, but its slope also gradually decreases to a 1. The higher the learning rate, the faster it reaches the highest accuracy in the long run. This might imply that since we get an accuracy of 1 fastest using the learning rate of 0.1, we should use it. 0.001  is well below one so we can’t use that, and 0.01 does not train as fast as 0.1.

**d)Compare the validation set accuracy across the three learning rates. What trends
do you observe in your line charts? What do you think this implies about
choosing a learning rate?**
--> The accuracy for 0.1 is similar to before where it achieves a high accuracy pretty fast with a big initial slope. It reaches an accuracy of 0.95, but decreases, and settles at an accuracy of 0.925. The learning rate of 0.01 has a rate of increase/slope slightly less than the one with 0.1, and also reaches a max accuracy of 0.95 although it settles at 0.925. The learning rate of 0.001 starts at an accuracy of 0.7 but decreases. However, it also slowly  grows, and converges to 0.925. This might imply that the learning rate may not matter for this case as all 3 learning rates have a similar accuracy at the end. We might want to use 0.01 since it will be faster than 0.001 and has less chances of settling for less optimal minima

**4. For the seismic.csv data set, use 10 hidden neurons, a learning rate n = 0.01, a training
percentage of 60%, and your favorite random seed. Using five different thresholds
(0.05, 0.1, 0.5, 0.9, 0.95) for converting the predicted probabilities into predicted labels,
calculate the accuracy and recalls of your trained neural network on the test set**

**a)What were the test set accuracies you observed for each threshold value? How
did they change as the threshold changed?**
--> Five different thresholds (random seed 1000)

accuracy on test set (0.05 threshold) = 0.4894

accuracy on test set (0.1 threshold) = 0.7524

accuracy on test set (0.5 threshold) = 0.9246

accuracy on test set (0.9 threshold) = 0.9246

accuracy on test set (0.95 threshold) = 0.9246

As the threshold increased the overall accuracy also increased. 

**b)What were the recalls on each label that you observed for each threshold value?
How did they change as the threshold changed?**
--> Recalls

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

**c) Remembering that a label of 1 is predicting a seismic event, and a label of 0 is
predicting that there is no seismic event, which threshold do you think is ideal for
this data set? Why do you think this threshold is the best?**

--> The threshold of 0.05 would be the best at predicting seismic events since it had the highest recall_1 at 0.87. This means that 87% of the times it predicted there would be a seismic event it was correct. The downside to this is that the overall accuracy is only 48% so there may be many false positive predictions, however its better to have a false positive than predict incorrectly. 

----
 Harsha Rauniyar and Austin Alcancia
