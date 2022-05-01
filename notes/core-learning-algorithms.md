# Core Learning Algorithms

###### tags: `FreeCodeCamp-MLwithPython`

* [Linear Regression](#linear-regression)
    * [Training and Testing Data](#training-and-testing-data)
    * [The Training Proces](#the-training-process)
* [Classification](#classification)
    * [Building the Model](#building-the-model)
* [Clustering](#clustering)
* [Hidden Markov Models](#hidden-markov-models)
* [Sources](#sources)

---

## Linear Regression

Linear regression is one of the most basic forms of machine learning and is used to predict numeric values. 
It follows a simple concept: If data points are related linearly, we can generate a line of best fit for these points and use it to predict future values.

An example of data set:

![](https://i.imgur.com/I19exJ1.png)

"Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points." (https://www.investopedia.com/terms/l/line-of-best-fit.asp)

A line of best fot for this graph:

![](https://i.imgur.com/ZdV8wCC.png)

Once we've generated this line for our dataset, we can use its equation to predict future values. We just pass the features of the data point we would like to predict into the equation of the line and use the output as our prediction.


Tutorial of using a linear model to predict the survival rate of passengers from the titanic dataset: click [here](https://www.tensorflow.org/tutorials/estimator/linear).

### Training and Testing Data

When we train models, we need two sets of data: training and testing.

The training data is what we feed to the model so that it can develop and learn. It is usually a much larger size than the testing data.

The testing data is what we use to evaulate the model and see how well it is performing. We must use a seperate set of data that the model has not been trained on to evaluate it.

This is because we need our model to be able to make predictions on NEW data, data that we have never seen before. If we simply test the model on the data that it has already seen we cannot measure its accuracy accuratly. We can't be sure that the model hasn't simply memorized our training data.

### The Training Process

We will not feed the entire dataset to our model at once, but simply small batches of entries. We will feed these batches to our model multiple times according to the number of epochs.

An epoch is simply one stream of our entire dataset. The number of epochs we define is the amount of times our model will see the entire dataset. We use multiple epochs in hope that after seeing the same data multiple times the model will better determine how to estimate it.

## Classification

Classification is used to seperate data points into classes of different labels.

Tutorial of classification: click [here](https://www.tensorflow.org/tutorials/estimator/premade).


### Building the Model

For classification tasks there are variety of different estimators/models that we can pick from. TensorFlow provides several pre-made classifier Estimators, including:

* **Linear Classifier** `tf.estimator.LinearClassifier` for classifiers based on linear models.
* **DNN Classifier** `tf.estimator.DNNClassifier` for deep models that perform multi-class classification.
* **DNN Linear Combined Classifier** `tf.estimator.DNNLinearCombinedClassifier` for wide & deep models.


## Clustering

Clustering is a Machine Learning technique that involves the grouping of data points. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. (https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

Basic Algorithm for K-Means.

* Step 1: Randomly pick K points to place K centroids.
* Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
* Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
* Step 4: Reassign every point once again to the closest centroid.
* Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.


## Hidden Markov Models

"The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional) probability distribution []. Transitions among the states are governed by a set of probabilities called transition probabilities." (http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html)

A hidden markov model works with probabilities to predict future events or states. In this section we will learn how to create a hidden markov model that can predict the weather.

Tutorial of Hidden Markov Models: click [here](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel).

Components of a markov model:

* **States**: In each markov model we have a finite set of states. These states could be something like "warm" and "cold" or "high" and "low" or even "red", "green" and "blue". These states are "hidden" within the model, which means we do not direcly observe them.
* **Observations**: Each state has a particular outcome or observation associated with it based on a probability distribution. An example of this is the following: On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.
* **Transitions**: Each state will have a probability defining the likelyhood of transitioning to a different state. An example is the following: a cold day has a 30% chance of being followed by a hot day and a 70% chance of being follwed by another cold day.

## Sources

1. Chen, James. “Line Of Best Fit.” Investopedia, Investopedia, 29 Jan. 2020, www.investopedia.com/terms/l/line-of-best-fit.asp.
2. “Tf.feature_column.categorical_column_with_vocabulary_list.” TensorFlow, www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list?version=stable.
3. “Build a Linear Model with Estimators &nbsp;: &nbsp; TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/estimator/linear.
4. Staff, EasyBib. “The Free Automatic Bibliography Composer.” EasyBib, Chegg, 1 Jan. 2020, www.easybib.com/project/style/mla8?id=1582473656_5e52a1b8c84d52.80301186.
5. Seif, George. “The 5 Clustering Algorithms Data Scientists Need to Know.” Medium, Towards Data Science, 14 Sept. 2019, https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68.
6. Definition of Hidden Markov Model, http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html.
7. “Tfp.distributions.HiddenMarkovModel &nbsp;: &nbsp; TensorFlow Probability.” TensorFlow, www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel.
