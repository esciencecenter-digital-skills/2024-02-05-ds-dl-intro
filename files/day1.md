![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1
2024-02-01 Deep learning introduction

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-1)

Collaborative Document day 1: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-1)

Collaborative Document day 2: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-2)

Collaborative Document day 3: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-3)

Collaborative Document day 4: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-4)

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2024-02-05-ds-dl-intro/)

🛠 Setup

[link](https://esciencecenter-digital-skills.github.io/2024-02-05-ds-dl-intro/#setup)

Download files

- [Weather dataset](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)
- [BBQ labels](https://zenodo.org/record/4980359/files/weather_prediction_bbq_labels.csv?download=1)

## 👩‍🏫👩‍💻🎓 Instructors

Djura Smits, Sven van der Burg, Robin Richardson

## 🧑‍🙋 Helpers

Ewan Cahen 



## 🗓️ Agenda
| Time | Topic |
|--:|:---|
|09:00|Welcome and icebreaker|
|09:15|Introduction to Deep Learning|
|10:15|Coffee break|
|10:30|Introduction to Deep Learning|
|11:30|Coffee break|
|11:45|Classification by a Neural Network using Keras|
|12:45|Wrap-up|
|13:00|END|

## 🔧 Exercises

### Introduction to Deep Learning

#### Calculate the output for one neuron
Suppose we have:

- Input: X = (0, 0.5, 1)
- Weights: W = (-1, -0.5, 0.5)
- Bias: b = 1
- Activation function _relu_: `f(x) = max(x, 0)`

What is the output of the neuron?

_Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._


**Solution**

You can calculate the output as follows:

* Weighted sum of input: `0 * (-1) + 0.5 * (-0.5) + 1 * 0.5 = 0.25`
* Add the bias: `0.25 + 1 = 1.25`
* Apply activation function: `max(1.25, 0) = 1.25`

So, the neuron's output is `1.25`

#### Activation functions
Look at the following activation functions:

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/01_sigmoid.svg)
A. Sigmoid activation function

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/01_relu.svg)
B. ReLU activation function

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/01_identity_function.svg)
C. Identity (or linear) activation function

Combine the following statements with the correct activation function:

1. This function enforces the activation of a neuron to be between 0 and 1
2. This function is useful in regression tasks when applied to an output neuron
3. This function is the most popular activation function in hidden layers, since it introduces non-linearity in a computationally efficient way.
4. This function is useful in classification tasks when applied to an output neuron
5. (optional) For positive values this function results in the same activations as the identity function.
6. (optional) This function is not differentiable at 0
7. (optional) This function is the default for Dense layers (search the Keras documentation!)

**Solution**

1. A
2. C
3. B
4. A
5. B
6. B
7. C

#### Mean Squared Error
One of the simplest loss functions is the Mean Squared Error. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$ .
It is the mean of all squared errors, where the error is the difference between the predicted and expected value.
In the following table, fill in the missing values in the 'squared error' column. What is the MSE loss
for the predictions on these 4 samples?

| **Prediction** | **Expected value** | **Squared error** |
| -------------- | ------------------ | ----------------- |
| 1              | -1                 | 4                 |
| 2              | -1                 | ..                |
| 0              | 0                  | ..                |
| 3              | 2                  | ..                |
|                | **MSE:**           | ..                |


**Solution**

| **Prediction** | **Expected value** | **Squared error** |
|----------------|--------------------|-------------------|
| 1              | -1                 | 4                 |
| 2              | -1                 | 9                 |
| 0              | 0                  | 0                 |
| 3              | 2                  | 1                 |
|                | **MSE:**           | 3.5               |

#### Deep Learning Problems Exercise

Which of the following would you apply Deep Learning to?

1. Recognising whether or not a picture contains a bird.
2. Calculating the median and interquartile range of a dataset.
3. Identifying MRI images of a rare disease when only one or two example images available for training.
4. Identifying people in pictures after being trained only on cats and dogs.
5. Translating English into French.

**Solution**

1. and 5 are the sort of tasks often solved with Deep Learning.
2. is technically possible but solving this with Deep Learning would be extremely wasteful, you could do the same with much less computing power using traditional techniques.
3. will probably fail because there is not enough training data.
4. will fail because the Deep Learning system only knows what cats and dogs look like, it might accidentally classify the people as cats or dogs.

#### Deep Learning workflow exercise (in breakout rooms)

Think about a problem you would like to use Deep Learning to solve.

1. What do you want a Deep Learning system to be able to tell you?
2. What data inputs and outputs will you have?
3. Do you think you will need to train the network or will a pre-trained network be suitable?
4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you will use to train the network.




## Create the neural network
With the code snippets above, we defined a Keras model with 1 hidden layer with10 neurons and an output layer with 3 neurons.

* How many parameters does the resulting model have?
* What happens to the number of parameters if we increase or decrease the number of neurons in the hidden layer?



### (optional) Keras Sequential vs Functional API
So far we have used the Functional API of Keras. You can also implement neural networks using the Sequential model. As you can read in the documentation, the Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

* (optional) Use the Sequential model to implement the same network



## 🧠 Collaborative Notes

### About the eScience Center

More [eScience Center workshops](https://www.esciencecenter.nl/events/?f=workshops).

### Introduction to Deep Learning

#### What is Deep Learning?

Part of AI, subset of machine learning. Machine learning is about algorithms that learn patterns form taking in a lot of data. More data -> hopefully solve problems better.

Deep learning got more popular with hardware performance getting better, as it is very compute intensive. More layers means more weights to determine.

A definition of AI could be the study of creating something that resembles or is better than human intelligence.

Deep learning is all about neural networks, loosely based on how neurons in the human brain work.

A neuron has a bunch of inputs (pictures, records of a table) which have to be numbers. These numbers will be weighted. Sum these weighted inputs and add a constant bias. Apply an activation function to the result. This activation function is non-linear.

A neural network consists of layers of neurons, one input layer, one output layer and one or more hidden layers. The output from one layer is input for the next layer.

Neural networks learn from a loss function, which says how good or bad the outcome of the network was.

Deep learning is good for:

* pattern/object recognition, e.g. recognising dogs or cats in pictures
* segmenting images (or any data)
* translating between sets of data, e.g. natural languages
* create new data that looks like training data, e.g. art, deepfake videos

Deep learning is *not* good when

* small amount of training data available
* when an explanation is required on how the answer was derived
* classifying things that don't look like the training data

Deep learning is more interpolation than extrapolation.

Deep learning is overkill when

* logic operations, e.g. averages, ranges, [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz) (but it [can be done](https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow))
* modelling well defined systems
* basic computer vision tasks, e.g. blurring an image

To increase the amount of data you have, you can augment the existing data, like mirroring or rotating image. You can also use an existing trained network.

Deep learning workflow:

1. Formulate/ Outline the problem
2. Identify inputs and outputs
3. Prepare data
4. Choose a pre-trained model or build a new architecture from scratch
5. Choose a loss function and optimizer
6. Train the model
![](https://raw.githubusercontent.com/carpentries-incubator/deep-learning-intro/main/episodes/fig/training-0_to_1500.svg)
7. Perform a Prediction/Classification
8. Measure Performance
9. Refine the model
10. Share Model

#### Testing your setup

We work with Jupyter Lab.

Activate your conda evironment:

```bash
conda activate dl_workshop
```

Start Jupyter Lab:

```bash
jupyter lab
```

Open a notebook and test the dependencies:

```python
import sklearn
print('sklearn version: ', sklearn.__version__)

import seaborn
print('seaborn version: ', seaborn.__version__)

import pandas
print('pandas version: ', pandas.__version__)

import tensorflow
print('Tensorflow version: ', tensorflow.__version__)
```

TensorFlow might generate warnings about not having a GPU, this is fine for this workshop.

### Classification by a Neural Network using Keras

#### 2. Identify Inputs and Outputs
```python
import seaborn as sns
```

```python
penguins = sns.load_dataset('penguins')
```

```python=
penguins.head()
```

```python=
penguins.shape
```

```python=
sns.pairplot(penguins, hue="species")
```

From the plot we see that Gentoo is clearly distinguishable from the other two classes on a number of features.

Model inputs and outputs:

input: bill length, bill depth, flipper length and body mass
output: species

#### 3. Prepare data

```python=
penguins_filtered = penguins.drop(columns=['island', 'sex'])
```

```python=
# Clean missing values (Drop rows with any NaN values)
penguins_filtered = penguins_filtered.dropna()
```

```python=
# Extract the columns corresponding to features
features = penguins_filtered.drop(columns=['species'])
```

```python=
import pandas as pd
```

Get the target as one-hot encoding
```python=
target = pd.get_dummies(penguins_filtered['species'])
```

How many output neurons do we need for our network?
3 output neurons (corresponding to each type of penguin)


```python=
from sklearn.model_selecion import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
```

```python=
X_train.shape
```

#### 4. Build an architecture from scratch (or choose a pretrained model)

```python=
from tensorflow import keras
```

```python=
from numpy.random import seed
seed(1)

keras.utils_set_random_seed(2)
```

```python=
# input layer
inputs = keras.Input(shape=X_train.shape[1])
```

```python=
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
```

```python=
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
```

```python=
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()
```


### Sven's summary:
* Did everyone get the instruction email only on saturday? And not on the monday the week before the workshop? Or only people that registered on short notice?
* Indeed, some questions in the exercises were a bit unfair, because we didn't explain everything. Feel free to ask for clarification!
* We will try to keep the pace from yesterday's live coding part 


## 📚 Resources

* [TensorFlow playground](https://playground.tensorflow.org/)
* [Detecting COVID-19 in chest X-ray images](https://arxiv.org/abs/2003.09871)
* [Forecasting building energy load](https://ieeexplore.ieee.org/document/7793413)
* [Protein function prediction](https://pubmed.ncbi.nlm.nih.gov/29039790/)
* [Simulating Chemical Processes](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)
* [Help to restore ancient murals](https://heritagesciencejournal.springeropen.com/articles/10.1186/s40494-020-0355-x)

