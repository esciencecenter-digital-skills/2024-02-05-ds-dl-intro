![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2
2024-02-01 Deep learning introduction

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-2)

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

Djura Smits, Sven van den Burg, Robin Richardson

## 🧑‍🙋 Helpers

Ewan Cahen 


## 🗓️ Agenda
| Time | Topic |
|--:|:---|
|09:00|Welcome and icebreaker|
|09:15|Classification by a Neural Network using Keras|
|10:15|Coffee break|
|10:30|Classification by a Neural Network using Keras|
|11:30|Coffee break|
|11:45|Monitor the training process|
|12:45|Wrap-up|
|13:00|END|

## ⛸ Welcome and icebreaker
* Yesterday's feedback
* Icebreaker: What is something you are proud of?
* Recap yesterday
    * What is deep learning?
    * How are neural networks build up?
    * How do I build a neural network from scratch in Keras?
    * For a single neuron: $\text{output} = f(\sum_{i} (x_i*w_i) + \text{bias})$, where $f$ is an activation function, $w_i$ is the weight of input $i$ and $x_i$ is the input value.

## 🔧 Exercises

### Classification by a Neural Network using Keras

#### The Training Curve
Looking at the training curve we have just made.

1. How does the training progress?
   * Does the training loss increase or decrease?
   * Does it change quickly or slowly?
   * Does the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?

When the training process does not go well:

3. (optional) Something went wrong here during training. What could be the problem, and how do you see that in the training curve?
Also compare the range on the y-axis with the previous training curve.
![](https://codimd.carpentries.org/uploads/upload_9a7323086d120cf1de9be691a79ca64e.png)




**Solution:**

1. The training loss decreases quickly. It drops in a smooth line with little jitter.
This is ideal for a training curve.
2. The results of the training give very little information on its performance on a test set.
  You should be careful not to use it as an indication of a well trained network.
3. (optional) The loss does not go down at all, or only very slightly. This means that the model is not learning anything.
It could be that something went wrong in the data preparation (for example the labels are not attached to the right features).
In addition, the graph is very jittery. This means that for every update step,
the weights in the network are updated in such a way that the loss sometimes increases a lot and sometimes decreases a lot.
This could indicate that the weights are updated too much at every learning step and you need a smaller learning rate
(we will go into more details on this in the next episode).
Or there is a high variation in the data, leading the optimizer to change the weights in different directions at every learning step.
This could be addressed by presenting more data at every learning step (or in other words increasing the batch size).
In this case the graph was created by training on nonsense data, so this a training curve for a problem where nothing can be learned really.

#### Confusion Matrix
You measured the performance of the neural network you trained by
visualizing a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?

**Your answers:**

* Room 1:
    1. 
    2. 
    3. 
* Room 2:
    1.  No 
    2.  No
    3. More variables (dataset), different way of spliting the data, increase epochs.
* Room 3:
    1. 
    2. 
    3. 


### Monitor the training process

#### Exercise: Architecture of the network (in breakout rooms)
As we want to design a neural network architecture for a regression task,
see if you can first come up with the answers to the following questions:

1. What must be the dimension of our input layer?
2. We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the `softmax` activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression?
Hint: A layer with `relu` activation, with `sigmoid` activation or no activation at all?
3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?



##### Solution:
1. 89
2. one neuron with linear activation function
3. 2 output nodes, with linear/no activaiton function


## Exercise: Gradient descent

Answer the following questions:

### 1. What is the goal of optimization?

- A. To find the weights that maximize the loss function
- B. To find the weights that minimize the loss function

### 2. What happens in one gradient descent step?

- A. The weights are adjusted so that we move in the direction of the gradient, so up the slope of the loss function
- B. The weights are adjusted so that we move in the direction of the gradient, so down the slope of the loss function
- C. The weights are adjusted so that we move in the direction of the negative gradient, so up the slope of the loss function
- D. The weights are adjusted so that we move in the direction of the negative gradient, so down the slope of the loss function

### 3. When the batch size is increased:
(multiple answers might apply)

- A. The number of samples in an epoch also increases
- B. The number of batches in an epoch goes down
- C. The training progress is more jumpy, because more samples are consulted in each update step (one batch).
- D. The memory load (memory as in computer hardware) of the training process is increased

**Your answers:**



### Solution:
1. B To find the weights that minimize the loss function. The loss function quantifies the total error of the network, we want to have the smallest error as possible, hence we minimize the loss.
2. D The weights are adjusted so that we move in the direction of the negative gradient, so down the slope of the loss function. We want to move towards the global minimum, so in the opposite direction of the gradient.
3. B & D are correct
A. The number of samples in an epoch also increases (**incorrect**, an epoch is always 
ined as passing through the training data for one cycle)
B. The number of batches in an epoch goes down (**correct**, the number of batches is the samples in an epoch divided by the batch size)
C. The training progress is more jumpy, because more samples are consulted in each update step (one batch). (**incorrect**, more samples are consulted in each update step, but this makes the progress less jumpy since you get a more accurate estimate of the loss in the entire dataset)
D. The memory load (memory as in computer hardware) of the training process is increased (**correct**, the data is begin loaded one batch at a time, so more samples means more memory usage)


## 🧠 Collaborative Notes

### Classification by a Neural Network using Keras

#### 5. Choose a loss function and optimizer

In our example, we choose Categorical Crossentropy, which is useful for categorising items with a "probability".

The goal of training is to minimise the loss function, as the loss function calculates the error. This is done by back propagation.

To compile the model:

```python
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
```

The optimizer `adam` is the way the model learns, which is usually a good choice.

#### 6. Train model

```python
history = model.fit(X_train, y_train, epochs=100)
```

Epochs is the amount of training iterations.

Make a line plot of the training loss:

```python
sns.lineplot(x=history.epoch, y=history.history['loss'])
```

#### 7. Perform a prediction/classification

To make a prediction:

```python
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
prediction
```

Get the predictions as species, by taking the maximum value in each row and taking the corresponding column:

```python
predicted_species = prediction.idxmax(axis="columns")
predicted_species
```

#### 8. Measuring performance

We will use a confusion matrix to evaluate the performance:


```python
from sklearn.metrics import confusion_matrix

true_species = y_test.idxmax(axis="columns")

matrix = confusion_matrix(true_species, predicted_species)
print(matrix)
```

To make it visual, let's do some preparations:

```python
# Convert to a pandas dataframe
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'
```

To display the heatmap:

```python
sns.heatmap(confusion_df, annot=True)
```

#### 9. Refine the model

The training data is quite unbalanced:

```python
penguins['species'].value_counts()
```

So getting better data might help.

One could also drop features that don't seem to distinguish the species.

We will go deeper into this later.

#### 10. Share model

To save the model:

```python
model.save('my_first_model')
```

To load it again:

```python
pretrained_model = keras.models.load_model('my_first_model')
```

Look at a summary of the pretrained model:

```python
pretrained_model.summary()
```

### Monitor the training process

We will use a regression model here, because we don't want to classify things, but get a prediction.

#### 1. Formulate / Outline the problem: weather prediction

We want to predict the hours of sunshine of the next day.

#### 2. Identify inputs and outputs

Load the dataset you downloaded earlier:

```python
import pandas as pd

filename_data = "weather_prediction_dataset_light.csv"
data = pd.read_csv(filename_data)
data.head()
```

If you don't have the dataset, you can load it directly in your notebook:

```python
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
```

Let's have a quick look at the data:

```python
data.columns
```

Look at the shape of the data:

```python
data.shape
```

#### 3. Prepare data

We will only take the first three years of the data and clean it up:

```python
nr_rows = 365*3 # 3 years
# data
X_data = data.loc[:nr_rows] # Select first 3 years
X_data = X_data.drop(columns=['DATE', 'MONTH']) # Drop date and month column

# labels (sunshine hours the next day)
y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"]
```

The range `1:(nr_rows + 1)` is to get the data of the following day.

Split the data into training and test set:

```python
from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
```

Split the remaining 30% into two equal parts:

```python
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)
```

Training set: Used to fit the model to the data
Validation set: Use to check how model is training during training process. Check for overfitting. Use for refining the model
Test set: Holy dataset that you only use at the very end as an independent unbiased final check.

#### 4. Choose a pretrained model or start building architecture from scratch
```python=
from tensorflow import keras

def create_nn():
    # Input layer
    inputs = keras.Input(shape=X_data.shape[1],
                         name='input')
    # Hidden layers
    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)
    
    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_model")

model = create_nn()
model.summary()
```

#### Intermezzo: How do neural networks learn?
![image](https://hackmd.io/_uploads/Bkk_Z51sa.png)

Loss: quantifies how wrong your model is
On the X-axis: Weight of a single neuron

##### Gradient descent:
1. Make predictions based on initial weight, 
2. Compute loss
3. Compute gradient of the loss with respect to the weights
4. Update the weights by taking a small step in the direction of the negative gradient, down the slope.

This is a highly simplified visualization of the loss 'landscape', in reality the landscape is much more complex with many local minima.

Batch: Subset of the training data that you use for one update step in the gradient descent.

Batch size: The number of samples within a batch



#### 5. Choose a loss function and optimizer

```python=
model.compile(loss='mse',
              optimizer='adam',
              metrics=[keras.metrics.RootMeanSquaredError()])
```

Let's put this into a function
```python=
def compile_model(model):
    model.compile(loss='mse',
              optimizer='adam',
              metrics=[keras.metrics.RootMeanSquaredError()])
    
```

```python=
compile_model(model)
```

Train the model:
```python=
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=2)
```

Let's plot our history
```python=
import seaborn as sns
import matplotlib.pyplot as plt

def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    
plot_history(history, 'root_mean_squared_error')
```

#### 7. Perform a prediction
```python=
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
```

#### 8. Measure Performance

```python=
# We define a function that we will reuse in this lesson
def plot_predictions(y_pred, y_true, title):
    plt.style.use('ggplot')  # optional, that's only to define a visual style
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel("predicted sunshine hours")
    plt.ylabel("true sunshine hours")
    plt.title(title)
```

```python=
plot_predictions(y_train_predicted, y_train, title="Predictions on the training set")
```

```python=
plot_predictions(y_test_predicted, y_test, title="Predictions on the test set")
```

```python=
# We define a function that we will reuse in this lesson
def plot_predictions(y_pred, y_true, title):
    plt.style.use('ggplot')  # optional, that's only to define a visual style
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel("predicted sunshine hours")
    plt.ylabel("true sunshine hours")
    plt.title(title)
```


### Sven's summary
- Please indicate when we go a bit too fast!

## 📚 Resources

[Weather dataset](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)

```python
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
```
