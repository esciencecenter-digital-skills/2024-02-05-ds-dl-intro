![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 4
2024-02-01 Deep learning introduction

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-4)

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

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda
| Time | Topic |
|--:|:---|
|09:00|Welcome and icebreaker|
|09:15|Advanced layer types|
|10:15|Coffee break|
|10:30|Advanced layer types|
|11:30|Coffee break|
|11:45|Outlook|
|12:45|Wrap-up and post-workshop survey|
|13:00|END|

## Recap from yesterday:
- Can we have the image of one neuron in one of the collab notes?
    - ![image](https://hackmd.io/_uploads/Bk2ePWGo6.png)
- what other types of data/real life applications can be used with CNN?
    - exercise later
- Why not softmax as output layer for the cifar nn?
    - We'll see that later
- I don't understand the answer of convolutional neural network exercise 2
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise

- Why are more images created with a convolutional layer?
- If we are creating more layers with convolutional process, then how it helps reducing the parameters? I guess I didnt get the concept of convolunional network
- Why do we have the flatten layer and what does it do exactly?

## 🔧 Exercises

### Advanced layer types

#### Network depth (in breakout rooms)
What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters?
Try it out. Create a `model` that has an additional `Conv2d` layer with 50 filters after the last MaxPooling2D layer. Train it for 20 epochs and plot the results.

**HINT**:
The model definition that we used previously needs to be adjusted as follows:
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layer here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
```


**Solution:**

We add an extra Conv2D layer after the second pooling layer:
```python
def create_nn_extra_layer():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x) #
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x) # estra layer
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x) # a new Dense layer
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model

model = create_nn_extra_layer()
```

With the model defined above, we can inspect the number of parameters:
```python
model.summary()
```
```output
Model: "cifar_model"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
input_7 (InputLayer)        [(None, 32, 32, 3)]       0
conv2d_16 (Conv2D)          (None, 30, 30, 50)        1400
max_pooling2d_10 (MaxPoolin  (None, 15, 15, 50)       0
g2D)
conv2d_17 (Conv2D)          (None, 13, 13, 50)        22550
max_pooling2d_11 (MaxPoolin  (None, 6, 6, 50)         0
g2D)
conv2d_18 (Conv2D)          (None, 4, 4, 50)          22550
flatten_6 (Flatten)         (None, 800)               0
dense_11 (Dense)            (None, 50)                40050
dense_12 (Dense)            (None, 10)                510
=================================================================
Total params: 87,060
Trainable params: 87,060
Non-trainable params: 0
_________________________________________________________________
```
The number of parameters has decreased by adding this layer.
We can see that the conv layer decreases the resolution from 6x6 to 4x4,
as a result, the input of the Dense layer is smaller than in the previous network.
To train the network and plot the results:
```python
compile_model(model)
history = model.fit(train_images, train_labels, epochs=20,
                   validation_data=(val_images, val_labels))
plot_history(history, ['accuracy', 'val_accuracy'])
```
![](https://raw.githubusercontent.com/carpentries-incubator/deep-learning-intro/main/episodes/fig/04_training_history_2.png){alt="Plot of training accuracy and validation accuracy vs epochs for the trained model"}
```python
plot_history(history, ['loss', 'val_loss'])
```

![](https://raw.githubusercontent.com/carpentries-incubator/deep-learning-intro/main/episodes/fig/04_training_history_loss_2.png){alt="Plot of training loss and validation loss vs epochs for the trained model"}

#### Why and when to use convolutional neural networks
1. Would it make sense to train a convolutional neural network (CNN) on the penguins dataset and why?
2. Would it make sense to train a CNN on the weather dataset and why?
3. (Optional) Can you think of a different machine learning task that would benefit from a CNN architecture?

**Your answers:**

**Solution:**

1. No that would not make sense. Convolutions only work when the features of the data can be ordered  in a meaningful way. Pixels for example are ordered in a spatial dimension.  This kind of order cannot be applied to the features of the penguin dataset. If we would have pictures or audio recordings of the penguins as input data it would make sense to use a CNN architecture.
2. It would make sense, but only if we approach the problem from a different angle then we did before. Namely, 1D convolutions work quite well on sequential data such as timeseries. If we have as our input a matrix of the different weather conditions over time in the past x days, a CNN would be suited to quickly grasp the temporal relationship over days.
3. Some example domains in which CNNs are applied:
  - Text data
  - Timeseries, specifically audio
  - Molecular structures

## 🧠 Collaborative Notes

### Advanced layer types

#### Code from yesterday

```python
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()
```

```python
n = 5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

```python
train_images = train_images / 255.0
val_images = val_images / 255.0
```

```python
from matplotlib import pyplot as plt
plt.imshow(train_images[0])
```

#### 4. Choose a pretrained model or start building architecture from scratch

##### Pooling layers

Recuces an image by clustering pixels.

An examlpe is taking the max of each cluster of pixels; this is called Max-Pool.

This is a way to extract features at different length scales.

```python
def create_nn():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x) # a new maxpooling layer
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x) # a new maxpooling layer (same as maxpool)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x) # a new Dense layer
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model

model = create_nn()
model.summary()
```

Notice that the dimensions of the images get divided by two in every pooling layer. Furthermore, the pooling layers don't have parameters.

If the dimensions of your image are odd instead of even, it depends on the implementation; look at the documentation to dee the default and/or make a different choice.

#### 5. Choose a loss function and optimize

Remember that our target class is represented by a single integer, whereas the output of our network has 10 nodes, one for each class.
So, we should have actually one-hot encoded the targets and used a softmax activation for the neurons in our output layer!
Luckily, there is a quick fix to calculate crossentropy loss for data that
has its classes represented by integers, the `SparseCategoricalCrossentropy()` function. 
Adding the argument `from_logits=True` accounts for the fact that the output has a linear activation instead of softmax.
This is what is often done in practice, because it spares you from having to worry about one-hot encoding.


```python
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
compile_model(model)
```

#### 6. Train the model

```python
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(val_images, val_labels))
```

#### 7. Perform a Prediction/Classification

We will skip that this time.

#### 8. Measure performance

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_history(history, metrics):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics(str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
plot_history(history, ['accuracy', 'val_accuracy'])
```

```python
plot_history(history, ['loss', 'val_loss'])
```

It seems that the model is overfitting somewhat, because the validation accuracy and loss stagnates.

Let's compare with a model that only consists of dense layers:

```python
def create_dense_model():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(50, activation='relu')(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    return keras.models.Model(inputs=inputs, outputs=outputs,
                              name='dense_model')

dense_model = create_dense_model()
dense_model.summary()
```

Concolutional networks are much more efficient in their usage of parameters.

```python
compile_model(dense_model)
history_dense = dense_model.fit(train_images, train_labels, epochs=30,
                    validation_data=(val_images, val_labels))
plot_history(history_dense, ['accuracy', 'val_accuracy'])
```

#### 9. Refine the model

Convolutional and Pooling layers are also applicable to different types of data than image data. Whenever the data is ordered in a (spatial) dimension, and *translation invariant* features are expected to be useful, convolutions can be used. Think for example of time series data from an accelerometer, audio data for speech recognition, or 3d structures of chemical compounds.

##### Dropout layers

This is a way to keep enough neurons to keep learning, but not to overfit.

It randomly turns on or off inputs.

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/neural_network_sketch_dropout.png)

I'ts to prevent some neurons from specialising too much. You force redundancy in the model.

It is only during training that we randomly turn off neurons, not when making predictions.

```python
def create_nn_with_dropout():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.8)(x) # This is new!
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model

model_dropout = create_nn_with_dropout()
model_dropout.summary()
```

The $0.8$ means that each input has $80\%$ chace to be dropped.

Train the model:

```python
compile_model(model_dropout)

history = model_dropout.fit(train_images, train_labels, epochs=20,
                    validation_data=(val_images, val_labels))
```

Plot the accuracy:

```python
plot_history(history, ['accuracy', 'val_accuracy'])

val_loss, val_acc = model_dropout.evaluate(val_images,  val_labels, verbose=2)
```

Plot the loss:

```python
plot_history(history, ['loss', 'val_loss'])
```

#### 10. Share model
Let's save our model

```python
model.save('cnn_model')
```

### Outlook

#### Organising deep learning projects
As you might have noticed already in this course, deep learning projects can quickly become messy.
Here follow some best practices for keeping your projects organized:

##### 1. Organise experiments in notebooks
Jupyter notebooks are a useful tool for doing deep learning experiments.
You can very easily modify your code bit by bit, and interactively look at the results.
In addition you can explain why you are doing things in markdown cells.
- As a rule of thumb do one approach or experiment in one notebook.
- Give consistent and meaningful names to notebooks, such as: `01-all-cities-simple-cnn.ipynb`
- Add a rationale on top and a conclusion on the bottom of each notebook

[_Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007) provides further advice on how to maximise the usefulness and reproducibility of experiments captured in a notebook.

##### 2. Use Python modules
Code that is repeatedly used should live in a Python module and not be copied to multiple notebooks.
You can import functions and classes from the module(s) in the notebooks.
This way you can remove a lot of code definition from your notebooks and have a focus on the actual experiment.

##### 3. Keep track of your results in a central place
Always evaluate your experiments in the same way, on the exact same test set.
Document the results of your experiments in a consistent and meaningful way.
You can use a simple spreadsheet such as this:

| MODEL NAME              | MODEL DESCRIPTION                          | RMSE | TESTSET NAME  | GITHUB COMMIT | COMMENTS |
|-------------------------|--------------------------------------------|------|---------------|---------------|----------|
| weather_prediction_v1.0 | Basel features only, 10 years. nn: 100-50  | 3.21 | 10_years_v1.0 |  ed28d85      |          |
| weather_prediction_v1.1 | all features, 10 years. nn: 100-50         | 3.35 | 10_years_v1.0 |  4427b78      |          |

You could also use a tool such as [Weights and Biases](https://wandb.ai/site) for this.

You now understand the basic principles of deep learning and are able to implement your own deep learning pipelines in Python. But there is still so much to learn and do!

#### Next steps

Here are some suggestions for next steps you can take in your endeavor to become a deep learning expert:

1. Learn more by going through a few of the learning resources we have compiled for you
2. Apply what you have learned to your own projects. Use the deep learning workflow to structure your work. Start as simple as possible, and incrementally increase the complexity of your approach.
3. Compete in a [Kaggle](https://www.kaggle.com/competitions) competition to practice what you have learned.
4. Get access to a GPU. Your deep learning experiments will progress much quicker if you have to wait for your network to train in a few seconds instead of hours (which is the order of magnitude of speedup you can expect from training on a GPU instead of CPU). Tensorflow/Keras will automatically detect and use a GPU if it is available on your system without any code changes. A simple and quick way to get access to a GPU is to use [Google Colab](https://colab.google/)


## 📚 Resources

- [Image kernels explained](https://setosa.io/ev/image-kernels/)
- [The convolutional neural network cheat sheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#)
- [Hugging face](https://huggingface.co/): The AI community building the future.
- [MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra](https://doi.org/10.1186/s13321-021-00558-4)
- [Using ms2deepscore: How to load data, train a model, and compute similarities.](https://github.com/matchms/ms2deepscore/blob/0.4.0/notebooks/MS2DeepScore_tutorial.ipynb)
- [ChatGPT Chemistry Assistant for Text Mining and the Prediction of MOF Synthesis](https://doi.org/10.1021/jacs.3c05819)
- [Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007)
- [Weights and Biases](https://wandb.ai/site)
- [Cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/)
