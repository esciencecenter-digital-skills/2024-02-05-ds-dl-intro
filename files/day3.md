![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 3
2024-02-01 Deep learning introduction

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](http://tinyurl.com/2024-02-05-ds-dl-intro-day-3)

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
|09:15|Monitor the training process|
|10:15|Coffee break|
|10:30|Monitor the training process|
|11:30|Coffee break|
|11:45|Advanced layer types|
|12:45|Wrap-up|
|13:00|END|

## ⛸ Welcome and icebreaker
- Icebreaker: If you could be an animal, which animal would you pick?
- Feedback yesterday
- Recap:
    - Measuring performance of the model
    - How do neural networks learn? Gradient descent, batches.
    - Choosing activation functions: 
        - Relu in hidden layers. 
        - In output layer: Linear activation for regression
        - In output layer: Softmax for classification
    - 
## 🔧 Exercises
### Exercise: Reflecting on our results (in breakout rooms)
1. Is the performance of the model as you expected (or better/worse)?
2. Is there a noteable difference between training set and test set? And if so, any idea why?
3. (Optional) When developing a model, you will often vary different aspects of your model like which features you use, model parameters and architecture. It is important to settle on a single-number ev (aluation metric to compare your models.
    * What single-number evaluation metric would you choose here and why?


**Solution:**

3. The metric that we are using: RMSE would be a good one. You could also consider Mean Squared Error, that punishes large errors more (because large errors create even larger squared errors).
It is important that if the model improves in performance on the basis of this metric then that should also lead you a step closer to reaching your goal: to predict tomorrow's sunshine hours. 
If you feel that improving the metric does not lead you closer to your goal, then it would be better to choose a different metric.

### Exercise: Baseline
1. Looking at this baseline: Would you consider this a simple or a hard problem to solve?
2. (Optional) Can you think of other baselines?

**Solution:**

1. This really depends on your definition of hard! The baseline gives a more accurate prediction than just
randomly predicting a number, so the problem is not impossible to solve with machine learning. However, given the structure of the data and our expectations with respect to quality of prediction, it may remain hard to find a good algorithm which exceeds our baseline by orders of magnitude.
2. There are a lot of possible answers. A slighly more complicated baseline would be to take the average over the last couple of days.

### Exercise: (in breakout rooms) Try to reduce the degree of overfitting by lowering the number of parameters
We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

* Is it possible to get rid of overfitting this way?
* Does the overall performance suffer or does it mostly stay the same?
* (optional) How low can you go with the number of parameters without notable effect on the performance on the validation set?



**Solution:**

Let's first adapt our `create_nn` function so that we can tweak the number of nodes in the 2 layers
by passing arguments to the function:

```python
def create_nn(nodes1=100, nodes2=50):
   # Input layer
   inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')
   # Dense layers
   layers_dense = keras.layers.Dense(nodes1, 'relu')(inputs)
   layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)
   # Output layer
   outputs = keras.layers.Dense(1)(layers_dense)
   return keras.Model(inputs=inputs, outputs=outputs, name="model_small")
```

Let's see if it works by creating a much smaller network with 10 nodes in the first layer,
and 5 nodes in the second layer:

```python
model = create_nn(10, 5)
model.summary()
```
```
Model: "model_small"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 89)]              0
_________________________________________________________________
dense_9 (Dense)              (None, 10)                900
_________________________________________________________________
dense_10 (Dense)             (None, 5)                 55
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 6
=================================================================
Total params: 961
Trainable params: 961
Non-trainable params: 0
```

Let's compile and train this network:
```python
compile_model(model)
history = model.fit(X_train, y_train,
                   batch_size = 32,
                   epochs = 200,
                   validation_data=(X_val, y_val))
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

![](https://raw.githubusercontent.com/carpentries-incubator/deep-learning-intro/main/episodes/fig/03_training_history_3_rmse_smaller_model.png){alt='Plot of RMSE vs epochs for the training set and the validation set with similar performance across the two sets.'}

1. With this smaller model we have reduced overfitting a bit, since the training and validation loss are now closer to each other, and the validation loss does now reach a plateau and does not further increase.
We have not completely avoided overfitting though. 
2. In the case of this small example model, the validation RMSE seems to end up around 3.2, which is much better than the 4.08 we had before. Note that you can double check the actual score by calling `model.evaluate()` on the test set.
3. In general, it quickly becomes a complicated search for the right "sweet spot", i.e. the settings for which overfitting will be (nearly) avoided but the model still performs equally well. A model with 3 neurons in both layers seems to be around this spot, reaching an RMSE of 3.1 on the validation set. 
Reducing the number of nodes further increases the validation RMSE again.

### Exercise: (in breakout rooms) Simplify the model and add data + next steps

You may have been wondering why we are including weather observations from
multiple cities to predict sunshine hours only in Basel. The weather is
a complex phenomenon with correlations over large distances and time scales,
but what happens if we limit ourselves to only one city?

1. Since we will be reducing the number of features quite significantly,
we should afford to include more data. Instead of using only 3 years, use
8 or 9 years!
2. Remove all cities from the training data that are not for Basel.
You can use something like:
```python
cols = [c for c in X_data.columns if c[:5] == 'BASEL']
X_data = X_data[cols]
```
3. Now rerun the last model we defined which included the BatchNorm layer.
Recreate the scatter plot comparing your prediction with the baseline
prediction based on yesterday's sunshine hours, and compute also the RMSE.
Note that even though we will use many more observations than previously,
the network should still train quickly because we reduce the number of
features (columns).
Is the prediction better compared to what we had before?

#### (Optional) What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
Usually models are "well behaving" in the sense that small changes to the architectures also only result in small changes of the performance (if any).
It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
Applying common sense is often a good first step to make a guess of how much better results *could* be.
In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
But how much better our model could be exactly, often remains difficult to answer.

4. What changes to the model architecture might make sense to explore?
5. Ignoring changes to the model architecture, what might notably improve the prediction quality?
6. (Optional) Try to train a model on all years that are available,
and all features from all cities. How does it perform?
7. (Optional) Try one of the most fruitful ideas you have. Does it improve the model?

### Advanced layer types

#### Number of features CIFAR-10
How many features does one image in the CIFAR-10 dataset have?

* A. 32
* B. 1024
* C. 3072
* D. 5000



**Solution:**

The correct solution is C: 3072. There are 1024 pixels in one image (32 * 32), each pixel has 3 channels (RGB). So 1024 * 3 = 3072.

#### Number of parameters
Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?

* A. 307200
* B. 307300
* C. 100
* D. 3072


**Solution:**

The correct answer is B: Each entry of the input dimensions, i.e. the `shape` of one single data point, is connected with 100 neurons of our hidden layer, and each of these neurons has a bias term associated to it. So we have `307300` parameters to learn.
```python
width, height = (32, 32)
n_hidden_neurons = 100
n_bias = 100
n_input_items = width * height * 3
n_parameters = (n_input_items * n_hidden_neurons) + n_bias
n_parameters
```
```output
307300
```
We can also check this by building the layer in Keras:
```python
inputs = keras.Input(shape=n_input_items)
outputs = keras.layers.Dense(100)(inputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
```
```output
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3072)]            0
_________________________________________________________________
dense (Dense)                (None, 100)               307300
=================================================================
Total params: 307,300
Trainable params: 307,300
Non-trainable params: 0
_________________________________________________________________
```

### Convolutional neural network exercise (in breakout rooms)
#### 1. Border pixels
What, do you think, happens to the border pixels when applying a convolution?

**Solution**:

There are different ways of dealing with border pixels.
You can ignore them, which means that your output image is slightly smaller then your input.
It is also possible to 'pad' the borders, e.g. with the same value or with zeros, so that the convolution can also be applied to the border pixels.
In that case, the output image will have the same size as the input image.

#### 2. Number of model parameters
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise


**Solution**:

We have 100 matrices with 3 * 3 * 3 = 27 values each so that gives 27 * 100 = 2700 weights. This is a magnitude of 100 less than the fully connected layer with 100 units! Nevertheless, as we will see, convolutional networks work very well for image data. This illustrates the expressiveness of convolutional layers.

#### 3. Convolutional Neural Network
So let us look at a network with a few convolutional layers. We need to finish with a Dense layer to connect the output cells of the convolutional layer to the outputs for our classes.

```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```
Inspect the network above:

* What do you think is the function of the `Flatten` layer?
* Which layer has the most parameters? Do you find this intuitive?
* (optional) Pick a model from https://paperswithcode.com/sota/image-classification-on-cifar-10 . Try to understand how it works.

**Solution:**

* The Flatten layer converts the 28x28x50 output of the convolutional layer into a single one-dimensional vector, that can be used as input for a dense layer.
* The last dense layer has the most parameters. This layer connects every single output 'pixel' from the convolutional layer to the 10 output classes.
That results in a large number of connections, so a large number of parameters. This undermines a bit the expressiveness of the convolutional layers, that have much fewer parameters.


## 🧠 Collaborative Notes

### Monitor the training process

#### 8. Measure performance

Evaluate the metrics:

```python
train_metrics = model.evaluate(X_train, y_train, return_dict=True)
test_metrics = model.evaluate(X_test, y_test, return_dict=True)
print('Train RMSE: {:.2f}, Test RMSE: {:.2f}'.format(train_metrics['root_mean_squared_error'], test_metrics['root_mean_squared_error']))
```

How bad is our model really? What to aim for? This is generally tricky and unclear. Try to come up with a **baseline**.

The goal of a baseline is to get a feeling of what is good. You want your model to perform at least as good as the baseline.

In the weather example, a baseline for tomorrow's sunshine hours could be today's sunshine hours.

```python
y_baseline_prediction = X_test['BASEL_sunshine']
plot_predictions(y_baseline_prediction, y_test, title='Baseline predictions on the test set')
```

Now evaluate the baseline:

```python
from sklearn.metrics import mean_squared_error
rmse_baseline = mean_squared_error(y_test, y_baseline_prediction, squared=False)
print('Baseline:', rmse_baseline)
print('Neural network: ', test_metrics['root_mean_squared_error'])
```

#### 9. Refine the model

Create a new model:

```python
model = create_nn()
compile_model(model)
```

Fit the model:

```python
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val))
```

Let's make a plot:

```python
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

Overfitting can happen when your model is too complex for your dataset. In this case, you can try to adapt the model to use less parameters.

Another technique is **early stopping**.

As the name suggests, this technique just means that you stop the model training if things do not seem to improve anymore. More specifically, this usually means that the training is stopped if the validation loss does not (notably) improve anymore.

Let's create a new model:

```python
model = create_nn()
compile_model(model)
```

To use early stopping:

```python
from tensorflow.keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=10
    )

history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])
```

And make a plot:

```python
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

A very common step in classical machine learning pipelines is to scale the features, for instance by using sckit-learn's `StandardScaler`.
This can in principle also be done for deep learning.
An alternative, more common approach, is to add **BatchNormalization** layers ([documentation of the batch normalization layer](https://keras.io/api/layers/normalization_layers/batch_normalization/)) which will learn how to scale the input values.
Similar to dropout, batch normalization is available as a network layer in Keras and can be added to the network in a similar way.
It does not require any additional parameter setting.

The `BatchNormalization` can be inserted as yet another layer into the architecture.

```python
def create_nn():
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.BatchNormalization()(inputs) # This is new!
    layers_dense = keras.layers.Dense(100, 'relu')(layers_dense)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    # Defining the model and compiling it
    return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")

model = create_nn()
compile_model(model)
model.summary()
```

We can train the model again as follows:

```python
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])

plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

You can also use **TensorBoard** to automate things and get a nice overview:

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # You can adjust this to add a more meaningful model name
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(X_train, y_train,
                   batch_size = 32,
                   epochs = 200,
                   validation_data=(X_val, y_val),
                   callbacks=[tensorboard_callback],
                   verbose = 2)
```

You can launch the tensorboard interface from a Jupyter notebook, showing all trained models: 

```
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

#### 10. Save model

```python
model.save('my_tuned_weather_model')
```

### Advanced layer types

#### 1. Formulate / Outline the problem: Image classification

```python
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()
```


If you get an SSL error, run this first:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

Let's use the first 5000 images only:

```python
n = 5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

Let's look at an image

```python
from matplotlib import pyplot as plt
plt.imshow(train_images[0])
```

#### 2. Identify inputs and outputs

Let's look at what the data looks like:

```python
train_images.min(), train_images.max()
```

We have 8-bit colours.

```python
train_labels.shape
```

```python
train_labels.min(), train_labels.max()
```

We have 10 labels.

#### 3. Prepare data

The RGB values of the pixels are between 0 an 255. We should rescale them to be between 0 and 1:

```python
train_images = train_images / 255.0
val_images = val_images / 255.0
```


#### 4. Choose a pretrained model or start building architecture from scratch

Normally, you would use a pretrained workshop, but for educational purposes, we will build one ourselves.

##### Convolutional layers

e can decrease the number of units in our hidden layer, but this also decreases the number of patterns our network can remember. Moreover, if we increase the image size, the number of weights will 'explode', even though the task of recognizing large images is not necessarily more difficult than the task of recognizing small images.

The solution is that we make the network learn in a 'smart' way. The features that we learn should be similar both for small and large images, and similar features (e.g. edges, corners) can appear anywhere in the image (in mathematical terms: *translation invariant*). We do this by making use of a concepts from image processing that precede Deep Learning.

A **convolution matrix**, or **kernel**, is a matrix transformation that we 'slide' over the image to calculate features at each position of the image. For each pixel, we calculate the matrix product between the kernel and the pixel with its surroundings. A kernel is typically small, between 3x3 and 7x7 pixels. We can for example think of the 3x3 kernel:
```output
[[-1, -1, -1],
 [0, 0, 0]
 [1, 1, 1]]
```
This kernel will give a high value to a pixel if it is on a horizontal border between dark and light areas.
Note that for RGB images, the kernel should also have a depth of 3.

In the following image, we see the effect of such a kernel on the values of a single-channel image. The red cell in the output matrix is the result of multiplying and summing the values of the red square in the input, and the kernel. Applying this kernel to a real image shows that it indeed detects horizontal edges.

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/04_conv_matrix.png){alt='Example of a convolution matrix calculation' style='width:90%'}

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/04_conv_image.png){alt='Convolution example on an image of a cat to extract features' style='width:100%'}

In our **convolutional layer** our hidden units are a number of convolutional matrices (or kernels), where the values of the matrices are the weights that we learn in the training process. The output of a convolutional layer is an 'image' for each of the kernels, that gives the output of the kernel applied to each pixel.

Let's create a convolutional network:

```python
def create_conv_net():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")
    
    return model

model = create_conv_net()
model.summary()
```



## 📚 Resources

- [PDF about the CIFAR-10 training set](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [Image kernels explained](https://setosa.io/ev/image-kernels/)
- [The convolutional neural network cheat sheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#)