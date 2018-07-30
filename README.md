# Keras Learning

- [Regression](https://github.com/Eurus-Holmes/keras_learning/blob/master/regression.ipynb)

目的是对一组数据进行拟合。

**1. 用 Sequential 建立 model**

**2. 再用 model.add 添加神经层，添加的是 Dense 全连接神经层**

参数有两个，输入数据和输出数据的维度，
本代码的例子中 x 和 y 是一维的。

如果需要添加下一个神经层的时候，不用再定义输入的纬度，因为它默认就把前一层的输出作为当前层的输入。
在这个例子里，只需要一层就够了。

``` python
# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
```

- [Classifier](https://github.com/Eurus-Holmes/keras_learning/blob/master/Classifier_MNIST.ipynb)
- [CNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/MNIST_CNN.ipynb)
- [Regression_RNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/Regression_RNN_LSTM.py)

**1. 搭建模型，仍然用 Sequential**

**2. 然后加入 LSTM 神经层**
```python
model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)
```

`batch_input_shape`,  就是在后面处理批量的训练数据时它的大小是多少，有多少个时间点，每个时间点有多少个数据。

`output_dim` , 就是 LSTM 里面有二十个 unit。

`return_sequences`,  就是在每个时间点，要不要输出output，默认的是 false，现在我们把它定义为 true。如果等于 false，就是只在最后一个时间点输出一个值。

`stateful`, 默认的也是 false，意义是批和批之间是否有联系。直观的理解就是我们在读完二十步，第21步开始是接着前面二十步的。也就是第一个 batch中的最后一步与第二个 batch 中的第一步之间是有联系的。

**3. 有个不同点是 TimeDistributed**

在上一个回归问题中，我们是直接加 Dense 层，因为只在最后一个输出层把它变成一个全连接层。
而这个问题是每个时间点都有一个 output，那需要 dense 对每一个 output 都进行一次全连接的计算。




- [Classifier_RNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/Classifier_MNIST_RNN.ipynb)
- [Autoencoder](https://github.com/Eurus-Holmes/keras_learning/blob/master/Autoencoder.py)
- [Save&Reload](https://github.com/Eurus-Holmes/keras_learning/blob/master/save%26reload.py)

------------------
# Why Keras?
# Keras: Deep Learning for humans

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

[![Build Status](https://travis-ci.org/keras-team/keras.svg?branch=master)](https://travis-ci.org/keras-team/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

## You have just found Keras.

Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

Use Keras if you need a deep learning library that:

- Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Runs seamlessly on CPU and GPU.

Read the documentation at [Keras.io](https://keras.io).

Keras is compatible with: __Python 2.7-3.6__.


------------------


## Getting started: 30 seconds to Keras

The core data structure of Keras is a __model__, a way to organize layers. The simplest type of model is the [`Sequential`](https://keras.io/getting-started/sequential-model-guide) model, a linear stack of layers. For more complex architectures, you should use the [Keras functional API](https://keras.io/getting-started/functional-api-guide), which allows to build arbitrary graphs of layers.

Here is the `Sequential` model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```


You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:

```python
model.train_on_batch(x_batch, y_batch)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

In the [examples folder](https://github.com/keras-team/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.


------------------
