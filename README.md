# Keras Learning

----------


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

----------


- [Classifier](https://github.com/Eurus-Holmes/keras_learning/blob/master/Classifier_MNIST.ipynb)

数据用的是 Keras 自带 MNIST 这个数据包，再分成训练集和测试集。

x 是一张张图片，y 是每张图片对应的标签，即它是哪个数字。

简单介绍一下相关模块：

`models.Sequential`用来一层一层地去建立神经层;

`layers.Dense` 意思是这个神经层是全连接层;

`layers.Activation` 激活函数;

`optimizers.RMSprop` 优化器采用 RMSprop，加速神经网络训练方法。

```python
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
```

在回归网络中用到的是 `model.add` 一层一层添加神经层，这里直接在模型的里面加多个神经层。
好比一个水管，一段一段的，数据是从上面一段掉到下面一段，再掉到下面一段。

第一段就是加入 Dense 神经层。32 是输出的维度，784 是输入的维度。

第一层传出的数据有 32 个feature，传给激活单元。
激活函数用到的是 relu 函数。
经过激活函数之后，就变成了非线性的数据。

然后再把这个数据传给下一个神经层，这个 Dense 我们定义它有 10 个输出的 feature。同样的，此处不需要再定义输入的维度，因为它接收的是上一层的输出。

接下来再输入给下面的 softmax 函数，用来分类。

```python
# Another way to build your neural net
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

----------


- [CNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/MNIST_CNN.ipynb)

数据仍然是用 mnist。

**1. 建立网络第一层，建立一个 Convolution2D，参数有 filter 的数量。**

`filter` 就是滤波器，用32个滤波器扫描同一张图片，每个滤波器会总结出一个 feature。每个滤波器会生成一整张图片，有32个滤波器就会生成32张代表不同特征的图片；

`nb_row、nb_col` 代表这个滤波器有多少行多少列；

`border_mode` 代表这个滤波器在过滤时候用什么方式，这里采用 same。
因为是第一层，所以需要定义输入数据的维度，1, 28, 28 就是图片图片的维度。
滤波器完成之后，会生成32层的数据，但是图片的长和宽是不变的，仍然是28×28。

之后再加一个 `relu` 激活函数。

```python
# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',     # Padding method
    dim_ordering='th',      # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
    input_shape=(1,         # channels
                 28, 28,)    # height & width
))
model.add(Activation('relu'))
```

**2. Pooling 是一个向下取样的过程。**

它可以缩小生成出来的长和宽，高度不需要被压缩。

`pool_size` 是向下取样的时候，考虑多长多宽的图片。

`strides` 步长，是取完一个样之后要跳几步再取样，再跳几步再取样。

```python
# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    # Padding method
))
```

**3. 接下来建立第二个神经层。**

有 64 个 filter，5, 5 的长宽，再跟着一个激活函数；

再跟着一个 `MaxPooling2D` 取样。

```python
# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
```

**4. 接下来进入全联接层。**

用 `Flatten` 把卷出来的三维的层，抹平成二维的；

接下来就加一个 `Dense` 全联接层，抹平就是为了可以把这一个一个点全连接成一个层；

接着再加一个激活函数。

```python
# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
```

**5. 在第二个全连接层，输出 10 个 unit, 用 softmax 作为分类。**

```python
# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
```

----------


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

----------


- [Classifier_RNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/Classifier_MNIST_RNN.ipynb)

RNN 是一个序列化的神经网络，我们处理图片数据的时候，也要以序列化的方式去考虑。

图片是由一行一行的像素组成，我们就一行一行地去序列化地读取数据。最后再进行一个总结，来决定它到底是被分辨成哪一类。

用到的参数含义：

`TIME_STEPS` 是要读取多少个时间点的数据，如果一次读一行需要读28次；

`INPUT_SIZE` 每次每一行读取多少个像素；

`BATCH_SIZE` 每一批训练多少张；

`BATCH_INDEX` 用来生成数据；

`OUTPUT_SIZE` 分类结果的长度，0到9，所以长度为 10；

`CELL_SIZE` 网络中隐藏层要放多少个 unit；

`LR` 是学习率。

**1. 用 Sequential**

建立模型，就是一层一层地加上神经层。

```python
# build RNN model
model = Sequential()
```

**2. 加上 SimpleRNN**

`batch_input_shape` 就是在后面处理批量的训练数据时它的大小是多少，有多少个时间点，每个时间点有多少个像素。

```python
# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))
```

**3. 加 Dense 输出层**

输出 output 长度为 10，接着用 softmax 激活函数用于分类。

```python
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
```

**4. 在训练的时候有一个小技巧，就是怎么去处理批量**

输出结果时每 500 步输出一下测试集的准确率和损失。

需要用到 BATCH_INDEX，一批批地截取数据，下一批的时候，这个 BATCH_INDEX 就需要累加，后面的时间点和步长没有变化都是28。

y 的批量和 x 的处理是一样的，只不过 y 只有二维，所以它只有两个参数。

后面有一个判断语句，如果这个 index 大于训练数据的总数，index 就变为 0，再从头开始一批批处理。

```python
# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
```

----------


- [Autoencoder](https://github.com/Eurus-Holmes/keras_learning/blob/master/Autoencoder.py)

自编码，简单来说就是把输入数据进行一个压缩和解压缩的过程。

原来有很多 Feature，压缩成几个来代表原来的数据，解压之后恢复成原来的维度，再和原数据进行比较。

做的事情是把 `datasets.mnist` 数据的 28×28＝784 维的数据，压缩成 2 维的数据，然后在一个二维空间中可视化出分类的效果。

模型结构：

`encoding_dim`, 要压缩成的维度.

```python
# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))
```

建立 `encoded` 层和 `decoded` 层，再用 `autoencoder` 把二者组建在一起。训练时用 `autoencoder` 层。

**1. encoded 用4层 Dense 全联接层**

激活函数用 `relu`，输入的维度就是前一步定义的 `input_img`。

接下来定义下一层，它的输出维度是64，输入是上一层的输出结果。

在最后一层，我们定义它的输出维度就是想要的 `encoding_dim＝2`。

**2. 解压的环节，它的过程和压缩的过程是正好相反的**

相对应层的激活函数也是一样的，不过在解压的最后一层用到的激活函数是 `tanh`。因为输入值是由 -0.5 到 0.5 这个范围，在最后一层用这个激活函数的时候，它的输出是 -1 到 1，可以是作为一个很好的对应。

```python
# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)
```

接下来直接用 `Model` 这个模块来组建模型

输入就是图片，输出是解压的最后的结果。

```python
# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)
```

当我们想要看由 784 压缩到 2维后，这个结果是什么样的时候，也可以只单独组建压缩的板块，此时它的输入是图片，输出是压缩环节的最后结果。

```python
# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)
```

----------


- [Save&Reload](https://github.com/Eurus-Holmes/keras_learning/blob/master/save%26reload.py)

----------

 - [Nested_LSTM](https://github.com/Eurus-Holmes/keras_learning/blob/master/Nested_LSTM.md)

----------

 - [CNN-RNN](https://github.com/Eurus-Holmes/keras_learning/blob/master/CNN-RNN.md)

----------

 - [keras_layer-summary](https://github.com/Eurus-Holmes/keras_learning/blob/master/keras_layer-summary.md)

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
