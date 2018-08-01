# BatchNormalization layer

通常在线性向非线性转变时使用,如下：

```python
model.add(Dense(100,input_dim=20))
model.add(BatchNormalization())
model.add(Activation('relu'))
```

> 作用: 能够保证权重的尺度不变，因为BatchNormalization在激活函数前对输入进行了标准化。

另一个流行的做法：
在2D卷积与激活函数前，进行Normalization，如下：

```python
def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return BatchNormalization()(Activation(activation='relu')(Convolution2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))
```

在一些code中，`bias=False`的意思简单明了，目的是为了减少参数。

```python
model.add(Dense(64, bias=False))
```

----------
# Maxpooling layer

> 池化层是基于采样的离散过程（sample-based discretization process）。听起来好复杂的样子，简单来说，即对input进行采样，降低input的维度，减少了参数（简化了计算），增强了模型的泛化能力，也降低了overfitting的可能性。

![Figure 1](https://github.com/Eurus-Holmes/keras_learning/raw/master/images/1.png)

<p align="center">Max_pooling</p>

简单来说，就是用2\*2的filter过滤原输入，得到每一个2\*2格子中值最大的元素，构成新的输入。
注意，这些2\*2的格子没有Overlap。

直观的例子如下，经过**Max_pooling**后，一帧图像相当于被采样成一个更小的图像，但保存了原始图像的大量特征：

![Max_pooling示例](http://upload-images.jianshu.io/upload_images/2528310-d7761035ec7517a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/500)

Max_pooling layer只能用于图像吗？不是的，目前CNN在text classification领域也大有可为。只要是CNN, max_pooling layer都可以尝试一下。

----------
# pad_sequences & Masking layer

上面提到，文本数据也可以用CNN来处理。很多人有疑问，CNN的输入通常是一个（图像）矩阵，而文本中句子或者文章的长度不一，CNN如何将长短不一的输入转化成矩阵呢？
答案是——**pad_sequences**。

```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='pre', value=0.)
```

`sequences`: 所有的句子

`maxlen`: 把所有句子的长度都定为maxlen，如果是None，则maxlen自动设置为sequences中最长的那个

`padding`: 在句子前端（pre）或后端(post)填充

`truncating`: 截断过长的句子，从前端(pre)或者后端(post)截断

`value`: 填充的数值

假设我们使用0填充，那么多无用的0，会不会影响结果呢？

对于CNN来说，是不影响的。

对于RNN也需要**pad_sequences**，此时填充的0是多余的，就需要使用**Masking layer**忽略其中的0值。

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

----------
# Flatten layer & Dense layer

Flatten层用来将输入“压平”，即把多维的输入一维化，**常用在从卷积层(Convolution)到全连接层(Dense)的过渡**。

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
# now: model.output_shape == (None, 65536)
model.add(Dense(32))
```

也就是说，Convolution卷积层之后是无法直接连接Dense全连接层的，需要把Convolution层的数据压平（Flatten），然后就可以直接加Dense层了。上面code最后一行, Dense(32)表示output的shape为(*,32)。

注意，当Dense作为第一层时，需要specify输入的维度，之后就不用了，如下：

```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

----------
# Dropout layer

如dropout的名字，就是要随机扔掉当前层一些weight，相当于废弃了一部分Neurons。
它还有另一个名字 dropout regularization, 所以你应该知道它有什么作用了：

 1. 降低模型复杂度，增强模型的泛化能力，防止过拟合。
 2. 顺带降低了运算量。。。

```python
model = Sequential()
model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
```

----------


