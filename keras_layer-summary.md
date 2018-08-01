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


