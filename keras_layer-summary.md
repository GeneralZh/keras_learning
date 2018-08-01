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

![1](https://leanote.com/api/file/getImage?fileId=5b616403ab64414e6300176a)
*Max_pooling*

简单来说，就是用$2*2$的$filter$过滤原输入，得到每一个$2*2$格子中值最大的元素，构成新的输入。
注意，这些$2*2$的格子没有$Overlap$。

直观的例子如下，经过**Max_pooling**后，一帧图像相当于被采样成一个更小的图像，但保存了原始图像的大量特征：

![Max_pooling示例]](http://upload-images.jianshu.io/upload_images/2528310-d7761035ec7517a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/500)

----------


