> 本文主要采用CNN,RNN对时序数据进行二分类。

----------
# CNN处理时序数据的二分类

```python

model = Sequential()
model.add(Conv1D(128, 3, padding='same', input_shape=(max_lenth, max_features)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(256, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(128, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling1D())   #时序的时间维度上全局池化
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.binary_crossentropy])  
```

----------
# 双层RNN处理时序数据的二分类

```python
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from keras import backend as K
import my_callbacks
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
max_lenth = 23
max_features = 12
training_iters = 2000
train_batch_size = 800
test_batch_size = 800
n_hidden_units = 64  
lr = 0.0003
cb = [
    my_callbacks.RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=200, verbose=2,mode='max')
]
model = Sequential()
model.add(keras.layers.core.Masking(mask_value=0., input_shape=(max_lenth, max_features)))
model.add(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,#多层时需设置为true
              return_state=False, go_backwards=False, stateful=False, unroll=False))   #input_shape=(max_lenth, max_features),
model.add(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=False,
              return_state=False, go_backwards=False, stateful=False, unroll=False))   #input_shape=(max_lenth, max_features),
model.add(Dropout(0.5))
 
model.add(Dense(1))
model.add(BatchNormalization())
model.add(keras.layers.core.Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.binary_crossentropy])  
model.fit(x_train, y_train, batch_size=train_batch_size, epochs=training_iters, verbose=2,
          callbacks=cb,validation_split=0.2,
shuffle=True, class_weight=class_weight, sample_weight=None, initial_epoch=0)
pred_y = model.predict(x_test, batch_size=test_batch_size)
score = roc_auc_score(y_test,pred_y)
```

----------
# 加入attention机制的 双向RNN（attention即对所有时刻的输出乘上对应的权重相加作为最终输出）

```python
from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
 
 
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
 
 
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
 
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
```

```python
model = Sequential()
model.add(keras.layers.core.Masking(mask_value=0., input_shape=(max_lenth, max_features)))
model.add(Bidirectional(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,#多层时需设置为true
              return_state=False, go_backwards=False, stateful=False, unroll=False),merge_mode='concat'))   #input_shape=(max_lenth, max_features),
model.add(Bidirectional(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
              return_state=False, go_backwards=False, stateful=False, unroll=False),merge_mode='concat'))   #input_shape=(max_lenth, max_features),
model.add(Dropout(0.5))
model.add(AttentionWithContext())
model.add(Dense(1))
model.add(BatchNormalization())
model.add(keras.layers.core.Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.binary_crossentropy])
```

----------
# CNN-RNN融合

```python
class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)
 
    def build(self, input_shape):
        input_shape = input_shape
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        return x
 
    def get_output_shape_for(self, input_shape):
        return input_shape
 
model_left = Sequential()
model_left.add(keras.layers.core.Masking(mask_value=0., input_shape=(max_lenth, max_features)))  #解决不同长度的序列问题
model_left.add(GRU(units=left_hidden_units,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,#多层时需设置为true
              return_state=False, go_backwards=False, stateful=False, unroll=False))
model_left.add(GRU(units=left_hidden_units,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
              return_state=False, go_backwards=False, stateful=False, unroll=False))
model_left.add(NonMasking())   #Flatten()不支持masking,此处用于unmask
model_left.add(Flatten())
 
## FCN
model_right = Sequential()
model_right.add(Conv1D(128, 3, padding='same', input_shape=(max_lenth, max_features)))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(Conv1D(256, 3))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(Conv1D(128, 3))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(GlobalAveragePooling1D())
model_right.add(Reshape((1,1,-1)))
model_right.add(Flatten())
 
model = Sequential()
model.add(Merge([model_left,model_right], mode='concat'))
 
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
model.fit([left_x_train,right_x_train], y_train, batch_size=train_batch_size, epochs=training_iters, verbose=2,
          callbacks=[cb],validation_split=0.2,
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
pred_y = model.predict([left_x_test,right_x_test], batch_size=test_batch_size)
score = roc_auc_score(y_test,pred_y)
```

----------
