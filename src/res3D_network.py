import sys
sys.path.append('./')
sys.path.append('../')
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.objectives import *
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from tensorflow.contrib.rnn import *
from metric import *
from keras.regularizers import *

import keras.backend as K

#标准化后，reku，dropout
def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)

    layer = Activation(params["conv_activation"])(layer)   #relu

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)   #0.2

    return layer
#添加一层卷积层
def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    from keras.regularizers import l1_l2,l1,l2
    layer = Conv2D(
        filters=num_filters, #核的数量
        kernel_size=filter_length,   #核的尺寸
        strides=subsample_length,    #步长
        padding='same',      #填充
        kernel_initializer=params["conv_init"]
        ,kernel_regularizer=l2(0.01))(layer)  #初始化核：正态分布
    return layer


#根据json文件 添加一系列卷积层
def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:  # 遍历步长列表 不带残差的网络结构
        layer1 = add_conv_weight(
            layer,
            params["conv_filter_length"],  # 卷积核的长度：16
            params["conv_num_filters_start"],  # 卷积核数量：32
            subsample_length=subsample_length,  # 每一层的步长
            **params)
        layer2 = add_conv_weight(
            layer,
            8,  # 卷积核的长度：
            params["conv_num_filters_start"],  # 卷积核数量：32
            subsample_length=subsample_length,  # 每一层的步长
            **params)
        layer3 = add_conv_weight(
            layer,
            16,  # 卷积核的长度：
            params["conv_num_filters_start"],  # 卷积核数量：32
            subsample_length=subsample_length,  # 每一层的步长
            **params)
        layer = concatenate([layer1, layer2, layer3], axis=3)
        layer = add_conv_weight(
            layer,
            1,  # 卷积核的长度：
            params["conv_num_filters_start"],  # 卷积核数量：32
            subsample_length=subsample_length,  # 每一层的步长
            **params)
        layer = _bn_relu(layer, **params)  # 多个卷积层后来个relu
    return layer


# 这一个函数是 每四次池化通道扩充一次   【【（bn +relu）：第一次不加 其他都加】 +卷积】X2  最后add池化
def resnet_block(
        layer,
        num_filters,    #————————#卷积核的数量
        subsample_length,  #池化的长度，就是核的尺寸
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)   #创建一个和x一样的张量value为0
        return K.concatenate([x, y], axis=3)  #通道增加一倍

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 4
        shape[3] *= 2     #通道数增加一倍？
        return tuple(shape)

    shortcut = MaxPooling2D(pool_size=subsample_length)(layer)  #先池化一下 #subsample_length是池化尺寸
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0   #conv_increase_channels_at：4
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut) #增加通道数？

    for i in range(params["conv_num_skip"]):  #2
        if not (block_index == 0 and i == 0):  #relu下
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0, #一开始不drop，之后才有
                **params)
        layer = add_conv_weight(    #添加一个卷积层
            layer,
            params["conv_filter_length"], #卷积长度 固定16
            num_filters, ##核的数量
            subsample_length if i == 0 else 1, #卷积的步长
            **params)
    layer = Add()([shortcut, layer])  #添加到池化之后
    return layer

#计算卷积层核的数量  # 32
#第一次  Int取整了
def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):

    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],  #长度
        params["conv_num_filters_start"], #数量
        subsample_length=1,  #卷积的步长
        **params)
    layer = _bn_relu(layer, **params) #relu

    for index, subsample_length in enumerate(params["conv_subsample_lengths"]): #生成带有index的迭代器，总共16次
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)  #32  计算每一个resent_block中卷积核的数量
        layer = resnet_block(
            layer,
            num_filters,  ##核的数量
            subsample_length,#池化长度以及卷积步长
            index,   #block数量
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer,Feature, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.core import Reshape
    from keras.layers.wrappers import TimeDistributed
    # from keras.layers.core import Flatten
    # layer=Flatten()(layer)
    # layer=Dense(256,activation='relu')(layer)
    #
    # layer = Dropout(0.5)(layer)


    layer = GlobalAveragePooling2D()(layer)
    layer = Dropout(0.5)(layer)
    layer=concatenate([layer,Feature])
    layer=Dense(128)(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(params["num_categories"])(layer)
    return Activation('sigmoid')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam,sgd

    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy',Normal,AF,FDAWB,CRBBB,LAFB,PVC,PAC,ER,TWC,ALL])

    #     optimizer = sgd(
    #         lr=0.01,
    #         clipnorm=params.get("clipnorm", 1),decay=0.0001,momentum=0.5)
    #
    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer=optimizer,
    #                   metrics=['accuracy'])
def my_complex_loss_graph(Feature):
    target=Dense(32)(Feature)
    target=LeakyReLU()(target)

    return target



def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    Feature=Input(shape=params['Feature_shape'], dtype='float32',
                   name='feature')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)



    output = add_output_layer(layer,Feature, **params)
    model = Model(inputs=[inputs,Feature], outputs=[output])
    add_compile(model, **params)
    return model
