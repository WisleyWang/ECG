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
# import keras
from keras.regularizers import l1_l2, l1, l2
import keras.backend as K

from keras.engine.topology import Layer
#标准化后，reku，dropout


def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import LeakyReLU

    layer = BatchNormalization()(layer)
    # layer = l2(0.003)(layer)
    layer = Activation(params["conv_activation"]) (layer)   #relu
    layer=PReLU()(layer)
    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)   #0.2

    return layer

def reg_l2(weight_matrix):

    x=np.dot(weight_matrix.T,weight_matrix)
    E = 1-np.eye(x.shape)
    y=x*E
    return 2.2*np.linalg.norm(y)


####
class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        # print("WQ.shape", WQ.shape)

        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        # print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)



#添加一层卷积层
def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D

    layer1 = SeparableConv1D(
        filters=num_filters,  # 核的数量
        kernel_size=filter_length,  # 核的尺寸
        strides=subsample_length,  # 步长

        padding='same',  # 填充
        kernel_regularizer=reg_l2,
        kernel_initializer=params["conv_init"])(layer)  # 初始化核：正态分布
    layer2 = SeparableConv1D(
        filters=num_filters,  # 核的数量
        kernel_size=32,  # 核的尺寸

        strides=subsample_length,  # 步长
        padding='same',  # 填充
        kernel_regularizer=reg_l2,
        kernel_initializer=params["conv_init"])(layer)  # 初始化核：正态分布


    layer = concatenate([layer1, layer2], axis=2)
    layer = SeparableConv1D(
        filters=num_filters,  # 核的数量
        kernel_size=subsample_length,  # 核的尺寸
        strides=1,  # 步长
        padding='same',  # 填充
        kernel_initializer=params["conv_init"],
        kernel_regularizer=reg_l2)(layer)

    layer = _bn_relu(layer, **params)  # 多个卷积层后来个relu
     
    return layer


#根据json文件 添加一系列卷积层
def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]: #遍历步长列表 不带残差的网络结构
        layer1 = add_conv_weight(
                    layer,
                    params["conv_filter_length"],  #卷积核的长度：16
                    params["conv_num_filters_start"], #卷积核数量：32
                    subsample_length=subsample_length, #每一层的步长
                    **params)
        layer2=add_conv_weight(
                    layer,
                    64,  #卷积核的长度：
                    params["conv_num_filters_start"], #卷积核数量：32
                    subsample_length=subsample_length, #每一层的步长
                    **params)
        layer3=add_conv_weight(
                    layer,
                    128,  #卷积核的长度：
                    params["conv_num_filters_start"], #卷积核数量：32
                    subsample_length=subsample_length, #每一层的步长
                    **params)
        layer=concatenate([layer1,layer2,layer3],axis=2)

        layer=add_conv_weight(
                    layer,
                    1,  #卷积核的长度：
                    params["conv_num_filters_start"], #卷积核数量：32
                    subsample_length=subsample_length, #每一层的步长
                    **params)
        layer = _bn_relu(layer, **params)  #多个卷积层后来个relu
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
        return K.concatenate([x, y], axis=2)  #通道增加一倍

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2     #通道数增加一倍？
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)  #先池化一下 #subsample_length是池化尺寸
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

    # layer = Self_Attention(16)(layer)
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

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.core import Reshape
    from keras.layers.wrappers import TimeDistributed
    # from keras.layers.core import Flatten
    # layer=Flatten()(layer)
    # layer=Dense(256,activation='relu')(layer)
    #
    # layer = Dropout(0.5)(layer)


    # layer = GlobalAveragePooling1D()(layer)
    layer=Flatten()(layer)



    layer = Dropout(0.3)(layer)
    # layer=concatenate([layer,Feature])
    layer=Dense(params['num_categories'])(layer)
    # layer = Dropout(0.5)(layer)
    # out1 = Dense(1,activation='relu')(layer)
    # out2 = Dense(1, activation='relu')(layer)
    # out3 = Dense(1, activation='relu')(layer)
    # out4 = Dense(1, activation='relu')(layer)
    # out5 = Dense(1, activation='relu')(layer)
    # out6 = Dense(1, activation='relu')(layer)
    # out7 = Dense(1, activation='relu')(layer)
    # out8 = Dense(1, activation='relu')(layer)
    # out9 = Dense(1, activation='relu')(layer)
    # layer = concatenate([out1,out2,out3,out4,out5,out6,out7,out8,out9])
    return Activation('sigmoid')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam,sgd

    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))
    if params["num_categories"]==9:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',Normal,AF,FDAWB,CRBBB,LAFB,PVC,PAC,ER,TWC,ALL])
        # model.compile(loss=DSC_loss,
        #               optimizer=optimizer,
        #               metrics=['accuracy',Normal,AF,FDAWB,CRBBB,LAFB,PVC,PAC,ER,TWC,ALL])
    elif params["num_categories"]==1:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',f1])
    else:
        print('error in model.compile')

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
    # Feature=Input(shape=params['Feature_shape'], dtype='float32',
    #                name='feature')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)



    output = add_output_layer(layer, **params)
    model = Model(inputs=inputs, outputs=output)
    add_compile(model, **params)
    return model








'''
定义损失函数
'''

smooth = 1.
from keras import backend as K
######################################### Dice Similarity Coefficient  Dice和DSC一个东西 ######################################################
#https://blog.csdn.net/a362682954/article/details/81179276
def DSC(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

'''
定义损失函数
'''

smooth = 1.
from keras import backend as K
######################################### Dice Similarity Coefficient  Dice和DSC一个东西 ######################################################
#https://blog.csdn.net/a362682954/article/details/81179276
def DSC(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def DSC_loss(y_true,y_pred):
    return 1- DSC(y_true,y_pred)
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed


