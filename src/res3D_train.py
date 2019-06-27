import sys
sys.path.append('./')
sys.path.append('../')
from res3D_network import *
from process import *
import pandas as pd
import numpy as py
from sklearn.model_selection import train_test_split
from keras.callbacks import *
import keras
from keras.models import *

params = {
    # 网络参数
    "input_shape": (24,1024,1),
    'Feature_shape':(2,),
    # "conv_subsample_lengths": [],
    # "conv_subsample_lengths": [1, 2, 1, 2,1,2],
    "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "conv_filter_length": (3,16),
    "conv_num_filters_start": 16,
    "conv_init": "he_normal",
    "conv_activation": "relu",
    "conv_dropout": 0.5,
    # 网络结构选择
    'is_regular_conv': False,  # 常规卷积
    "compile": True,  # 残差网络
    "conv_num_skip": 2,
    # 多少增加一次通道数
    "conv_increase_channels_at": 4,
    # 窗口长度
    "window": 1024,

    "learning_rate": 0.001,

    "generator": True,

    "save_dir": "saved",
    "num_categories": 9,
    # 训练参数
    "batch_size":128,
    'STEPS_PER_EPOCH': 80,
    'EPOCHS': 50,
    ###
    'name':'2D_resnet'
}
model= build_network(**params)
model.summary()
reference_dir="//media/jdcloud/reference.csv"
BASE_DIR='//media/jdcloud/Train/'
files,labels=get_train_data(reference_dir)
train_data, val_data,train_label,val_label = train_test_split(files,labels,test_size=0.3, random_state=42)
val_data, test_data,val_label,test_label = train_test_split(val_data,val_label,test_size=0.4, random_state=42)

# 加载模型
# model = model_from_json(open('./' + 'test' + 'final_network_architecture.json').read())
# 保存模型
json_string = model.to_json()
open('./'  + params['name']+'_architecture.json', 'w').write(json_string)

# # load weight

for i in range(0,30):
    # model.load_weights(params['name'] + '_beast' + '_weight' + '.hdf5')
    add_compile(model, **params)
    best_weights_filepath = params['name']+'_beast' +'_weight'+'.hdf5'
    earlyStopping=EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                               save_best_only=True, mode='auto',save_weights_only=True)
    reducelr= ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=4, verbose=0, mode='auto', epsilon=0.0001, cooldown=2, min_lr=0)
  #
    history=model.fit_generator(get_data2fit(train_data,train_label,BASE_DIR=BASE_DIR,**params) , steps_per_epoch=params['STEPS_PER_EPOCH'], epochs=params['EPOCHS'],
                            validation_data=get_data2fit(val_data,val_label,BASE_DIR=BASE_DIR,**params),validation_steps=128,
                           callbacks=[earlyStopping,saveBestModel,reducelr],shuffle=True)
'''
  #
  #                              callbacks=[earlyStopping,saveBestModel,reducelr],shuffle=True)
  # # validation_data=generate_3Ddata(val_data,val_label,number=2,BASE_DIR=BASE_DIR,**params),validation_steps=15,
  #   x,y=get_data2fit(train_data, train_label, size=500, BASE_DIR=BASE_DIR, **params)
# model.fit(x,y,batch_size=20,epochs=30,validation_split=0.1,shuffle=True,callbacks=[earlyStopping,saveBestModel,reducelr])
'''