import keras.backend as K
from network import *
import pandas as pd
import sys
sys.path.append('./')
sys.path.append('../')
from new_process import *
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
import keras


if __name__ == '__main__':
    params = {
        # 网络参数
        "input_shape": (4096, 24),
        'Feature_shape': (2,),
        # "conv_subsample_lengths": [],
        # "conv_subsample_lengths": [1, 2, 1, 2,1,2],
        "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "conv_filter_length": 16,
        "conv_num_filters_start": 32,
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
        "window": 4096,

        "learning_rate": 0.001,

        "generator": True,

        "save_dir": "saved",
        "num_categories": 9,
        # 训练参数
        "batch_size": 132,
        'STEPS_PER_EPOCH': 60,
        'EPOCHS': 60,
        ###
        'name': 'final_model'
    }
    BASE_DIR = "/media/jdcloud1/Test/TEST8000/"

    model = model_from_json(open(params['name']+'__architecture.json').read())
    model.summary()

    #加载权重  resnet_feature
    model.load_weights(params['name']+'_beast' +'_weight'+'.hdf5')

    files=os.listdir(BASE_DIR)
    #files.sort(key=lambda x:int(x[3:-4]))
    test_data=list_save_all(files,BASE_DIR=BASE_DIR) #shape like (500,3) 500data include "data","age","sex"
    # testdata[:,0] means we just input the ecg data, not including age and sex
    test_data[:,0]=data_process(test_data[:,0], filter=True, normalize=False)

    # print(files)
    loop=20

    labels=[]
    File_name=[]
    for j in range(len(files)):
        print(files[j])
        File_name.append(files[j].split('.')[0])
        file=test_data[j]

        batch_x = np.array([Slices(file[0], window=params['window'],noises=False) for i in range(loop)])
        # batch_f=np.array(  [file[1:]  for i in range(loop)])

        y_pred = model.predict(batch_x, batch_size=loop, verbose=2)

        label = []
        # label.append(file)
        for i in range(9):
            if np.average(y_pred[:,i])>=0.4:
                label.append(1)
            else:
                label.append(0)
        label=np.array(label)
        # print(label==1)
        label=np.where(label==1)
        label=label[0].tolist()
        step=8-len(label)     #define how many , we need to add
        # a possibility that all the prediction is <0.5
        # so we assign this to label 0
        if step==8:
            label.append(0)
            step = 8 - len(label)
        # add the ,
        if step:
            for i in range(step):
                label.append('')
        labels.append(label)
        print(label)
    all_save_data = pd.DataFrame(columns=[ 'label1', 'label2', 'label3', 'label4',
                                          'label5', 'label6', 'label7', 'label8'], index=File_name,data=labels)
    all_save_data.index.name='File_name'
    all_save_data.to_csv('answers.csv', index=True, sep=",")
    print('successfully save to csv file answer.csv')
