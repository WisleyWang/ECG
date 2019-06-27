import keras
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import  sklearn
from scipy.io import loadmat
import os
import pandas as pd
from math import isnan
import biosppy
import random
import sys
sys.path.append('./')
sys.path.append('../')
import random
from tqdm import tqdm
from sklearn.decomposition import PCA


'''
get files and labels
only use in train data
'''
def get_train_data(files_dir):
    reference = pd.read_csv(files_dir,
                            sep=',')  # File_name,label1,label2,label3,label4,label5,label6,label7,label8
    reference.columns = ['name', '0', '1', '2', '3', '4', '5', '6', '7']
    # files=reference['name'].tolist()
    files = []
    labels = []
    for i in range(len(reference['name'].tolist())):
        lists = reference.loc[i].tolist()
        files.append(lists[0])
        label = np.zeros(9)
        for j in lists[1:]:
            if not isnan(j):
                label[int(j)] = 1
        labels.append(label)
    return files,labels


# from a mat get data,sex,age , data_row=['I','II','III','aVF','aVR','aVL','V1','V2','V3','V4','V5','V6']
def get_feature(wav_file, BASE_DIR='//media/jdcloud/Train/'):
    mat=loadmat(BASE_DIR+wav_file)
    keys= ['I','II','III','aVF','aVR','aVL','V1','V2','V3','V4','V5','V6']
    data=np.asarray([mat[i] for i in keys]).squeeze(axis=1)   ##not slice
    sex=mat['sex'][0]

    if sex == 'male':
        sex = 1
    elif sex == 'female':
        sex = 2
    else:
        sex = 3
    age=mat['age'][0][0]
    return data,sex,age


'''
 according to the scv file to read the data from train_dir
 the order will follow the labels from the function get_train_data()
 it will return a array like [[12lead,sex,age],[12lead,sex,age]] so you can use slice to take the ecg data
'''
def list_save_all(files_name,BASE_DIR='//media/jdcloud/Train/'):
    all_data = []
    for i in tqdm(files_name):
        data, sex, age = get_feature(i, BASE_DIR=BASE_DIR)
        # feature=get_powerft(data)
        all_data.append([data, age, sex])


    return np.array(all_data)

############# find power feature--------------------------------
def get_powerft(data):
    all_feature=[]
    _, feature1, feature2, feature3, feature4, feature5 = biosppy.signals.eeg.get_power_features(signal=data.T,
                                                                                                 sampling_rate=500,
                                                                                                 size=2, overlap=0.5)
    pca = PCA(n_components=8, copy=True)
    for i in range(5):
        feature= pca.fit_transform(eval('feature{}'.format(i + 1)).T)
        all_feature.append(feature.flatten())  #8*12=96
    return list(np.asarray(all_feature).flatten())  ### 8*12*5=480


# visual a data ,data_shape=(12,-1)
def visual(data):
# ###  visual
    fig,axs=plt.subplots(12,1) #
    for i in range(12):
        axs[i].plot(data[i])
    plt.show()

'''
#####################################  here is for data process  #####################################
'''

# 进行归一化
def Normalize(v):
    v=np.array(v)
    v= (v - v.mean(axis=1).reshape((v.shape[0], 1))) / (v.max(axis=1).reshape((v.shape[0], 1))
                                                        -v.min(axis=1).reshape((v.shape[0], 1)))
    return v

'''
filter the data
'''
# def one_lead_ecg_filter(signal,sampling_rate=500):
#     filtered,_,_=st.filter_signal(signal=signal,ftype='FIR',band='bandpass',
#                                   order=int(0.3*sampling_rate),frequency=[3,45],sampling_rate=sampling_rate)
#     return filtered
#
# def twel_lead_ecg_filter(data,sampling_rate=500):
#     # data=data.T
#     for i in range(12):
#         data[i]=one_lead_ecg_filter(data[i],sampling_rate)
#     return data
import pywt

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    bior = pywt.Wavelet('bior2.6')
    level=8;
    # 分解
    coeffs = pywt.wavedec(data, bior,level)
    # 高频系数置零
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    coeffs[len(coeffs)-8] *= 1
    # 重构
    meta = pywt.waverec(coeffs, bior)
    return meta

def twel_lead_ecg_filter(data,sampling_rate=500):
    # data=data.T
    for i in range(12):
        data[i]=wavelet_denoising(data[i])
    return data


def data_process(data,filter=False,normalize=False):

    propce_data = data
    if filter :
        '''
        it will the data in 24lead------12lead noise and 12lead filter
        '''
        for i in tqdm(range(len(data))):
            propce_data[i] = twel_lead_ecg_filter(data[i])
        print("\nsuccessfully filter the data\n")

####------------------------------------------------------------------------------------
    # for i in tqdm(range(len(data))):
    #     newdata.append(np.concatenate((np.array(propce_data[i]),np.array(data[i])),axis=0))
    newdata=propce_data
##------------------------------------------------------------------------------------------------



    if normalize:
        for i in tqdm(range(len(data))):
            newdata[i]=Normalize(newdata[i])
        print("\nsuccessfully Normalize the data\n")

    return newdata

'''
#####################################  !!!here is for Main Function!!!  #####################################
'''
def get_process_data(files_name,labels,augment=False,augument_rate=0.8,filter=False,normalize=False,noise=False):

    if augment:
        all_data, labels = data_augment(files_name, labels, rate=augument_rate, showresult=False)
    else:
        all_data = list_save_all(files_name)
    print("\n successfully read all the data.\n")

    all_data[:,0]=data_process(all_data[:,0],filter=filter,normalize=normalize)

    print("\n successfully preprocess all the data.\n")
    #print("\n*********************************************** START TRAINING ***********************************************\n")

    return all_data,labels


####slice
def Slices(data,window,noises,test_data=False):
    start = np.random.random_integers(0, data.shape[1] - window)
    data= data[:, start:start + window]
    if noises:
        if not test_data:
            if np.random.randint(0, 2):
                noise = np.random.normal(0, 0.01, data.shape)
                data += noise
    return data.T


def get_data_train(train_data,train_label,noises=False,batch_size=100,window=4500,BASE_DIR='',test_data=False):
    assert BASE_DIR is not None, "please input BASE_DIR"


    #K表示batch的index，i表示随机产生的0到测试集数量的数
    # print(train_data.shape)
    while True:
        batch_list = []
        label_list = []
        # batch_size=np.random.random_integers(30,250)
        for i in range(batch_size):
            j = np.random.random_integers(0, len(train_data)-1)
            batch_list .append (train_data[j])
            label_list .append (train_label[j])
        batch_list=np.array(batch_list)
        label_list=np.array(label_list)
        batch_x = np.array([Slices(file,window,noises,test_data=test_data) for file in batch_list[:,0]])
        # batch_f=np.array([file[1:] for file in batch_list[:]])
        batch_y = np.array([label for label in label_list[:]])
        yield (batch_x, batch_y)





'''
here is for data augment
'''
def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x):
    result = []
    for i in range(12):
        new_y=amplify(x[i])
        result.append(new_y)
    return np.array(result)

'''
attention!!!
if u use data_augment,you donnot have to read the data 
what you need is input the csv file to the function
'''
def data_augment(train_data, train_label, rate=0.8, showresult=False,):
    column_index = ['label0', 'label1', 'label2', 'label3', 'label4',
                    'label5', 'label6', 'label7', 'label8']
    all_train_data = pd.DataFrame(columns=column_index, index=train_data, data=train_label)
    quantity_info = all_train_data.apply(lambda x: x.sum(), axis=0)
    max_quantity = quantity_info.max()  # max_quantity

    abc = []
    abc_label = []

    for i in tqdm(range(8)):
        quantity_need = int((max_quantity - len(all_train_data[all_train_data[column_index[i]] == 1.0])) * rate)
        #     print(quantity_need)
        now_class_one = all_train_data[all_train_data[column_index[i]] == 1.0].index

        for j in range(quantity_need):
            seed = np.random.randint(0, len(now_class_one) - 1)
            now_data = get_feature(now_class_one[seed], BASE_DIR='//media/jdcloud/Train/')
            new_now_data = [augment(now_data[0]), now_data[1], now_data[2]]

            abc.append(new_now_data)
            abc_label.append(all_train_data.loc[now_class_one[seed]].tolist())
    train_data=list_save_all(train_data)
    # concat the daya
    train_data = np.concatenate((np.array(train_data),np.array(abc)),axis=0)
    train_label = np.concatenate((train_label , abc_label),axis=0)
    # print(train_data.shape,train_label.shape)
    # shuffle
    # cc = list(zip(train_data, train_label))
    # random.shuffle(cc)
    # train_data[:], train_label[:] = zip(*cc)

    if showresult:
        new_all_train_data = pd.DataFrame(columns=column_index, data=train_label)
        quantity_info = new_all_train_data.apply(lambda x: x.sum(), axis=0)
        print(quantity_info)

    return np.array(train_data), np.array(train_label)
