
import numpy as np
import matplotlib.pyplot as plt
import  sklearn
from scipy.io import loadmat
import os
import pandas as pd
from math import isnan
import pywt
import biosppy
import cv2
from io import StringIO
from pyts.image import GramianAngularField


# 进行归一化
def normalize(v):
    v= (v - v.mean(axis=1).reshape((v.shape[0], 1))) / (v.max(axis=1).reshape((v.shape[0], 1))
                                                        -v.min(axis=1).reshape((v.shape[0], 1)))
    return v

# # loadmat打开文件
# def get_feature(wav_file, window, BASE_DIR):
#     mat = loadmat(BASE_DIR + wav_file)
#     dat = mat["data"]
#     start=np.random.random_integers(0,dat.shape[1]-window)
#     # feature = dat[0:12]
#     return normalize(dat[:,start:start+window]).T


# from a mat get data,sex,age , data_row=['I','II','III','aVF','aVR','aVL','V1','V2','V3','V4','V5','V6']
def get_feature(wav_file, BASE_DIR='//media/jdcloud/Train/'):
    mat=loadmat(BASE_DIR+wav_file)
    keys= ['I','II','III','aVF','aVR','aVL','V1','V2','V3','V4','V5','V6']
    data=np.asarray([mat[i] for i in keys]).squeeze(axis=1)   ##not slice
    sex=mat['sex'][0]
    age=mat['age'][0][0]
    return data,sex,age

# visual a data ,data_shape=(12,-1)
def visual(data):
# ###  visual
    fig,axs=plt.subplots(12,1) #
    for i in range(12):
        axs[i].plot(data[i])
    plt.show()

## get files and labels
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



####slice
def Slices(file,window,BASE_DIR='//media/jdcloud/Train/'):
    data,sex,age=get_feature(file, BASE_DIR=BASE_DIR)
    # data=normalize(data)
    start = np.random.random_integers(0, data.shape[1] - window)

    # feature = dat[0:12]
    # return normalize(data[:, start:start + window]).T



    data= data[:, start:start + window]
    # if np.random.randint(0,2):
    #     noise=np.random.normal(0,0.1,data.shape)
    #     data=data+noise
    # data=twel_lead_ecg_filter(data)
    return data.T

def get_otherfeature(file,BASE_DIR='//media/jdcloud/Train/'):
    data, sex, age = get_feature(file, BASE_DIR=BASE_DIR)

    if sex == 'male':
        sex = 1
    elif sex == 'female':
        sex = 2
    else:
        sex = 3
    feature = [sex, age]
    return feature


'''

here is for filter!!!
'''

from biosppy import tools as st
def one_lead_ecg_filter(signal,sampling_rate=500):
    filtered,_,_=st.filter_signal(signal=signal,ftype='FIR',band='bandpass',
                                  order=int(0.3*sampling_rate),frequency=[3,45],
                                           sampling_rate=sampling_rate)
    return filtered


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

# def filter_all_data(data):
#     # for i in tqdm(range(len(data))):
#     print(data.shape)
#     print(data[1].shape)
#     for i in range(len(data)):
#         data[i]=twel_lead_ecg_filter(data[i])
#     return data




def get_data_train(train_data,train_label,batch_size=100,window=2560,BASE_DIR=''):
    assert BASE_DIR is not None, "please input BASE_DIR"
    # batch_train_datas=np.empty(shape=(batch_size,window,12))
    # batch_train_labels=np.zeros(shape=(batch_size,2))
    # step=len(train_data)//batch_size

    #K表示batch的index，i表示随机产生的0到测试集数量的数
    while True:
        batch_list = []
        label_list = []
        for i in range(batch_size):
            j = np.random.random_integers(0, len(train_data) - 1)
            batch_list.append(train_data[j])
            label_list.append(train_label[j])

        # batch_list = train_data[i * batch_size: i * batch_size + batch_size]

        # label_list = train_label[i * batch_size: i * batch_size + batch_size]
        batch_x = np.array([Slices(file,window,BASE_DIR=BASE_DIR) for file in batch_list[:]])
        # batch_f=np.array([get_otherfeature(file,BASE_DIR=BASE_DIR) for file in batch_list[:]])
        batch_y = np.array([label for label in label_list[:]])
        yield batch_x, batch_y


def Slice_3D(file,BASE_DIR='//media/jdcloud/Train/',**params):
    data, sex, age = get_feature(file, BASE_DIR=BASE_DIR)
    # data=normalize(data)
    start = np.random.random_integers(0, data.shape[1] - params['window'])
    img=np.zeros(params['input_shape'])

    data1=data[:,start:start+params['window']]
    data2=twel_lead_ecg_filter(data1)
    img[:,:,0]=np.concatenate([data1,data2],axis=0)
    # kernel = np.ones((4, 4), np.uint8)
    # print(data.shape,params['input_shape'][2])
    # new_data = (data[:,start:start + params['window']]+1)*300
    # new_data=np.trunc(new_data).astype(np.int)
    # one_img = np.zeros((620, params['window'] + 10))

    # gasf = GramianAngularField(image_size=256, method='summation')
    # now_img = gasf.fit_transform(new_data)
    # print(now_img.shape)
    # exit()
    # for k in range(params['input_shape'][2]):
    #     fig = plt.figure()
    #     # one_img[:,:]=0
    #     # for i in range(params['window']-1):
    #     #     if ((new_data[k,i]-new_data[k,i-1]>10)&(new_data[k,i]<new_data[k,i+1])):
    #     #         one_img[new_data[k, i] - 30:new_data[k, i] +2, i + 3:i + 3 + 4] = 1
    #     #     elif ((new_data[k,i]-new_data[k,i-1]<-15)&(new_data[k,i]>new_data[k,i+1])):
    #     #         one_img[new_data[k, i] :new_data[k, i]+50 , i + 3:i + 3 + 4] = 1
    #     #     else:
    #     #         one_img[new_data[k, i] - 2:new_data[k, i] + 2, i + 3:i + 3 + 4] = 1
    #
    #     # # If we haven't already shown or saved the plot, then we need to
    #
    #     plt.rcParams['figure.dpi']=250
    #     plt.rcParams['savefig.dpi']=250
    #     plt.plot(new_data[k])
    #     plt.xticks([]), plt.yticks([])
    #     for spine in plt.gca().spines.values():
    #         spine.set_visible(False)
    #     # draw the figure first...
    #     fig.canvas.draw()
    #
    #     # Now we can save it to a numpy array.
    #     one_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     plt.close()
    #     # im_gray=plt_img[:,:,0]
    #     # one_img = cv2.erode(one_img, kernel, iterations=1)
    #     one_img=cv2.GaussianBlur(one_img[:,:,0],(3,3),1)
    #     one_img = cv2.resize(one_img, (params['input_shape'][1], params['input_shape'][0]), interpolation=cv2.INTER_LANCZOS4)
    #     img[:,:,k]=one_img
    # img[img>0]+=20

    # print(img.shape)

    return img


def generate_3Ddata(train_data,train_label,BASE_DIR='',**params):
    assert BASE_DIR is not None, "please input BASE_DIR"
    # batch_train_datas=np.empty(shape=(batch_size,window,12))
    # batch_train_labels=np.zeros(shape=(batch_size,2))
    step = len(train_data) // params['batch_size']
    # K表示batch的index，i表示随机产生的0到测试集数量的数

    while True:
        for i in range(step):
            batch_list = train_data[i * params['batch_size']: i * params['batch_size'] + params['batch_size']]

            label_list = train_label[i * params['batch_size']: i * params['batch_size'] + params['batch_size']]

            batch_x = np.array([Slice_3D(file, BASE_DIR=BASE_DIR,**params) for file in batch_list[:]])
            batch_f = np.array([get_otherfeature(file, BASE_DIR=BASE_DIR) for file in batch_list[:]])
            batch_y = np.array([label for label in label_list[:]])
            yield ([batch_x, batch_f], batch_y)

def get_data2fit(train_data,train_label,BASE_DIR='',**params):
        while True:
            batch_list = []
            label_list = []
            # batch_size=np.random.random_integers(30,250)
            for i in range(params['batch_size']):
                j = np.random.random_integers(0, len(train_data) - 1)
                batch_list.append(train_data[j])
                label_list.append(train_label[j])

            batch_x = np.array([Slice_3D(file, BASE_DIR=BASE_DIR, **params) for file in batch_list[:]])
            batch_f = np.array([get_otherfeature(file, BASE_DIR=BASE_DIR) for file in batch_list[:]])
            batch_y = np.array([label for label in label_list[:]])
            print(batch_x.shape)
            yield [batch_x, batch_f], batch_y

# # #
# BASE_DIR='//media/jdcloud/Train/'
# files=os.listdir(BASE_DIR)
#
#
# params = {
#     # 网络参数
#     "input_shape": (512,1024, 12),
#     'Feature_shape':(2,),
#     # "conv_subsample_lengths": [],
#     # "conv_subsample_lengths": [1, 2, 1, 2,1,2],
#     "conv_subsample_lengths": [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2,1,1,1,2],
#     "conv_filter_length": (3,3,3),
#     "conv_num_filters_start": 32,
#     "conv_init": "he_normal",
#     "conv_activation": "relu",
#     "conv_dropout": 0.5,
#     # 网络结构选择
#     'is_regular_conv': False,  # 常规卷积
#     "compile": True,  # 残差网络
#     "conv_num_skip": 2,
#     # 多少增加一次通道数
#     "conv_increase_channels_at": 4,
#     # 窗口长度
#     "window": 2048,
#
#     "learning_rate": 0.001,
#
#     "generator": True,
#
#     "save_dir": "saved",
#     "num_categories": 9,
#     # 训练参数
#     "batch_size":20,
#     'STEPS_PER_EPOCH': 100,
#     'EPOCHS': 60,
#     ###
#     'name':'3D_resnet'
# }
# # ###(['__globalsfor__', 'V3', 'V2', 'I', '__header__', 'V4', 'aVR', 'sex', 'V1', 'V5', 'III', 'V6', 'aVL', '__version__', 'aVF', 'II', 'age']
# # ###data ,sex, age = get_feature(files[4], BASE_DIR=BASE_DIR)
# img=Slice_3D(files[2],BASE_DIR='//media/jdcloud/Train/',**params)
# for k in range(12):
#     plt.imshow(img[:,:,k],'gray')
#     plt.show()
# print(img.shape)
# if __name__ == '__main__':
#     pass

# ###count
#     dict_sex={}
#     dict_age={}
#     name=[]
#     for file in files:
#        data,sex,age=get_feature(wav_file=file)
#        print('data_shape:',data.shape)
#        # print('sex:',sex)
#        # print('age:',age)
#        if sex not in dict_sex.keys():
#            dict_sex[sex]=1
#        else:
#             dict_sex[sex]=dict_sex[sex]+1
#        if sex =='U':
#            name.append(file)
#        if age not in dict_age.keys():
#            dict_age[age] = 1
#        else:
#            dict_age[age] = dict_age[age] + 1
#
#     print(dict_sex)
#     print(dict_age)


#     print(name)