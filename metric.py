


'''
using this






import metric

metrics=['accuracy',Normal,AF,FDAWB,CRBBB,LAFB,PVC,PAC,ER,TWC,ALL]






'''





from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.objectives import *
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from tensorflow.contrib.rnn import *


##---------------------------------------------------
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


##____Normal 	AF 	FDAWB 	CRBBB 	LAFB 	PVC 	PAC 	ER 	TWC 	得分__

def Normal(y_true, y_pred):
    return f1(y_true[:,0],y_pred[:,0])
def AF(y_true, y_pred):
    return f1(y_true[:,1],y_pred[:,1])
def FDAWB(y_true, y_pred):
    return f1(y_true[:,2],y_pred[:,2])
def CRBBB(y_true, y_pred):
    return f1(y_true[:,3],y_pred[:,3])
def LAFB(y_true, y_pred):
    return f1(y_true[:,4],y_pred[:,4])
def PVC(y_true, y_pred):
    return f1(y_true[:,5],y_pred[:,5])
def PAC(y_true, y_pred):
    return f1(y_true[:,6],y_pred[:,6])
def ER(y_true, y_pred):
    return f1(y_true[:,7],y_pred[:,7])
def TWC(y_true, y_pred):
    return f1(y_true[:,8],y_pred[:,8])
def ALL(y_true, y_pred):
    score=K.zeros(1)
    for i in range(9):
        score+=f1(y_true[:,i],y_pred[:,i])
    return score/9

