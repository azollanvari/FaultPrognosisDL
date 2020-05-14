#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:49:24 2018

@author: amin
@editors: saeid & qasymjomart
"""

#%% training data
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import scipy.io as io
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
# data_dir= '/Users/kassymzhomart.kunanb/Documents/TFP/4Saied_DNN_Transformer/'

# os.chdir(data_dir)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


#%%REd data from csv
#fname = os.path.join(data_dir, 'Train.csv') #for csv
# there are 8 subjects, and 'subjects' array is to control which subjects to load 
#f = open(fname)
#dataTr = f.read()
#f.close()
#
#lines = dataTr.split('\n')
#header = lines[0].split(',')
#Now, convert all lines of data into a Numpy array.
#import numpy as np
#
#float_data_train = np.zeros((len(lines), len(header)))
#for i, line in enumerate(lines):
#    values = [float(x) for x in line.split(',')[0:]]
#    float_data_train[i, :] = values
#
#from matplotlib import pyplot as plt
#temp = float_data_train[:, 1] #voltage
#plt.plot(range(len(temp)), temp)


#%%
#lines = lines[1:]
#print(header)
file_ind = ['2.5', '3', '3.5', '4', '4.5', '5', 
            '5.5', '6', '7', '8', '8.4', '9',
            '9.5', '10', '11', '12', '13', '15']



Fs=3000
st=0.02 #stationary interval in terms of second
L=int(st*Fs) #block length

StartOfSignal=[80000, 45000, 52000, 70000, 70000, 42000, 30000, 34000,
               50000, 57000, 56000, 75000, 47000, 28000, 50000, 50000,
               50000, 48000]

#%% divide between train, validation, and test


data_train_list = []
data_valid_list = []
data_test_list = []
import numpy as np
k=-1
for file in file_ind:
    k = k + 1
#   there is a file for each subject in erp-data folder named by the following format: subject1.mat
    f = io.loadmat('load_current_'+file+'A.mat')
#   go through matlab maps to get the data
    a = float(file)*np.ones((len(f['Data1_AI_1']), 1))
    b = np.double(f['Data1_AI_1'])
    a = a[StartOfSignal[k]:]
    b = b[StartOfSignal[k]:]
    N=len(b)
    I=np.floor(N/L)-1  #total number of observations (N/L)
    Ntest=int(np.floor(I/4))   #we set 1/4 of I for test
    Nvalid = int(np.floor(3*I/16)) #validation is 1/4 of the 3/4*I (training) = 3/16
    Ntrain=int(I-Nvalid-Ntest)
    train_ind_max = Ntrain*L
    valid_ind_max = train_ind_max+Nvalid*L
    test_ind_max = valid_ind_max+Ntest*L
    
    data_temp_train = np.concatenate((a[0:train_ind_max], b[0:train_ind_max]), axis=1)
    data_temp_valid = np.concatenate((a[train_ind_max:valid_ind_max], b[train_ind_max:valid_ind_max]), axis=1)
    data_temp_test = np.concatenate((a[valid_ind_max:test_ind_max], b[valid_ind_max:test_ind_max]), axis=1)
    data_train_list.append(data_temp_train)
    data_valid_list.append(data_temp_valid)
    data_test_list.append(data_temp_test)

#lines = lines[1:]
#print(header)
#print(len(lines))

data_train = np.concatenate(data_train_list, axis=0) #convert the list to np arrays
data_valid = np.concatenate(data_valid_list, axis=0)
data_test = np.concatenate(data_test_list, axis=0)

from matplotlib import pyplot as plt
temp = data_train[:, 1] #voltage
plt.plot(range(len(temp)), temp)

from matplotlib import pyplot as plt
temp = data_valid[:, 1] #voltage
plt.plot(range(len(temp)), temp)


from matplotlib import pyplot as plt
temp = data_test[:, 1] #voltage
plt.plot(range(len(temp)), temp)



#%% Normalize using mean and std of training
#mean = data_train.mean(axis=0)
#data_train -= mean
#std = data_train.std(axis=0)
#data_train /= std
#
#
#data_valid -= mean
#data_valid /= std
#data_test -= mean
#data_test /= std


dmin=data_train.min(axis=0)
dmax=data_train.max(axis=0)
max_min = dmax - dmin
data_train  = (data_train-dmin)/max_min
data_valid  = (data_valid-dmin)/max_min
data_test  = (data_test-dmin)/max_min

#%% data generator

window = L
step = 1
delay = 0
batch_size = 32

def generator(data, window, delay, min_index, max_index,
              shuffle=False, batch_size=batch_size, step=step):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + window
    
    while 1:
        if shuffle:
            sample_ind = np.random.randint(
                    min_index, max_index//window, size=batch_size)
            rows = sample_ind*window
        else:
            if i >= max_index:
                    i = min_index + window
            rows = np.arange(i, min(i + batch_size*window, max_index), window)
            i = rows[-1]+window
        samples = np.zeros((len(rows),
                            window // step, 
                            (data.shape[-1]-1))) #second argument is the number of time stamps (1440/6)
                            # first argument is number of samples (indepedent)
                            #third arg is e.g. 12 (size of features)
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - window, rows[j], step)
            samples[j] = data[indices,1:] #indexing reducing dimenion but slicing doesn't
            #d_min = np.amin(data[indices,1:],axis=0)
            #samples[j] = (data[indices,1:]-d_min)/(np.amax(data[indices,1:],axis=0)-d_min)
            #d_min = np.amin(data[indices,1:],axis=0)
            targets[j] = data[rows[j]-1 + delay][0] #target is the first column
        yield samples, targets



#%%
train_gen = generator(data_train,
                      window=window,
                      delay=delay,
                      min_index=0,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size) #see what None does
#%%
val_gen = generator(data_valid,
                    window=window,
                    delay=delay,
                    min_index=0,
                    max_index=None,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

#%%
test_gen = generator(data_test,
                    window=window,
                    delay=delay,
                    min_index=0,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)

#%%
val_steps = data_valid.shape[0]//(window*batch_size)
test_steps = data_test.shape[0]//(window*batch_size)


#import numpy as np
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


units = [[16],[16,16,16],[16,16,16,16,16],[16,16,16,16,16,16,16],
         [32],[32,32,32],[32,32,32,32,32],[32,32,32,32,32,32,32],
         [64],[64,64,64],[64,64,64,64,64],[64,64,64,64,64,64,64], 
         [128],[128,128,128],[128,128,128,128,128],[128,128,128,128,128,128,128],
         [256],[256,256,256],[256,256,256,256,256],[256,256,256,256,256,256,256]]

units_test = [[64]]

# In[16]:


#import keras
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, GRU, CuDNNLSTM, CuDNNGRU, Dense
#from decimal import *
import math
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD
#from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
#import pickle
from keras import backend as K


for num_units in units: 
    #K.clear_session()
    filenameFig = 'Current_Bidirectional_LSTM'
    
    for num_unit in num_units:
      filenameFig = filenameFig + '_' + str(num_unit)
    filename = filenameFig
    filename_model = filename + '.h5'
    
    #keras.callbacks.LearningRateScheduler
    
    # callbacks_list = [
    #     keras.callbacks.EarlyStopping(
    #             monitor='val_loss',
    #             patience=150,),
    #             keras.callbacks.ModelCheckpoint(
    #                     filepath=filename + '.h5',
    #                     monitor='val_loss',
    #                     save_best_only=True,
    #                     )
    #             ]
    
    model = Sequential()
    
    #model.add(layers.Reshape((window//step,1)))
    if len(num_units) == 1:
         
        model.add(Bidirectional(CuDNNLSTM(num_units[0]),
                         input_shape=(window//step,1)))
    
    else:
        model.add(Bidirectional(CuDNNLSTM(num_units[0],
                     return_sequences=True),
                     input_shape=(window//step,1)))

        
        for i in range(1,len(num_units) - 1):
            model.add(Bidirectional(CuDNNLSTM(num_units[i],
                                 return_sequences=True)))
        
        
        model.add(Bidirectional(CuDNNLSTM(num_units[-1])))

    model.add(Dense(1))
    #print('Here!')
    
    #model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=['mae','mape'])
    model.compile(optimizer=RMSprop(lr=0.0001), loss='mse', metrics=['mae','mape'])
    history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=150,
                              validation_data=val_gen,
                              validation_steps=val_steps)
    #save model
    model.summary()
    
    
    #save model  
        
    model.save(filename + '.h5')    
    
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(history.history, handle)

    data_test_for_evaluate = data_valid[:,1:].reshape((len(data_valid)//window, window, 1))
    targets_test = data_valid[:,:1].reshape((len(data_valid)//window, window, 1))
    sample = np.zeros((1, window // step, 
                                (data_valid.shape[-1]-1)))
    predicted_targets = np.zeros((len(data_test_for_evaluate),))
    true_targets = np.zeros((len(data_test_for_evaluate),))
    
    for i in range(0,len(data_test_for_evaluate)):
        true_targets[i] = targets_test[i,window-1]
    target_mean = true_targets.mean(axis=0)
    
    for i in range(0,len(data_test_for_evaluate)):
        sample[0] = data_test_for_evaluate[i,]
        predicted_targets[i]=model.predict(sample)
    
    MSE = sum(abs(predicted_targets-true_targets)**2)/len(true_targets)
    MAE = sum(abs(predicted_targets-true_targets))/len(true_targets)
    
    RRSE = 100 * np.sqrt(MSE * len(true_targets) / (sum(abs(true_targets-target_mean)**2)))
    RAE = 100 * MAE * len(true_targets) / sum(abs(true_targets-target_mean)) 
    
    print('MSE: ', MSE)
    print('MAE: ', MAE)
    print('RRSE: ', RRSE)
    print('RAE: ', RAE)
    print('target_mean: ', target_mean)
    print('len(true_targets): ', len(true_targets))
    print(sum(abs(true_targets-target_mean)**2))
    print(sum(abs(true_targets-target_mean))/len(true_targets))
    #plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    # if we would like to read from a saved "history"
    #import matplotlib.pyplot as plt
    #import math
    #fig=plt.figure()
       
    epoch_count = range(1, len(history.history['loss']) + 1)
    #plt.plot(epoch_count, np.array(d['loss']), 'b--', labe$\mathit{M}$=$\mathit{L}$='training MAE')
    #plt.plot(epoch_count, np.array(d['val_loss']), 'r-', labe$\mathit{M}$=$\mathit{L}$='validation MAE')
    plt.plot(epoch_count, np.array(history.history['loss']), 'b--')
    plt.plot(epoch_count, np.array(history.history['val_loss']), 'r-')
    y=history.history['val_loss']
    ymin = min(y)
    xpos = y.index(min(y))
    xmin = epoch_count[xpos]
    y=history.history['val_mae']
    yymin = min(y)
    
    print('MSE by formula: ', MSE, ' MSE by model: ', ymin)
    
    string1 = 'MSE = ' + '%.2E' % float(ymin)
    string2 = '\n' + 'RAE = ' + to_str(round(RAE,2)) + '%' + '\n' + 'RRSE = ' + to_str(round(RRSE,2)) + '%'
    string = string1 + string2
    ax.annotate(string, xy=(xmin, ymin),xycoords='data',
                 xytext=(-80, 85), textcoords='offset points',
                 bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                 size=12,
                 arrowprops=dict(arrowstyle="->"))
    plt.title('$\mathit{N}$=' + str(len(num_units)) + ', $\mathit{M}$=$\mathit{L}$=' + str(num_units[0]))
    #xint = range(min(epoch_count), 15,2)
    xint = range(min(epoch_count)-1, math.ceil(max(epoch_count)),20)
    plt.xticks(xint)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend(loc="best")
    filename1 = filename + '_loss' 
    fig.set_size_inches(5.46, 3.83)
    fig.savefig(filename1 + '.pdf', bbox_inches='tight')
    
    
    #1st element of score: MSE (keras)
    #2nd element of score: MSE 
    #3nd element of score: MAE
    #4th element of score: RRSE
    #5th element of score: RAE
    score = []
    score.append(ymin)
    score.append(MSE)
    score.append(MAE)
    score.append(RRSE)
    score.append(RAE)
    filenameTXT = filename + '.txt'
    np.savetxt(filenameTXT, score)
    
    K.clear_session()
    del model





