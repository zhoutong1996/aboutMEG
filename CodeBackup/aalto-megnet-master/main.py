# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:14:22 2018

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
os.chdir('/m/nbe/scratch/braindata/izbrv/megnet/')
import tensorflow as tf
import numpy as np
from time import time
from utils import logger#, compute_covariance
from models import Model
#%%
savepath = '/m/nbe/scratch/braindata/izbrv/camcan_preproc/'

pretrained = False

h_params = dict(architecture='lf-cnn', #lf-cnn oe eegnet
                sid = 'camcan',
                n_epochs = 10000, #total number of iterations
                eval_step = 5, #evaluate validation loss each 1000 iterations
                n_batch = 200, #training batch size
                n_classes = 2, # number of classes
                n_ch=204, #number of channels
                n_t=64) #number of time points
params = dict(l1_lambda=3e-4,
              n_ls=32, #number of latent factors
              learn_rate=3e-4,
              dropout = .5,
              patience = 3,# patientce for early stopping
              min_delta = 5e-6,
              nonlin_in = tf.identity, #input layer activation for var-cnn and lf-cnn
              nonlin_hid = tf.nn.relu, #convolution layer activation for var-cnn and lf-cnn
              nonlin_out = tf.identity, #output layer activation for var-cnn and lf-cnn
              filter_length=7, #convolutional filter length for var-cnn and lf-cnn
              pooling = 2, #convlayer pooling factor for var-cnn and lf-cnn
              stride = 1, #stride parameter for convolution filter
              test_upd_batch = 20) #pseudo-real time test batch size

results = []
dpath = '/m/nbe/scratch/braindata/izbrv/camcan_preproc/'
train_paths = [''.join([dpath,'camcan_train.tfrecord'])]
val_paths = [''.join([dpath,'camcan_val.tfrecord'])]
data_paths = (train_paths,val_paths)

start = time()
model = Model(h_params,params,data_paths,savepath)
if pretrained:
    model.load()
else:
    model.train_tfr()


stop = time() - start
#%%
test_accs = model.evaluate_performance(dpath+'camcan_test.tfrecord', batch_size=120)
prt_test_acc, prt_logits = model.evaluate_realtime(dpath+'camcan_test.tfrecord', batch_size=120, step_size=params['test_upd_batch'])
results.append({'val_acc':model.v_acc[0], 'test_init':np.mean(test_accs), 'test_upd':np.mean(prt_test_acc), 'train_time':stop, 'sid':h_params['architecture']})
logger(savepath,h_params,params,results[-1])

#%%
#visualization

model.load() # restore "across-subject" parameters 
model.compute_patterns(output='patterns')
model.plot_patterns(sensor_layout='Vectorview-grad', sorting='best', spectra=False, scale=False)