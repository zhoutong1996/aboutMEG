# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:12:48 2017
Read/ Filter 0.1-45Hz/ Epoch/ Downsample/ Get labels/ Shuffle/  Split/ Scale/ Serialize/ Save
mne version = 0.15.2/ python2.7
@author: zubarei1
"""

"pool: 102+101+3+4 (right), and 1+2+103+104(left)"

import mne
import numpy as np
import os
from glob import glob
import tensorflow as tf
path0 = '/m/nbe/scratch/restmeg/data/'
data_path =  path0 +'/camcan/cc700/mri/pipeline/release004/BIDSsep/megraw/'
sub_dirs = glob(data_path+'sub-CC*')
savepath = '/m/nbe/scratch/braindata/izbrv/camcan_preproc/'
raw_suff = '/meg/passive_raw_tsss_mc.fif'

  
def scale_type(X,intrvl=36):
    """Perform scaling based on pre-stimulus baseline"""
    X0 = X[:,:,:intrvl]
    X0 = X0.reshape([X.shape[0],-1])
    X -= X0.mean(-1)[:,None,None]
    X /= X0.std(-1)[:,None,None]
    X = X[:,:,intrvl:]
    return X
       
def npy_to_tfrecords(X_,y_,output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for X,y in zip(X_,y_):
         # Feature contains a map of string to feature proto objects
         feature = {}
         feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
         feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
         # Construct the Example proto object
         example = tf.train.Example(features=tf.train.Features(feature=feature))
         # Serialize the example to a string
         serialized = example.SerializeToString()
         # write the serialized object to the disk
         writer.write(serialized)
    writer.close()
    

jj = 0
i = 0
for sub in sub_dirs:
    fname = sub+raw_suff
    if os.path.isfile(sub+'/meg/task_raw_tsss_mc.fif'):
        os.remove(sub+'/meg/task_raw_tsss_mc.fif')
    try:
        raw = mne.io.RawFIF(fname, preload=True, verbose='CRITICAL')        
        events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,output='onset')
        events = mne.merge_events(events,[6,7,8],10)
        picks = mne.pick_types(raw.info,meg='grad')
        fmin = 1.
        fmax= 45.
        raw = raw.filter(l_freq=fmin,h_freq=fmax)
        epochs = mne.epochs.Epochs(raw,events,tmin=-.3,tmax=.5,decim=8.,detrend=1,reject={'grad':4000e-13},picks=picks)
        epochs.equalize_event_counts(['9','10'])
        del raw
        data = epochs.get_data()
        data = scale_type(data,36)
        labels = epochs.events[:,2]-9
        del epochs
        i+=1
        print(i)
        if i == 1:
            X = data
            y = labels
        elif i > 1:
            X = np.concatenate([X,data])
            y = np.concatenate([y,labels])         
        print(X.shape)
        if i >1 and i%200==0:
            print('writing')
            shuffle= np.random.permutation(X.shape[0])
            val_size = int(round(0.1*X.shape[0]))
            X = X[shuffle,...]
            X = X.astype(np.float32)
            y = y[shuffle,...]
            
            X_val = X[:val_size,...]
            y_val = y[:val_size,...]
            X_train = X[val_size:,...]
            y_train = y[val_size:,...]
            
            npy_to_tfrecords(X_train,y_train,''.join([savepath,'camcan_',str(jj),'_train.tfrecord']))
            npy_to_tfrecords(X_val,y_val,''.join([savepath,'camcan_',str(jj),'_val.tfrecord']))
            jj+=1
            i =0
            del X, y
            
    except IOError:
        continue
    except ValueError:
        continue
    except KeyError:
        continue

npy_to_tfrecords(X,y,''.join([savepath,'camcan_test.tfrecord']))  