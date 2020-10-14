# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:12:48 2017
Read/ Filter 0.1-45Hz/ Epoch/ Downsample/ Get labels/ Save
mne version = 0.15.2/ python2.7
@author: zubarei1
"""



path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/raw_data/'
 

import mne
import numpy as np
data_path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/raw_data/'
save_path = '/m/nbe/scratch/braindata/izbrv/fb_mi/'
fn_prefs = ['sub' + str(i) for i in range(1,19)]
raw_suff = '/motor_imag.fif'

raws = [mne.io.RawFIF(data_path+fname+raw_suff, \
                        preload=False, verbose=False) for fname in fn_prefs]
#some more preprocessing for speed
#epochs = []
for jj in range(len(raws)):
    raw = raws.pop(0)
    events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003)
    raw.load_data()
    raw.pick_types(meg='mag')
    #raw.filter(l_freq=1.,h_freq=45.,method='iir')
    epochs = mne.epochs.Epochs(raw,events, baseline=(1.7,2.),\
                    tmin=1.7,tmax=3.,decim=1)
    epochs = epochs['5','6']
    events = epochs.events[:,2]
    interval = epochs.time_as_index([0.])[0]
    #epochs.equalize_event_counts(['5','6','2'])
    
    del raw
    times = epochs.times
    data = epochs.get_data()   
    data = mne.filter.filter_data(data, epochs.info['sfreq'], l_freq=1., 
                                h_freq=45.,  method='iir',verbose=False)        
    
    data = data[:,:,::8]
    y = np.array(epochs.events[:,2]==5,dtype=np.int)

    del epochs
    try:
        assert len(y) == data.shape[0]
        #print(data.shape, y.shape, np.mean(y))    
        def onehot(y):
            n_classes = len(set(y))
            out = np.zeros((len(y),n_classes))
            for i,ii in enumerate(y):
                out[i][ii] +=1
            return out
        y_onehot = onehot(y)
        np.savez(save_path+'mi_'+str(jj),X=data, y=y, y_onehot=y_onehot)

#%%
#import matplotlib.pylab as plt
#evokeds = [epochs[evid].average() for evid in ['1','2','4','8','16','24','32','64']]
##names = {'1':'vis_L', '2':'vis_R', '4':'vis_RL', '8':'ss_R', '16':'ss_L', '32':'aud_L', '64':'aud_R', '24':'ss_RL'}
#names = ['vis_L', 'vis_R', 'vis_RL', 'ss_R', 'ss_L', 'ss_RL', 'aud_L', 'aud_R']
#positions = np.linspace(0.1,0.9,len(names))
#times = evokeds[0].times
#tableau20 = [(31, 119, 180), (174, 199, 232), (158, 218, 229), 
#             (255, 127, 14), (255, 187, 120),   (214, 39, 40),
#             (44, 160, 44), (152, 223, 138),  (255, 152, 150),    
#             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
#             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
#             (188, 189, 34), (219, 219, 141), (23, 190, 207), ]    
##  
## Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
#tableau20=np.array(tableau20)/255. 
#
#def my_callback_erf(ax, ch_idx):
#    for i, evd in enumerate(evokeds):
#        ax.plot(times, evd.data[ch_idx,...], color=tableau20[i], lw=3 )
#        plt.axhline(0, color='grey', linestyle='-')        
#        #ax.xlim([-200, 600])
#        #ax.ylim([-2.5e-12, 2.75e-12])
#        #ax.ylim([-4.5, 5.])
#        ax.set_xlabel = 'Time, ms'
#        ax.set_ylabel = 'Amplitude, $\fT'
#        plt.rc('xtick', labelsize=28)
#        plt.rc('ytick', labelsize=28)
#"""Re-plot original ERPs/ERFs"""
#
#from mne.viz import iter_topography
#for ax, idx in iter_topography(evokeds[0].info, fig_facecolor='white',
#                               axis_facecolor='white', axis_spinecolor='white',
#                               on_pick=my_callback_erf):
#    for i, evd in enumerate(evokeds):
#        ax.plot(times,evd.data[idx,...], color=tableau20[i])
#        #ax.set_ylim([totmin, totmax])
#        #ax.set_xlim([-100, 600])
#                            
#    for cond, col, pos in zip(names, tableau20, positions):
#        plt.figtext(pos,0.5, cond, color=col, fontsize=14)
#        plt.show()