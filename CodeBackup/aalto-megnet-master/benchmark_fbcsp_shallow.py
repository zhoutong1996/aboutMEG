import os
savepath = '/l/DLforMEG/'
os.chdir('/m/nbe/work/zubarei1/DLforMEG/deeplearningproject/')
from utils import leave_one_subj_out#, read_data_sets, reduce_dataset
from time import time
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.util import np_to_var, var_to_np
from numpy.random import RandomState
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint


h_params = dict(n_iter =30000,
                n_batch =100,
                n_classes=5,
                n_ch=204,
                n_t=64,
                eval_interval=1000,
                patience=3)

              
#mode = 'megset'
grid_search = False
scale = True
dataset = 'megset' # options are mi3, megset8, mesget5
model_constraint = MaxNormDefaultConstraint()
if dataset == 'mi3':
    dpath = '/m/nbe/scratch/braindata/izbrv/mi_data_3/'
    data_paths0 = [dpath+'mi_'+str(i)+'.npz' for i in range(19)]
    interval = 36
    n_classes = 3
    #pm_paths = [dpath+'pm_'+str(i)+'.npz' for i in range(19) if i!=12]
elif 'megset' in dataset:
    dpath = '/m/nbe/work/zubarei1/DLforMEG/megset_RZ/'#'/m/nbe/project/megset/megset_RZ/'##scratch/braindata/izbrv/megset_RZ/'
    data_paths0 = [dpath+'megdata_sub0'+str(i)+'.npz' for i in range(1,8)]
    interval = 36
    n_classes = 5
results = []
for path in data_paths0:
                    min_val_loss = 999.
                    patience_cnt = 0
                    data_paths = copy.copy(data_paths0)
                    data_paths.remove(path)
                    megdata = leave_one_subj_out(data_paths,holdout=path, val_size=0.1, scale=True, crop=36,reduce=True)
                    start = time()
                    X_train = megdata.train.meg_data.astype(np.float32)
                    y_train = np.argmax(megdata.train.labels,1).astype(np.int64) #2,3 -> 0,1
                    X_val = megdata.validation.meg_data.astype(np.float32)
                    y_val = np.argmax(megdata.validation.labels,1).astype(np.int64) #2,3 -> 0,1
                    y_test = np.argmax(megdata.test.labels,1).astype(np.int64) #2,3 -> 0,1
                    cuda = torch.cuda.is_available()
                    set_random_seeds(seed=20170629, cuda=cuda)
                    in_chans = X_train.shape[1]
                    # final_conv_length = auto ensures we only get a single output in the time dimension
                    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                            input_time_length=X_train.shape[2],
                                            n_filters_time=40, filter_time_length=13, 
                                            n_filters_spat=40, pool_time_length=37, 
                                            pool_time_stride=8, pool_mode='mean', 
                                            split_first_layer=True, batch_norm=True, 
                                            batch_norm_alpha=0.1, drop_prob=0.5,
                                            final_conv_length='auto').create_network()
                                            

                    if cuda:
                        model.cuda()
                    optimizer = optim.Adam(model.parameters())
                    rng = RandomState((2017,6,30))
                    for i in range(h_params['n_iter']+1):
                        batch_X, batch_y = megdata.train.next_batch(h_params['n_batch'])
                        batch_X = batch_X[:,:,:,None]
                        batch_y = np.argmax(batch_y,1)
                        net_in = np_to_var(batch_X)
                        if cuda:
                            net_in = net_in.cuda()
                        net_target = np_to_var(batch_y)
                        if cuda:
                            net_target = net_target.cuda()
                        optimizer.zero_grad()
                        # Compute outputs of the network
                        outputs = model(net_in)
                        # Compute the loss
                        loss = F.nll_loss(outputs, net_target)
                        # Do the backpropagation
                        loss.backward()
                        # Update parameters with the optimizer
                        optimizer.step()
                        model_constraint.apply(model)
                            # Print some statistics each epoch
                        model.eval()
                        if i % h_params['eval_interval'] == 0:
                            
                            net_in = np_to_var(X_val[:,:,:,None])
                            if cuda:
                                net_in = net_in.cuda()
                            net_target = np_to_var(y_val)
                            if cuda:
                                net_target = net_target.cuda()
                            outputs = model(net_in)
                            val_loss = var_to_np(F.nll_loss(outputs, net_target))
                            print("{:.1f} Loss: {:.5f}".format(i,float(var_to_np(loss))))
                            predicted_labels = np.argmax(var_to_np(outputs), axis=1)
                            val_acc = np.mean(y_val  == predicted_labels)
                            print("{:.1f} Accuracy: {:.3f}".format(i,val_acc))
                            if (min_val_loss - val_loss) < 1e-8:
                                      patience_cnt +=1
                                      #print('*')
                            if min_val_loss >= val_loss:
                                min_val_loss = val_loss
                            if patience_cnt >= h_params['patience']:
                                print("early stopping...", i)
                                break

                    net_in = np_to_var(X_val[:,:,:,None])
                    if cuda:
                        net_in = net_in.cuda()
                        net_target = np_to_var(y_val)
                    if cuda:
                        net_target = net_target.cuda()
                    outputs = model(net_in)
                    val_pred = np.argmax(var_to_np(outputs), axis=1)
                    v_acc =  np.mean(val_pred==y_val)
                    
                    net_in = np_to_var(megdata.test.meg_data.astype(np.float32)[:,:,:,None])
                    if cuda:
                        net_in = net_in.cuda()
                        net_target = np_to_var(y_val)
                    if cuda:
                        net_target = net_target.cuda()
                    outputs = model(net_in)
                    test_pred = np.argmax(var_to_np(outputs), axis=1)
                    test_accuracy = np.mean(test_pred==y_test)
                    stop = time() - start
                    #print('CNN: ',stop)
                    results.append([v_acc,test_accuracy,'-',stop])
                    print([v_acc,test_accuracy,'-',stop])
                    
print(results)
                            
                        