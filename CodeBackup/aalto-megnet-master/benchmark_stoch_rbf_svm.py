import os
savepath = '/l/DLforMEG/'
os.chdir('/m/home/home6/62/zubarei1/data/Desktop/projects/papers/DLforMEG/megnet/')
from utils import leave_one_subj_out#, read_data_sets
from time import time
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import log_loss
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler#, RobustScaler
from sklearn.model_selection import GridSearchCV,StratifiedKFold

def onehot(y,n_classes=None):
    if not n_classes:
        n_classes = len(set(y))
    out = np.zeros((len(y),n_classes))
    for i,ii in enumerate(y):
        out[i][ii] +=1
    return out
    
def reduce_dataset(features, labels):
    for jj in [0,3]:
        del_ind = np.where(labels[:,jj]==1)[0]
        features = np.delete(features,del_ind,axis=0)
        labels = np.delete(labels,del_ind,axis=0)
    merge_ind = np.where(labels[:,5])[0]
    labels[merge_ind,6] = 1
    labels = np.delete(labels,[2,5,7],axis=1)
    return features, labels

import copy
mode = 'across'
grid_search = True
rbf = False
dataset = 'mi3' # options are mi3, megset8, mesget5
if dataset == 'mi3':
    dpath = '/m/nbe/scratch/braindata/izbrv/mi_data_3/'
    data_paths0 = [dpath+'mi_'+str(i)+'.npz' for i in range(19)]
    interval = 36
    n_classes = 3

elif 'megset' in dataset:
    dpath = '/m/nbe/project/megset/megset_RZ/'##scratch/braindata/izbrv/megset_RZ/'
    data_paths0 = [dpath+'megdata_sub0'+str(i)+'.npz' for i in range(1,8)]
    interval = 36
    n_classes = 5
results = []
for path in data_paths0:
                    data_paths = copy.copy(data_paths0)
                    megdata = leave_one_subj_out(data_paths,holdout=path, val_size=0.1, scale=True, crop=36)
                    
                    X_train = megdata.train.meg_data.reshape(megdata.train.labels.shape[0],-1)
                    X_val = megdata.validation.meg_data.reshape(megdata.validation.labels.shape[0],-1)
                    X_test = megdata.test.meg_data.reshape(megdata.test.labels.shape[0],-1)
                    if dataset == 'megset5':
                        X_test, y_test = reduce_dataset(X_test,megdata.test.labels)
                        X_train, y_train = reduce_dataset(X_train,megdata.train.labels)
                        X_val, y_val = reduce_dataset(X_val,megdata.validation.labels)
                        y_train = np.argmax(y_train,1)
                        y_val = np.argmax(y_val,1)                     
                        y_test = np.argmax(y_test,1)
                    else:
                        y_train = np.argmax(megdata.train.labels,1)
                        y_val = np.argmax(megdata.validation.labels,1)                     
                        y_test = np.argmax(megdata.test.labels,1)
                    del megdata#, save some memory
                    scaler = StandardScaler()
                    """replace here with your favourite estimator"""
                    start = time()
                    if grid_search:
                        
                                    
                        param_grid_svm = {'alpha': [3e-3,1e-3,3e-4, 1e-4,3e-5,1e-5,1e-6][::-1]}# 3e+3, 1e+3, 3e+2,  1e+2, 30, 10, 3, 1][::-1],
                                          ##'feature_map__gamma':[1e-6,1e-5, 3e-5, 1e-4,1e-3]}#,3e-4,1e-3,3e-3,1e-2,3e-2]}
                        if rbf:
                            fm = Nystroem(gamma=1e-5, n_components=X_train.shape[0], random_state=1)
                            X_train = fm.fit_transform(X_train)
                            X_val = fm.transform(X_val)

                        nfolds = 3
                        cv = StratifiedKFold(nfolds) 
                        clf0 = SGDClassifier(loss="hinge", penalty="l2",
                                                              learning_rate = 'optimal',eta0= 3e-4,
                                                               max_iter=5000, tol=1e-1)
                        
                        clf = GridSearchCV(clf0, param_grid_svm, cv=nfolds, scoring='accuracy',n_jobs=4)
                        clf.fit(X_train, y_train)
                        bmc =clf.best_estimator_ #

                    """until here"""
                    min_loss = np.inf
                    m_patience = 3
                    patience = 0
                    
                    for i in range(30000):
                        subset = np.random.choice(X_train.shape[0], 100)
                        Xtt = X_train[subset,...]
                        ytt = y_train[subset,...]
                        
                        bmc.partial_fit(Xtt,ytt)
                        if i%500==0:
                            y_hat = onehot(bmc.predict(X_val),n_classes)
                            
                            loss = log_loss(y_true=onehot(y_val,n_classes),y_pred=y_hat,normalize=True)
                            print(loss)
                            if loss >= min_loss -1e-6:
                                patience+=1
                            else:
                                min_loss = loss
                            if patience >= m_patience:
                                print('Early stopping')
                                break
                    
                    v_acc = np.sum(bmc.predict(X_val)==y_val)/float(len(y_val))
                    if rbf:
                        X_test = fm.transform(X_test)
                    test_score = np.sum(bmc.predict(X_test)==y_test)/float(len(y_test))
                    stop = time() - start
                    
                    prt_batch_pred = []
                    batch_size = 20
                    n_test_points = X_test.shape[0]//batch_size
                    for jj in range(n_test_points):
                        Xt = X_test[jj*batch_size:(jj+1)*batch_size,...]
                        yt = y_test[jj*batch_size:(jj+1)*batch_size,...]
                        score = np.sum(bmc.predict(Xt)==yt)/float(batch_size)
                        prt_batch_pred.append(score)                                                
                        bmc.partial_fit(Xt,yt)
                    
                    print([v_acc,test_score,np.mean(prt_batch_pred),stop])                     
                    results.append([v_acc,test_score,np.mean(prt_batch_pred),stop])
print('Done!')
print(results)