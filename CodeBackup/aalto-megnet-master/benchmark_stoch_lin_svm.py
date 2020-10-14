import os
savepath = '/l/DLforMEG/'
os.chdir('/m/home/home6/62/zubarei1/data/Desktop/projects/papers/DLforMEG/megnet/')
from utils import leave_one_subj_out#,reduce_dataset #read_data_sets, 
from time import time
#from utils import cv_svm
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler#, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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
scale = False
dataset = 'mi3' # options are mi3, megset8, mesget5
if dataset == 'mi3':
    dpath = '/m/nbe/scratch/braindata/izbrv/mi_data_3/'
    data_paths0 = [dpath+'mi_'+str(i)+'.npz' for i in range(19)]
    interval = 36
    #pm_paths = [dpath+'pm_'+str(i)+'.npz' for i in range(19) if i!=12]
elif 'megset' in dataset:
    dpath = '/m/nbe/project/megset/megset_RZ/'##scratch/braindata/izbrv/megset_RZ/'
    data_paths0 = [dpath+'megdata_sub0'+str(i)+'.npz' for i in range(1,8)]
    interval = 36
elif 'camcan' in dataset:
    dpath = '/m/nbe/scratch/braindata/izbrv/camcan_preproc/'##scratch/braindata/izbrv/megset_RZ/'
    data_paths0 = [dpath+'subs_'+str(i)+'_train.npz' for i in range(2)]
    interval = 36
results = []
for path in data_paths0:
                    data_paths = copy.copy(data_paths0)
                    #data_paths.remove(path)
                    #print(path)
                    if mode == 'across':
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
                    """replace here with your estimator"""
                    start = time()
                    if grid_search:
                        param_grid_svm = {'alpha': [0.0003, 1e-4,3e-5,1e-5,3e-6,1e-6][::-1]}
                        if scale:
                            X_train = scaler.fit_transform(X_train)
                            X_val = scaler.fit_transform(X_val)
                            X_test = scaler.fit_transform(X_test)

                        nfolds = 3
                        
                        clf0 = SGDClassifier(loss="hinge", penalty="l2",
                                             learning_rate='constant',eta0= 3e-4,
                                             max_iter=5000, tol=1e-6,
                                             warm_start=True)#SVC(C=1,kernel='linear',cache_size=2000)
                        clf = GridSearchCV(clf0, param_grid_svm, cv=nfolds, scoring='accuracy',n_jobs=4)
                        clf.fit(X_train, y_train)
                        bmc =clf.best_estimator_ #
                        print(bmc)
                    else:
                        if scale:
                            bmc = Pipeline([('sc', scaler), ('clf', bmc)])
                        else:
                            bmc = SGDClassifier(alpha=1e-4,loss="hinge", penalty="l2")#SVC(C=1,kernel='linear',cache_size=2000)SVC(kernel='rbf',C=3e+4, gamma=1e-5,cache_size=2000)
                        
                    """until here"""
                    
                    bmc.fit(X_train,y_train)
                    v_acc = np.sum(bmc.predict(X_val)==y_val)/float(len(y_val))
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