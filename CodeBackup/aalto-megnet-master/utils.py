"""
@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import numpy as np
import csv
import tensorflow as tf

class DataSet(object):
    def __init__(self, meg_data, labels):
        """Construct a DataSet
        inputs:
        meg_data - array of shape (n_trials, n_channels, n_times)
        labels - array of one-hot values of shape (n_trials, n_classes)"""
        assert meg_data.shape[0] == labels.shape[0], (
          'meg_data.shape: %s labels.shape: %s' % (meg_data.shape,
                                                 labels.shape))
        self._num_examples = meg_data.shape[0]
        meg_data = meg_data.astype(np.float32)
        self._meg_data = meg_data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def meg_data(self):
        return self._meg_data
    @property
    def labels(self):
        return self._labels
    def next_batch(self, batch_size, compute_evokeds=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._meg_data = self._meg_data[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if compute_evokeds:
            return self._meg_data[start:end], self._labels[start:end], 
        return self._meg_data[start:end], self._labels[start:end]

class DataSets(object):
        pass
    

def leave_one_subj_out(data_paths, holdout=None, val_size=0.1, scale=None,crop=None, reduce=False):
    """ Imports and pools data from a list of paths to *.npz files, containing X and y,
        Shuffles and splits the data into training and validation sets
        If holdout is specified shuffles it and assignes to a test set
        Scales and crops the data
        inputs:
        data_paths - list of strings, paths to the data files
        holdout - string, path to held-out data file to be used as test set
        val_size - float, (0,1], proportion of the data to be used as validation set
        scale - bool, perform scaling
        crop - int, index of the time-sample corresponing to the stimulus onset "0" in the MEG measurement
        output:
        data_set - Instance of DataSets
        usage:
        data_set.train.meg_data - np.array of shape (n_trials, n_channels, n_times)
        data_set.validation.meg_data - np.array of shape (n_trials, n_channels, n_times)
        data_set.train.labels - array of one-hot values of shape (n_trials, n_classes)
        data_set.validation.labels - array of one-hot values of shape (n_trials, n_classes)
        if holdout is specified
        data_set.test.meg_data - np.array of shape (n_trials, n_channels, n_times)
        data_set.test.labels - array of one-hot values of shape (n_trials, n_classes)
        """
    data_sets = DataSets()
    features = None
    if isinstance(data_paths,list):
        for i, dp in enumerate(data_paths):
            with np.load(dp) as data:
                if dp == holdout:
                    print('holdout subject: %s' % holdout[-9:-4])
                    
                    shufflet= np.random.permutation(len(data['y']))                    
                    X_test = data['X'][shufflet,...].astype(np.float32)
                    y_test = data['y_onehot'][shufflet].astype(np.float32)
                    if reduce:
                        X_test, y_test = reduce_dataset(X_test,y_test)
                    assert X_test.shape[0] == y_test.shape[0]
                elif not np.any(features):
                    features = data["X"].astype(np.float32)
                    labels = data["y_onehot"].astype(np.float32)
#                    if not crop:
#                        crop = data['interval']
#                        print("cropping at ind: #", crop)
                    
                else:
                    features = np.vstack([features,data["X"]]).astype(np.float32)
                    labels = np.vstack([labels,data["y_onehot"]]).astype(np.float32)
                    
    else:
        print('data_paths should be a list!')
        return
    assert features.shape[0] == labels.shape[0]
    val_size = int(round(val_size*labels.shape[0]))
    if scale:
        features = scale_type(features,scale)
        if holdout:
            X_test = scale_type(X_test,scale)        
   
    if crop:
        features = features[:,:,crop:]
        if holdout:
            X_test = X_test[:,:,crop:]
    if reduce:
        features, labels = reduce_dataset(features,labels)
    shuffle= np.random.permutation(len(features))
    features = features[shuffle,...]
    labels = labels[shuffle,...]
    X_val = features[:val_size,...]
    y_val = labels[:val_size,...]
    X_train = features[val_size:,...]
    y_train = labels[val_size:,...]
    data_sets.train = DataSet(X_train, y_train)
    data_sets.validation = DataSet(X_val, y_val)
    if holdout:
        data_sets.test = DataSet(X_test,y_test)
    return data_sets

def scale_type(X,interval=None):
    """Perform scaling based on pre-stimulus baseline"""
    if interval == None:
        interval = np.arange(X.shape[-1])        
    elif isinstance(interval,int):
        interval = np.arange(interval)
    elif isinstance(interval,tuple):
        interval = np.arange(interval[0],interval[1])
    X0 = X[:,:,interval]
    if X.shape[1]==306:
        magind = np.arange(2,306,3)
        gradind = np.delete(np.arange(306),magind)
        X0m = X0[:,magind,:].reshape([X0.shape[0],-1])
        X0g = X0[:,gradind,:].reshape([X0.shape[0],-1])
        
        X[:,magind,:] -= X0m.mean(-1)[...,None,None]
        X[:,magind,:] /= X0m.std(-1)[:,None,None]
        X[:,gradind,:] -= X0g.mean(-1)[:,None,None]
        X[:,gradind,:] /= X0g.std(-1)[:,None,None]
    else:      
        X0 = X0.reshape([X.shape[0],-1])
        X -= X0.mean(-1)[:,None,None]
        X /= X0.std(-1)[:,None,None]
    return X
    
def logger(savepath,h_params, params, results):
    """Log perfromance"""
    log = dict()
    log.update(h_params)
    log.update(params)
    log.update(results)
    for a in log:
        if hasattr(log[a], '__call__'):
            log[a] = log[a].__name__
    header = ['architecture','sid','val_acc','test_init', 'test_upd', 'train_time',
              'n_epochs','eval_step','n_batch','n_classes','n_ch','n_t',
              'l1_lambda','n_ls','learn_rate','dropout','patience','min_delta',
              'nonlin_in','nonlin_hid','nonlin_out','filter_length','pooling',
              'test_upd_batch', 'stride']
    with open(savepath+'-'.join([h_params['architecture'],'training_log.csv']), 'a') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames=header)
        #writer.writeheader()
        writer.writerow(log)
        
def plot_cm(y_true,y_pred,classes=None, normalize=False,):
    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools
    
    cm = confusion_matrix(y_true, y_pred)
    title='Confusion matrix'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    if not classes:
        classes = np.arange(len(np.unique(y_true)))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel='True label',
    plt.xlabel='Predicted label'
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
def reduce_dataset(features, labels):
    for jj in [0,3]: 
        del_ind = np.where(labels[:,jj]==1)[0]
        features = np.delete(features,del_ind,axis=0)
        labels = np.delete(labels,del_ind,axis=0)
    merge_ind = np.where(labels[:,5])[0]
    labels[merge_ind,6] = 1
    new_labels = labels[:,[1,2,4,7,6]]
    return features, new_labels
    
    
class RtDatasSet():
    def __init__(self, X, y):
        """Construct a DataSet
        inputs:
        meg_data - array of shape (n_trials, n_channels, n_times)
        labels - array of one-hot values of shape (n_trials, n_classes)"""
        assert X.shape[0] == y.shape[0], ('X: %s y: %s' % (X.shape, y.shape))
        self._n_segments = X.shape[0]
        X = X.astype(np.float32)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def meg_data(self):
        return self._meg_data
    @property
    def labels(self):
        return self._labels
        
    def next_batch(self, batch_size=1, compute_evokeds=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._meg_data = self._meg_data[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if compute_evokeds:
            return self._meg_data[start:end], self._labels[start:end], 
        return self._meg_data[start:end], self._labels[start:end]

#class DataSets(object):
#        pass


#filenames = ["file1.tfrecord", "file2.tfrecord", ..."fileN.tfrecord"]

    # for version 1.5 and above use tf.data.TFRecordDataset

    # example proto decode

    