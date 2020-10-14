# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:39:39 2018


@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
from layers import ConvDSV, ConvLayer, spatial_dropout, Dense, vgg_block
import tensorflow as tf
import numpy as np


class Model():
    """Model object"""
    def __init__(self, h_params=None, params=None,data_paths=[],savepath='',val_size=int(19*24)):
        self.h_params = h_params
        self.params = params
        self.savepath = savepath
        self.sess = tf.Session()        
        self.keep_prob = params['dropout']
        self.train_dataset  = tf.data.TFRecordDataset(data_paths[0])
        self.val_dataset  = tf.data.TFRecordDataset(data_paths[1])
        # Parse the record into tensors.
        self.train_dataset = self.train_dataset.map(self._parse_function)
        self.val_dataset = self.val_dataset.map(self._parse_function)
        # Generate batches
        self.train_dataset = self.train_dataset.batch(h_params['n_batch'])#.repeat()
        self.val_dataset = self.val_dataset.batch(val_size).repeat()
        #create iterators
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.val_iter = self.val_dataset.make_initializable_iterator()
        self.training_handle = self.sess.run(self.train_iter.string_handle())
        self.validation_handle = self.sess.run(self.val_iter.string_handle())
        self.sess.run(self.train_iter.initializer)
        self.sess.run(self.val_iter.initializer)
        self.X, self.y_ = self.iterator.get_next()
        #initialize computational graph
        handles = self._buildGraph()
        self.saver = tf.train.Saver(max_to_keep=1)
        (self.y_pred, self.cost, self.train_step, 
         self.correct_prediction, self.accuracy,self.cross_entropy, self.reg, self.p_classes) = handles
    
    def _parse_function(self,example_proto):
            keys_to_features = {'X':tf.FixedLenFeature((self.h_params['n_ch'],self.h_params['n_t']), tf.float32),
                                  'y': tf.FixedLenFeature((), tf.int64, default_value=0)}
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            return parsed_features['X'], parsed_features['y']
            
    def _buildGraph(self):
        """Build computational graph"""
        if self.h_params['architecture'] == 'eegnet':
            X1 = tf.expand_dims(self.X,-1)                    
            vc1 = ConvDSV(n_ls=8, nonlin_out=tf.identity, inch=1,
                          filter_length=32, domain='time', stride=1, 
                          pooling=1, conv_type='2d')
            vc1o = vc1(X1)
            bn1 = tf.layers.batch_normalization(vc1o)
            dwc1 = ConvDSV(n_ls=1, nonlin_out=tf.identity, inch=8, 
                           padding = 'VALID', filter_length=204, domain='space', 
                           stride=1, pooling=1, conv_type='depthwise')
            dwc1o = dwc1(bn1)
            bn2 = tf.layers.batch_normalization(dwc1o)
            out2 = tf.nn.elu(bn2)
            out22 = spatial_dropout(out2,self.keep_prob)
            sc1 = ConvDSV(n_ls=8, nonlin_out=tf.identity, inch=8,
                         filter_length=8, domain='time', stride=1, pooling=1, 
                         conv_type='separable')
            sc1o = sc1(out22)
            bn3 = tf.layers.batch_normalization(sc1o)
            out3 = tf.nn.elu(bn3)
            out4 = tf.nn.avg_pool(out3,[1,1,4,1],[1,1,4,1], 'SAME')
            out44 = spatial_dropout(out4,self.keep_prob)
            sc2 = ConvDSV(n_ls=16, nonlin_out=tf.identity, inch=8,
                          filter_length=8, domain='time', stride=1, pooling=1,
                          conv_type='separable')
            sc2o = sc2(out44)
            bn4 = tf.layers.batch_normalization(sc2o)
            out5 = tf.nn.elu(bn4)
            out6 = tf.nn.avg_pool(out5,[1,1,4,1],[1,1,4,1], 'SAME') #fix typo here out5
            out66 = spatial_dropout(out6,self.keep_prob)
            out7 = tf.reshape(out66,[-1,16*4])#16*4
            fc_out = Dense(size=self.h_params['n_classes'], nonlin=tf.identity,dropout=self.keep_prob)
            y_pred = fc_out(out7)
            
        elif self.h_params['architecture'] == 'lf-cnn':
            self.conv = ConvLayer(n_ls=self.params['n_ls'], 
                             filter_length=self.params['filter_length'],
                             pool=self.params['pooling'],
                             nonlin_in=self.params['nonlin_in'],
                             nonlin_out=self.params['nonlin_hid'],
                             conv_type='lf')
                    
            self.fin_fc = Dense(size=self.h_params['n_classes'], 
                           nonlin=self.params['nonlin_out'],
                           dropout=self.keep_prob)
            y_pred = self.fin_fc(self.conv(self.X))
            
        elif self.h_params['architecture'] == 'var-cnn':
            self.conv = ConvLayer(n_ls=self.params['n_ls'], 
                             filter_length=self.params['filter_length'],
                             pool=self.params['pooling'], 
                             nonlin_in=self.params['nonlin_in'],
                             nonlin_out=self.params['nonlin_hid'],
                             conv_type='var')          
            self.fin_fc = Dense(size=self.h_params['n_classes'], 
                           nonlin=self.params['nonlin_out'],
                           dropout=self.keep_prob)
            y_pred = self.fin_fc(self.conv(self.X))
            
        elif 'vgg' in self.h_params['architecture']:
            X1 = tf.expand_dims(self.X,-1)
            inch = 1
            if X1.shape[1]==306:
                X1 = tf.concat([X1[:,0:306:3,:],X1[:,1:306:3,:],X1[:,2:306:3,:]],axis=3)
                inch = 3
                print(X1.shape)
            vgg_dict = dict(n_ls=self.params['n_ls'], nonlin_out=self.params['nonlin_hid'], 
                            inch=inch,padding = 'SAME', filter_length=(3,3), domain='2d', 
                           stride=1, pooling=1, conv_type='2d')
            vgg1 = vgg_block(2,ConvDSV,vgg_dict)
            out1 = vgg1(X1)
            
            vgg_dict['inch'] = vgg_dict['n_ls']
            vgg_dict['n_ls'] *=2
            vgg2 = vgg_block(2,ConvDSV,vgg_dict)
            out2 = vgg2(out1)
#            
            vgg_dict['inch'] = vgg_dict['n_ls']
            vgg_dict['n_ls'] *=2
            vgg3 = vgg_block(4,ConvDSV,vgg_dict)
            out3 = vgg3(out2)
            
            vgg_dict['inch'] = vgg_dict['n_ls']
            vgg_dict['n_ls'] *=2
            vgg4 = vgg_block(4,ConvDSV,vgg_dict)
            out4 = vgg4(out3)
#            
            vgg_dict['inch'] = vgg_dict['n_ls']
            vgg5 = vgg_block(4,ConvDSV,vgg_dict)
            out5 = vgg5(out4)
            
#            
            fc_1 = Dense(size=4096, nonlin=self.params['nonlin_hid'],dropout=self.keep_prob)
            fc_2 = Dense(size=4096, nonlin=self.params['nonlin_hid'],dropout=self.keep_prob)
            fc_out = Dense(size=self.h_params['n_classes'], nonlin=self.params['nonlin_out'],dropout=self.keep_prob)
            y_pred = fc_out(fc_2(fc_1(out5)))
            #y_pred = fc_out(fc_1(out2))

            
        else:
            print('Supported achitecures are: eegnet, lf-cnn, var-cnn: training aborted!')
            return
        p_classes = tf.nn.softmax(y_pred)
        cross_entropy = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=y_pred))
        #add L1 regularization
        regularizerss = [tf.reduce_sum(tf.abs(var)) for var in
                                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
                                           if 'weights' in var.name]# 'dense'
        l1_reg = self.params['l1_lambda'] * tf.add_n(regularizerss) 
        reg = l1_reg
        cost = cross_entropy + reg# add regularization
        #Optimizers, accuracy etc
        train_step = tf.train.AdamOptimizer(self.params['learn_rate']).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), self.y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        print('Initialization complete!')
        return (y_pred, cost, train_step, correct_prediction,
                accuracy, cross_entropy, reg, p_classes)
                
    def train(self,megdata,pretrained=False):
        min_val_loss =  np.inf
        val_dict = {self.X: megdata.validation.meg_data, 
                        self.y_: megdata.validation.labels, 
                        self.keep_prob: 1.}                    
        with self.sess.as_default():
            if pretrained:
                self.saver.restore(self.sess,''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))
                min_val_loss = self.cost.eval(feed_dict=val_dict)
                self.v_acc = self.accuracy.eval(feed_dict=val_dict)
                print('Restored: Loss =', min_val_loss)
            else:
                self.sess.run(tf.global_variables_initializer())
                patience_cnt = 0
                for i in range(self.h_params['n_iter']+1):
                    batch = megdata.train.next_batch(self.h_params['n_batch'])
                    train_dict = {self.X: batch[0], self.y_: batch[1], self.keep_prob: self.params['dropout']}
                    if i % self.h_params['eval_step'] == 0:
                        train_dict[self.keep_prob] = 1.
                        train_accuracy = self.accuracy.eval(train_dict)
                        val_accuracy = self.accuracy.eval(feed_dict=val_dict)
                        val_loss = self.cost.eval(feed_dict=val_dict)
                        print('step %d, train_acc %g, val_acc %g, val loss %g, val reg %g' % (i, train_accuracy, val_accuracy, val_loss,self.reg.eval(feed_dict=val_dict)))
                        print('loss= %g' % (self.cross_entropy.eval(feed_dict=train_dict)))
                        if i >self.h_params['eval_step']*20:
                            if min_val_loss >= val_loss:
                                self.v_acc = val_accuracy
                                min_val_loss = val_loss
                                self.saver.save(self.sess, ''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))
                            else:
                                patience_cnt +=1
                                print('*')
                            if patience_cnt >= self.params['patience']:
                                print("early stopping...")
                                #restore the best model
                                self.saver.restore(self.sess,''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))
                                break     
                    self.train_step.run(feed_dict=train_dict)
    
    def train_tfr(self):       
        """Uses Datasets in Tensorflow Records format"""
        
        ii=0
        min_val_loss =  np.inf
        patience_cnt = 0
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.h_params['n_epochs']):
            self.sess.run([self.train_iter.initializer,self.val_iter.initializer])
            self.train_dataset.shuffle(buffer_size=10000)
            while True:
                
                try:
                    
                    _, loss,acc = self.sess.run([self.train_step,self.cost,self.accuracy],feed_dict={self.handle: self.training_handle})
                    ii+=1
                except tf.errors.OutOfRangeError:
                    break            
            
            if i %self.h_params['eval_step']==0:
                self.v_acc, v_loss = self.sess.run([self.accuracy,self.cost],feed_dict={self.handle: self.validation_handle})
                print('epoch %d, train_loss %g, train acc %g val loss %g, val acc %g' % (i, loss,acc, v_loss, self.v_acc))
                
                if min_val_loss >= v_loss:
                    min_val_loss = v_loss
                    self.saver.save(self.sess, ''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))
                else:
                    patience_cnt +=1
                    print('*')
                if patience_cnt > self.params['patience']:
                    print("early stopping...")
                    #restore the best model
                    self.saver.restore(self.sess, ''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))                
                    break
        
                
        
    
    def compute_patterns(self, megdata=None,output='patterns'):
        from sklearn.covariance import ledoit_wolf
        if self.h_params['architecture'] == 'lf-cnn':
            spatial = self.conv.W.eval(session=self.sess)
            self.filters = np.squeeze(self.conv.filters.eval(session=self.sess))
            self.patterns = spatial

            if 'patterns' in output:
                if isinstance(self.val_dataset,tf.data.Dataset):
                    data = self.sess.run(self.X,feed_dict={self.handle:self.validation_handle})
                elif megdata:
                    data = megdata.train.meg_data
                
                data = data.transpose([0,2,1])
                data= data.reshape([-1,data.shape[-1]])
                self.dcov,_ = ledoit_wolf(data)
                self.patterns = np.dot(self.dcov,self.patterns)
            if 'full' in output:
                #lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
                lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
                self.lat_prec = np.linalg.inv(lat_cov)
                self.patterns = np.dot(self.patterns,self.lat_prec)
            self.out_weights = self.fin_fc.w.eval({self.handle:self.validation_handle},session=self.sess)
            self.out_biases = self.fin_fc.b.eval(session=self.sess)
            self.out_weights = np.reshape(self.out_weights,[self.params['n_ls'],-1,self.h_params['n_classes']])
        else:
            print('Parameter interpretation only available for LF-CNN!')
    
    def plot_patterns(self,sensor_layout='Vectorview-grad',sorting='l2',spectra=True,scale=False):
        from mne import channels, evoked, create_info
        import matplotlib.pyplot as plt
        from scipy.signal import freqz
        self.ts=[]
        lo = channels.read_layout(sensor_layout)
        info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
        self.fake_evoked = evoked.EvokedArray(self.patterns,info)
        nfilt = min(self.params['n_ls']//self.h_params['n_classes'],8)
        #define which components to plot       
        if sorting is 'l2':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=2))            
        elif sorting is 'l1':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=1))            
        elif sorting is 'contribution':
            #One col per class
            nfilt = 3
            order = []
            for i in range(self.h_params['n_classes']):
                inds = np.argsort(self.out_weights[...,i].sum(-1))[::-1]
                order+=list(inds[:nfilt])
            order = np.array(order)
        elif sorting == 'abs':
            nfilt = self.h_params['n_classes']
            order = []
            for i in range(self.h_params['n_classes']):
                pat= np.argmax(np.abs(self.out_weights[...,i].sum(-1)))
                
                order.append(pat)
                #print(pat.shape)
                #self.ts.append(t)
            order = np.array(order)
        elif sorting is 'best':
            #One pattern per class
            nfilt = self.h_params['n_classes']
            order = []
            for i in range(self.h_params['n_classes']):
                pat,t= np.where(self.out_weights[...,i]==np.max(self.out_weights[...,i]))
                order.append(pat[0])
                self.ts.append(t)
            order = np.array(order)
        elif sorting is 'best_neg':
            nfilt=self.h_params['n_classes']
            #One row per class
            order = []
            for i in range(self.h_params['n_classes']):
                pat= np.argmin(self.out_weights[...,i].sum(-1))
                order.append(pat)
        elif sorting is 'worst':
            nfilt=self.h_params['n_classes']
            #One row per class
            order = []
            weight_sum = np.sum(np.abs(self.out_weights).sum(-1),-1)
            pat= np.argsort(weight_sum)
            print(weight_sum[pat])
            order = np.array(pat[:nfilt])

        elif isinstance(sorting,list):
            nfilt = len(sorting)
            order = np.array(sorting)
        
        else:
            order = np.arange(self.params['n_ls'])
        self.fake_evoked.data[:,:len(order)] = self.fake_evoked.data[:,order]
        if scale:
            self.fake_evoked.data[:,:len(order)] /= self.fake_evoked.data[:,:len(order)].max(0)
        self.fake_evoked.data[:,len(order):] *=0
        self.out_filters = self.filters[:,order]
        order = np.array(order)
        
        if spectra:
            z = 2
            fs = 125
        else: 
            z = 1
        nrows = max(1,len(order)//nfilt)
        ncols = min(nfilt,len(order))
        #print('rows:',nrows, ' cols:', ncols)
        f, ax = plt.subplots(z*nrows, ncols,sharey=True)
        ax = np.atleast_2d(ax)
        for i in range(nrows):
            if spectra:
                for jj,flt in enumerate(self.out_filters[:,i*ncols:(i+1)*ncols].T):
                    w, h = freqz(flt,1)
                    ax[z*i+1, jj].plot(w/np.pi*fs/2, np.abs(h)) 
                    #ax[z*i+1, jj].set_ylim(0,1.5)
            self.fake_evoked.plot_topomap(times=np.arange(i*ncols,  (i+1)*ncols, 1.), 
                                     axes=ax[z*i], colorbar=False, #vmin=0, 
                                     vmax=np.percentile(self.fake_evoked.data[:,:len(order)],99), 
                                     scalings=1,
                                     time_format='')
                           
                    
        #f.tight_layout()
        return f
        
    def load(self):
        self.saver.restore(self.sess,''.join([self.savepath,self.h_params['architecture'],'-',self.h_params['sid']]))
        self.v_acc = self.sess.run([self.accuracy],feed_dict={self.handle: self.validation_handle})
        
    def plot_out_weihts(self,fs=None):
        from matplotlib import pyplot as plt
        if fs:
            times = np.arange(self.h_params['n_t']//2)/(2./fs)
        else:
            times = np.arange(self.h_params['n_t'])
        f,ax = plt.subplots(1,self.h_params['n_classes'])
        for i in range(self.h_params['n_classes']):
            F = self.out_weights[...,i]
            pat,t= np.where(F==np.max(F))
            ax[i].pcolor(times,np.arange(32),F,cmap='bone_r',vmin=0.0,vmax=.25)
            ax[i].plot(times[t]+.008,pat+.5,markeredgecolor='red',markerfacecolor='none',marker='s',markersize=10,markeredgewidth=2)
        plt.show()
            
                
    def evaluate_performance(self, data_path, batch_size=120):
        """Compute intial test accuracy"""
        test_dataset  = tf.data.TFRecordDataset(data_path).map(self._parse_function)  
        test_dataset = test_dataset.batch(batch_size)
        test_iter = test_dataset.make_initializable_iterator()
        acc = []
        self.sess.run(test_iter.initializer)
        test_handle = self.sess.run(test_iter.string_handle())
        while True:
            try:
                acc.append(self.sess.run(self.accuracy,feed_dict={self.handle: test_handle}))
            except tf.errors.OutOfRangeError:
                print('Finished: acc: %g +\- %g' % (np.mean(acc), np.std(acc)))
                break
                    
        return acc
             
    def evaluate_realtime(self,data_path, batch_size=120, step_size=1):
        """Compute test accuracy batch by batch with incremental updates"""
        prt_batch_pred = []
        prt_logits = []
        n_test_points = batch_size//step_size
        count=0
        
        test_dataset  = tf.data.TFRecordDataset(data_path).map(self._parse_function)  
        test_dataset = test_dataset.batch(step_size)
        test_iter = test_dataset.make_initializable_iterator()
        self.sess.run(test_iter.initializer)
        test_handle = self.sess.run(test_iter.string_handle())
        
        while True:
            try:
                self.load()
                count += 1
                preds = 0
                for jj in range(n_test_points):
                    pred, probs, _ = self.sess.run([self.correct_prediction,
                                                     self.p_classes,self.train_step],
                                                     feed_dict={self.handle: test_handle})
                    #for _ in range(step_size):
                     #self.sess.run(self.train.)
                    preds +=np.mean(pred)
                    prt_logits.append(probs)
                prt_batch_pred.append(preds/n_test_points)
                #print(count)
            except tf.errors.OutOfRangeError:
                print('prt_done: count: %d, acc: %g +\- %g' % (count,np.mean(prt_batch_pred), np.std(prt_batch_pred)))
                break
#                    for _ in range(self.params['test_upd_batch']):
#                        self.train_step.run(feed_dict={self.X: test_batch[0], self.y_: test_batch[1], 
#                                                      self.keep_prob: self.params['dropout']})
        return prt_batch_pred,np.concatenate(prt_logits)
    
    def tf_cov(self,elem):
        x = elem['X']
        mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx
                                         

                            
               