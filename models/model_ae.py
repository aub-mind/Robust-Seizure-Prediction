"""
This file contains the class to build Model graph

"""
import os
import numpy as np
from models.helping_functions import *
from utils.save_load import load_ae, savefile
import tensorflow as tf
import time
from keras.utils import to_categorical


class CNN_GRU:
    """ CNN_GRU model class"""
    
    def __init__(self, dim, dataset, noise_limit, graph):

        self.chanels = dim[1]  # electrodes
        self.win_length = dim[0]
        self.dataset = dataset
        self.noise_limit = noise_limit
        self.graph = graph

        # Placeholders
        self.input = tf.placeholder(tf.float32, [None, self.win_length,
                                                 self.chanels], name="input")
        # drop out rates
        self.cnn_keep_rate = tf.placeholder_with_default(1.0, shape=(),
                                                         name='keep_rate')
        self.gru_keep_rate = tf.placeholder_with_default(1.0, shape=(),
                                                         name='keep_rate')
        self.lamda = tf.placeholder_with_default(1.0, shape=(), name='lambda')
        
        self.Y = tf.placeholder("float", [None, 2], name="Y")
        
        # create the new variable for the adversarial noise.
        self.ADVERSARY_VARIABLES = 'adversary_variables'
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.ADVERSARY_VARIABLES]
        
        self.x_noise = tf.Variable(
                tf.zeros([1, self.win_length, self.chanels], dtype=tf.float32),
                name='x_noise', trainable=False,
                collections=collections)
 
        self.x_ae2 = self.input + self.x_noise
        # limit to clip the noise
        self.x_noise_clip = tf.assign(
                self.x_noise,
                tf.clip_by_value(self.x_noise,
                                 -self.noise_limit,
                                 self.noise_limit)
                                        )
        self.build()

    def build(self):
        """ Builds CNN_GRU model graph"""

        with self.graph.as_default():
            print('Building new model')
            with tf.variable_scope('conv_1'):
                conv1 = tf.layers.conv1d(inputs=self.x_ae2, filters=64,
                                         kernel_size=5, strides=2,
                                         activation=tf.nn.leaky_relu,
                                         name='conv1')   
                max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2,
                                                     strides=2)
                print("conv1 shape:", conv1.get_shape())
               
            with tf.variable_scope('conv_2'):
                conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, 
                                         kernel_size=3, strides=2, 
                                         activation=tf.nn.leaky_relu,
                                          name='conv2')
                max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2,
                                                     strides=2)
                print("conv2 shape:", conv2.get_shape())
                
            with tf.variable_scope('conv_3'):
                conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64,
                                         kernel_size=3, strides=1,
                                         activation=tf.nn.leaky_relu,
                                         name='conv3')
                max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2,
                                                     strides=2)
                print("conv3 shape:", conv3.get_shape())
                
            with tf.variable_scope('conv_4'):
                conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=128,
                                         kernel_size=2, strides=1,
                                         activation=tf.nn.leaky_relu,
                                         name='conv4')
                max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2,
                                                     strides=2)
                print("conv4 shape:", conv4.get_shape())

            last_cnn_layer = max_pool_4
            # get the number of features of the last CNN layer
            last_filter_dim = last_cnn_layer.get_shape().as_list()[2] 
            print('filters', last_filter_dim)
            # GRU Layer
            gru_size = 128  # no. of GRU cells in one layer                     
            with tf.variable_scope('GRU'):
                # features = tf.keras.layers.Flatten()(last_cnn_layer)
                gru_output = tf.keras.layers.GRU(gru_size)(last_cnn_layer)
                output = tf.layers.dense(gru_output, 2, name='dense_1')
     
            print("Output shape: ", output.get_shape())
            print("Done building the graph........................")
            
            ##################################################################
            # get the embedding for further analisys
            self.embeddings1 = gru_output  # the output of the GRU
            self.cost_Y = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                            logits=output, labels=self.Y)
                    )
            
            l2 = 0.001 * sum(
                tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.total_loss = self.cost_Y+l2
            self.Y_op = tf.train.AdamOptimizer().minimize(self.total_loss)
            
           # applying gradient clipping
            optimizer = tf.train.AdamOptimizer()
            gvs = optimizer.compute_gradients(self.total_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.Y_op = optimizer.apply_gradients(capped_gvs)
           
#            print("Trainable parameters without AE: ")
#            print([var.name for var in tf.trainable_variables()])
            # Optimizing with adversarial examples
            adversary_variables = tf.get_collection(self.ADVERSARY_VARIABLES)
            print("Trainable parameters with AE: ",
                  [var.name for var in adversary_variables])
            # Optimizing without adversarial examples
            l2_ae = self.lamda * tf.nn.l2_loss(self.x_noise, name='L2')
            self.loss_adversary = self.cost_Y + l2_ae
            self.optimizer_ae = tf.train.AdamOptimizer().minimize(
                    self.loss_adversary, var_list=adversary_variables,
                    name='adversarial_optimizer')
            # convert logits to probabilities
            self.predictions = tf.nn.softmax(output, name='Preds')
           
    def init_noise(self, sess):
        """Initialize adversarial noise"""
        
        sess.run(tf.variables_initializer([self.x_noise]))

    def optimize_ae(self, sess, actual, X, epochs):
        self.init_noise(sess)
        X2=X
        for i in range(epochs):
            feed_dict = {self.Y: actual, self.input: X2, self.lamda: 0.001}
            ae,_, loss, n = sess.run(
                    [self.x_ae2, self.optimizer_ae, 
                     self.loss_adversary, self.x_noise], feed_dict=feed_dict)
            sess.run(self.x_noise_clip)
            #if(norm(n.reshape(-1), 2) <= int(norm(X.reshape(-1), 2)*0.02)):
            X2 = ae
#            if(i % 50 == 0):
#                print('adversary loss: ', round(loss,2))
        return X2
    
    def generate_ae(self, n, sess, X, y, steps, target, path_ae):
        """ Generates AE examples
        
        Args:
        X: EEG input data of size [(train+valid) size, window_length,
        electrodes]
        y: lables, size [(train+valid) size]
        n:number of adversarial examples to generate
        steps: steps specified for convergence
        
        Returns:
        data: generated AEs size (n)
        lables: lables of generated AEs size (n)
    
        """
        X_file = os.path.join(str(path_ae), 'augmented_input%s.hkl' % (target))
        y_file = os.path.join(str(path_ae), 'augmented_lables%s.hkl' % (target))
#        input_ae, y_ae = load_ae(path_ae, target)
#        if input_ae is not None:
#            print("AE cache is not None")
#            return  input_ae, y_ae
        
        percentage = 1/3
        inter_count = int(n * percentage)  # interictal
        pre_count = n-inter_count  # preictal
        print('ae_interictal: ', inter_count)
        print('ae_preictal: ', pre_count)
        summary = np.zeros([2, 1])
        preictal = np.array([0, 1])
        interictal = np.array([1, 0])
        states = [interictal, preictal]
        data = []
        lables = []
        start_time = time.time()
        for i in range(n):
            if (i % 1 == 0):
                print("Generated Examples: " + str(i))
            # randomly chose a state
            # num=np.random.randint(2)
            if(summary[1] < pre_count):
                state = states[1]
                summary[1] += 1
            else:
                state = states[0]
                summary[0] += 1
            # generating an example
            target = np.flip(state).reshape(-1, 2)
            
            idx=np.random.randint(0, y.shape[0])
            # find the corresponding input to the
            while (y[idx] != state).all():
                idx = np.random.randint(0, y.shape[0])
            inp = X[idx, :, :].reshape(-1, X.shape[1], X.shape[2])
            ae = self.optimize_ae(sess, target, inp, steps)
            # check if the AE is different from the input
            data.append(ae)
            lables.append(y[idx].reshape(-1, 2))
            execution_time = time.time() - start_time
            print("Time to generate {} examples = {}".format(i, 
                  round(execution_time, 2)))

        input_ae = np.concatenate((data), axis=0)
        y_ae = np.concatenate((lables), axis=0)
        print('AEs Generated = intericta: ',
              int(summary[0][0]),'preictal: ',int(summary[1][0]))
        print('\n')
        #saving generated AEs
        savefile(input_ae, X_file)
        savefile(y_ae, y_file)
        return input_ae, y_ae

    def feed_forward(self, sess, X):
        preds = sess.run(self.predictions, feed_dict={self.input: X})
        return preds
    
    def train(self, session, epochs, nb_batches,
              batch_size, X_train1, y_train1, X_val, y_val, verbose):
        """ Train the model and print results at each epoch"""
        
        start_time = time.time()
        for epoch in range(epochs):
            for batch in range(nb_batches):
                epoch_x1, epoch_y1 = next_batch(X_train1,
                                                y_train1, batch, batch_size)
                feed_dict = {self.input: epoch_x1, self.Y: epoch_y1, 
                             self.cnn_keep_rate:1, self.gru_keep_rate:1}
                session.run(self.Y_op, feed_dict=feed_dict)
                
        #validation 
            feed_dict = {self.input: X_val, self.Y: y_val,
                         self.cnn_keep_rate:1, self.gru_keep_rate:1}
            session.run(self.Y_op, feed_dict=feed_dict)  
            X_val = X_val.reshape(-1, self.win_length, self.chanels)
            preds_val, Y_loss = session.run([self.predictions, self.cost_Y],
                                            feed_dict=feed_dict)
            execution_time = time.time() - start_time
            auc, acc, sens_val, _= calc_metrics(y_val, preds_val)
            if verbose:
                print('sensitivity: ', round(sens_val, 2),
                      'AUC: ', round(auc, 2), 'Y_loss: ', round(Y_loss, 2),
                      'acc: ', round(acc, 2), ' time: ', round(execution_time, 2))    
        print("Finished training ............................................")
        
        
    def testing(self, session, X_test, y_test):
        """ Helping function to perform model testing"""            
        
        test1 = X_test.reshape(-1, self.win_length, self.chanels)  
        print("Testing.......................................................")
        feed_dict = {self.input: test1, self.Y: y_test, self.cnn_keep_rate: 1,
                     self.gru_keep_rate: 1}
        test_loss1, preds, emb = session.run([self.cost_Y, self.predictions, 
                                              self.embeddings1], feed_dict=feed_dict)
        auc_test, acc, sensitivity, false_alarm = calc_metrics(y_test, preds)

        print('Sensitivity: ', round(sensitivity, 2), ' AUC: ',
              round(auc_test, 2), ' acc: ', round(acc, 2))
                
        return auc_test, acc, sensitivity, false_alarm, test_loss1, emb, preds

    def train_eval_test_ae(self, session, patient, ictal_X, ictal_y,
                           interictal_X, interictal_y, settings,
                           validation='cv', mode='without_AE', batch_size=256,
                           epoch=50, percentage=0.3, verbose=False):
        """
        Train the model in two modes:
            AE: training with adversarial examples
            without_AE: regular training

        Within each mode of training there are two modes for model evaluation:
            cv: one seizure leave out cross validation
            test: devide the data to training evaluation and testing
    
        """
        dataset = self.dataset
        if(dataset == 'CHBMIT'):
            path_ae = 'Adversarial_examples\\MIT\\'
        elif(dataset == 'FB'):
            path_ae = 'Adversarial_examples\\FB\\'
        
        print('##############################################################')
        
        
        target = patient
        auc_score_total = 0
        
        history = dict(AUC_AVG=[], y_pred=[], y_test=[], test_loss=[],
                       test_sensitivity=[], test_false_alarm=[], embed1=[])
        
        if validation == "cv":
            ind = 0
            loo_folds = train_val_cv_split(ictal_X, ictal_y,
                                            interictal_X, interictal_y, 0.1)
            print("Training ....................................")

            for X_train1, y_train1, X_val, y_val, X_test, y_test in loo_folds:
                y_val = to_categorical(y_val)
                y_test = to_categorical(y_test)
                y_train1 = to_categorical(y_train1)
                
                print('validation shape ', y_val.shape, ' test shape ',
                      y_test.shape, 'train shape', y_train1.shape)
                nb_batches = int(X_train1.shape[0] / batch_size)
                print('\n')
                print('Fold:', ind+1)
                # reinitialize graph variables at each fold
                session.run(tf.global_variables_initializer())
                self.init_noise(session)
                # initialize saver to save the trained model
                saver = tf.train.Saver()
                save_dir = 'checkpoints_%s/pat%s' % (dataset, target)
                
                # Create the directory if it does not exist.
                if not os.path.exists(save_dir):
                     os.makedirs(save_dir)
                save_path = os.path.join(save_dir, 'CNN_GRU_%s' % (target))
                #Training
                self.train(session, epoch, nb_batches, batch_size, X_train1, 
                           y_train1, X_val, y_val, verbose)
                ind+=1
                if mode == 'AE':
                    print('Training with AEs...........................................')
                    X = np.concatenate((X_train1, X_val))
                    y = np.concatenate((y_train1, y_val))
                    print('Number of AEs to generate: ',
                          int(X.shape[0] * percentage))

                     # size of AEs to generate from the data % 
                    input_ae, lables_ae = self.generate_ae(
                            int(X.shape[0]*percentage), session, X, y, 100, target, path_ae)
                    X_train1 = np.concatenate((X_train1, input_ae), axis=0)
                    y_train1 = np.concatenate((y_train1, lables_ae), axis=0)
                    del(input_ae)
                    del(lables_ae)
                    X_train1, y_train1 = shuffle_data(X_train1, y_train1)
                    print('validation shape ', y_val.shape, ' test shape ',
                          y_test.shape, 'train shape', y_train1.shape)
                    
                    nb_batches = int(X_train1.shape[0] / batch_size)
                    print('\n')
                    #training with AEs
                    self.train(session, 15, nb_batches, batch_size,
                               X_train1, y_train1, X_val, y_val, verbose)
                # Testing
                auc_test, acc, sensitivity, false_alarm, test_loss1, \
                    emb, preds = self.testing(session, X_test, y_test)
                auc_score_total+=auc_test
               
                history = collect_results(history, sensitivity, false_alarm, 
                                        preds, y_test, test_loss1, emb)
              
            if mode == 'AE':
                filename = os.path.join(
                         str(settings['resultsCV_AE']), 'history%s.hkl' %(target))
            else: 
                filename = os.path.join(
                         str(settings['resultsCV']), 'history%s.hkl' %(target))

            print("Average AUC patient : ", int(target), ' ', auc_score_total/ind)
            history['AUC_AVG'].append(auc_score_total/ind)
            
            savefile(history, filename)
            saver.save(sess=session, save_path=save_path)
            print('Finished LOSOCV with AEs .................................')

        elif validation == "test":
            
            print('Training with AEs.........................................')
            X_train1, y_train1, X_val, y_val, X_test, y_test = \
                train_val_test_split(ictal_X, ictal_y,
                                     interictal_X, interictal_y, 0.1, 0.2)
            y_val = to_categorical(y_val)
            y_test = to_categorical(y_test)
            y_train1 = to_categorical(y_train1)
            print('validation shape ', y_val.shape, ' test shape ', 
                  y_test.shape, 'train shape', y_train1.shape)
    
            nb_batches = int(X_train1.shape[0] / batch_size)
            # reinitialize graph variables at each fold
            session.run(tf.global_variables_initializer())
            self.init_noise(session)
            saver = tf.train.Saver()
            save_dir = 'checkpoints_%s/pat%s' % (dataset, target)
            # Create the directory if it does not exist.
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'CNN_GRU_%s' % (target)) 
            # Training
            self.train(session, epoch, nb_batches,
                       batch_size, X_train1, y_train1, X_val, y_val, verbose)
        
            if mode == 'AE':
                X = np.concatenate((X_train1, X_val))
                y = np.concatenate((y_train1, y_val))
                print('Number of AEs to generate: ', int(X.shape[0]*percentage))
                
                input_ae, lables_ae = self.generate_ae(
                            int(X.shape[0]*percentage), session, X, y, 100, target, path_ae)
                X_train1 = np.concatenate((X_train1, input_ae), axis=0)
                y_train1 = np.concatenate((y_train1, lables_ae), axis=0)
                del(input_ae)
                del(lables_ae)
                X_train1, y_train1 = shuffle_data(X_train1, y_train1)
                print('validation shape ', y_val.shape, ' test shape ',
                      y_test.shape, 'train shape', y_train1.shape)
                
                nb_batches = int(X_train1.shape[0] / batch_size)
                self.train(session, 15, nb_batches, batch_size,
                           X_train1, y_train1, X_val, y_val, verbose)
            
            # Testing
            auc_test, acc, sensitivity, false_alarm, \
                test_loss1, emb, preds = self.testing(session, X_test, y_test)
                
            auc_score_total += auc_test
            
            history = collect_results(history, sensitivity, false_alarm,
                                      preds, y_test, test_loss1, emb)
            # interictal
            if mode == 'AE':
                filename = os.path.join(
                     str(settings['results_AE']), 'test_hist_%s.hkl' % (target))
            else: 
                filename = os.path.join(
                     str(settings['results']), 'test_hist_%s.hkl' % (target))
            
            savefile(history, filename)
            saver.save(sess=session, save_path=save_path)
            print('Finished evaluation......')
            print('\n')
