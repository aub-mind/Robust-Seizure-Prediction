import os
import os.path
import numpy as np
from utils.load_signals_modified_MIT2 import PrepData 
from utils.prep_data2 import  train_val_loo_split, train_val_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import KFold 
import time
import hickle as hkl
from sklearn.metrics import roc_auc_score, auc

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy.linalg import norm

from sklearn.metrics import f1_score
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
#%%

def init_noise(sess):
    sess.run(tf.variables_initializer([x_noise]))
    
def data_loading(target,dataset):
    print ('Data Loading...........')
    if(dataset=="CHBMIT"):
        
        settings={"dataset": "CHBMIT","datadir": "C:\\Users\\anh21\\OneDrive - American University of Beirut\\Epilepsy Project\\Seizure-prediction-CNN-master\\CHBMIT",
          "cachedir": "C:\\Users\\anh21\\OneDrive - American University of Beirut\\Epilepsy Project\\Seizure-prediction-CNN-master\\CHBMIT_SZPred",
          "resultdir": "results",
           "resultsCV":"results_CV",
           "score":"score\\without_filter_MIT",
            "score2":"score\\with_filter_MIT",
            "window":"score\\MIT_1min",
             "score3":"score\\hyp_tuning_MIT",
              "score4":"score\\soa_MIT"
            }
    elif dataset=="FB":
        settings={"dataset": "FB","datadir": "C:\\Users\\anh21\\OneDrive - American University of Beirut\\Epilepsy Project\\Seizure-prediction-CNN-master\\FB",
      "cachedir": "C:\\Users\\anh21\\OneDrive - American University of Beirut\\Epilepsy Project\\Seizure-prediction-CNN-master\\FB_cache",
      "resultdir": "results",
       "score":"score\\without_filter_FB",
            "score2":"score\\with_filter_FB",
            "score3":"score\\hyp_tuning_FB",
             "score4":"score\\soa_FB"
        }

    ictal_X, ictal_y =PrepData(target, type='ictal', settings=settings).apply()
    interictal_X, interictal_y =PrepData(target, type='interictal', settings=settings).apply()


    
    return  ictal_X, ictal_y ,interictal_X, interictal_y,settings

        

def next_batch(X, y, counter, batch_size):
    start_index = counter * batch_size
    end_index = start_index + batch_size
    if end_index > X.shape[0]: end_index = X.shape[0]
    return X[start_index:end_index], y[start_index:end_index]        
        
def savefile(history,file):
    hkl.dump(history,file)
    
    
#%%
#Build the graph

def cnn_gru(x, keep_rate):
    
    with tf.variable_scope('conv_1'):
        conv1 = tf.layers.conv1d(inputs=x, filters=64, kernel_size=5, strides=3, activation = tf.nn.leaky_relu,padding='same',name='conv1')
       
        print ("conv1 shape:", conv1.get_shape())
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
       # drop_out1 = tf.nn.dropout(max_pool_1, cnn_keep_rate)
    with tf.variable_scope('conv_2'):
        conv2 = tf.layers.conv1d(inputs=max_pool_1 , filters=64, kernel_size=3, strides=2, activation = tf.nn.leaky_relu,padding='same',name='conv2')
            
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
       # drop_out2 = tf.nn.dropout(max_pool_2, cnn_keep_rate)
        print ("conv2 shape:", conv2.get_shape())
    with tf.variable_scope('conv_3'):
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=2, strides=1, padding='same', activation = tf.nn.leaky_relu,name='conv3')
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
       # drop_out3 = tf.nn.dropout(max_pool_3, cnn_keep_rate)
#     print("conv5 shape:", conv5.get_shape())
#    with tf.variable_scope('conv_4'):
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=128, kernel_size=2, strides=1, padding='same', activation = tf.nn.leaky_relu,name='conv4')
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
       # drop_out4 = tf.nn.dropout(max_pool_4, cnn_keep_rate)
#    with tf.variable_scope('conv_5'):
#        conv5 = tf.layers.conv1d(inputs=drop_out4, filters=256, kernel_size=3, strides=1, padding='same', activation = tf.nn.leaky_relu,name='conv3')
#        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')
#        drop_out5 = tf.nn.dropout(max_pool_5, cnn_keep_rate)
    
    last_cnn_layer = max_pool_4
    window=last_cnn_layer.get_shape().as_list()[1]
    last_filter_dim = last_cnn_layer.get_shape().as_list()[2] #get the number of features of the last CNN layer
    print('filters',last_filter_dim)
    print('window_size',window)
    gru_size = 128 # no. of GRU cells in one layer (Note: 2^N)
   
    # 							RECURRENT NETWORK
    # 								 GRU 
    with tf.variable_scope('GRU'):
        #features = tf.keras.layers.Flatten()(last_cnn_layer)
        gru_output=tf.keras.layers.GRU(gru_size)(last_cnn_layer)
        
        #this line is the slowest 
        output_Y = tf.layers.dense(gru_output, 2, name='dense_1')
        
    print("Output ",output_Y.get_shape())
    print("Done building the graph.........")
    
#################################################################################
    #get the embedding for further analisys
    embeddings1=gru_output[-1] # the output of the GRU

    # 							
    return output_Y,embeddings1
# In[24]:
dataset='FB'
if dataset=='CHBMIT':
    patients = [
           '1',
             '2',
               '3',
             '5',
         '9',
            '10',
            '13',
           '14',
             '18',
            '19',
           '20',
         '21',
         '23'
        ]
elif dataset=='FB':
    patients = [
#          '1',
#            '3',
#            '4',
#            '5',
#             '6',
#            #  '13',
#             '14',
#             '15',
#             '16',
#             '17', 
#             '18',
            # '19',
            '20',
           '21'
        ]
    

# the actual training

for r in range(len(patients)):
    validation="cv" #cv: LOSOCV , test: 70% train,30% test
    target =patients[r]
    #loading the original data for each patient
    ictal_X, ictal_y ,interictal_X, interictal_y,settings=data_loading(patients[r],dataset)
    
    batch_size=256
    epoch=50
    
    auc_score_total=0
    total_test_sensitivity = 0
    total_test_false_alarm = 0
    
    history = dict(AUC_AVG=[],y_pred=[],y_test=[],test_loss=[],test_sensitivity=[],test_false_alarm=[],embed1=[])
    start_time = time.time()
    Win_Length=ictal_X.shape[2]
    Num_chanels=ictal_X.shape[3]
    
    #Placeholders 
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, [None, Win_Length, Num_chanels], name="X")   
    cnn_keep_rate = tf.placeholder_with_default(1.0, shape=(),name='keep_rate')#drop out rate  50%
    gru_keep_rate = tf.placeholder_with_default(1.0, shape=(),name='keep_rate')#drop out rate  50%
    
    Y = tf.placeholder("float", [None, 2], name="Y") # e.g. take training 2 person's data
    
    #can create the new variable for the adversarial noise. It will not be trainable
    ADVERSARY_VARIABLES = 'adversary_variables'
    collections = [ ADVERSARY_VARIABLES]
    x_noise = tf.Variable(tf.zeros([1,Win_Length,Num_chanels],dtype=tf.float32),
                      name='x_noise', trainable=False,
                      collections=collections)
    adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
    x_ae2=x+x_noise
    noise_limit=int(norm(interictal_X.reshape(-1),2)*0.075) #  7.5% of the max value
    x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))
    output_Y,embedding1=cnn_gru(x_ae2,cnn_keep_rate)
    cost_Y = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_Y,labels =Y) ) 
    l2 = 0.0005* sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()) # adding l2 norm
    total_loss=cost_Y+l2
    
    Y_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)
    
    
    lamda=tf.placeholder_with_default(1.0, shape=(),name='lambda')
    l2_loss=lamda * tf.nn.l2_loss(x_noise,name='L2')
    adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
    #cost_Y = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_Y,labels =Y) ,name='cost_y') 
    loss_adversary = cost_Y+ l2_loss
    optimizer_adversary = tf.train.AdamOptimizer().minimize(loss_adversary, var_list=adversary_variables,name='adv_Adam')
    # Calculating accuracy measure 
    #convert logits to probabilities
    predictions=tf.nn.softmax(output_Y,name='Preds')
        
    ind = 0
    #openning the session
    session=tf.Session()
    if validation=="cv":
        
        loo_folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.1)
        
        for X_train1, y_train1, X_val, y_val, X_test, y_test in loo_folds:
            
            y_val = to_categorical(y_val)
            y_test = to_categorical(y_test)
            y_train1 = to_categorical(y_train1)
            print('validation shape ',y_val.shape,' test shape ',y_test.shape, 'train shape', y_train1.shape)
    
            nb_batches = int(X_train1.shape[0] / batch_size)
            print('Fold:', ind+1)
            #reinitialize graph variables at each fold
            session.run(tf.global_variables_initializer())
            init_noise(session)
            saver = tf.train.Saver()
            save_dir = 'checkpoints_%s/pat%s'%(dataset,target)
             #Create the directory if it does not exist.
            if not os.path.exists(save_dir):
                 os.makedirs(save_dir)
            save_path = os.path.join(save_dir,'CNN_GRU_%s'%(target)) 
            #Training
            for k in range(epoch):
                for j in range(nb_batches):
                    epoch_x1,epoch_y1=next_batch(X_train1,y_train1,j,batch_size)
                    session.run(Y_op, feed_dict={x: epoch_x1, Y: epoch_y1, cnn_keep_rate:1,gru_keep_rate:1})
                session.run(Y_op, feed_dict={x: X_val, Y: y_val, cnn_keep_rate:1,gru_keep_rate:1})  
                X_val=X_val.reshape(-1,Win_Length,Num_chanels)
                preds_val,Y_loss=session.run( [predictions,cost_Y], feed_dict={x: X_val,Y: y_val,cnn_keep_rate: 1,gru_keep_rate:1})
                auc_score_val= roc_auc_score(y_val,preds_val)
                preds_val2=np.argmax(preds_val,axis=1) 
                y_val2=np.argmax(y_val,axis=1)
                #f1_val= f1_score(y_val2,preds_val2,average='weighted') 
                execution_time = time.time() - start_time
    
                c=confusion_matrix(y_val2, preds_val2)
                sens_val=c[1][1]/(c[1][1]+c[1][0])
                fpr_val=c[0][1]/(c[0][1]+c[0][0])
                acc=(c[1][1]+c[0][0])/(c[1][1]+c[0][0]+c[1][0]+c[0][1])
                
                #f1=f1_score(y_val2, preds_val2,'micro') 
                
                print('sensitivity: ',round( sens_val,2),'FPR: ', round(fpr_val,2),'Y_loss: ', round(Y_loss,2),'acc: ', round(acc,2),' time: ',execution_time)
            
            test1=X_test.reshape(-1,Win_Length,Num_chanels)  
            print("Testing..........................")
            test_loss1,preds,emb1=session.run([cost_Y, predictions,embedding1], feed_dict={x: test1, Y: y_test,cnn_keep_rate: 1,gru_keep_rate:1 })
            
            auc_score1=roc_auc_score(y_test,preds)
            y_pred=np.argmax(preds,axis=1) 
            y_test_2=np.argmax(y_test,axis=1)
            #auc_score1=roc_auc_score(y_test_2,y_pred)
            print('test AUC_score: ', auc_score1)
            auc_score_total+=auc_score1
            c=confusion_matrix(y_test_2, y_pred)
            sensitivity=c[1][1]/(c[1][1]+c[1][0])
            false_alarm=c[0][1]/(c[0][1]+c[0][0])
            acc=(c[1][1]+c[0][0])/(c[1][1]+c[0][0]+c[1][0]+c[0][1])
                
            #false_alarm=c[0][1]
           
            #f1_test=f1_score(y_test_2,y_pred, average='weighted') 
           # print('test f1_score: ', f1_test)
            total_test_sensitivity += sensitivity
            total_test_false_alarm += false_alarm
            #auc_score_total+=auc_score
            print('Sensitivity Test: ', sensitivity,'False Alarm Rate Test: ',false_alarm,'acc: ',acc,' time: ',execution_time)
            print('test AUC_score: ', auc_score1)
            ind += 1
            history['test_sensitivity'].append(sensitivity);history['test_false_alarm'].append(false_alarm)
            history['y_pred'].append(preds)
            history['y_test'].append(y_test)
            history['test_loss'].append(test_loss1)
            history['embed1'].append(emb1)
            # interictal
            
        filename = os.path.join(
                 str(settings['score4']), 'history%s.hkl' %(target))
        print("Average AUC patient : ",int(target),' ',auc_score_total/ind)
        history['AUC_AVG'].append(auc_score_total/ind)
        
        savefile(history,filename)
        saver.save(sess=session, save_path=save_path)
        print('Finished LOSOCV ......')
    
    
    elif validation=="test":
        X_train1, y_train1, X_val, y_val, X_test, y_test  = train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y,0.1,0.2)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)
        y_train1 = to_categorical(y_train1)
        print('validation shape ',y_val.shape,' test shape ',y_test.shape, 'train shape', y_train1.shape)

        nb_batches = int(X_train1.shape[0] / batch_size)
        
        #reinitialize graph variables at each fold
        session.run(tf.global_variables_initializer())
        init_noise(session)
        saver = tf.train.Saver()
        save_dir = 'checkpoints_%s/pat%s'%(dataset,target)
         #Create the directory if it does not exist.
        if not os.path.exists(save_dir):
             os.makedirs(save_dir)
        save_path = os.path.join(save_dir,'CNN_GRU_%s'%(target)) 
        #Training
        for k in range(epoch):
            for j in range(nb_batches):
                epoch_x1,epoch_y1=next_batch(X_train1,y_train1,j,batch_size)
                session.run(Y_op, feed_dict={x: epoch_x1, Y: epoch_y1, cnn_keep_rate: 1,gru_keep_rate:1})
            
            X_val=X_val.reshape(-1,Win_Length,Num_chanels)
            preds_val,Y_loss=session.run( [predictions,cost_Y], feed_dict={x: X_val,Y: y_val,cnn_keep_rate: 1,gru_keep_rate:1})
            session.run(Y_op, feed_dict={x: X_val, Y: y_val, cnn_keep_rate:1,gru_keep_rate:1})  
           # auc_score_val= roc_auc_score(y_val,preds_val)
            preds_val2=np.argmax(preds_val,axis=1) 
            y_val2=np.argmax(y_val,axis=1)
            #f1_val= f1_score(y_val2,preds_val2,average='weighted') 
            execution_time = time.time() - start_time

            c=confusion_matrix(y_val2, preds_val2)
            sens_val=c[1][1]/(c[1][1]+c[1][0])
            fpr_val=c[0][1]/(c[0][1]+c[0][0])
            acc=(c[1][1]+c[0][0])/(c[1][1]+c[0][0]+c[1][0]+c[0][1])
            #fpr_val=c[0][1]
            #f1=f1_score(y_val2, preds_val2,'micro') 
            
            print('Evaluation sensitivity: ',round( sens_val,2),'FPR: ', round(fpr_val,2),' time: ',execution_time)
        
#        
        test1=X_test.reshape(-1,Win_Length,Num_chanels)  
        print("Testing..........................")
        test_loss1,preds,emb1=session.run([cost_Y, predictions,embedding1], feed_dict={x: test1, Y: y_test,cnn_keep_rate: 1,gru_keep_rate:1 })
        
        auc_score1=roc_auc_score(y_test,preds)
        auc_score_total+=auc_score1
        y_pred=np.argmax(preds,axis=1) 
        y_test_2=np.argmax(y_test,axis=1)
        #auc_score1=roc_auc_score(y_test_2,y_pred)
        print('test AUC_score: ', auc_score1)
        
        c=confusion_matrix(y_test_2, y_pred)
        sensitivity=c[1][1]/(c[1][1]+c[1][0])
        false_alarm=c[0][1]/(c[0][1]+c[0][0])
        acc=(c[1][1]+c[0][0])/(c[1][1]+c[0][0]+c[1][0]+c[0][1])
        #f1_test=f1_score(y_test_2,y_pred, average='weighted') 
       # print('test f1_score: ', f1_test)
        total_test_sensitivity += sensitivity
        total_test_false_alarm += false_alarm
        #auc_score_total+=auc_score
        print('Sensitivity Test: ', sensitivity,'False Alarm Rate Test: ',false_alarm,'acc: ',acc,' time: ',execution_time)
        print('test AUC_score: ', auc_score1)
        ind += 1
        history['test_sensitivity'].append(sensitivity);history['test_false_alarm'].append(false_alarm)
        history['y_pred'].append(preds)
        history['y_test'].append(y_test)
        history['test_loss'].append(test_loss1)
        history['embed1'].append(emb1)
        # interictal
        
        filename = os.path.join(
                 str(settings['score4']), 'history%s.hkl' %(target))
        print("Average AUC patient : ",int(target),' ',auc_score_total/ind)
        history['AUC_AVG'].append(auc_score_total/ind)
        
        savefile(history,filename)
        saver.save(sess=session, save_path=save_path)
        print('Finished regular training......')

#%%
