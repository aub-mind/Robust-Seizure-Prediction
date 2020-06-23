"""
This file contains functions to help loading and preparing the EEG CHBMIT and
FB datasets.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def calc_metrics(y, preds):
    """ Calculate the matrics of interest """

    auc_score = roc_auc_score(y, preds)
    preds = np.argmax(preds, axis=1)
    y = np.argmax(y, axis=1)
    c = confusion_matrix(y, preds)
    sens = c[1][1]/(c[1][1]+c[1][0])
    fpr = c[0][1]/(c[0][1]+c[0][0])
    acc = (c[1][1]+c[0][0])/(c[1][1]+c[0][0]+c[1][0]+c[0][1])

    return auc_score, acc, sens, fpr


def next_batch(X, y, counter, batch_size):
    """ Returns the next batch from the preprocessed data

    Args:
    X: EEG input data of size [batch_size, window_length, electrodes]
    y: lables, size [batch_size]
    counter (int): the batch number
    
    Returns:
    batch of data and corresponding lables
    """

    start_index = counter * batch_size
    end_index = start_index + batch_size
    if end_index > X.shape[0]:
        end_index = X.shape[0]
    return X[start_index: end_index], y[start_index: end_index]

def collect_results(history, sensitivity, false_alarm, preds,
                    y_test, test_loss1, emb):
    history['test_sensitivity'].append(sensitivity)
    history['test_false_alarm'].append(false_alarm)
    history['y_pred'].append(preds)
    history['y_test'].append(y_test)
    history['test_loss'].append(test_loss1)
    history['embed1'].append(emb)

    return history


def gn_augment(data, y, std=1):
    """ Helping function for augmentation with gaussian noise"""
    n = np.random.randn(data.shape[0], data.shape[1], data.shape[2])*std
    return data+n, y

def shuffle_data(X_train, y_train):
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    
    return X_train[s], y_train[s]

def train_val_cv_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio, is_shuffeling=True):
    """ Prepare data for leave-one-out cross-validation
    For each fold, one seizure is taken out for testing, the rest for training
    Interictal are concatenated and split into N (no. of seizures) parts,
    each interictal part is combined with one seizure
    
    ِArgs:
    ictal_X: EEG input data of size [n_folds, samples_in fold, window_length, electrodes] 
    ictal_y: lables, size [(train+valid) size]
    interictal_X: EEG input data of size [n_folds, samples_in fold, window_length, electrodes] 
    interictal_y: lables, size [(train+valid) size]
    
    Returns:
    train, val, test splitted data (X_train, y_train, X_val, y_val, X_test, y_test) 
    """
    
    if len(ictal_y) > len(interictal_y):
        nfold = len(interictal_y)
    else:
        nfold = len(ictal_y)
    print('number of folds= ', nfold)

#    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/nfold))
#    print ('interictal_fold_len',interictal_fold_len)
    
    for i in range(nfold):
        X_test_ictal = ictal_X[i] 
        y_test_ictal = ictal_y[i] 
       
      
        X_test_interictal = interictal_X[i]
        y_test_interictal = interictal_y[i]
#        
        if i==0: #lower bound (case the first instance is the one getting held out)
            X_train_ictal = np.concatenate(ictal_X[1:], axis=0)
            y_train_ictal = np.concatenate(ictal_y[1:], axis=0)

            X_train_interictal = np.concatenate(interictal_X[1:], axis=0)
            y_train_interictal = np.concatenate(interictal_y[1:], axis=0)
        elif i < nfold-1: # in-between

            X_train_ictal = np.concatenate((ictal_X[: i], ictal_X[i+1:]), axis=0)
            if len(X_train_ictal.shape) > 3:
                X_train_ictal = np.concatenate((X_train_ictal), axis=0)
            y_train_ictal =np.concatenate((ictal_y[: i], ictal_y[i+1:]), axis=0)
            if len(y_train_ictal.shape) > 1:
                y_train_ictal = np.concatenate((y_train_ictal), axis=0)
                
            X_train_interictal = np.concatenate([interictal_X[: i], interictal_X[i+1:]], axis=0)
            y_train_interictal = np.concatenate([interictal_y[: i], interictal_y[i+1:]], axis=0)
        
            if len(X_train_interictal.shape) > 3:
                X_train_interictal=np.concatenate((X_train_interictal),axis=0)
                y_train_interictal=np.concatenate((y_train_interictal),axis=0)
        else:  
            X_train_ictal = np.concatenate(ictal_X[: i], axis=0)
            y_train_ictal = np.concatenate(ictal_y[: i], axis=0)

            X_train_interictal = np.concatenate(interictal_X[: i], axis=0)
            y_train_interictal = np.concatenate(interictal_y[: i], axis=0)

        
        '''
        Downsampling interictal training set so that the 2 classes
        are balanced
        '''
        
        X_train_interictal = X_train_interictal[0: X_train_ictal.shape[0]]
        y_train_interictal = y_train_interictal[0: y_train_ictal.shape[0]]
        X_test_interictal = X_test_interictal[0: X_test_ictal.shape[0]]
        y_test_interictal = y_test_interictal[0: y_test_ictal.shape[0]]
#        down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
#        if down_spl > 1:
#            X_train_interictal = X_train_interictal[::down_spl]
#            y_train_interictal = y_train_interictal[::down_spl]
#        elif down_spl == 1:
#            X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
#            y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]
##        
#        down_spl = int(np.floor(y_test_interictal.shape[0]/y_test_ictal.shape[0]))
#        if down_spl > 1:
#            X_test_interictal = X_test_interictal[::down_spl]
#            y_test_interictal = y_test_interictal[::down_spl]
        

        # print ('Balancing:', y_train_ictal.shape,y_train_interictal.shape)
        
        
        X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],
                                                X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]), axis=0)
        
        y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], 
                                                y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]), axis=0)
        if is_shuffeling == True:
            X_train, y_train = shuffle_data(X_train, y_train)
        
        X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],
                                              X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]), axis=0)
        y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):], 
                                              y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]), axis=0)

        X_test = np.concatenate((X_test_ictal, X_test_interictal), axis=0)
        y_test = np.concatenate((y_test_ictal, y_test_interictal), axis=0)
       # X_test = norm.transform(X_test.reshape(-1,X_test_ictal.shape[2])).reshape(-1, X_test_ictal.shape[1], X_test_ictal.shape[2])
        
        yield (X_train, y_train, X_val, y_val, X_test, y_test)
        
        
        
def train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio, test_ratio, is_shuffeling=True):
    
    """
    Prepare data for train, val, test
    
    Args:
    ictal_X: EEG input data of size [n_folds, samples_in fold, window_length, electrodes] 
    ictal_y: lables, size [(train+valid) size]
    interictal_X: EEG input data of size [n_folds, samples_in fold, window_length, electrodes] 
    interictal_y: lables, size [(train+valid) size]
    
    Returns:
    train, val, test splitted data (X_train, y_train, X_val, y_val, X_test, y_test) 
    
    """
    
    num_sz = len(ictal_y)
    num_sz_test = int(np.ceil(test_ratio*num_sz))
    print ('Total %d seizures. Last %d is used for testing.' %(num_sz, num_sz_test))
    interictal_X = interictal_X[0: len(ictal_y)]
    interictal_y = interictal_y[0: len(ictal_y)]
#    if isinstance(interictal_y, list):
    interictal_X = np.concatenate(interictal_X,axis=0)
    interictal_y = np.concatenate(interictal_y,axis=0)
#    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))
  


    X_test_ictal = np.concatenate(ictal_X[-num_sz_test:],axis=0)
    y_test_ictal = np.concatenate(ictal_y[-num_sz_test:],axis=0)

    X_test_interictal = interictal_X[-num_sz_test:]
    y_test_interictal = interictal_y[-num_sz_test:]

    X_train_ictal = np.concatenate(ictal_X[:-num_sz_test],axis=0)
    y_train_ictal = np.concatenate(ictal_y[:-num_sz_test],axis=0)

    X_train_interictal =  interictal_X[:-num_sz_test]
    y_train_interictal = interictal_y[:-num_sz_test]

    print (y_train_ictal.shape,y_train_interictal.shape)

    ''' Downsampling interictal training set so that the 2 classes
    are balanced
    '''
    
    down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
    if down_spl > 1:
        X_train_interictal = X_train_interictal[::down_spl]
        y_train_interictal = y_train_interictal[::down_spl]
    elif down_spl == 1:
        X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
        y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

    print ('Balancing:', y_train_ictal.shape,y_train_interictal.shape)

    X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],
                                            X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
    y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],
                                            y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)

#    if len(X_train_ictal) < (len(X_train_interictal)):
#            f = int(0.3 * len(X_train_ictal))
#    else:
#            f = int(0.3 * len(X_train_interictal))
#    idx = np.random.choice(f, f, replace=False)
#    X_train_ictal_gn, y_train_ictal_gn = gn_augment(X_train_ictal[idx], y_train_ictal[idx], std=0.0001)
#    X_train_interictal_gn, y_train_interictal_gn = gn_augment(X_train_interictal[idx], y_train_interictal[idx], std=0.0001)
#
#        
#    X_train = np.concatenate((X_train,X_train_ictal_gn, X_train_interictal_gn), axis=0)
#    y_train = np.concatenate((y_train,y_train_ictal_gn, y_train_interictal_gn), axis=0)
#    del(X_train_ictal_gn,y_train_ictal_gn,X_train_interictal_gn,y_train_interictal_gn)
    if is_shuffeling == True:
         X_train, y_train = shuffle_data(X_train, y_train)
    
    X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],
                                          X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]), axis=0)
    y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],
                                          y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]), axis=0)



    X_test = np.concatenate((X_test_ictal, X_test_interictal), axis=0)
    y_test = np.concatenate((y_test_ictal, y_test_interictal), axis=0)

    print ('X_train, X_val, X_test',X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_eeg(eeg_sig, pred, y, title, is_xlabel=False):

    eeg_len = eeg_sig.shape[0]
    states = {0: 'interictal', 1: 'preictal'}
    x = np.linspace(0.0, eeg_len/256, eeg_len)
    plt.figure()
    plt.grid(which='major', color='#666666', linestyle='--')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle=':', alpha=0.5)
    plt.style.use('seaborn-deep')
    plt.plot(x, eeg_sig)
    if is_xlabel == True:
        plt.xlabel('Time(s) \n Network Output:  %s \n Network Prediction:  %s' %
                   (str(pred), states[np.argmax(y)]), fontsize=14)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('uVolt', fontsize=14)
    plt.savefig('{}.png'.format(title), dpi=400, bbox_inches='tight')
    plt.show()


def generate_adversarial(model, sess, target, X, y, epochs):
    """
    Generate and visualize the generated AE with the coresponding input and
    model prediction

    Args:
    model: tensorflow model from built with CNN_GRU class
    X: EEG input data of size [(train+valid) size, window_length,
    electrodes]
    y: one hot encoded lables, size [(train+valid) size]
    sess: tensorflow session
    target: the target class of the adversarial example
    """

    actual = np.flip(target).reshape(-1, 2)  # the lable of the generated AE
    epochs = epochs
    # find a sample of target category
    idx = np.random.randint(0, y.shape[0])
    while (y[idx] != target).all():
        idx = np.random.randint(0, y.shape[0])
    inp = X[idx, :, :].reshape(-1, 7168, X.shape[2])

    print("y: ", y[idx], 'index :', idx)
    pred2 = model.feed_forward(sess, inp)
    pred2 = (np.round(np.array(pred2[0]), 3))
    print('Network Prediction on original signal: ' +
          str(pred2) + '\n')

    title = 'Preictal EEG Signal'
    plot_eeg(inp[0, 0:2000, 1], pred2, y[idx], title, is_xlabel=True)

    ae = model.adversarial(sess, actual, inp, epochs)
    # check if the AE is different from the input
    print('is x_ae == x -->', (ae == inp).all())
    # model prediction on generated AE
    pred = model.feed_forward(sess, ae)
    pred = (np.round(np.array(pred[0]), 3))

    title = 'Adversarial Example'
    plot_eeg(ae[0, 0:2000, 1], pred, y[idx], title, is_xlabel=True)
    # get the adversarial noise
    n = sess.run(model.x_noise)
    title = 'EEG AE Noise'
    plot_eeg(n[0, 0:2000, 1], pred, y[idx], title, is_xlabel=False)
