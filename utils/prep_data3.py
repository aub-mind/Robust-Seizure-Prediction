import numpy as np

def train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio):
    '''
    Prepare data for leave-one-out cross-validation
    :param ictal_X:
    :param ictal_y:
    :param interictal_X:
    :param interictal_y:
    :return: (X_train, y_train, X_val, y_val, X_test, y_test)
    '''
    #For each fold, one seizure is taken out for testing, the rest for training
    #Interictal are concatenated and split into N (no. of seizures) parts,
    #each interictal part is combined with one seizure
    nfold = len(ictal_y)

    print('number of folds= ',nfold)
    interictal_X = np.concatenate(interictal_X,axis=0)
    interictal_y = np.concatenate(interictal_y,axis=0)
    
    
    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/nfold))
    print ('interictal_fold_len',interictal_fold_len)
    
    for i in range(nfold):
        X_test_ictal = ictal_X[i]#.reshape(1,ictal_X.shape[1],ictal_X.shape[2]) # take the ith row (i,:,:) and reshape it
        y_test_ictal = ictal_y[i]#.reshape(-1,1)
       
        
        X_test_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
        y_test_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]

        if i==0:
            X_train_ictal = np.concatenate(ictal_X[1:],axis=0)
            y_train_ictal = np.concatenate(ictal_y[1:],axis=0)

            X_train_interictal = interictal_X[(i + 1) * interictal_fold_len + 1:]
            y_train_interictal = interictal_y[(i + 1) * interictal_fold_len + 1:]
        elif i < nfold-1:
             
            X_train_ictal =np.concatenate((ictal_X[:i],ictal_X[i + 1:]),axis=0)
            if len(X_train_ictal.shape)>3:
                X_train_ictal=np.concatenate((X_train_ictal),axis=0)
            y_train_ictal =np.concatenate((ictal_y[:i],ictal_y[i + 1:]),axis=0)
            if len(y_train_ictal.shape)>1:
                y_train_ictal=np.concatenate((y_train_ictal),axis=0)

            X_train_interictal = np.concatenate([interictal_X[:i * interictal_fold_len], interictal_X[(i + 1) * interictal_fold_len + 1:]],axis=0)
            y_train_interictal = np.concatenate([interictal_y[:i * interictal_fold_len], interictal_y[(i + 1) * interictal_fold_len + 1:]],axis=0)
       
        else:
            X_train_ictal = np.concatenate(ictal_X[:i],axis=0)
            y_train_ictal = np.concatenate(ictal_y[:i],axis=0)

            X_train_interictal = interictal_X[:i * interictal_fold_len]
            y_train_interictal = interictal_y[:i * interictal_fold_len]

       
            
        
            

        print ('ictal shape: ',y_train_ictal.shape,'interictal shape: ',y_train_interictal.shape)

        '''
        Downsampling interictal training set so that the 2 classes
        are balanced
        '''
        down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
        if down_spl > 1:
            X_train_interictal = X_train_interictal[::down_spl]
            y_train_interictal = y_train_interictal[::down_spl]
        elif down_spl == 1:
            X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
            y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

       
        
        X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
        y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
        
       
      
#        s = np.arange(X_train.shape[0])
#        np.random.shuffle(s)
#        X_train[s]
#        y_train[s]
        
        X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)
        y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):], y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)
        
        
        y_train[y_train==2] = 1
        y_val[y_val==2] = 1
        
        
        X_test = np.concatenate((X_test_ictal, X_test_interictal),axis=0)
        y_test = np.concatenate((y_test_ictal, y_test_interictal),axis=0)
        
        # remove overlapped ictal samples in test-set
        X_test = X_test[y_test != 2]
        y_test = y_test[y_test != 2]
        
        yield (X_train, y_train, X_val, y_val, X_test, y_test)
        
        
        





