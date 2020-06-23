from sklearn.metrics import classification_report
import pandas as pd
import hickle as hkl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc,roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#loading the results for patient
def load_results(path,dataset):
    if(dataset=='CHBMIT'):
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
    elif (dataset=='FB'):
        
        patients = [
           '1',
            '3',
            '4',
            '5',
            '6',
#             '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21' ]
    data=dict()
    for i in patients:
        d=hkl.load(path+'history%s.hkl'%(i))
        data[i]=d
    return data, patients

def calculate_fpr(pred, test_w):
    """Calculate FPR based on et. al work"""
    fpr=0
    seg=[]
    test=[]

    test=np.array(test_w)
    # count all false positive predictions
    pred = pred[np.where(test == 0)[0]]  
    
#    pdb.set_trace()
    n_five_mins = (len(pred)/10) + 1
    # convert the 30 sec predictions to 5 min predictions 
    # ie one prediction each 10 windows of 30 seconds
    counter = 0
    while (counter + 1) <= n_five_mins:
        win = pred[counter*10: (counter+1)*10]
        r = np.count_nonzero(win)
        if r >= 8:
            seg.append(1)
        else: seg.append(0)
        counter += 1
    
    j=0
#    pdb.set_trace()
    if len(seg)>0:
        while j+7 <= len(seg):
        # the alarm raised for 35 minutes
            if seg[j] == 1:
                j += 7
                fpr += 1  # count the number of false positive alarms
            else:
                j += 1 
        
        fpr= fpr/(n_five_mins*5/60)  # calculate the FPR/h
    else:
        fpr=0
    return fpr

def data_prep(data):
    y_test=np.array(data['y_test'])
    y_pred=np.array(data['y_pred'])
    y_test=np.concatenate(y_test,axis=0)
    y_pred=np.concatenate(y_pred,axis=0)
    return  y_test, y_pred


def auc_results(data, patients):
   
    results=[]
    for i in patients:
        test1,pred1=data_prep(data[i])
        auc_score=data[i]['AUC_AVG'][0]
        #auc_score=roc_auc_score(test1.reshape(-1),pred1.reshape(-1))
        results.append(round(auc_score,4))
    return np.array(results)

def summary_results(patients, data):
   
    avg_std=0
    avg_std_m=0
    avg_fpr=0
    avg_fpr_std=0
    
    count=0
    for i in patients:
         
        sen_mean=np.array(data[i]['test_sensitivity']).mean()*100
        test=np.argmax(data[i]['y_test'][0],axis=1)
        avg_std+=sen_mean
        fpr_list=[]
        for j in range(len(data[i]['y_pred'])):
            pred= np.argmax(data[i]['y_pred'][0], axis=1)
            test= np.argmax(data[i]['y_test'][0], axis=1)
            fpr_list.append(calculate_fpr(pred,test))
        fpr=np.mean(fpr_list)
        fpr_std=np.std(fpr_list)
        avg_fpr+=fpr
        avg_fpr_std+=fpr_std
        sen_std=np.array(data[i]['test_sensitivity']).std()*100
        avg_std_m+=sen_std
        
        avg_fpr_std+=fpr_std
        
        print('patient'+ i,' sensitivity: ',round(sen_mean,4),'sen_std: ',round(sen_std,4),' FPR: ',round(fpr,4),'fpr_std: ',round(fpr_std,4))
        count+=1
    print("AVG_sens: ",avg_std/13,' std: ',avg_std_m/13,' AVG_FPR: ',avg_fpr/13,' std: ',avg_fpr_std/13)
    