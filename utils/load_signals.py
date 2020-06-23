import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import mne
from utils.save_load import save_hickle_file, load_hickle_file


# the purpose of this function is to return ictal and interictal parts for a particular hour	
def load_signals_CHBMIT(data_dir, target, data_type):
    print ('load_signals_CHBMIT for Patient', target)
    from mne.io import RawArray, read_raw_edf
    from mne.channels import read_montage
    from mne import create_info, concatenate_raws, pick_types
    from mne.filter import notch_filter

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'),header=0)
    #print (onset)
    osfilenames,szstart,szstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
    osfilenames = list(osfilenames)
    #print ('Seizure files:', osfilenames)

    segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'),header=None)
    nsfilenames = list(segment[segment[1]==0][0])

    nsdict = {
            '0':[]
    }
    targets = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23'
    ]
    for t in targets:
        nslist = [elem for elem in nsfilenames if
                  elem.find('chb%s_' %t)!= -1 or
                  elem.find('chb0%s_' %t)!= -1 or
                  elem.find('chb%sa_' %t)!= -1 or
                  elem.find('chb%sb_' %t)!= -1 or
                  elem.find('chb%sc_' %t)!= -1]
        nsdict[t] = nslist
    #nsfilenames = shuffle(nsfilenames, random_state=0)

    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'),header=None)
    sifilenames,sistart,sistop = special_interictal[0],special_interictal[1],special_interictal[2]
    sifilenames = list(sifilenames)

    def strcv(i):
        if i < 10:
            return '0' + str(i)
        elif i < 100:
            return str(i)

    strtrg = 'chb' + strcv(int(target))    
    dir = os.path.join(data_dir, strtrg)
    text_files = [f for f in os.listdir(dir) if f.endswith('.edf')]
    #print (target,strtrg)
    print (text_files)

    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]
        print ('ictal files', filenames)
    elif data_type == 'interictal':
        filenames = [filename for filename in text_files if filename in nsdict[target]]
        print ('interictal files', filenames)

    totalfiles = len(filenames)
    print ('Total %s files %d' % (data_type,totalfiles))
    for filename in filenames:

        exclude_chs = []
        if target in ['4','9']:
            exclude_chs = [u'T8-P8']

        if target in ['13','16']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
        elif target in ['4']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
        else:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']


        rawEEG = read_raw_edf('%s/%s' % (dir, filename),
                              #exclude=exclude_chs,  #only work in mne 0.16
                              verbose=0,preload=True)

        rawEEG.pick_channels(chs)
        print (rawEEG.ch_names)
        
        rawEEG.notch_filter(freqs=np.arange(60,121,60))
        tmp = rawEEG.to_data_frame()
        print('tmp as dataframe')

        tmp = tmp.as_matrix()


        if data_type == 'ictal':
            SOP = 30 * 60 * 256
            # get seizure onset information
            indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
            if len(indices) > 0:
                print ('%d seizures in the file %s' % (len(indices),filename))
                prev_sp = -1e6
                for i in range(len(indices)):
                    st = szstart[indices[i]]*256 - 5 * 60 * 256 #SPH=5min
                    sp = szstop[indices[i]]*256
                    #print ('Seizure %s %d starts at %d stops at %d last sz stop is %d' % (filename, i, (st+5*60*256),sp,prev_sp))

                    # take care of some special filenames
                    if filename[6]=='_':
                        seq = int(filename[7:9])
                    else:
                        seq = int(filename[6:8])
                    if filename == 'chb02_16+.edf':
                        prevfile = 'chb02_16.edf'
                    else:
                        if filename[6]=='_':
                            prevfile = '%s_%s.edf' %(filename[:6],strcv(seq-1))
                        else:
                            prevfile = '%s_%s.edf' %(filename[:5],strcv(seq-1))

                    if st - SOP > prev_sp:
                        prev_sp = sp
                        if st - SOP >= 0:
                            data = tmp[st - SOP : st]
                        else:
                            if os.path.exists('%s/%s' % (dir, prevfile)):
                                rawEEG = read_raw_edf('%s/%s' % (dir, prevfile), preload=True,verbose=0)
                                rawEEG.pick_channels(chs)
                                prevtmp = rawEEG.to_data_frame()
                                prevtmp = prevtmp.as_matrix()
                                if st > 0:
                                    data = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                                else:
                                    data = prevtmp[st - SOP:st]

                            else:
                                if st > 0:
                                    data = tmp[:st]
                                else:
                                    #raise Exception("file %s does not contain useful info" % filename)
                                    print ("WARNING: file %s does not contain useful info" % filename)
                                    continue
                    else:
                        prev_sp = sp
                        continue

                    print ('data shape', data.shape)
                    if data.shape[0] == SOP:
                        yield(data)
                    else:
                        continue

        elif data_type == 'interictal':
            if filename in sifilenames:
                st = sistart[sifilenames.index(filename)]
                sp = sistop[sifilenames.index(filename)]
                if sp < 0:
                    data = tmp[st*256:]
                else:
                    data = tmp[st*256:sp*256]
            else:
                data = tmp
            print ('data shape', data.shape)
            yield(data)
            
def load_signals_FB(data_dir, target, data_type):
    print ('load_signals_FB for Patient', target)

    def strcv(i):
        if i < 10:
            return '000' + str(i)
        elif i < 100:
            return '00' + str(i)
        elif i < 1000:
            return '0' + str(i)
        elif i < 10000:
            return str(i) 

    if int(target) < 10:
        strtrg = '00' + str(target)
    elif int(target) < 100:
        strtrg = '0' + str(target)

    if data_type == 'ictal':

        SOP = 30*60*256
        target_ = 'pat%sIktal' % strtrg
        dir = os.path.join(data_dir, target_)
        df_sz = pd.read_csv(
            os.path.join(data_dir,'seizure.csv'),index_col=None,header=0)
        df_sz = df_sz[df_sz.patient==int(target)]
        df_sz.reset_index(inplace=True,drop=True)

        print (df_sz)
        print ('Patient %s has %d seizures' % (target,df_sz.shape[0]))
        for i in range(df_sz.shape[0]):
            data = []
            filename = df_sz.iloc[i]['filename']
            st = df_sz.iloc[i]['start'] - 5*60*256
            print ('Seizure %s starts at %d' % (filename, st))
            for ch in range(1,7):
                filename2 = '%s/%s_%d.asc' % (dir, filename, ch)
                if os.path.exists(filename2):
                    tmp = np.loadtxt(filename2)
                    seq = int(filename[-4:])
                    prevfile = '%s/%s%s_%d.asc' % (dir, filename[:-4], strcv(seq - 1), ch)

                    if st - SOP >= 0:
                        tmp = tmp[st - SOP:st]
                    else:
                        prevtmp = np.loadtxt(prevfile)
                        if os.path.exists(prevfile):
                            if st > 0:
                                tmp = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                            else:
                                tmp = prevtmp[st - SOP:st]
                        else:
                            if st > 0:
                                tmp = tmp[:st]
                            else:
                                raise Exception("file %s does not contain useful info" % filename)

                    tmp = tmp.reshape(1, tmp.shape[0])
                    data.append(tmp)

                else:
                    raise Exception("file %s not found" % filename)
            if len(data) > 0:
                concat = np.concatenate(data)
                print (concat.shape)
                yield (concat)

    elif data_type == 'interictal':
        target_ = 'pat%sInteriktal' % strtrg
        dir = os.path.join(data_dir, target_)
        text_files = [f for f in os.listdir(dir) if f.endswith('.asc')]
        prefixes = [text[:8] for text in text_files]
        prefixes = set(prefixes)
        prefixes = sorted(prefixes)

        totalfiles = len(text_files)
        print (prefixes, totalfiles)

        done = False
        count = 0

        for prefix in prefixes:
            i = 0
            while not done:

                i += 1

                stri = strcv(i)
                data = []
                for ch in range(1, 7):
                    filename = '%s/%s_%s_%d.asc' % (dir, prefix, stri, ch)

                    if os.path.exists(filename):
                        try:                           
                            tmp = np.loadtxt(filename)
                            tmp = tmp.reshape(1, tmp.shape[0])
                            data.append(tmp)
                            count += 1
                        except:
                            print ('OOOPS, this file can not be loaded', filename)                    
                    elif count >= totalfiles:
                        done = True
                    elif count < totalfiles:
                        break
                    else:
                        raise Exception("file %s not found" % filename)

                if i > 99999:
                    break

                if len(data) > 0:
                    yield (np.concatenate(data))
                    
def FB_notch_filter(data):
        d=np.zeros((data.shape))
        for i in range((data.shape[1])):
            d[:,i]=mne.filter.notch_filter(data[:,i],256,freqs=np.arange(50,101,50))
        return d
                 
class PrepData():
    def __init__(self, target, type, settings):
        print("**********")
        self.target = target
        self.settings = settings
        self.type = type
        self.global_proj = np.array([0.0]*114)

    def read_raw_signal(self):
        if self.settings['dataset'] == 'CHBMIT':
            print("*************")
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)
            return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)
        elif self.settings['dataset'] == 'FB':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)
            return load_signals_FB(self.settings['datadir'], self.target, self.type)
        elif self.settings['dataset'] == 'Kaggle2014Pred':
            if self.type == 'ictal':
                data_type = 'preictal'
            else:
                data_type = self.type
            return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)

        return 'array, freq, misc'
    #this function returns windowed ictal and interictal signals with window length of 28sec and overlap 50%
    
    
            
    def preprocess(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq  # re-sample to target frequency
        numts =28

#        df_sampling = pd.read_csv(
#            'sampling_%s.csv' % self.settings['dataset'],
#            header=0,index_col=None)
#        trg = int(self.target)
#        print (df_sampling)
#        print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
#        ictal_ovl_pt = \
#            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_pt=0.50   # overlap 50%
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt*numts) 
        interictal_ovl_len = int(targetFrequency*ictal_ovl_pt*numts)

        def process_raw_data(mat_data):            
            print ('Loading data')       
            X=[]
            y=[]
            #scale_ = scale_coef[target]
            for data in mat_data:
                if self.settings['dataset'] == 'FB':
                
                    data = data.transpose()
                X_temp = []
                y_temp = []
                
                if ictal:
                    y_value=1
                else:
                    y_value=0

                totalSample = int(data.shape[0]/targetFrequency/numts) + 1
                print('Total samples= ',totalSample)
                window_len = int(targetFrequency*numts)
                print ('ictal_ovl_len =', ictal_ovl_len)
                scaler = StandardScaler()

                data = scaler.fit_transform(data)
#                print(data.shape)

                if ictal:
                        i=0
                        while(window_len + (i)*ictal_ovl_len <= data.shape[0]):
                            win = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len,:] # segmenting the signal with overlap 50%
                            X_temp.append(win)  #here we need to return only the windowed signal
                            y_temp.append(y_value)
                     
                            i+=1
                else:
#                        for j in range(totalSample):
#                            if((j+1)*window_len <= data.shape[0]):
#                                s = data[j*window_len:j*window_len + window_len,:] 
#                                X_temp.append(s)  #here we need to return only the windowed signal
#                                y_temp.append(y_value)
                    i=0
                    while(window_len + (i)*interictal_ovl_len <= data.shape[0]):
                            win = data[i*interictal_ovl_len:i*interictal_ovl_len + window_len,:] # segmenting the signal with overlap 50%
                            X_temp.append(win)  #here we need to return only the windowed signal
                            y_temp.append(y_value)
                     
                            i+=1
                y_temp=np.array(y_temp, dtype='float32')
                X_temp=np.array(X_temp, dtype='float32')
                
#                if 'X' in locals():
#                    X=np.vstack((X,X_temp[np.newaxis,:,:,:]))
#                else:
#                    X =np.empty((0,X_temp.shape[0],X_temp.shape[1],X_temp.shape[2]))
#                    X=np.vstack((X,X_temp[np.newaxis,:,:,:]))
#                
#                if 'y' in locals():
#                    y=np.vstack((y,y_temp[np.newaxis,:]))
#                else:
#                    y=np.empty((0,y_temp.shape[0]))
#                    y=np.vstack((y,y_temp[np.newaxis,:]))
                    
                    
                X.append(X_temp)
                y.append(y_temp) 
            n_samples = np.concatenate(X, axis=0).shape[0]
            # ensure that each fold has the same number of samples
            folds = int(n_samples/len(X_temp))
            max_len = folds * X_temp.shape[0]
            if max_len < n_samples:
                X=np.concatenate(X, axis=0)
                y=np.concatenate(y, axis=0)
                X=X[0:max_len]
                y=y[0:max_len]
                X=X.reshape(-1,X_temp.shape[0],X_temp.shape[1],X_temp.shape[2])
                y=y.reshape(-1,y_temp.shape[0])
            elif max_len == n_samples:
                X = np.array(X, dtype='float32')
                y = np.array(y, dtype='float32')
               
                
            if ictal or interictal:
#                X=np.array(X)#.reshape(-1,X_temp.shape[0],X_temp.shape[1],X_temp.shape[2])
#                y=np.array(y)#.reshape(-1,y_temp.shape[0])

                print ('Xshape ', X.shape, 'y shape', y.shape)
                return X, y
            else:
#                X=np.array(X)#.reshape(-1,X_temp.shape[0],X_temp.shape[1],X_temp.shape[2])
#                y=np.array(y)#.reshape(-1,y_temp.shape[0],y_temp.shape[1],y_temp.shape[2])
##                
                print ('X', X.shape)
                return X,y

        data = process_raw_data(data_)

        return  data

    def apply(self):

        filename = '%s_%s' % (self.type, self.target)

        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))


        if cache is not None:
            print("Cache is not known")
            return cache

        data = self.read_raw_signal()


        X, y = self.preprocess(data)
        #saving part
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y