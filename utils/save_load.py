import os
import hickle as hkl

def save_hickle_file(filename, data):
    filename = filename + '.hickle'
    print ('Saving to %s' % filename)

    
    hkl.dump(data,filename)

def load_hickle_file(filename):
    filename = filename + '.hickle'
    if os.path.isfile(filename):
        print ('Loading %s ...' % filename)
        data = hkl.load(filename)
        return data
    return None

def savefile(history, file):
    hkl.dump(history, file)


def load_ae(path, target):
    """ Load generated adversarial examples"""
    input_filename = path+'augmented_input'+target+'.hkl'
    lables_filename = path+'augmented_lables'+target+'.hkl'
    if os.path.isfile(input_filename):
        print ('Loading %s ...' % input_filename)
        data = hkl.load(input_filename)
        lables = hkl.load(lables_filename)
        return data, lables
    
    return None, None
