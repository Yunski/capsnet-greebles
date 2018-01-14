import os
import scipy.io
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

def random_shift(image):
    return tf.contrib.keras.preprocessing.image.random_shift(image, 0.35, 0.35)


def load_affnist(batch_size, samples_per_epoch=None, is_training=True, use_val_only=False):
    path = os.path.join('data', 'affnist')
    if is_training:
        train_data = loadmat(os.path.join(path, 'training.mat'))
        validation_data = loadmat(os.path.join(path, 'validation.mat'))
              
        X_val = validation_data['affNISTdata']['image'].transpose()
        X_val = X_val.reshape((X_val.shape[0], 40, 40, 1)).astype(np.float32) / 255
        Y_val = validation_data['affNISTdata']['label_int']
        Y_val = Y_val.reshape((Y_val.shape[0])).astype(np.int32)

        X_val, Y_val = shuffle(X_val, Y_val)
        num_val_batches = len(Y_val) // batch_size
 
        Y_train = train_data['affNISTdata']['label_int']
        Y_train = Y_train.reshape((Y_train.shape[0])).astype(np.int32)

        num_train_batches = samples_per_epoch // batch_size if samples_per_epoch else len(Y_train) // batch_size
        if use_val_only:
            print("use_val_only=True")
            return [], X_val, [], Y_val, num_train_batches, num_val_batches 
        
        X_train = train_data['affNISTdata']['image'].transpose()
        X_train = X_train.reshape((X_train.shape[0], 40, 40, 1)).astype(np.float32) / 255
        X_train = np.array([random_shift(image) for image in X_train])

        X_train, Y_train = shuffle(X_train, Y_train)

        return X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches
    else:
        dataset = loadmat(os.path.join(path, 'test.mat'))
        X_test = dataset['affNISTdata']['image'].transpose()
        X_test = X_test.reshape((320000, 40, 40, 1)).astype(np.float32) / 255
        Y_test = dataset['affNISTdata']['label_int']
        Y_test = Y_test.reshape((320000)).astype(np.int32)
        
        num_test_batches = len(Y_test) // batch_size
        
        return X_test, Y_test, num_test_batches


def loadmat(filename):
    '''
    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

