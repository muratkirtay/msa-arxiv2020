import numpy as np
import pickle
import tensorflow as tf
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def visualize_model(model, model_path):
    """ Save the MLP model in .png format"""
    tf.keras.utils.plot_model(model, to_file= model_path, show_shapes=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

def get_model_summary(model):
    """ Adapted from stackoverflow, to convert model summary as string.
        It is created for debugging and logging puposes
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def save_model_params(model, sensor_path):
    """ Save the final configuration of the network in h5 format """
    model.save(sensor_path)

def normalize_inputs(x_train, x_val, x_test):
    """ Perform normaliztion for each inputs"""
    return x_train.astype('float32')/255.0, x_val.astype('float32')/255.0, x_test.astype('float32')/255.0

def normalize_depth_input(x_train, x_val, x_test):
    """ Normalize depth input witn min-max normaliztion"""
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.fit_transform(x_test)

    return x_train.astype('float32'), x_val.astype('float32'), x_test.astype('float32')

def load_rsense_inputs():
    """ Return the flattened inputs for the Realsense camera """
    Xcolor_tr = pickle.load(open('../recognition_io/training_rsense.pkl', 'rb'), encoding='latin1')
    Xcolor_val = pickle.load(open('../recognition_io/validation_rsense.pkl', 'rb'), encoding='latin1')
    Xcolor_tst = pickle.load(open('../recognition_io/testing_rsense.pkl', 'rb'), encoding='latin1')

    return Xcolor_tr, Xcolor_val, Xcolor_tst

def load_ileft_inputs():
    """ Return the flattened inputs for the icubleft camera """
    Xcolor_tr = pickle.load(open('../recognition_io/training_ileft.pkl', 'rb'), encoding='latin1')
    Xcolor_val = pickle.load(open('../recognition_io/validation_ileft.pkl', 'rb'), encoding='latin1')
    Xcolor_tst = pickle.load(open('../recognition_io/testing_ileft.pkl', 'rb'), encoding='latin1')

    return Xcolor_tr, Xcolor_val, Xcolor_tst

def load_iright_inputs():
    """ Return the flattened inputs for the icubright camera """
    Xcolor_tr = pickle.load(open('../recognition_io/training_iright.pkl', 'rb'), encoding='latin1')
    Xcolor_val = pickle.load(open('../recognition_io/validation_iright.pkl', 'rb'), encoding='latin1')
    Xcolor_tst = pickle.load(open('../recognition_io/testing_iright.pkl', 'rb'), encoding='latin1')

    return Xcolor_tr, Xcolor_val, Xcolor_tst


def load_rdepth_inputs():
    """ Return the flattened inputs for the icubright camera """
    Xcolor_tr = pickle.load(open('../recognition_io/training_depth.pkl', 'rb'), encoding='latin1')
    Xcolor_val = pickle.load(open('../recognition_io/validation_depth.pkl', 'rb'), encoding='latin1')
    Xcolor_tst = pickle.load(open('../recognition_io/testing_depth.pkl', 'rb'), encoding='latin1')

    return Xcolor_tr, Xcolor_val, Xcolor_tst


def load_outputs():
    """ Return the object ids for all sensors """
    y_train = pickle.load(open('../recognition_io/y_training.pkl', 'rb'), encoding='latin1')
    y_val = pickle.load(open('../recognition_io/y_validation.pkl', 'rb'), encoding='latin1')
    y_tst = pickle.load(open('../recognition_io/y_testing.pkl', 'rb'), encoding='latin1')

    return y_train, y_val, y_tst

def reshape_input_vect(vect, size=(32,32,1)):
    """ Reshape input vector for convnet operations. """
    inp_vect = []
    for i in range(vect.shape[0]):
        inp_vect.append(np.reshape(vect[i], (32,32,1)))

    return np.asarray(inp_vect, dtype=np.int32)

def save_results(sensor, trace_acc, trace_loss, pred_probs, pred_y):
    """ Save teh various metrics during the training .pkl format"""
    pickle.dump(trace_acc, open('results/'+sensor+'/trace_acc_'+sensor+'.pkl', 'wb'))
    pickle.dump(trace_loss, open('results/'+sensor+'/trace_loss_'+sensor+'.pkl', 'wb'))
    pickle.dump(pred_probs, open('results/'+sensor+'/pred_probs_'+sensor+'.pkl', 'wb'))
    pickle.dump(pred_y, open('results/'+sensor+'/pred_y_'+sensor+'.pkl', 'wb'))



