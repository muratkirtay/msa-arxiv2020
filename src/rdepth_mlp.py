import numpy as np
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import time
import logging as lg
import model_helpers as hlp



def depth_mlp_model(input_len, hidden_units, outputs):
    """ Build three hidden layered network """

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_len, )),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dense(outputs, activation='softmax')
    ])

    return model

def main():

    desc = "rdepth  modality processing for object recognition"
    fname = 'runtime_logs/rdepth_mlp.log'  # 0-change this line for different sensor

    init = time.time()
    lg.basicConfig(filename=fname, format='%(message)s', level=lg.INFO)
    lg.info('Experiment started date/time: %s', time.ctime())

    # 1-change vishpath
    vis_path, path_h5s = 'visualization/rdepth_mlp_modelg.png', 'modelh5s/'
    Xtr, Xval, Xtst = hlp.load_rdepth_inputs() # 2-change sensory input
    y_train, y_val, y_tst = hlp.load_outputs()
    nof_objects, hidden_units, size = 100, 256 ,1024 #(32,32,1)
    nof_epochs, no_change_eps = 100, 30

    lg.info("I/O shapes----------------------------------------------")
    lg.info("Input train, validation, test: %s, %s, %s ", str(Xtr.shape), str(Xval.shape), str(Xtst.shape))
    lg.info("Output train, validation, test: %s, %s, %s ", str(y_train.shape), str(y_val.shape), str(y_tst.shape))

    Xtr, Xval, Xtst = hlp.normalize_depth_input(Xtr, Xval, Xtst)

    model = depth_mlp_model(size, hidden_units, nof_objects) #unimodalnet(size, nof_objects)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    estop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=no_change_eps)
    traces = model.fit(Xtr, y_train, callbacks=[estop_callback], validation_data=(Xval, y_val), epochs=nof_epochs, verbose=2)
    trace_acc, trace_loss = traces.history['accuracy'], traces.history['loss']

    test_loss, test_acc = model.evaluate(Xtst, y_tst, verbose=2)

    # save model params and results for postprocessing
    hlp.save_model_params(model, 'modelh5s/rdepth_mlp_model.h5') # 3- change the name of h5 file
    pred_probs = model.predict(Xtst)
    pred_y = pred_probs.argmax(axis=-1)

    hlp.save_results("rdepth_mlp", trace_acc, trace_loss, pred_probs, pred_y) # 4- change the name of result folder

    print("---" * 10)
    print("Test accuracy: {}".format(test_acc))

    # logging info
    hlp.visualize_model(model, vis_path)
    lg.info("Model Summary-----------------------------------------")
    lg.info(' %s ' % hlp.get_model_summary(model))
    lg.info('... Test accuracy: %f, final loss: %f ' % (test_acc, test_loss))
    lg.info("------------------------------------------------------")
    lg.info('Experiment Finished Date/Time: %s', time.ctime())
    lg.info("------------------------------------------------------")


if __name__ == '__main__':
	main()
