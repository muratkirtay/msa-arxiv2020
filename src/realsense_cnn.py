import numpy as np
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
import logging as lg
import model_helpers as hlp


def unimodalnet(size, nof_objects):
    """ Build Keras Sequential model to perfom ConvNet"""

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=size))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(nof_objects, activation='softmax'))

    return model


def main():

    desc = "Realsense color modality processing for object recognition"
    fname = 'runtime_logs/realsense_cnn.log'  # 0-change this line for different sensor

    init = time.time()
    lg.basicConfig(filename=fname, format='%(message)s', level=lg.INFO)
    lg.info('Experiment started date/time: %s', time.ctime())

    # 1-change vishpath
    vis_path, path_h5s = 'visualization/realsense_modelg.png', 'modelh5s/'
    Xtr, Xval, Xtst = hlp.load_rsense_inputs() # 2-change sensory input
    y_train, y_val, y_tst = hlp.load_outputs()
    nof_objects, size = 100, (32,32,1)
    nof_epochs, no_change_eps = 100, 30

    Xtr = hlp.reshape_input_vect(Xtr, size)
    Xval = hlp.reshape_input_vect(Xval, size)
    Xtst = hlp.reshape_input_vect(Xtst, size)

    lg.info("I/O shapes----------------------------------------------")
    lg.info("Input train, validation, test: %s, %s, %s ", str(Xtr.shape), str(Xval.shape), str(Xtst.shape))
    lg.info("Output train, validation, test: %s, %s, %s ", str(y_train.shape), str(y_val.shape), str(y_tst.shape))

    Xtr, Xval, Xtst = hlp.normalize_inputs(Xtr, Xval, Xtst )

    model = unimodalnet(size, nof_objects)

    # perform convnet
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    estop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=no_change_eps)
    traces = model.fit(Xtr, y_train, callbacks=[estop_callback], validation_data=(Xval, y_val), epochs=nof_epochs)
    trace_acc, trace_loss = traces.history['accuracy'], traces.history['loss']

    test_loss, test_acc = model.evaluate(Xtst, y_tst, verbose=2)

    # save model params and results for postprocessing
    hlp.save_model_params(model, 'modelh5s/realsense_cnn_model.h5') # 3- change the name of h5 file
    pred_probs, pred_y = model.predict(Xtst), pred_probs.argmax(axis=-1)
    
    hlp.save_results("realsense_cnn", trace_acc, trace_loss, pred_probs, pred_y) # 4- change the name of result folder

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
