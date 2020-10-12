import numpy as np
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import time
import logging as lg
import model_helpers as hlp

def create_mlp_model(inpsk, input_len, hidden_units, outputs):
    """ Build three hidden layered network for depth modality """

    inps = layers.Dense(hidden_units, input_dim=1024, activation='relu')(inpsk)
    x = layers.Dense(hidden_units, activation='relu')(inps)
    x = layers.Dense(hidden_units, activation='relu')(x)
    x = layers.Dense(hidden_units, activation='relu')(x)

    return x


def create_model(inputs, size):
    """ Build a covnet model for color modality. 
        TODO: check for pretrained networks, see pyimagesearch post
        title with 3-ways-to-create-a-keras-model-
    """
    conv1_resense = layers.Conv2D(32, (3, 3), activation='relu')
    x = conv1_resense(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    return x


def main():

    desc = "Intermediate fusion for object recognition"
    fname = 'runtime_logs/fusion_intermediate.log'  # 0-change this line for different sensor

    init = time.time()
    lg.basicConfig(filename=fname, format='%(message)s', level=lg.INFO)
    lg.info('Experiment started date/time: %s', time.ctime())

    # 1-change vishpath
    vis_path, path_h5s = 'visualization/fusion_intermediate.png', 'modelh5s/'
    Xtr, Xval, Xtst = hlp.load_rsense_inputs() # 2-change sensory input
    Xtr_il, Xval_il, Xtst_il = hlp.load_ileft_inputs()
    Xtr_ir, Xval_ir, Xtst_ir = hlp.load_iright_inputs()
    Xtr_d, Xval_d, Xtst_d = hlp.load_rdepth_inputs()

    y_train, y_val, y_tst = hlp.load_outputs()
    nof_objects, size = 100, (32,32,1)
    nof_epochs, no_change_eps = 100, 30
    train_size = 3600
    hidden_units, size_d = 128 ,1024

    # icub left camera
    Xtr_il = hlp.reshape_input_vect(Xtr_il, size)
    Xval_il = hlp.reshape_input_vect(Xval_il, size)
    Xtst_il = hlp.reshape_input_vect(Xtst_il, size)

    # icub right camera
    Xtr_ir = hlp.reshape_input_vect(Xtr_ir, size)
    Xval_ir = hlp.reshape_input_vect(Xval_ir, size)
    Xtst_ir = hlp.reshape_input_vect(Xtst_ir, size)

    # realsense color camera
    Xtr = hlp.reshape_input_vect(Xtr, size)
    Xval = hlp.reshape_input_vect(Xval, size)
    Xtst = hlp.reshape_input_vect(Xtst, size)

    lg.info("I/O shapes----------------------------------------------")
    lg.info("Input train, validation, test: %s, %s, %s ", str(Xtr.shape), str(Xval.shape), str(Xtst.shape))
    lg.info("Output train, validation, test: %s, %s, %s ", str(y_train.shape), str(y_val.shape), str(y_tst.shape))

    Xtr, Xval, Xtst = hlp.normalize_inputs(Xtr, Xval, Xtst )
    Xtr_il, Xval_il, Xtst_il = hlp.normalize_inputs(Xtr_il, Xval_il, Xtst_il)
    Xtr_ir, Xval_ir, Xtst_ir = hlp.normalize_inputs(Xtr_ir, Xval_ir, Xtst_ir)
    Xtr_d, Xval_d, Xtst_d = hlp.normalize_depth_input(Xtr_d, Xval_d, Xtst_d)

    inputs_mm1 = keras.Input(shape=size)
    inputs_mm2 = keras.Input(shape=size)
    inputs_mm3 = keras.Input(shape=size)
    inputs_mm4 = keras.Input(shape=(1024,))

    model_realsense = create_model(inputs_mm1, size_d)
    model_ileft = create_model(inputs_mm2, size)
    model_iright = create_model(inputs_mm3, size)
    model_depth = create_mlp_model(inputs_mm4, 1024, hidden_units, nof_objects)


    # combine the layers
    combined_layers = layers.concatenate([model_realsense, model_ileft, model_iright])
    in_class_joint = layers.Dense(256, activation='relu')(combined_layers)
    combine_color_depth = layers.concatenate([in_class_joint, model_depth])
    combine_color_depth = layers.Flatten()(combine_color_depth)
    combine_color_depth = layers.Dense(nof_objects, activation='softmax')(combine_color_depth)

    mmodel = keras.Model(inputs=[inputs_mm1, inputs_mm2, inputs_mm3, inputs_mm4], outputs=combine_color_depth)
    keras.utils.plot_model(mmodel, 'fusion_intermediate.png')

    mmodel.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    estop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=no_change_eps)
    traces = mmodel.fit([Xtr, Xtr_il, Xtr_ir, Xtr_d], y_train, callbacks=[estop_callback], validation_data=([Xval,Xval_il, Xval_ir, Xval_d], y_val), epochs=nof_epochs)
    trace_acc, trace_loss = traces.history['accuracy'], traces.history['loss']

    test_loss, test_acc = mmodel.evaluate([Xtst, Xtst_il, Xtst_ir,  Xtst_d], y_tst, verbose=2)

    hlp.save_model_params(mmodel, 'modelh5s/fusion_intermediate.h5') # 3- change the name of h5 file
    pred_probs = mmodel.predict([Xtst, Xtst_il, Xtst_ir,  Xtst_d])
    pred_y = pred_probs.argmax(axis=-1)

    hlp.save_results("fusion_intermediate", trace_acc, trace_loss, pred_probs, pred_y) # 4- change the name of result folder

    print("---" * 10)
    print("Test accuracy: {}".format(test_acc))

    # logging info
    hlp.visualize_model(mmodel, vis_path)
    lg.info("Model Summary-----------------------------------------")
    lg.info(' %s ' % hlp.get_model_summary(mmodel))
    lg.info('... Test accuracy: %f, final loss: %f ' % (test_acc, test_loss))
    lg.info("------------------------------------------------------")
    lg.info('Experiment Finished Date/Time: %s', time.ctime())
    lg.info("------------------------------------------------------")


if __name__ == '__main__':
	main()
