

from keras import backend as K
import keras
import numpy as np
import relighting_model as rm
import glob
import tensorflow as tf
import os, errno
import computeaph as caph
import time
import gc
import cv2
import tensorflowjs as tfjs
import argparse
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)


parser = argparse.ArgumentParser(description='NeuralRTI  fitter')
parser.add_argument('-d', '--data_path', default='RealRTI/item3/train/')
args = parser.parse_args()
def fit_NeuralRTI(data_path):
    #data_path = 'RealRTI/item3/train'
    header_file = np.zeros((1, 4), np.float32)

    model_savedir = data_path + '/model_files/'
    dmodel = os.path.join(model_savedir, 'decoder.hdf5')

    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)
    f = open(data_path + '/' 'dirs.lp')
    data = f.read()
    f.close
    linesn = data.split('\n')
    numLight = int(linesn[0])
    lines = linesn[1:numLight + 1]

    L = np.zeros((numLight, 3), np.float32)
    # since the light direction file can be comma, tab separated i used the line below to identify it
    if (len(lines[1].split(' ')) == 4):
        sep = ' '
    else:
        sep = '\t'
    #### read light directions
    for i, l in enumerate(lines):
        s = l.split(sep)
        if len(s) == 4:
            L[i, 0] = float(s[1])
            L[i, 1] = float(s[2])
            L[i, 2] = float(s[3])
        elif len(s) == 3:
            L[i, 0] = float(s[0])
            L[i, 1] = float(s[1])
            L[i, 2] = float(s[2])

    te = os.listdir(data_path)[5]
    imgtpype = te[te.rfind("."):]
    if imgtpype == '.lp':
        te = os.listdir(data_path)[1]
        imgtpype = te[te.rfind("."):]

    start_time = time.time()

    filenames = sorted(glob.glob(data_path + '/*' + imgtpype))  # training images file name
    image1 = cv2.imread(filenames[0])
    nShape = np.shape(image1)
    height = nShape[0]
    width = nShape[1]
    # cv2_img=cv2.imread(filenames[len(filenames)-1],1)
    # # cv2.namedWindow("input image")
    # #
    # cv2.imshow('input image',cv2_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gtpixvals, Inpixvals, Inldirs = caph.compute_aph(filenames, 1, 0, L)
    l = len(filenames)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # 原版 sess = tf.Session(config=config)
    sess = tf.compat.v1.Session(config=config)

    # config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    # # # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1
    # sess = tf.Session(config=config)
    # K.set_session(sess)
    tf.compat.v1.keras.backend.set_session(sess)
    encoder, decoder, model = rm.relight_model(l * 3, 9)

    callbck1 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                 mode='auto',
                                                 min_delta=0.0001, cooldown=0, min_lr=0)
    callbck2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1,
                                             mode='auto',
                                             baseline=None, restore_best_weights=True)
    callbck3 = keras.callbacks.TerminateOnNaN()
    #print(model.summary())

    keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    sgd = keras.optimizers.SGD(lr=0.05)
    model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
    history = model.fit([Inpixvals, Inldirs], gtpixvals, batch_size=64, epochs=30, verbose=1, shuffle=True,
                        validation_split=0.1, callbacks=[callbck1, callbck2, callbck3])

    _, ftimg, _ = caph.compute_aph(filenames, 1, 1)  # groundtruthimage, fitted images, relighting light dirn
    encoded = encoder.predict(ftimg)  # encode

    minv = np.amin(encoded)
    maxv = np.amax(encoded)
    header_file[0, 0] = minv
    header_file[0, 1] = maxv
    header_file[0, 2] = height
    header_file[0, 3] = width

    nv = (255 * (encoded - minv)) / (maxv - minv)
    nv = nv.astype('uint8')
    np.save(model_savedir + '/encoded', nv)
    np.save(model_savedir + '/header', header_file)
    # ov = ((nv * (maxv - minv)) / 255) + minv
    tfjs.converters.save_keras_model(decoder, model_savedir+'/json')
    decoder.save(dmodel)
    del gtpixvals, Inpixvals, Inldirs
    gc.collect()
    print('done!')
    print("--- %s seconds ---" % (time.time() - start_time))

#####################################################
if  __name__=='__main__':
    fit_NeuralRTI(args.data_path)


