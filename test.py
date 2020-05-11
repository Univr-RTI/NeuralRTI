
from keras.models import load_model
import cv2
import numpy as np
import keras
import os
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser(description='NeuralRTI  fitter')
parser.add_argument('-mod_files', '--model_files', default='RealRTI/item3/train/models_files')
parser.add_argument('-ld', '--light_dir')


args = parser.parse_args()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess = tf.compat.v1.Session(config=config)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
tf.compat.v1.keras.backend.set_session(sess)

def test(model_files, light_dir):
    if not os.path.exists('relighted'):
        os.makedirs('relighted')
    f = open(light_dir + '/dirs.lp')
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
    for i, l in enumerate(lines):
        s = l.split(sep)

        if len(s) == 4:
            L[i, 0] = float(s[1])
            L[i, 1] = float(s[2])
            L[i, 2] = float(s[3])

    encoded = np.load(model_files + '/encoded.npy')
    headerinfo=np.load(model_files + '/header.npy')
    minv = headerinfo[0,0]
    maxv = headerinfo[0,1]
    height= np.int32(headerinfo[0,2])
    width= np.int32(headerinfo[0,3])
    ov = ((encoded * (maxv - minv)) / 255) + minv
    decoder = load_model((model_files + '/decoder.hdf5'))  # load decoder model
    for dt in range(0, numLight):
        ld = np.transpose(L[dt, 0:2])

        lt = np.tile(ld, (height * width, 1))
        lt = np.reshape(lt, (height * width, 2))
        outputs = decoder.predict_on_batch([ov, lt])
        outputs = outputs.clip(min=0, max=1)
        outputs = np.reshape(outputs, (height, width, 3))
        outputs *= 255
        outputs = outputs.astype('uint8')
        cv2.imwrite('relighted/relighted' + str(dt).zfill(2) + '.png', outputs)
    print('done!'+ "\n")
    print('The relighted image saved in ', os.getcwd() + '/Relighted' + "\n")

if  __name__=='__main__':
    test(args.model_files, args.light_dir)


















#
#
#
# for d in range(11,12):
#     if isinstance(datasetcont[0], list):
#         datasets = datasetcont[d][0]
#         length=1
#     else:
#         datasets = os.listdir(datasetcont[d])
#         length=len(datasets)
#     for dataset in range(4, 5):
#         if isinstance(datasetcont[0], list):
#             object = datasetcont[d][0]
#             data_path = 'RealRTI/' + object + '/train'
#             data_patht = 'RealRTI/' + object + '/test/'
#             encmodels = 'RealRTI/' + object + '/models_9coeffs/encoder'
#             decmodels = 'RealRTI/' + object + '/models_9coeffs/decoder'
#             encmodels = sorted(glob.glob(encmodels + '/*.hdf5'))
#             decmodels = sorted(glob.glob(decmodels + '/*.hdf5'))
#         else:
#             object = datasets[dataset]
#             data_path = datasetcont[d] + '/' + object + '/train'
#             data_patht = datasetcont[d] + '/' + object + '/test/'
#             encmodels = datasetcont[d] + '/' + object + '/models_9coeffs/encoder'
#             decmodels = datasetcont[d] + '/' + object + '/models_9coeffs/decoder'
#             encmodels = sorted(glob.glob(encmodels + '/*.hdf5'))
#             decmodels = sorted(glob.glob(decmodels + '/*.hdf5'))
#
#         if not os.path.exists(data_patht + 'relighted1b_4l'):
#             os.makedirs(data_patht + 'relighted1b_4l')
#         ################## load test relighting direction
#
#         te = os.listdir(data_path)[0]
#         imgtpype = te[te.rfind("."):]
#         if imgtpype == '.lp':
#             te = os.listdir(data_path)[1]
#             imgtpype = te[te.rfind("."):]
#
#         ################################################################################################################
#         if 'synth' in object:  # synthetic dataset
#             filenames = sorted(glob.glob(data_path + '/*' + imgtpype))  # training images file name
#             l = len(filenames)
#             image1 = cv2.imread(filenames[0])
#             nShape = np.shape(image1)
#             height = nShape[0]
#             width = nShape[1]
#             scale = 1
#             ts = 1
#             f = open(data_patht + '/' 'dirs.lp')
#             data = f.read()
#             f.close
#             linesn = data.split('\n')
#             numLight = int(linesn[0])  # the last line is empty (how to fix it?)
#             lines = linesn[1:numLight + 1]
#
#             L = np.zeros((numLight, 3), np.float32)
#             # since the light direction file can be comma, tab separated i used the line below to identify it
#             if (len(lines[1].split(' ')) == 4):
#                 sep = ' '
#             else:
#                 sep = '\t'
#
#             for i, l in enumerate(lines):
#                 s = l.split(sep)
#
#                 if len(s) == 4:
#                     L[i, 0] = float(s[1])
#                     L[i, 1] = float(s[2])
#                     L[i, 2] = float(s[3])
#
#             _, ftimg, _ = caph.compute_aph(filenames, scale,
#                                            ts)  # groundtruthimage, fitted images, relighting light dirn
#             keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#
#             encoder = load_model(encmodels[0])  # load encoder model
#             # encoder.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error')
#             # print(encoder.summary())
#             encoded = encoder.predict(ftimg)  # encode
#
#             # coeffl=np.shape(encoded)[1]
#             # for coefind in range(0,coeffl):
#             #     coeffd=encoded[:,coefind]
#             #     coeffimg=np.reshape(coeffd,(height,width))
#             #     cv2.imshow('coeff',coeffimg)
#             #     cv2.waitKey(0)
#             #np.save(data_patht + 'rawencoding', encoded)
#             mnmx=np.zeros((1,2), np.float32)
#             minv = np.amin(encoded)
#             maxv = np.amax(encoded)
#             mnmx[0,0]=minv
#             mnmx[0,1]=maxv
#             nv = (255 * (encoded - minv)) / (maxv - minv)
#             nv = nv.astype('uint8')
#             nv1=np.reshape(nv,(height,width,9))
#
#             np.save(data_patht + '/encoded', nv)
#             np.save(data_patht +'minmax',mnmx)
#             ov = ((nv * (maxv - minv)) / 255) + minv
#
#             # ###########  check saved data ###############
#             # chkload=np.load(data_patht + '/encoded' + '.npy')
#             #
#             # coeffl=np.shape(chkload)[1]
#             # for coefind in range(0,coeffl):
#             #     coeffd=chkload[:,coefind]
#             #     coeffimg=np.reshape(coeffd,(height,width))
#             #     cv2.imshow('coeff',coeffimg)
#             #     cv2.waitKey(0)
#
#
#             # encoded = encoded.astype
#             decoder = load_model(decmodels[0])  # load decoder model
#             print(decoder.summary())
#
#             if image1.dtype == 'uint16':
#                 divider = 65535
#                 img_type = 'uint16'
#             else:
#                 divider = 255
#                 img_type = 'uint8'
#
#             for dt in range(0, numLight):
#                 start_time = time.time()
#                 ld = np.transpose(L[dt, 0:2])
#                 lt = np.tile(ld, (height * width, 1))
#                 lt = np.reshape(lt, (height * width, 2))
#                 outputs = decoder.predict_on_batch([ov, lt])
#                 outputs = outputs.clip(min=0, max=1)
#
#                 outputs = np.reshape(outputs, (height, width, 3))
#                 outputs *= divider
#                 outputs = outputs.astype(img_type)
#                 # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
#                 # cv2.resizeWindow('test', width*2, height*2)
#                 # cv2.imshow('test', outputs)
#                 # cv2.waitKey(0)
#                 print("--- %s seconds ---" % (time.time() - start_time))
#                 cv2.imwrite(data_patht + 'relighted1b_4l/' + 'relighted' + str(dt).zfill(2) + imgtpype,
#                             outputs)
#
#         else:  # real datasets
#
#             ####################
#             f = open(data_path + '/' 'dirs.lp')
#             data = f.read()
#             f.close
#             linesn = data.split('\n')
#             numLight = int(linesn[0])
#             lines = linesn[1:numLight + 1]
#
#             L = np.zeros((numLight, 3), np.float32)
#             lcheck = lines[1]
#             # since the light direction file can be comma, tab separated i used the line below to identify it
#             if (len(lcheck.split(' ')) == 4):
#                 sep = ' '
#             else:
#                 sep = '\t'
#
#             for i, l in enumerate(lines):
#                 s = l.split(sep)
#
#                 if len(s) == 4:
#                     L[i, 0] = float(s[1])
#                     L[i, 1] = float(s[2])
#                     L[i, 2] = float(s[3])
#
#             imageindex = datasetcont[d][1:len(datasetcont[d])]  # image number used for leave out one test
#             for dt in range(0, len(imageindex)):
#                 index = imageindex[dt]
#                 filenames = sorted(glob.glob(data_path + '/*' + imgtpype))  # training images file name
#                 filenames = np.delete(filenames, index, 0)
#                 l = len(filenames)
#                 image1 = cv2.imread(filenames[0])
#                 nShape = np.shape(image1)
#
#                 scale = 1
#                 ts = 1
#                 _, ftimg, _ = caph.compute_aph(filenames, scale,
#                                                ts)  # groundtruthimage, fitted images, relighting light dirn
#                 keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#
#                 encoder = load_model(encmodels[dt])  # load encoder model
#                 #print(encoder.summary())
#                 encoded = encoder.predict(ftimg)  # encode
#                 mnmx = np.zeros((1, 2), np.float32)
#                 minv = np.amin(encoded)
#                 maxv = np.amax(encoded)
#                 mnmx[0, 0] = minv
#                 mnmx[0, 1] = maxv
#                 nv = (255 * (encoded - minv)) / (maxv - minv)
#                 nv = nv.astype('uint8')
#                 np.save(data_patht + '/encoded' + str(index), nv)
#                 np.save(data_patht + 'minmax' + str(index), mnmx)
#                 ov = ((nv * (maxv - minv)) / 255) + minv
#                 decoder = load_model(decmodels[dt])  # load decoder model
#                 if image1.dtype == 'uint16':
#                     divider = 65535
#                     img_type = 'uint16'
#                 else:
#                     divider = 255
#                     img_type = 'uint8'
#                 nShape = np.shape(image1)
#                 height = nShape[0]
#                 width = nShape[1]
#
#                 ld = np.transpose(L[index, 0:2])
#                 lt = np.tile(ld, (height * width, 1))
#                 lt = np.reshape(lt, (height * width, 2))
#                 start_time = time.time()
#                 outputs = decoder.predict_on_batch([ov, lt])
#                 print("--- %s seconds ---" % (time.time() - start_time))
#
#                 outputs = outputs.clip(min=0, max=1)
#
#                 outputs = np.reshape(outputs, (height, width, 3))
#                 outputs *= divider
#                 outputs = outputs.astype(img_type)
#                 # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
#                 # cv2.resizeWindow('test', width*2, height*2)
#                 # cv2.imshow('test', outputs)
#                 # cv2.waitKey(100)
#
#                 cv2.imwrite(data_patht + 'relighted1b_4l/' + 'relighted' + str(index).zfill(2) + imgtpype, outputs)
#
#         ############################################################
