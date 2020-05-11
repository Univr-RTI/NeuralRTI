
import gc
import cv2
import numpy as np
def compute_aph(filename, scale, ts, L = None):
    image_path = filename[3]
    cv2_im = cv2.imread(image_path, 1)
    nShape = np.shape(cv2_im)
    height = nShape[0]
    width = nShape[1]
    if cv2_im.dtype == 'uint16':
        divider = 65535
    else:
        divider = 255

    ########################################################
    numLight=len(filename)
    imgcont = np.zeros((numLight, height * width * 3),np.float32)
    for i, filename in enumerate(sorted(filename)):
        cv2_im = cv2.imread(filename, 1) / divider
        cv2_im = cv2_im.astype('float16')
        img = cv2_im.ravel()
        imgcont[i, :] = img
    del filename, img, cv2_im
    if ts==0: # prepare appearance profile for training
        h = height * width * numLight
        allimgcont = np.zeros((h, (numLight * 3)), np.float16)
        gtimages = np.zeros((h, 3), np.float16)
        ldcont = np.zeros((h, 2), np.float16)

        for j in range(0, numLight):
            I = np.delete(imgcont, j, 0)
            rlimg = imgcont[j, :]
            rlimg = np.reshape(rlimg, (height * width, 3))
            Iv = np.reshape(imgcont, (numLight, height * width, 3))
            Iv = np.transpose(Iv, (1, 0, 2))
            Iv = np.reshape(Iv, (height * width, numLight* 3))
            allimgcont[(height * width * j):(height*width*(j+1)), :] = Iv
            ld = np.transpose(L[j, 0:2])
            lt = np.tile(ld, (height * width, 1))
            lt = np.reshape(lt, (height * width, 2))
            gtimages[(height * width * j):(height*width*(j+1)),:]=rlimg;
            ldcont[(height * width * j):(height*width*(j+1)),:]=lt;
            del Iv, I, rlimg, lt,ld
            gc.collect()
        del imgcont
        gc.collect()

    else:  # prepare appearance profile for testing
        Iv = np.reshape(imgcont, (numLight, height * width, 3))
        Iv = np.transpose(Iv, (1, 0, 2))
        Iv = np.reshape(Iv, (height * width, numLight * 3))
        allimgcont=Iv
        ldcont=0
        gtimages=0
    return gtimages, allimgcont, ldcont
