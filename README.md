# NeuralRTI

Neural Reflectance Transformation Imaging, CGI 2020
Tinsae Gebrechristos Dulecha,  Filippo Andrea Fanni, Federico Ponchio, Fabio Pellacini and Andrea Giachetti.

<iframe width="600"  src="https://www.youtube.com/embed/Try-izE6nvk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Getting Started

This is a Keras implementation of a NeuralRTI, a pixel based encoding and relighting of RTI data.


### Prerequisites

- Python3.5+
- Keras2.0+
- numpy
- OpenCV(cv2,used for image I/O)
- glob (used for reading out a list of images)

Tested on:
- Ubuntu 16.04/17.10/18.04, Python 3.5.2, Keras 2.3.1, Tensorflow(-gpu) 2.1.0

**************** Running training *********************************************** 

For training NeuralRTI (on one of our benchmark dataset), please download Synthetic dataset from [SynthRTI](https://github.com/Univr-RTI/SynthRTI), and Real datset from [RealRTI](https://github.com/Univr-RTI/RealRTI) and extract it anywhere. 

Then run the following script:

python train.py --data_path [data-path]

**************************** example ***************************************************

python train.py --data_path exampledataset

******* You can find the output (the encoded npy file, header info(min,max, height and width of image), decoder model and decoder model converted into json file) in exampledataset/model-files



************** Testing, relighting from different light directions **********************************

python test.py --model_files [data-path]/model-files --light_dir [path to light directions]

******************************* example ***********************************************************

python test.py --model_files exampledataset/model_files --light_dir exampledataset/test_lightdirs

***********  How to train NeuralRTI on other datasets?

If you want to run this code on other dataset, please first arrange your dataset in the same manner of our example dataset. The required files are:
- Images (any format)
- lights (named dirs.lp)
please don't forget to arrange the light direction file. 


This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

******** Acknowledgments *************************

This work was supported by the DSURF (PRIN 2015) project funded by the Italian Ministry of University and Research and by the MIUR Excellence Departments 2018-2022.



