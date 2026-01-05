# To run the script, open a terminal in the root directory and run:
#   >>python -m src.exmaples.trainYourFirstModel


from tensorflow import keras
from src.CNN.util import dataPrep
from src.CNN.CNN import CNN
from src.CNN.layers.ConvolutionLayer import ConvolutionLayer
from src.CNN.layers.FullyConnected import FullyConnected
from src.CNN.layers.MaxPooling import MaxPooling
from src.CNN.modules.loss import crossEntropyLoss
import numpy as np


# ======== Set up Environment ======== #
# for convinience, its better to use a notebook
# this file is devided into 4 major parts:
#       1) preparing data to train
#       2) setiing up network layer
#       3) traing the network
#       4) saving you model



# ======== preparing data to train ======== #
# first you need to load your training data. In MNIST hand-written dataset we trust!
# for the sake simplisity, we use 'tensorflow.keras' to load the data for us. However
# feel free to load any data set, in any way you want!
#       installation: pip install 

# To feed the data to our code, we need to normalize the data (0.0 to 1.0) and use on-hot-encoding
# inside 'src/layers/util' directory, you can find dataPrep.py which is responsible for data prepration
# use dataPrep_MNISTdigitClassification to do just that
# pass in your training data and their lables followd by lable_class
# In case of MNIST, it would be 10 (since we have 0-9 digits)

# start with small samples for proof of concept
# !! max smaples for mnist is 60000 !!
SAMPLES_RANGE = 1000

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()

X, Y = dataPrep.dataPrep_MNISTdigitClassification( X_TRAIN[:SAMPLES_RANGE],
                                                   Y_TRAIN[:SAMPLES_RANGE],
                                                   10
                                                   )

# ======== Setting up network layers ======== #
# Network is extremly sensitive to intitial values that results to unstable accuracy \
# and that can also  result in loss > 2.6
# to remedy this, either use epoch > 40 or setup the network using large layers
# here we are using a (16, 32, 16) network with one pooling layer of size 2

# NOTE::
# Currently the app supports _poolSize = 1 and can handle only one pooling layer
# This will be fixed in later versions
l1 = ConvolutionLayer(
    _kernelSize       =5,
    _stride           =1,
    _numberOfKernels  =16
)

p1 = MaxPooling(
    _poolSize = 2
)
    
l2 = ConvolutionLayer(
    _kernelSize       =3,
    _stride           =1,
    _numberOfKernels  =32
)

l3 = ConvolutionLayer(
    _kernelSize       =5,
    _stride           =1,
    _numberOfKernels  =16
)

lfc = FullyConnected(
    _numberOfOutputNodes = 10
)

cnn = CNN()
cnn.addLayer(l1)
cnn.addLayer(p1)
cnn.addLayer(l2)
cnn.addLayer(l3)
cnn.addLayer(lfc)


# ======== traing the network ======== #
EPOCH = 6
BATCH_SIZE = 128
LR = 0.04

def trainSGD(_epochs, _batchSize, _learningRate, _xtrain, _ytrain):
        numberOfSamples = len(_xtrain)

        indicies = np.arange(numberOfSamples)

        criterian = crossEntropyLoss(reduction=None)

        for epoch in range(_epochs):
            epochLoss = 0
            correctPrediction = 0

            np.random.shuffle(indicies)
            xtrainShuffled = _xtrain[indicies]
            ytrainShuffled = _ytrain[indicies]

            for i in range(0, numberOfSamples, _batchSize):
                batchLoss = 0
                batchCorrect = 0
                batchSize = min(_batchSize, numberOfSamples - i)

                for j in range(i, i+ batchSize):
                    cnn.forward(xtrainShuffled[j])
                    batchCorrect += cnn.compute_accuracy(cnn.activations[-1].reshape(1,-1), ytrainShuffled[j].reshape(1,-1))
                    batchLoss += criterian(cnn.activations[-1], ytrainShuffled[j]) #loss
                    cnn.backprop(ytrainShuffled[j].reshape(-1,1), _SGD=True)
                
                for layer in cnn.layers:
                    if hasattr(layer, 'updateParam'):
                        layer.updateParam(_learningRate, batchSize)
            
                epochLoss += batchLoss / batchSize
                correctPrediction += batchCorrect
            
            accuracy = correctPrediction / numberOfSamples
            avgLoss = epochLoss / (numberOfSamples // _batchSize)
                    
            print("epoch {}: loss={}, accuracy={:.5%}".format(epoch, avgLoss,accuracy))

trainSGD(EPOCH, BATCH_SIZE, LR, X, Y)

# If done correctly, you will get an output like this:
# NOTE: ignore any "Extreme Softmax Value". It will be fixed in later patches \
# And any error will be added to a log file to keep the output terminal clean

# Lazy Initialize Initiated:
# Max Pooling lazyly initialized
# Lazy Initialize Initiated:
# Lazy Initialize Initiated:
# epoch 0: loss=2.4447028612993127, accuracy=25.90000%
# epoch 1: loss=1.7246047708146033, accuracy=54.10000%
# epoch 2: loss=1.3437532449051361, accuracy=65.90000%
# epoch 3: loss=0.9507033919057104, accuracy=78.30000%
# epoch 4: loss=0.8946962667044405, accuracy=78.30000%
# epoch 5: loss=0.7091818574675656, accuracy=84.20000%

# Congrats model is trained
# if you are happy with the accuracy, move on to the next section


# ======== saving your model ======== #

# models are stored in '.npz' format
# its an uncompressed format to store numpy arrays. read more
#                               https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
# you can call CNN.saveModel(), passing in the file name
# the model will be save in the same directory, where the srcipt was ran from
# So if you open a shell in the root directory, you can find your model in
#    ./CNN-FROM-SCRATCH/

MODEL_NAME = "LookMyFirstModel"
# cnn.saveModel(MODEL_NAME)
