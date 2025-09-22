import numpy as np

def onHotEncode(lables, numClasses):
    onHot = np.zeros((len(lables), numClasses))
    
    for i, label in enumerate(lables):
        onHot[i, label] = 1
    return onHot


# TODO: Make a separate class that deals with different way to prep data and channels
def dataPrep_MNISTdigitClassification(X, Y, numClasses):
    
    # normalize data -> (0, 1)
    X = X.astype('float32')
    X = X(X / 255.0 - 0.1307) / 0.3081
    
    X = X[:, np.newaxis, :, :] # (nData: smaples, nChannels: 1, Width: 28, Height: 28)

    yOneHot = onHotEncode(Y, numClasses)

    return X, yOneHot
