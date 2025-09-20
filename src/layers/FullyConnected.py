import numpy as np
from .util.ActivationFunctions import ActivationFunction

class FullyConnected:

    def __init__(self, _numberOfOutputNodes, _verbose=False):
        # expects the input to be flattened -> temp: flatten data in the method for now
        # #TODO: data format validation
        self.mNin = None
        self.mNout  = _numberOfOutputNodes
        self.mBiases = np.zeros((_numberOfOutputNodes, 1))
        self.mWeights = None
        self.nablaweightsAcc = None
        self.nablaBiasesAcc  = None
        self.mIsLazyInit = False
        if _verbose:
            self._debugConstructor()

    def costDerevitive(self, _activation, _y):
        return _activation - _y
    
    def feedforward(self, _input):
        _input = _input.flatten()

        if (self.mIsLazyInit == False):
            self.mNin = _input.shape[0]
            self.mWeights = np.random.randn(self.mNout, self.mNin) * np.sqrt(2.0 / self.mNin) # Xavier initialization
            self.mIsLazyInit = True
        z = np.dot(self.mWeights, _input.reshape(-1,1)) + self.mBiases 
        return z
    
    def backprop(self, _in, _y, _learningRate, _SGD=False, _verbose=False):
        
        # expects flattened data -> temp: flatten the data in the method for now
        _in = _in.flatten()
        # TODO: shape validation

        _in = _in.reshape(-1,1)
        _y  = _y.reshape(-1,1)

        preactivation = np.dot(self.mWeights, _in) + self.mBiases
        activation = ActivationFunction.softmax(preactivation)
        # delta = self.costDerevitive(activation, _y) * ActivationFunction.primeReLU(preactivation) # shape: (n,1) -> for RELU
        delta = activation - _y # -> for Softmax


        # Gradient Clipping
        # delta = np.clip(delta, -1,1)
        
        nablaW = np.dot(delta, _in.T)
        nablaB = np.sum(delta, axis=1, keepdims=True) # what?

        # Weights/Biases Clipping
        # nablaB = np.clip(nablaB, -1,1)
        # nablaW = np.clip(nablaW, -1,1)


        if _SGD:
            if self.nablaweightsAcc is None:
                self.nablaBiasesAcc  = np.zeros_like(nablaB)
                self.nablaweightsAcc = np.zeros_like(nablaW)
            self.nablaweightsAcc += nablaW
            self.nablaBiasesAcc += nablaB
        else:
            self.mWeights -= _learningRate*nablaW
            self.mBiases  -= _learningRate*nablaB


        deltaPool = np.dot(self.mWeights.T,  delta)


        if _verbose:
            self._debugBackprop(
                _nablaWShape   = nablaW.shape,
                _weightsShape  = self.mWeights.shape,
                _nablaBShape   = nablaB.shape,
                _biasesShape   = self.mBiases.shape,
                _deltaShape    = delta.shape
                )
            
        return deltaPool

    def updateParam(self, _learningRate, _batchSize):

        if self.nablaweightsAcc is not None:
            self.mWeights -= _learningRate * (self.nablaweightsAcc/_batchSize)
            self.mBiases  -= _learningRate * (self.nablaBiasesAcc/_batchSize)
            
            # Reset accumulator
            self.nablaweightsAcc = None
            self.nablaBiasesAcc  = None 
            
    def _debugBackprop(self, _nablaWShape, _weightsShape, _nablaBShape, _biasesShape, _deltaShape):

        print("***Back Propagation Debugging For FC Layer***")

        print("   nablaW shape:   {}".format(_nablaWShape))
        print("   weights shape:  {}".format(_weightsShape))
        print("   nablaB shape:   {}".format(_nablaBShape))
        print("   biases shape:   {}".format(_biasesShape))
        print("   delta shape:    {}".format(_deltaShape))

    def _debugConstructor(self):
        print("number of outout Nodes: {}".format(self.mNout))
        print("biases shape: {}".format(self.mBiases.shape))
        print("*"*10)
        print("nubmer of input Nodes: {}".format(self.mNin))
        print("weights shape: {}".format(self.mWeights.shape))
        print("*"*10)
        print("bias tensor: ")
        print(self.mBiases)
        print("*"*10)
        print("weight tensor: ")
        print(self.mWeights)

    