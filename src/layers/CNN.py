from .ConvolutionLayer import ConvolutionLayer
from .FullyConnected import FullyConnected
from .MaxPooling import MaxPooling
from .util.ActivationFunctions import ActivationFunction
import numpy as np

class CNN:
    def __init__(self):
        self.layers      = []
        self.activations = []
        self.dummyOutput = np.arange(10)
        self.lr          = 0.1

    def addLayer(self, layer):
        self.layers.append(layer)
        
    def forward(self, _input, _verbose=False):
        self.activations = [_input]
        currecntActivaiton = _input

        for i, layer in enumerate(self.layers):
            

            if isinstance(layer, ConvolutionLayer):
                z = layer.feedforward(currecntActivaiton)
                a = ActivationFunction.ReLU(z)

            elif isinstance(layer, MaxPooling):
                a = layer.poolMax(currecntActivaiton) 
            elif isinstance(layer, FullyConnected):
                z = layer.feedforward(currecntActivaiton)
                a = ActivationFunction.softmax(z)
                
                # ========= TEMP DEBUG ========= #
                if np.max(a) > 0.9 or np.min(a) < 1e-10 :
                    print("💣 Extreme Softmax Value!")
            else:
                pass


            self.activations.append(a)
            currecntActivaiton=a

            if _verbose:
                self._debugForward(
                    _layerINDX         =   i, 
                    _layerType         =   type(layer).__name__, 
                    _inputShape        =   self.activations[-2].shape, 
                    _outputShape       =   z.shape, 
                    _activatedOutShape =   self.activations[-1].shape
                )
                
    def backprop(self, _output, _SGD=False, _verbose=False):
        fcLayer: FullyConnected = self.layers[-1]
        deltaPool = fcLayer.backprop(self.activations[-2], _output, self.lr, _SGD)

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            layerInput = self.activations[i]  # Input to this layer
            
            # if _verbose:
            #     self._debugBackprop(
            #         _index      =i,
            #         _layerName  =type(layer).__name__,
            #         _inputShape =layerInput.shape
            #         )

            if isinstance(layer, ConvolutionLayer):
                # Reshape delta to match this layer's output shape
                deltaReshaped = deltaPool.reshape(layer.mOutputShape)
                deltaPool = layer.backprop(layerInput, deltaReshaped, self.lr, _SGD)
            elif isinstance(layer, MaxPooling):
                deltaPool = layer.unpool(deltaPool)
            
            if _verbose:
                gradNorm = np.linalg.norm(deltaPool)
                print(f"Gradient norm: {gradNorm:.6f}")
                if gradNorm < 1e-8:  # Vanishing gradients
                    print("⚠️  Vanishing gradients detected!")

    def calculateLoss(self, _ypred, _ytrue):
        return np.mean((_ypred.flatten() - _ytrue) ** 2)
    
    def crossEntropy(self, _yPred, _yTrue):
        _yPred = np.clip(_yPred.flatten(), 1e-10, 1 - 1e-10)
        _yTrue = _yTrue.flatten()
        return -np.sum(_yTrue * np.log(_yPred))  # add eps for stability

    
    def compute_accuracy(self, _ypred, _ytrue):
        predictions = np.argmax(_ypred, axis=1)
        labels = np.argmax(_ytrue, axis=1)
        return np.mean(predictions==labels)
    
    def trainSGD(self, _epochs, _batchSize, _learningRate, _xtrain, _ytrain, _verbose=False):
        self.lr = _learningRate
        numberOfSamples = len(_xtrain)

        indicies = np.arange(numberOfSamples)


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
                    self.forward(xtrainShuffled[j])
                    batchCorrect += self.compute_accuracy(self.activations[-1].reshape(1,-1), ytrainShuffled[j].reshape(1,-1))
                    batchLoss += self.crossEntropy(self.activations[-1], ytrainShuffled[j]) #loss
                    self.backprop(ytrainShuffled[j].reshape(-1,1), _SGD=True)
                
                for layer in self.layers:
                    if hasattr(layer, 'updateParam'):
                        layer.updateParam(_learningRate, batchSize)
            
                epochLoss += batchLoss / batchSize
                correctPrediction += batchCorrect
            
            accuracy = correctPrediction / numberOfSamples
            avgLoss = epochLoss / (numberOfSamples // _batchSize)
                    
            print("epoch {}: loss={}, accuracy={:.5%}".format(epoch, avgLoss,accuracy))

    def _debugForward(self, _layerINDX, _layerType, _inputShape, _outputShape, _activatedOutShape):
        print("="*4 + " " + "layer #{}".format(_layerINDX) + " " + "="*4)
        print("layer type:                  {}".format(_layerType))
        print("input Shape type:            {}".format(_inputShape))
        print("output Shape type:           {}".format(_outputShape))
        print("activated Output Shape type: {}".format(_activatedOutShape))
        print("")

    def _debugBackprop(self, _index, _layerName, _inputShape):
        print("")
        print("="*5 + " " + "layer #" + str(_index) + " " + "="*5)
        print("layer type:              {}".format(_layerName))
        print("layer's input dimention: {}".format(_inputShape))
        print("")