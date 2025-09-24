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
    def whatThisPic(self, _imArr):
        '''
            classifies the number in the photo
            _imArr: flattened and nomrmalized image
        '''
        currecntActivaiton = _imArr

        for layer in self.layers:
            if isinstance(layer, ConvolutionLayer):
                z = layer.feedforward(currecntActivaiton)
                a = ActivationFunction.ReLU(z)

            elif isinstance(layer, MaxPooling):
                a = layer.poolMax(currecntActivaiton) 
            elif isinstance(layer, FullyConnected):
                z = layer.feedforward(currecntActivaiton)
                a = ActivationFunction.softmax(z)
            
            currecntActivaiton = a
        
        return np.argmax(a)
    
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

    def saveModel(self, filename):
        d = {}
        convCounter = 1
        poolingCounter = 1

        import sys
        if 'ConvolutionLayer' not in sys.modules(): from .ConvolutionLayer import ConvolutionLayer
        if 'FullyConnected'   not in sys.modules(): from .FullyConnected import FullyConnected
        if 'MaxPooling'       not in sys.modules(): from .MaxPooling import MaxPooling


        for layer in self.layers:
            if isinstance(layer, ConvolutionLayer):
                d[f"conv{convCounter}_config"]  = np.array([layer.mFilterCount, layer.mKernelSize, layer.mStride])
                d[f"conv{convCounter}_weights"] = layer.mWeights
                d[f"conv{convCounter}_biases"]  = layer.mBiases
                convCounter += 1

            if isinstance(layer, MaxPooling): 
                d[f"pooling{poolingCounter}_config"] = np.array([layer.mPoolSize])
                poolingCounter += 1
                
            if isinstance(layer, FullyConnected):
                d["fc_config"]  = np.array([layer.mNout])
                d["fc_weights"] = layer.mWeights
                d["fc_biases"]  = layer.mBiases

        np.savez(filename,
                 architecture = np.array([type(x).__name__ for x in self.layers]),
                 **d,
                 allow_pickle=True
                 )
        
    @classmethod
    def loadModel(cls, filename):
        # TODO #1: check if file exist
        # TODO #2: hadle if file couldnt be loaded
        model = np.load(filename)
        arcitecture = model["architecture"]

        classInstance = cls()

        # Constrcut layers based on arcitecture
        convCounter = 1
        poolingCounter = 1
        
        for layer in arcitecture:
            match layer:
                case "ConvolutionLayer":
                    config = model[f"conv{convCounter}_config"]
                    convLayer = ConvolutionLayer(
                        _kernelSize      = config[1],
                        _stride          = config[2],
                        _numberOfKernels = config[0],
                        _weights         = model[f"conv{convCounter}_weights"],
                        _Biases          = model[f"conv{convCounter}_biases"] 
                    )
                    classInstance.addLayer(convLayer)
                    convCounter += 1
                case "MaxPooling":
                    config = model[f"pooling{poolingCounter}_config"]
                    poolingLayer = MaxPooling(
                        _poolSize = config[0]
                    )
                    classInstance.addLayer(poolingLayer)
                    poolingCounter += 1
                case "FullyConnected":
                    config = model["fc_config"]
                    fcLayer = FullyConnected(
                        _numberOfOutputNodes = config[0],
                        _weights             = model["fc_weights"],
                        _biases              = model["fc_biases"]
                    )
                    classInstance.addLayer(fcLayer)
        return classInstance

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