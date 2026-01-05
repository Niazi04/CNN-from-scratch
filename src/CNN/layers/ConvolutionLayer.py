import numpy as np
from scipy import signal
from math import floor

class ConvolutionLayer:
    def __init__(self, _kernelSize, _stride, _numberOfKernels, _weights=None, _Biases=None):
        '''
            Using Lazy Initialization to construct the class.
            Input related shits like dimention and stuff will automatically be 
            calculated upon feedforward invokation.
            This way, user can add multiple layers without bothering to feed
            the dimentions of previous output to the input of new layer
        '''
        self.mFilterCount      = _numberOfKernels   # number of kernels per convolution layer
        self.mKernelSize       = _kernelSize    
        self.mStride           = _stride
        self.mLazilyIntialized = False
        self.mInitWithWeights  = False

        self.mInputShape       = None  # DxWxH
        self.mKernelDepth      = None 
        self.mOutputDimention  = None  #output diemntion WxH - not including the channels
        
        self.mOutputShape      = None  #output shape DxWxH
        self.zpreactive        = None
        
        if _weights is not None and _Biases is not None:
            self.mBiases           = _Biases
            self.mWeights          = _weights
            self.mInitWithWeights  = True
        else:
            self.mBiases           = None
            self.mWeights          = None

        self.nablaweightsAcc   = None # accumulator for SGD
        self.nablaBiasesAcc    = None # accumulator for SGD

    def feedforward(self, _input):
        self._LazyInit(_input.shape)
        output = np.zeros(self.mOutputShape)

        for filterIDX in range(self.mFilterCount):
            featureMap = np.zeros((self.mOutputDimention, self.mOutputDimention))
            for channelIDX in range(self.mKernelDepth):
                kernel = self.mWeights[filterIDX, channelIDX]
                channelData = _input[channelIDX]
                convolved = signal.convolve2d(channelData, kernel, mode="valid")


                #TODO: implement manual strided convolution
                if (self.mStride) > 1 :
                    convolved = convolved[::self.mStride, ::self.mStride]
                
                featureMap += convolved
            output[filterIDX] = featureMap + self.mBiases[filterIDX]
            self.zpreactive[filterIDX] = output[filterIDX]
        return output

        
    def backprop(self, _input, _deltaConvolutionActivated, _learningRate, _SGD=False, _verbose=False):
        # expects unpooled gradient
        # TODO: handle shape validation
        # days procastinated to implement it = 8
        _deltaConvolution = _deltaConvolutionActivated * (self.zpreactive > 0).astype(float)

        nablaInput = np.zeros(self.mInputShape)
        for inputChannelIDX in range(nablaInput.shape[0]):
            for kernelIDX in range(self.mFilterCount):
                rotatedKernel = np.rot90(self.mWeights[kernelIDX, inputChannelIDX], 2)
                nablaInput[inputChannelIDX] += signal.convolve2d(
                    _deltaConvolution[kernelIDX], 
                    self.mWeights[kernelIDX, inputChannelIDX], 
                    # rotatedKernel, 
                    "full"
                    )
                
        nablaKernel = np.zeros_like(self.mWeights)
        for deltaConvIDX in range(_deltaConvolution.shape[0]):
            for inputIDX in range(_input.shape[0]):
                nablaKernel[deltaConvIDX, inputIDX] = signal.correlate2d(_deltaConvolution[deltaConvIDX], _input[inputIDX], "valid")

        nablaBias = np.sum(_deltaConvolution, axis=(1, 2))

        if _SGD:
            if self.nablaweightsAcc is None:
                self.nablaBiasesAcc  = np.zeros_like(nablaBias)
                self.nablaweightsAcc = np.zeros_like(nablaKernel)
            self.nablaweightsAcc += nablaKernel
            self.nablaBiasesAcc += nablaBias

        else:
            self.mWeights -= _learningRate * nablaKernel
            self.mBiases  -= _learningRate * nablaBias

        if (_verbose):
            self._debugBackProp(
                _nablaWeightsShape = nablaKernel.shape,
                _nablaBiasessShape = nablaBias.shape,
                _deltaInputShape   = nablaInput.shape,
                _deltaConvShape    = _deltaConvolution.shape
            )

        return nablaInput
    
    def updateParam(self, _learningRate, _batchSize):
        # used for SGD optimization
        # updates all parameteres in one batch
        
        if self.nablaweightsAcc is not None:
            self.mWeights -= _learningRate * (self.nablaweightsAcc/_batchSize)
            self.mBiases  -= _learningRate * (self.nablaBiasesAcc/_batchSize)
            
            # Reset accumulator
            self.nablaweightsAcc = None
            self.nablaBiasesAcc  = None    

    def _LazyInit(self, _inputShape):
        if (self.mLazilyIntialized):
            # print("memeber variables already initialized")
            return
        print("Lazy Initialize Initiated:  ")
        inputChannelDepth, inputWidth, inputHeight = _inputShape

        self.mInputShape       = (inputChannelDepth, inputWidth, inputHeight)
        self.mKernelDepth      = inputChannelDepth
        self.mOutputDimention  = floor(
            (inputWidth - self.mKernelSize) / self.mStride
        ) + 1

        self.mOutputShape      = (self.mFilterCount,
                                  self.mOutputDimention,
                                  self.mOutputDimention) #output shape DxWxH
        self.zpreactive        = np.zeros(self.mOutputShape)
        
        if not self.mInitWithWeights:
            # If model is not loaded, prepare weights and biases

            self.mBiases           = np.zeros(self.mFilterCount)
            self.mWeights          = np.random.randn(self.mFilterCount,
                                                    self.mKernelDepth,
                                                    self.mKernelSize,
                                                    self.mKernelSize) * np.sqrt( 2.0 / (self. mKernelDepth *
                                                                                    self. mKernelSize *
                                                                                    self.mKernelSize)) # He initialization
            self.mInitWithWeights = True
        self.mLazilyIntialized = True

    def _debugBackProp(self, _nablaWeightsShape, _nablaBiasessShape, _deltaInputShape, _deltaConvShape):
        print("*********DIMENTIONS*********")
        print("layer weights:   {}".format(self.mWeights.shape))
        print("nabla weight:    {}".format(_nablaWeightsShape))
        print("layer biases:    {}".format(self.mBiases.shape))
        print("nabla bias:      {}".format(_nablaBiasessShape))
        print("delta input:     {}".format(_deltaInputShape))
        print("input of layert: {}".format(self.mInputShape))
        print("delta conv:      {}".format(_deltaConvShape))

    def _DebugPoolMax(self, _activatedFeature, _pooledLayer,_lastPoolMaxPos):
        print("feature map acitvated: ")
        print(_activatedFeature)
        print("================================")
        print("pooled layer: ")
        print(_pooledLayer)
        print("================================")
        print("pool pos: : ")
        print(_lastPoolMaxPos)