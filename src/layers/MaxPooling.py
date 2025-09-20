import numpy as np

class MaxPooling:
    def __init__(self, _poolSize):
        self.mPoolSize        = _poolSize
        self.mOutputDimention = None
        self.mLastPoolMax     = None
        self.mFilterCount     = None
        self.mOutputShape     = None
        self.mLazyInit        = False

    def _lazyInit(self, _outDimention, _filterCount):
        if (self.mLazyInit):
            # print("Pooling Layer already initialized")
            return
        
        print("Max Pooling lazyly initialized")

        # self.mOutputDimention =  _outDimention // self.mPoolSize
        self.mOutputDimention =  _outDimention
        self.mFilterCount     = _filterCount
        self.mOutputShape      = (self.mFilterCount,
                                  self.mOutputDimention,
                                  self.mOutputDimention) #output shape DxWxH
        self.mLastPoolMax      = np.zeros(self.mOutputShape)
        self.mLazyInit        = True

    def poolMax(self, _activatedFeature,  _verbose=False):
        # output of pooling will have a dimention of ( feature_d // pool_d)
        # since feature_d will be the same after activation, we dont need to
        # get the dimentions of _activatedFeature
        # pool_d is also the number of iterations needed to find the max value
        # for padding=0

        # TODO: handle data loss

        self._lazyInit(
            _outDimention =_activatedFeature.shape[1],
            _filterCount  =_activatedFeature.shape[0]
            )

        poolOutDimention = self.mOutputDimention//self.mPoolSize
        pooled = np.zeros((self.mFilterCount,poolOutDimention, poolOutDimention))
        for k in range(self.mFilterCount):
            for i in range(poolOutDimention):
                for j in range(poolOutDimention):
                    receptiveField = _activatedFeature[
                        k,
                        i*self.mPoolSize:i*self.mPoolSize + self.mPoolSize,
                        j*self.mPoolSize:j*self.mPoolSize+self.mPoolSize
                        ]
                    maxVal = np.max(receptiveField)
                    pooled[k, i, j] = maxVal

                    # store the positation of max element
                    # needed for unpooling

                    maxPos = (receptiveField == maxVal)
                    self.mLastPoolMax[
                        k,
                        i*self.mPoolSize:i*self.mPoolSize + self.mPoolSize,
                        j*self.mPoolSize:j*self.mPoolSize+self.mPoolSize
                        ] = maxPos
        
        if (_verbose):
            self._DebugPoolMax(_activatedFeature, 
                               pooled,
                               self.mLastPoolMax)
            
        return pooled
    
    def unpool(self, _poolGradient):
        unpooledGradient = np.zeros_like(self.mLastPoolMax)
        poolOutDimention = self.mOutputDimention // self.mPoolSize

        for k in range(self.mFilterCount):
            for i in range(poolOutDimention):
                for j in range(poolOutDimention):
                    receptiveSlice = np.s_[
                        k,
                        i*self.mPoolSize:i*self.mPoolSize + self.mPoolSize,
                        j*self.mPoolSize:j*self.mPoolSize + self.mPoolSize
                        ]
                    unpooledGradient[receptiveSlice] = _poolGradient[k, i, j] * self.mLastPoolMax[receptiveSlice]
        return unpooledGradient
    
    def _DebugPoolMax(self, _activatedFeature, _pooledLayer,_lastPoolMaxPos):
        print("feature map acitvated: ")
        print(_activatedFeature)
        print("================================")
        print("pooled layer: ")
        print(_pooledLayer)
        print("================================")
        print("pool pos: : ")
        print(_lastPoolMaxPos)