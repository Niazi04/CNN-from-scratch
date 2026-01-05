import numpy as np

class MSELoss:
    def __init__(self, reduction=None):
        self.reduction = reduction

    def forward(self, _ypred, _ytrue):
        return np.mean((_ypred.flatten() - _ytrue) ** 2)
        #TODO: handle redutction after implementing auto batch
        

    def __call__(self, _ypred, _ytrue):
        return self.forward( _ypred, _ytrue)   

class crossEntropyLoss:
    def __init__(self, reduction=None):
        self.reduction = reduction
    def forward(self, _yPred, _yTrue):

        # one-hot-encoded
        # TODO: single class lables
        _yPred = np.clip(_yPred.flatten(), 1e-10, 1 - 1e-10)
        _yTrue = _yTrue.flatten()
        loss = -np.sum(_yTrue * np.log(_yPred))
        return loss
        # TODO: enabel this after I implemented auto batch
        # if self.reduction == "mean":
        #     return np.mean(loss)
        # elif self.reduction == "sum":
        #     return np.sum(loss)
        # else:
        #     return loss
    def __call__(self, _yPred, _yTrue):
        return self.forward( _yPred, _yTrue)