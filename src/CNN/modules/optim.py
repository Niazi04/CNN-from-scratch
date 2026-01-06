from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "SGD"
]

class _optimizer(ABC):
    def __init__(self, _learningRate, ):
        self.lr        = _learningRate
        self.iteration = 0

    @abstractmethod
    def updateParam(self, layer):
        pass

    def step(self):
        self.iteration += 1

    def reset(self):
        self.iteration = 0

class SGD(_optimizer):
    def __init__(self, _learningRate, momentum=0.0):
        super().__init__(_learningRate)
        self.momentum = momentum
        self.velocity = {}
        self.wUpdated = False
        self.bUpdated = False
    
    def updateParam(self, layer):
        self.wUpdated = False
        self.bUpdated = False

        layerID = id(layer)

        gradW, gradB = layer.getGradient()

        if layerID not in self.velocity:
            self.velocity[layerID] = {
                'weights' : np.zeros_like(gradW) if gradW is not None else None,
                'biases'  : np.zeros_like(gradW) if gradB is not None else None 
            }
        
        if gradW is not None:
            velW = self.velocity[layerID]['weights']
            velW = self.momentum * velW - self.lr * gradW
            self.velocity[layerID]['weights'] = velW
            self.wUpdated = True
        if gradB is not None:
            velB = self.velocity[layerID]['biases']
            velB = self.momentum * velW - self.lr * gradB
            self.velocity[layerID]['biases'] = velB
            self.bUpdated = True


        if self.bUpdated or self.wUpdated:
            layer.updateParam(self.velocity[layerID]['weights'], self.velocity[layerID]['biases'])
        self.step()