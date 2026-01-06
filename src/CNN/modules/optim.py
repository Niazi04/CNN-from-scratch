from abc import ABC, abstractmethod

__all__ = [
    
]

class _optimizer(ABC):
    def __init__(self, _learningRate, ):
        self.lr        = _learningRate
        self.iteration = 0

    @abstractmethod
    def updatePAram(self):
        pass

    def step(self):
        self.iteration += 1

    def reset(self):
        self.iteration = 0

