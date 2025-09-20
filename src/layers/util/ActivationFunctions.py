import numpy as np

class ActivationFunction:
    @classmethod
    def sigmoid(cls, z):
        # z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))
    @classmethod
    def sigmoidPrime(cls, z):
        return cls.sigmoid(z) * (1 - cls.sigmoid(z))
    @classmethod
    def ReLU(cls, z): return np.maximum(0, z)
    @classmethod
    def primeReLU(cls, z): return 1.0 * (z > 0)
    @classmethod
    def softmax(cls, z):
        exp_z = np.exp(z - np.max(z))  # stability trick
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)