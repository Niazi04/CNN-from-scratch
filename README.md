
# CNN-from-scratch

  

A minimal implementation of Convolutional Neural Network using only Numpy and SciPy

  
## Installation
create a new directory and cd into it.
start by cloning the repo:
```bash
git clone https://github.com/Niazi04/CNN-from-scratch/tree/main
```
Intall dependencies:
```bash
python install -r requiremenets.txt
```
## Training Your First Model

For your fist model, we will use the classic MNIST Handwritten digit dataset. We use keras to import the data. Feel free to use whatever method u like.

1. Import all the necessary components

```python
from tensorflow import keras
from src.CNN.util import dataPrep
from src.CNN.CNN import CNN
from src.CNN.layers.ConvolutionLayer import ConvolutionLayer
from src.CNN.layers.FullyConnected import FullyConnected
from src.CNN.layers.MaxPooling import MaxPooling
from src.CNN.modules.loss import crossEntropyLoss
import numpy as np
```

2. Import the dataset
This system only uses CPU and NO cuda. That means its super slow so we only use a small smaple of the dataset to make it fast for a demo

```python
SAMPLES_RANGE = 1000

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()

X, Y = dataPrep.dataPrep_MNISTdigitClassification( X_TRAIN[:SAMPLES_RANGE],
                                                   Y_TRAIN[:SAMPLES_RANGE],
                                                   10
                                                  )
```

3. Create your model


Follow the details to train your model in `trainYourFirstModel.py`
The process is fully explained and limitations are mentioned


## Future Improvements


- [ ] Decouple model creation and training loop (ASAP)
- [ ] Decouple Loss Functions and Optimizers into their own classes
- [ ] Fix pool size problem
- [ ] Clean up random variable naming conventions
- [ ] Add different related optimizers
- [ ] Add different related Loss Functions
- [ ] Implement batch normalizer (similar to DataLoader in Pytorch)