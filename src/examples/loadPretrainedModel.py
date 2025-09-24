from src.layers.CNN import CNN
import matplotlib.pyplot as plt
from tensorflow import keras
from src.layers.util import dataPrep

(_xjunk, _yjunk), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()

# keep the script light
del _xjunk
del _yjunk

SAMPLES_RANGE = 500
X, Y = dataPrep.dataPrep_MNISTdigitClassification( X_TEST[:SAMPLES_RANGE],
                                                   Y_TEST[:SAMPLES_RANGE],
                                                   10
                                                   )


model = CNN.loadModel("LookMyFirstModel.npz")

IDX = 68
im = X[IDX]
im = im.squeeze()

plt.imshow(im, cmap="gray")
plt.title(f"Label: {Y[IDX]}")
plt.axis("off")
plt.show()
print(model.whatThisPic(X[IDX]))
