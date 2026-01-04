
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


Currently the code base and model creation is highly coupled. This will be fixed in future 
updates.

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