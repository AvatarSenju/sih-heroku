import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import torchvision
import torch
from torch import nn, optim
import random
from torchvision import datasets
# import albumentations as A
import os
import fastai
from fastai.vision import *
from fastai.metrics import error_rate,accuracy,fbeta 
BATCH_SIZE = 64



x=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy']


model_name='export1.pkl'
model_dir=os.getcwd()+ "/models"

learn = load_learner(model_dir,model_name)
img=open_image('appleext.jpg')
pred,pred_idx,probs = learn.predict(img)
print(x[pred_idx])