import numpy as np
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
from torch.optim import Adadelta
from tqdm import tqdm
import gc
import optuna
import pytorch_lightning as pl
import time
import pickle

from utils_mgr import readheavy, get_stft, clip_stft, DataAudio




def mean_std(n_train=1):
    
    print("reading train")
       
    train =  np.load(f"Audio/training_{n_train}.npy", allow_pickle = True)
    
    print("getting stft")
    stft = get_stft(train)
   
    del train
    gc.collect()

    # take each song and splits it into clips of n_samples 
    # creates 
    print("making clips")
    clip = clip_stft(stft, 128)
    print("making clips")
    
    print("stacking")
    stacked = np.stack(clip[:,0],axis=0)
    mean = np.mean(stacked,axis=(0,1,2))
    std = np.std(stacked,axis=(0,1,2))

    print("mean",mean)
    print("std",std)

    return mean,std

if __name__ == "__main__":
    mean,std = import_and_preprocess_data(n_train=2)

