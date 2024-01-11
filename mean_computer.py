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


"""
Script per processare i dati e salvarli in un dataloader.
Lo script prende le clip, le trasforma in stft e applica data augmentation e normalizzazione con torchvision.transforms.v2
In output ottengo un dataloader che posso usare per addestrare la rete. Il dataloader viene salvato in un file pickle. 
Tieni d'occhio il nome con cui salvi tale file per non sovrascrivere con file precedenti.

Lo script si divide in due parti: 
1) import_and_preprocess_data: legge i dati e crea le clip
2) preprox: applica le trasformazioni e crea il dataloader.

Dato che questo script è fatto di fretta e furia, bisogna modificarlo manualmente all'uopo.
Il mio pc è in grado di elaborare un solo file alla volta.
Ecco quindi che è necessario commentare tutte le reference a "train" o "valid" a seconda che si voglia elaborare il train o il validation set.
(eg se voglio creare il dataloader per il train, commento tutte le reference a "valid" e viceversa)

Ho fatto due funzioni diverse perché pare che in questo modo la memoria venga leggermente alleggerita. 
Potrebbe essere che a volte il kernel muoia lo stesso, ma stando attenti a chiudere/riaprire, spegnere/riaccendere dovrebbe funzionare (16GB di RAM richiesti)
"""


def import_and_preprocess_data(n_train=1):
    
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

