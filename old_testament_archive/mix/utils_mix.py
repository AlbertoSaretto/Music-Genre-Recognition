import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import os
import pytorch_lightning as pl
import os 
from torch.optim import Adadelta
import h5py

import torchvision.transforms.v2 as v2
import torch
from torchvision.transforms import Compose, ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings

import utils
import utils_mgr
from utils_mgr import getAudio, DataAudio

import pickle


class NNET1D(nn.Module):
        
    def __init__(self):
        super(NNET1D, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2)
        )
        

        self.fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 



# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LN1D(pl.LightningModule):
    
    def __init__(self, config=None):
       
        super().__init__()
        # super(NNET2, self).__init__() ? 
        
        print('Network initialized')
        
        self.net = NNET1D()
        self.val_loss = []
        self.train_loss = []
        self.best_val = np.inf
        

    # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.parameters(), lr = 0.5)


    def forward(self,x):
        return self.net(x)

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch) # Diego nota: aggiungere weights in base a distribuzione classi dataset?
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        x_batch = batch[0]
        label_batch = batch[1]

        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        val_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

            
        #validation_loss = np.append(validation_loss,1 - val_acc)
        #print(f"accuracy: {val_acc*100} %")
        
        # Should I save the model based on loss or on accuracy?7
        """
        LitNet doesnt need to save the model, it is done automatically by pytorch lightning
        if loss.item() < self.best_val:
            print("Saved Model")
            torch.save(self.net.state_dict(), "saved_models/nnet2/model.pt")
            self.best_val = loss.item()
        """


        self.val_loss.append(loss.item())
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        test_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

        self.log("test_loss", loss.item(), prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):

        return self.optimizer
    

class NNET2(nn.Module):
        
    def __init__(self,initialisation="xavier"):
        super(NNET2, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,513)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )
                

        self.fc = nn.Sequential(
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(150, 8),
            nn.Softmax(dim=1)
        )

        # I remove the initialization part because it's not needed


    def forward(self,x):
        
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1)) # Reshape x to fit in linear layers. Equivalent to F.Flatten
        return x 



# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LN2D(pl.LightningModule):
    
    def __init__(self, config=None,initialisation=None):
       
        super().__init__()       
        print('Network initialized')
        
        self.net = NNET2(initialisation)
        self.best_val = np.inf
        
        # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
            print("loaded parameters from pickle")
            print("optimzier parameters:", self.optimizer)
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.parameters())
                print("optimzier parameters:", self.optimizer)
        

    def forward(self,x):
        return self.net(x)

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch) 
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)
        """
        Validation accuracy is computed as follows.
        label_batch are the true labels. They are one-hot encoded (eg [1,0,0,0,0,0,0,0]). 
        out are the predicted labels. They are a 8-dim vector of probabilities.
        argmax checks what is the index with the highest probability. Each index is related to a Music Genre.
        If the indexes are equal the classification is correct.
        
        """
        val_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

        """
        LitNet doesnt need to save the model, it is done automatically by pytorch lightning
        if loss.item() < self.best_val:
            print("Saved Model")
            torch.save(self.net.state_dict(), "saved_models/nnet2/model.pt")
            self.best_val = loss.item()
        """
        # Saves lighting_logs used in Tensorboard.
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        test_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)
        # Saves lighting_logs to track test loss and accuracy. 
        self.log("test_loss", loss.item(), prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer
    

from torch.utils.data import Dataset
import warnings
import librosa
from utils_mgr import getAudio

class DataAudio_double(Dataset):

    def __init__(self, df, transform = None):
        
        # Get track index
        self.track_ids = df['index'].values

        #Get genre label
        self.label = df['labels'].values

        #Transform
        self.transform = transform

    def __len__(self):

        return len(self.track_ids)


    def create_input(self, i):
      
        # Get audio
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio, sr = getAudio(self.track_ids[i])

            #Select random clip from audio
            start = np.random.randint(0, (audio.shape[0]-2**18))
            audio = audio[start:start+2**18]
            
        
            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=513)[:,:128]
            mel = librosa.power_to_db(mel, ref=np.max).T
    
        
            return audio[np.newaxis,:], mel
        
            

    def __getitem__(self, idx):

        # get input and label
        try:
            audio,mel = self.create_input(idx)
            y = self.label[idx] 
        except:
            print("\nNot able to load track number ", self.track_ids[idx], " Loading next one\n")
            audio,mel = self.create_input(idx+1)
            y = self.label[idx]
        

        if self.transform:
            mel = self.transform(mel)
           
        return audio,mel,y



class DataAudioH5_double(Dataset):
    

    def __init__(self, dataset_folder="./h5_experimentation/h5Dataset", dataset_type="train", transform=None,input_type="2D"):
        
        self.x = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_x.h5"),"r")[f"{dataset_type}"]
        self.y = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_y.h5"),"r")[f"{dataset_type}"]
        self.transform = transform
        #Select type of input
        self.type = input_type

    def __len__(self):

        return len(self.x)

    def create_input(self, audio):
      
        # Get audio

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #Select random clip from audio
            start = np.random.randint(0, (audio.shape[0]-2**18))
            audio = audio[start:start+2**18]
            
        
            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            sr = 22050
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=513)[:,:128]
            mel = librosa.power_to_db(mel, ref=np.max).T
    
        
            return audio[np.newaxis,:], mel[np.newaxis,:] #maybe adding newaxis to mel could give problems. I added because debugging without transofrms
        
            

    def __getitem__(self, idx):

        
        # get input and label
    
        x = self.x[idx]
        audio,mel = self.create_input(x)
        y = self.y[idx] 

        if self.transform:
            mel = self.transform(mel)
           
        return audio,mel,y
