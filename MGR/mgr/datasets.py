import numpy as np
from torch.utils.data import Dataset
import librosa
import librosa
import warnings
import h5py
import os
from mgr.utils_mgr import getAudio, RandomApply
from sklearn import preprocessing



##############################################################################################################
#2D and 1D DataAudio
##############################################################################################################


class DataAudio(Dataset):

    def __init__(self, df, transform = None, PATH_DATA="data/",  net_type = "1D", test = False, mfcc=False, normalize = False):
        
        # Get track index
        self.track_ids = df['index'].values

        #Get genre label
        self.label = df['labels'].values

        #Transform
        self.transform = transform

        #Select type of input
        self.type = net_type

        #Test
        self.test = test

        #Path to data
        self.path = PATH_DATA

        #mfcc
        self.mfcc = mfcc

        #Normalize
        self.normalize = normalize

    def __len__(self):

        return len(self.track_ids)


    def create_input(self, i):
      
        # Get audio
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio, sr = getAudio(self.track_ids[i], PATH_DATA = self.path)

            #If test select clip window starting at half of the audio
            if(self.test):
                start = int(audio.shape[0]/2)
                audio = audio[start:start+2**18]

            else:
                #Select random clip from audio
                start = np.random.randint(0, (audio.shape[0]-2**18))
                audio = audio[start:start+2**18]
                
            if (self.type=="2D"):
                #Get 2D spectrogram
                stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=1024))
                
                spect = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128)[:,:256]

                if self.mfcc:
                    spect = librosa.feature.mfcc(S=librosa.power_to_db(spect), n_mfcc=20)
                    spect = spect.T
                    
                else:
                    spect = librosa.power_to_db(spect).T    

                return spect
            
            return audio[np.newaxis,:]  
        
    def __getitem__(self, idx):

        # get input and label
        try:
            x = self.create_input(idx)
            y = self.label[idx] 
        except:
            print("\nNot able to load track number ", self.track_ids[idx], " Loading next one\n")
            x = self.create_input(idx+1)
            y = self.label[idx+1]
        
        if self.normalize:
            #Scale data
            scaler = preprocessing.StandardScaler(copy=False)
            x = scaler.fit_transform(x)

        if self.transform:
            
            if self.type=="1D":
                # Audiogmentations library requires to specify the sample rate
                x = self.transform(x, 44100) 
            else:
                x = self.transform(x)
                
        return x,y
    


##############################################################################################################
#MixNet DataAudio
##############################################################################################################


class DataAudioMix(Dataset):

    def __init__(self, df, transform = None, PATH_DATA="data/", test = False, mfcc=False, normalize = False):
        
        # Get track index
        self.track_ids = df['index'].values

        #Get genre label
        self.label = df['labels'].values

        #Transform
        self.transform = transform

        #Test
        self.test = test

        #Path to data
        self.path = PATH_DATA

        #mfcc
        self.mfcc = mfcc

        #Normalize
        self.normalize = normalize

    def __len__(self):

        return len(self.track_ids)


    def create_input(self, i):
      
        # Get audio
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio, sr = getAudio(self.track_ids[i], PATH_DATA = self.path)

            #If test select clip window starting at half of the audio
            if(self.test):
                start = int(audio.shape[0]/2)
                audio = audio[start:start+2**18]

            else:
                #Select random clip from audio
                start = np.random.randint(0, (audio.shape[0]-2**18))
                audio = audio[start:start+2**18]           

            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=1024))
            
            spect = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128)[:,:256]

            if self.mfcc:
                spect = librosa.feature.mfcc(S=librosa.power_to_db(spect), n_mfcc=20)
                spect = spect.T
                
            else:
                spect = librosa.power_to_db(spect).T    

            
            return [audio[np.newaxis,:], spect]
        
    def __getitem__(self, idx):

        # get input and label
        try:
            x = self.create_input(idx)
            y = self.label[idx] 

        except:
            print("\nNot able to load track number ", self.track_ids[idx], " Loading next one\n")
            x = self.create_input(idx+1)
            y = self.label[idx+1]
        
        if self.normalize:
            #Scale data
            scaler = preprocessing.StandardScaler(copy=False)
            x[0] = scaler.fit_transform(x[0])
            x[1] = scaler.fit_transform(x[1])

    
        if self.transform['1D']:
            # Audiogmentations library requires to specify the sample rate
            x[0] = self.transform['1D'](x[0], 44100) 
        if self.transform['2D']:
            x[1] = self.transform['2D'](x[1])

        return x,y




