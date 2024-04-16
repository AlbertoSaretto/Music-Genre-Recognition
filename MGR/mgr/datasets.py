import numpy as np
from torch.utils.data import Dataset
import librosa
import librosa
import warnings
import h5py
import os
from mgr.utils_mgr import getAudio, RandomApply
from sklearn import preprocessing




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

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        
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
                 x = self.transform(x, 44100) # Using 44100, I should make this more robust using sr from previous function
        return x,y
    



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

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        
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
            x[0] = self.transform['1D'](x[0], 44100) # Using 44100, I should make this more robust using sr from previous function
        if self.transform['2D']:
            x[1] = self.transform['2D'](x[1])

        return x,y








'''
class DataAudioH5(Dataset):

    def __init__(self, dataset_folder="./h5_experimentation/", dataset_type="train", transform=None,input_type="2D"):
        
        self.x = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_x.h5"),"r")[f"{dataset_type}"]
        self.y = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_y.h5"),"r")[f"{dataset_type}"]
        self.transform = transform
        #Select type of input
        self.type = input_type

    def __len__(self):

        return self.x.len()

    def create_input(self, audio, sr=22050):

        """
        This function takes an audio clip and creates the input for the model
        """
      
        # Get audio

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        #Select random clip from audio
        start = np.random.randint(0, (audio.shape[0]-2**18))
        audio = audio[start:start+2**18]
        
        if self.type ==  "2D":
            
            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=513)[:,:128]
            mel = librosa.power_to_db(mel).T
            return mel
    
        return audio[np.newaxis,:]



    def __getitem__(self, idx):

        # get input and label

        audio = self.x[idx]
        x = self.create_input(audio)
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
           
        return x,y



class DataAudioH5_colab(Dataset):

    def __init__(self, file_x,file_y,dataset_folder="./h5_experimentation/", dataset_type="train", transform=None,input_type="2D"):
        
        self.x = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_x.h5"),"r")[f"{dataset_type}"]
        self.y = h5py.File(os.path.join(dataset_folder, f"{dataset_type}_y.h5"),"r")[f"{dataset_type}"]
        self.transform = transform
        #Select type of input
        self.type = input_type

    def __len__(self):

        return self.x.len()

    def create_input(self, audio,sr=22050):

        """
        This function takes an audio clip and creates the input for the model
        """
      
        # Get audio

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        #Select random clip from audio
        start = np.random.randint(0, (audio.shape[0]-2**18))
        audio = audio[start:start+2**18]
        
        if self.type ==  "2D":
            
            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=513)[:,:128]
            mel = librosa.power_to_db(mel).T
            return mel[np.newaxis,:]
    
        return audio[np.newaxis,:]



    def __getitem__(self, idx):

        # get input and label

        audio = self.x[idx]
        x = self.create_input(audio)
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
           
        return x,y
'''