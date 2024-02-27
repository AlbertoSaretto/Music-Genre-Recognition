import numpy as np
from torch.utils.data import Dataset
import librosa
import librosa
import warnings
import h5py
import os
from mgr.utils_mgr import getAudio


class DataAudio(Dataset):

    def __init__(self, df, transform = None, PATH_DATA="data/",  net_type = "1D", test = False):
        
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

    def __len__(self):

        return len(self.track_ids)


    def create_input(self, i):
      
        # Get audio

        # load audio track
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')

        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio, sr = getAudio(self.track_ids[i], self.path)

            #If test select clip window starting at half of the audio
            if(self.test):
                start = audio.shape[0]/2
                audio = audio[start:start+2**18]

            else:
                #Select random clip from audio
                start = np.random.randint(0, (audio.shape[0]-2**18))
                audio = audio[start:start+2**18]
                
            if(self.type=="2D"):
                #Get 2D spectrogram
                stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
                
                mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=513)[:,:128]
                mel = librosa.power_to_db(mel).T
                return mel
            
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
        

        if self.transform:
            
            if self.type=="1D":
                 # Audiogmentations library requires to specify the sample rate
                 x = self.transform(x,44100) # Using 44100, I should make this more robust using sr from previous function
            else:
                x = self.transform(x)
           
        return x,y




class DataAudioH5(Dataset):

    def __init__(self, dataset_folder="./h5_experimentation/", dataset_type="train", transform=None,input_type="2D"):
        
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
