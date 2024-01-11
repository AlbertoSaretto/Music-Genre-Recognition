import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import librosa
from tensorflow.keras.utils import to_categorical
import utils
import librosa
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import warnings



#Function to extract audio signal from .mp3 file
def getAudio(idx, AUDIO_DIR = 'Data/fma_small'):
    #Get the audio file path
    filename = utils.get_audio_path(AUDIO_DIR, idx)

    #Load the audio (sr = sampling rate, number of audio carries per second)
    x, sr = librosa.load(filename, sr=22050, mono=True)  #sr=None to consider original sampling rate

    return x, sr


#Function to convert labels into array of probabilities
def conv_label(label):
    le = LabelEncoder()
    y = le.fit_transform(label)
    y = to_categorical(y, 8)
    return y


#This function create a dataframe containing index and label for a subset, simplifying its structure and making it suitable 
#for the split into test, validation and validation sets, moreover labels are converted into numpy arrays
def create_subset(raw_sub):
    labels = conv_label(raw_sub['track']['genre_top']) 
    subset = pd.DataFrame(columns = ['index', 'genre_top', 'split', 'labels'])
    
    for j in range(len(raw_sub['track'].index)):
        index = raw_sub['track'].index[j]
        genre = raw_sub['track', 'genre_top'][index]
        split = raw_sub['set', 'split'][index]
        label = labels[j]
        # Create a new row as a Series
        new_row = pd.Series({'index': index, 'genre_top': genre, 'split':split, 'labels':label})
        subset.loc[len(subset)] = new_row
        
    return subset


#function for the creation of datasets arrays for CNNs learning
#Function for the creation of 1d audio signal dataset

def splitAudio(subset, data_type):
 
    data = np.array([0,0])
    
    #Select data type among full subset
    split = subset[subset['split'] == data_type]

    for index, row in split.iterrows(): 
        try: 
            audio = np.array(getAudio(row['index'])[0])
            y = row['labels']
            new_row = np.array([audio, y], dtype=object)
            data = np.vstack([data, new_row])
        except:
            print('problems with song: ', row['index'])
    
    #Remove first empty row
    data = data[1:]
    
    #Shuffle the dataset
    np.random.shuffle(data)
    
    return data


#Functions for the creation of .npy files for heavy arrays (split arrays in multiple files and then reconstrict it)

def saveheavy(data, name, n):  #functions to save havy arrays in multiple files
    l= len(data)
    if(n>1):
        for i in range(n-1):
            data_i = data[int((l/n)*i):int((l/n)*(i+1))]
            np.save(f'Data/Audio/{name}_{i+1}.npy', data_i)
    data_i = data[int((l/n)*(n-1)):l]
    np.save(f'Data/Audio/{name}_{n}.npy', data_i)


def readheavy(name, n, Dir):
    data = np.array([0,0])
    for i in range(n):
        new_row = np.load(f'{Dir}/{name}_{i+1}.npy', allow_pickle = True)
        data = np.vstack([data, new_row])
    return data[1:]  



def get_stft(a, get_log = False):
    if(get_log == False):
        for j in range(len(a)):
            audio = a[j, 0]
            stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))   
            a[j ,0] = stft 
        return a    
    if(get_log == True):
        for j in range(len(a)):
            audio = a[j, 0]
            stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))   
            log_stft = librosa.amplitude_to_db(stft)
            a[j ,0] = log_stft
        return a
    

def get_mel_from_clip(c_stft):
    for j in range(len(c_stft)):
        clip = c_stft[j,0]
        mel = librosa.feature.melspectrogram(sr=22050, S=clip**2, n_mels=513)
        c_stft[j,0] = librosa.power_to_db(mel)
    return c_stft   



def get_mfcc(a):
        for j in range(len(a)):
            audio = a[j, 0]
            stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512)) 
            mel = librosa.feature.melspectrogram(sr=22050, S=stft**2)  
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
            mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
            a[j ,0] = stft 
        return a




#Function to generate shorter data clips 

def clip_stft(a, n_samples):
    a_clip = np.array([0,0])
    for j in range(len(a)):
        full = a[j, 0].T
        n=0
        while (n<(len(full)-n_samples)):
            clip = full[n: (n+n_samples)]
            y = a[j, 1]
            new_row = np.array([clip, y], dtype=object)
            a_clip = np.vstack([a_clip, new_row])
            n+=int(n_samples/2)
    return a_clip[1:]


def clip_audio(df, n_samples):
    data_clip = np.array([0,0])
    for j in tqdm(range(len(df)), desc="Processing clips"):
        full = df[j, 0]
        n=0
        #while (n<(len(full)-n_samples)) and n<4096:
        while (n<(len(full)-n_samples)):
            clip = full[n: (n+n_samples)]
            y = df[j, 1]
            new_row = np.array([clip[np.newaxis,:], y], dtype=object)
            data_clip = np.vstack([data_clip, new_row])
            n+=int(n_samples/2)
            
           
    return data_clip[1:]

"""
#Class for the creation of torch manageble datasets, with Format one can select the desired input column 
class DataAudio(Dataset):

    def __init__(self, data, transform=None):
        self.x = data[:,0]
        self.y = data[:,1]
        self.transform = transform

    def __len__(self):
       
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)  # why not transofrming y?
            
        return x, y
"""


class DataAudio(Dataset):

    def __init__(self, df, transform = None, type = "1D"):
        
        # Get track index
        self.track_ids = df['index'].values

        #Get genre label
        self.label = df['labels'].values

        #Transform
        self.transform = transform

        #Select type of input
        self.type = type



    def __len__(self):

        return len(self.track_ids)


    def create_input(self, i):

        # Get audio

        # load audio track
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio, sr = getAudio(self.track_ids[i])

        #Select random clip from audio
        start = np.random.randint(0, (audio.shape[0]-2**17))
        audio = audio[start:start+2**17]
        
        if self.type ==  "2D":

            #Get 2D spectrogram
            stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=1024))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mel = librosa.feature.melspectrogram(sr=22050, S=stft**2, n_mels=513)[:,:128]
                mel = librosa.power_to_db(mel).T
            return mel
        
        return audio[np.newaxis,:]



    def __getitem__(self, idx):

        # get input and label

        x = self.create_input(idx) 
        y = self.label[idx]

        if self.transform:
            x = self.transform(x)

        return x,y


    

def pca_transform(clips, n_components=0.99):

    from sklearn.decomposition import PCA

    """
    It's necessary to manipulate the data in order to apply PCA, since it requires a 2D array as input
    First X is extracted, then it's reshaped in a 2D array with vstack,
    then PCA is applied and finally the data is reshaped again
    
    Examples of shapes, using import_and_preprocess_data with window of 22050/10

    clip.shape = (59243,2)
    X.shape = (59243,)
    np.vstack(X).shape = (59243, 2205)
    pca.fit_transform(X).shape = (59243, 1237)

    So to use a trick to substitue all X with X_ in the original dataset
    it's necessary to np.split(X_, 59243, axis=0)

    """
    # Extract X
    X = clips[:,0]
    # Reshape X in a 2D array
    X = np.vstack(X)
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_ = pca.fit_transform(X)
    
    # Substitute X with X_ in the original dataset
    clips[:,0] = np.split(X_, clips.shape[0], axis=0)

    return clips
