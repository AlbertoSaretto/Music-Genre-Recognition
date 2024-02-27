import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import librosa
from tensorflow.keras.utils import to_categorical
import mgr.utils as utils
import librosa
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import warnings
import h5py
import os
import gc

#Function to extract audio signal from .mp3 file
def getAudio(idx, sr_i=None, AUDIO_DIR = '/home/diego/Music-Genre-Recognition/data/fma_small/'):
    #Get the audio file path
    filename = utils.get_audio_path(AUDIO_DIR, idx)

    #Load the audio (sr = sampling rate, number of audio carries per second)
    x, sr = librosa.load(filename, sr=sr_i, mono=True)  #sr=None to consider original sampling rate

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
            np.save(f'data/audio_array/{name}_{i+1}.npy', data_i)
    data_i = data[int((l/n)*(n-1)):l]
    np.save(f'data/audio_array/{name}_{n}.npy', data_i)


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
    

def get_mel(a, get_log = False):
    if(get_log == False):
        for j in range(len(a)):
            audio = a[j, 0]
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            mel = librosa.feature.melspectrogram(sr=44100, S=stft**2, n_mels=513)[:,:128]     
            a[j ,0] = stft 
        return a    
    if(get_log == True):
        for j in range(len(a)):
            audio = a[j, 0]
            stft = np.abs(librosa.stft(audio, n_fft=4096, hop_length=2048))
            mel = librosa.feature.melspectrogram(sr=44100, S=stft**2, n_mels=513)[:,:128]   
            mel = librosa.power_to_db(mel) 
            a[j ,0] = mel
        return a
    

def get_mel_from_clip(c_stft):
    for j in range(len(c_stft)):
        clip = c_stft[j,0]
        mel = librosa.feature.melspectrogram(sr=44100, S=clip**2, n_mels=513)
        c_stft[j,0] = librosa.power_to_db(mel)
    return c_stft   




#Function to generate shorter data clips 

def clip_mel(a, n_samples):
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

def mean_2D_mel(dataset):
    
    print("getting stft")
    stft = get_stft(dataset)
   
    del dataset
    gc.collect()

    # take each song and splits it into clips of n_samples 
    # creates 
    print("making clips")
    clip = clip_mel(stft, 128)
    print("making clips")
    
    print("stacking")
    stacked = np.stack(clip[:,0],axis=0)
    mean = np.mean(stacked,axis=(0,1,2))
    std = np.std(stacked,axis=(0,1,2))

    return mean,std

def mean_1D(dataset):

    # take each song and splits it into clips of n_samples 
    # creates 
    print("making clips")
    clip = clip_audio(dataset, 2**18)
    print("making clips")
    
    print("stacking")
    stacked = np.stack(clip[:,0],axis=0)
    mean = np.mean(stacked,axis=(0,1,2))
    std = np.std(stacked,axis=(0,1,2))
    
    return mean,std


def import_and_preprocess_data(PATH_DATA="data/"):
    # Load metadata and features.
    tracks = utils.load(PATH_DATA+'fma_metadata/tracks.csv')


    #Select the desired subset among the entire dataset
    sub = 'small'
    raw_subset = tracks[tracks['set', 'subset'] <= sub] 
    
    #Creation of clean subset for the generation of training, test and validation sets
    meta_subset= create_subset(raw_subset)

    # Remove corrupted files
    corrupted = [98565, 98567, 98569, 99134, 108925, 133297]
    meta_subset = meta_subset[~meta_subset['index'].isin(corrupted)]

    #Split between taining, validation and test set according to original FMA split

    train_set = meta_subset[meta_subset["split"] == "training"]
    val_set   = meta_subset[meta_subset["split"] == "validation"]
    test_set  = meta_subset[meta_subset["split"] == "test"]

    return train_set, val_set, test_set


def display_mel(idx, n_samples, n_fft, n_mels, time_bin, sr_i=None):

    audio, sr = getAudio(idx, sr_i)

    #Select random clip from audio
    start = np.random.randint(0, (audio.shape[0]-n_samples))
    audio = audio[start:start+n_samples]
    
    #Get 2D spectrogram
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=int(n_fft/2)))              
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=n_mels)
    print('Original mel spectrogram (transpose shape is: ', mel.T.shape)
    mel = mel[:,:time_bin]
    mel = librosa.power_to_db(mel)
    
    # Plot the log mel spectrogram for visualization purpose 
    plt.figure(figsize=(7, 3))
    librosa.display.specshow(mel, sr=sr, hop_length=int(n_fft/2), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    return 0

def create_dataloaders(PATH_DATA="../data/",transforms=None,batch_size=64,num_workers=os.cpu_count(),net_type='1D'):
    from mgr.datasets import DataAudio
    from torch.utils.data import DataLoader
    
    train_set, val_set, test_set = import_and_preprocess_data(PATH_DATA)

    train_dataset  = DataAudio(train_set, transform = transforms, net_type=net_type)
    val_dataset    = DataAudio(val_set, transform = transforms, net_type=net_type)
    test_dataset   = DataAudio(test_set, transform = transforms, net_type=net_type, test = True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader

"""
compute_CM_terms and compute_metrics are used to compute the confusion matrix and the metrics

"""

def compute_CM_terms(out_net, label_batch):
    import torch
    import torch.nn.functional as F
    
    '''

    TO DO: CHECK IF SKLEARN METRICS ARE THE SAME

    Function to compute terms for the evaluation of confusion matrix and different metrics:

    TP: True Positive, the cases in which we predicted YES and the actual output was also YES.
    FP: False Positive, the cases in which we predicted YES and the actual output was NO.
    TN: True Negative, the cases in which we predicted NO and the actual output was also NO.
    FN: False Negative, the cases in which we predicted NO and the actual output was YES.
    '''

    out_pred = out_net.argmax(dim=1)
    out_pred_bool = F.one_hot(out_pred, num_classes=8).bool()
    
    # logical_and is element wise 'and', zeros element are always false
    # 0 == 0 -> False
    TP = torch.logical_and(out_pred_bool, label_batch).sum(dim=0)
    FP = torch.logical_and(out_pred_bool, ~label_batch).sum(dim=0)
    TN = torch.logical_and(~out_pred_bool, ~label_batch).sum(dim=0)
    FN = torch.logical_and(~out_pred_bool, label_batch).sum(dim=0)
    
    
    return TP, FP, TN, FN

def compute_metrics(out_net, label_batch):
    TP, FP, TN, FN = compute_CM_terms(out_net, label_batch)

    # Compute metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


#Function for main training of the network

def main_train(model_net, max_epochs=1, optimizer=None,  lr=1, 
               config=None, transforms=None, net_type='1D',batch_size=64,num_workers=os.cpu_count()):
    
    import pytorch_lightning as pl
    from mgr.models import LitNet
    import torch

    pl.seed_everything(0)

    if optimizer is None:
        optimizer = torch.optim.Adam(model_net.parameters(), lr=lr)
      
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=10,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )

    trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=5, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], ) # profiler="simple" remember to add this and make fun plots
    
    model = LitNet(model_net, lr=lr, config=config)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(transforms=transforms, net_type=net_type,batch_size=batch_size,num_workers=num_workers)

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)

    return model